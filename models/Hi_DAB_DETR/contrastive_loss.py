"""Contrastive loss."""

__all__ = [
    "get_lambda_coef",
    "loss_ptp",
    "loss_stp",
    "loss_sts",
]
from typing import Literal

import torch

from models.Hi_DAB_DETR.pmath import dist_matrix


def preprocess(
        lambda_coef: torch.Tensor,
        temperature: torch.Tensor,
        n_levels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess temperature and lambda coefficient tensors.

    Parameters
    ----------
    lambda_coef : torch.Tensor
        Tensor of shape (1,) or (L,) representing per-level weights.
    temperature : torch.Tensor
        Tensor of shape (1,) or (L,) representing per-level temperatures.
    n_levels : int
        Number of hierarchy levels (L).

    Returns
    -------
    temperature : torch.Tensor
        Expanded tensor of shape (1, 1, L) ready for broadcasting.
    lambda_coef : torch.Tensor
        Tensor of shape (L,) with validated per-level weights.

    Raises
    ------
    ValueError
        If shapes of `temperature` or `lambda_coef` do not match
        the expected dimensions.
    """
    # Check temperature
    if temperature.numel() == 1:
        temperature = temperature.expand(n_levels)
    if temperature.numel() != n_levels:
        msg = (f"Expected temperature of shape ({n_levels},), "
               f"but got {temperature.shape}")
        raise ValueError(msg)
    temperature = temperature.view(1, 1, n_levels)

    # Check lambda_coef
    if lambda_coef.numel() == 1:
        lambda_coef = lambda_coef.expand(n_levels)
    if lambda_coef.shape != (n_levels,):
        msg = (f"Expected lambda_coef of shape ({n_levels},), "
               f"but got {lambda_coef.shape}")
        raise ValueError(msg)

    return temperature, lambda_coef


def weighted_levelwise_loss(
        logits: torch.Tensor,
        positives: torch.Tensor,
        lambda_coef: torch.Tensor,
        n_levels: int,
        hierarchical_constraint: bool,  # noqa: FBT001
        level_mask: torch.Tensor | None,
    ) -> torch.Tensor:
    """
    Compute the weighted contrastive loss over multiple hierarchy levels.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape (B, *, L). Logits over candidates for each sample and
        level.
    positives : torch.Tensor
        Boolean tensor of shape (B, *, L). Mask indicating positive positions.
    lambda_coef : torch.Tensor
        Tensor of shape (L,). Weight assigned to each level.
    n_levels : int
        Total number of hierarchy levels.
    hierarchical_constraint : bool
        If `True`, applies the hierarchical constraint across levels.
        Only relevant for STP loss.
    level_mask : torch.Tensor or None
        Boolean tensor of shape (1, N, L), where N is the number of nodes and
        L is the number of levels. Indicates which nodes belong to each level
        in the hierarchy. Must be provided (not `None`) if
        hierarchical_constraint is `True`.

    Returns
    -------
    torch.Tensor
        Scalar loss value: weighted sum of per-level losses, normalized by
        number of levels.
    """
    # (B, *, L) - log_probs[i, j, l] log-softmax of the logits over the
    # candidate dimension (dim=1) for each sample i and level l.
    # 'j' indexes candidates.
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Hierarchical constraint
    if hierarchical_constraint:
        # (L,) - Number of nodes per level
        num_nodes_per_level = level_mask[0].sum(dim=0)
        # (L,) - Boolean vector: True if level l has ≥ 2 nodes
        valid_levels = (num_nodes_per_level >= 2)  # noqa: PLR2004
        # (B, N, L) - Broadcastable mask
        valid_levels_mask = valid_levels.view(1, 1, -1).expand_as(log_probs)
        # (B, N, L) - Mask non-positive entries with -inf to exclude them
        masked_log_probs = log_probs.masked_fill(
            ~positives | ~valid_levels_mask, float("-inf"))
        # (B, N, L) - cumulative max over previous levels
        # (excluding current level)
        cummax_vals = masked_log_probs.cummax(dim=2).values
        past_max = torch.full_like(cummax_vals, float("-inf"))
        past_max[:, :, 1:] = cummax_vals[:, :, :-1]
        # (B, L) - For each (i, l), get the max over all nodes j
        global_past_max = past_max.max(dim=1).values
        global_past_max = global_past_max.unsqueeze(1).expand_as(log_probs)
        # (B, N, L) - Apply the hierarchical constraint: for each (i, n, l),
        # ensure the log-prob is at least as large as the max log-prob
        # of positive nodes at previous levels.
        log_probs = torch.maximum(log_probs, global_past_max)

    # (B, *, L) - Zero out log-probs for non-positive positions.
    selected_log_probs = log_probs.masked_fill(~positives, 0.0)

    # (B, L) - Sum of log-probabilities of all positive candidates
    pos_log_probs_sum = selected_log_probs.sum(dim=1)
    n_pos = positives.sum(dim=1)
    # (B, L) - valid[i,l]: anchor i has at least one positive at level l.
    valid = (n_pos > 0)
    loss_per_sample_per_level = torch.zeros_like(pos_log_probs_sum)
    # (B, L) - Compute the negative mean across positive candidates.
    # Zero where no positives are available.
    loss_per_sample_per_level[valid] = -(
        pos_log_probs_sum[valid] / n_pos[valid])
    # (L,) - loss_per_level[l]: Sum of the per-anchor losses at level l.
    loss_per_level = loss_per_sample_per_level.sum(dim=0)
    # Weighted sum: each level l is weighted by its corresponding
    # lambda_coef[l]. The sum is then divided by the number of levels.
    return (lambda_coef * loss_per_level).sum() / n_levels


def loss_sts(
        node_ids: torch.Tensor,
        values: torch.Tensor,
        store_full_visit_path: torch.Tensor,
        store_count: torch.Tensor,
        store_memory: torch.Tensor,
        store_device: torch.device,
        lambda_coef: torch.Tensor,
        temperature: torch.Tensor,
        sampling_from_store: bool,  # noqa: FBT001
        k_pos: int | None = None,
    ) -> torch.Tensor:
    """
    Compute the sample-to-sample contrastive loss.

    Parameters
    ----------
    node_ids : torch.Tensor
        Tensor of shape (B,) containing the node IDs associated
        with each value.
    values : torch.Tensor
        Tensor of shape (B, D) containing the feature embeddings.
    store_full_visit_path : torch.Tensor
        Tensor of shape (N, P) containing the full visit path for each node.
    store_count : torch.Tensor
        Tensor of shape (N,) with the number of valid samples per node.
    store_memory : torch.Tensor
        Tensor of shape (N, M, D) representing the memory buffer.
    store_device : torch.device
        Device where the store data is allocated.
    lambda_coef : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level weights.
    temperature : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level temperatures.
    sampling_from_store : bool
        Whether to sample additional positives from the replay buffer.
    k_pos : int | None, optional
        Desired minimum number of positive samples per anchor per level.
        Only used if `sampling_from_store` is `True.
        The default is `None`.

    Returns
    -------
    torch.Tensor
        Scalar sample-to-sample contrastive loss.
    """
    # (N, L) - Full visit paths for all nodes in the hierarchy.
    path = store_full_visit_path
    # L: Number of hierarchy levels
    n_levels = path.shape[1]
    # Preprocess
    temperature, lambda_coef = preprocess(
        lambda_coef=lambda_coef,
        temperature=temperature,
        n_levels=n_levels,
    )

    # (B, L) - Full visit paths for each node in the batch
    gamma = path[node_ids]
    if sampling_from_store:
        # (B * L,) - Valid node IDs (excluding -1) used to identify sampling
        # targets
        sample_from_nodes = gamma[gamma != -1]
        # (U,), (U,) - For each unique node with x occurrences in gamma,
        # each of the x samples will have x-1 positives at the corresponding
        # level. We sample (k_pos - x + 1) additional values per node
        # to ensure each sample reaches at least `k_pos` positives.
        # Note: with batch size = 2, setting k_pos ≥ 2 is necessary
        # to avoid STS loss = 0.
        unique, count = torch.unique(sample_from_nodes, return_counts=True)
        # (U,) - Number of values to sample per node, clamped to the actual
        # number of stored elements.
        n_values_per_node = torch.clamp(
            k_pos - count + 1, min=torch.zeros_like(count),
            max=store_count[unique]
            )
        if n_values_per_node.sum() > 0:
            # For each node `u`, randomly permute its buffer
            # entries and select the first `n` samples
            sampled_values, sampled_node_ids = zip(*[
                (store_memory[u, torch.randperm(store_count[u],
                                                device=store_device)[:n]],
                u.repeat(n))
                for u, n in zip(unique, n_values_per_node) if n > 0
            ])
            # Update values, node_ids, and hierarchy paths.
            values = torch.cat([values, *sampled_values], dim=0)
            node_ids = torch.cat([node_ids, *sampled_node_ids], dim=0)
            gamma = path[node_ids]

    # B: Batch size
    batch_size = node_ids.shape[0]

    # Positives
    # (B, 1, L)  # noqa: ERA001
    gamma_i = gamma.unsqueeze(1)
    # (1, B, L)  # noqa: ERA001
    gamma_j = gamma.unsqueeze(0)
    # (B, B, L)  # noqa: ERA001
    same_node = (gamma_i == gamma_j)
    valid_i = (gamma_i != -1)
    valid_j = (gamma_j != -1)
    valid = valid_i & valid_j
    same_node = same_node & valid
    # No self-matching
    identity_mask = torch.eye(batch_size, dtype=torch.bool,
                              device=store_device).unsqueeze(-1)
    # (B, B, L) - positives[i,j,l] = True if i and j are positives at level l,
    # excluding self-matching.
    positives = same_node & (~identity_mask)

    # Distances
    # (B, B) - Tensor containing pairwise hyperbolic distances
    dists = dist_matrix(values, values)
    # (B, B, L)  # noqa: ERA001
    dists = dists.unsqueeze(-1).expand(-1, -1, n_levels)
    # (B, B, L) - Scale dists by the corresponding temperature.
    logits = -dists / temperature
    # (B, B, L) - Set logits[i,i,l] = -inf to remove self-matching from
    # the softmax denominator.
    logits = logits.masked_fill(identity_mask, float("-inf"))

    return weighted_levelwise_loss(
        logits=logits,
        positives=positives,
        lambda_coef=lambda_coef,
        n_levels=n_levels,
        hierarchical_constraint=False,
        level_mask=None,
    )


def loss_stp(
        node_ids: torch.Tensor,
        values: torch.Tensor,
        store_full_visit_path: torch.Tensor,
        store_mean: torch.Tensor,
        store_device: torch.device,
        lambda_coef: torch.Tensor,
        temperature: torch.Tensor,
        hierarchical_constraint: bool,  # noqa: FBT001
    ) -> torch.Tensor:
    """
    Compute the sample-to-prototype contrastive loss.

    Parameters
    ----------
    node_ids : torch.Tensor
        Tensor of shape (B,) containing the node IDs associated
        with each value.
    values : torch.Tensor
        Tensor of shape (B, D) containing the feature embeddings.
    store_full_visit_path : torch.Tensor
        Tensor of shape (N, P) containing the full visit path for each node.
    store_mean : torch.Tensor
        Tensor of shape (N, D) containing the current mean embedding
        for each node.
    store_device : torch.device
        Device where the store data is allocated.
    lambda_coef : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level weights.
    temperature : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level temperatures.
    hierarchical_constraint : bool
        If `True`, applies hierarchical constraint across levels.

    Returns
    -------
    torch.Tensor
        Scalar sample-to-prototype contrastive loss.
    """
    # (N, L) - Full visit paths for all nodes in the hierarchy.
    path = store_full_visit_path
    # L: Number of hierarchy levels
    n_levels = path.shape[1]
    # N: Number of hierarchy nodes
    n_nodes = path.shape[0]
    # B: Batch size
    batch_size = node_ids.shape[0]
    # Preprocess
    temperature, lambda_coef = preprocess(
        lambda_coef=lambda_coef,
        temperature=temperature,
        n_levels=n_levels,
    )

    # (B, L) - Full visit paths for each node in the batch
    gamma = path[node_ids]

    # Positives
    valid = (gamma != -1)
    # (B, N, L) - positives[i, n, l] = True if node n is the
    # target (positive) prototype for sample i at level l.
    positives = torch.zeros((batch_size, n_nodes, n_levels), dtype=torch.bool,
                            device=store_device)
    i_idx, l_idx = torch.nonzero(valid, as_tuple=True)
    n_idx = gamma[i_idx, l_idx]
    positives[i_idx, n_idx, l_idx] = True

    # Distances
    # (B, N) - Tensor containing pairwise hyperbolic distances between
    # samples and prototypes. N number of nodes in the hierarchy.
    dists = dist_matrix(values, store_mean)
    # (B, N, L)  # noqa: ERA001
    dists = dists.unsqueeze(-1).expand(-1, -1, n_levels)
    # (B, N, L) - Scale dists by the corresponding temperature.
    logits = -dists / temperature
    # (1, N, L) - Binary mask where level_mask[0, n, l] = True if node n
    # belongs to level l in the hierarchy. (level_mask[0, :, l] = H_l)
    level_mask = ((path == -1).diff(
        dim=1, append=torch.ones((n_nodes, 1), device=store_device)
        )== 1).unsqueeze(0)
    # Set logits[i,n,l] = -inf if node n is not in H_l.
    # This ensures the softmax denominator at each level includes only nodes
    # in H_l.
    logits = logits.masked_fill(~level_mask, float("-inf"))

    return weighted_levelwise_loss(
        logits=logits,
        positives=positives,
        lambda_coef=lambda_coef,
        n_levels=n_levels,
        hierarchical_constraint=hierarchical_constraint,
        level_mask=level_mask,
    )


def loss_ptp(
        store_full_visit_path: torch.Tensor,
        store_mean: torch.Tensor,
        store_device: torch.device,
        lambda_coef: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute the prototype-to-prototype contrastive loss.

    Parameters
    ----------
    store_full_visit_path : torch.Tensor
        Tensor of shape (N, P) containing the full visit path for each node.
    store_mean : torch.Tensor
        Tensor of shape (N, D) containing the current mean embedding
        for each node.
    store_device : torch.device
        Device where the store data is allocated.
    lambda_coef : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level weights.
    temperature : torch.Tensor
        Tensor of shape (1,) or (L,) with per-level temperatures.

    Returns
    -------
    torch.Tensor
        Scalar prototype-to-prototype contrastive loss.

    Notes
    -----
    This loss operates exclusively on the stored mean prototypes and their
    hierarchical relationships. As such, it is completely detached from the
    computational graph and does **not** backpropagate gradients to any model
    parameter. It is retained for structural consistency checks and debugging.
    """
    # (N, L) - Full visit paths for all nodes in the hierarchy.
    path = store_full_visit_path
    # L: Number of hierarchy levels
    n_levels = path.shape[1]
    # N: Number of hierarchy nodes
    n_nodes = path.shape[0]
    # Preprocess
    temperature, lambda_coef = preprocess(
        lambda_coef=lambda_coef,
        temperature=temperature,
        n_levels=n_levels,
    )

    # Positives
    # (N, L) - Binary mask where mask[n, l] = True if node n belongs
    # to level l in the hierarchy. (mask[:, l] = H_l)
    level_mask = ((path == -1).diff(
        dim=1, append=torch.ones((n_nodes, 1), device=store_device)) == 1)
    # (N, 1, L-1)  # noqa: ERA001
    parent_i = path[:, :-1].unsqueeze(1)  # path[n, l-1]
    # (1, N, L-1)  # noqa: ERA001
    parent_j = path[:, :-1].unsqueeze(0)
    same_parent = (parent_i == parent_j)
    # (N, N, L) - True if nodes share the same parent at level l-1
    same_parent = torch.cat([
        torch.zeros((n_nodes, n_nodes, 1), dtype=torch.bool,
                    device=store_device),
        same_parent
        ], dim=-1)
    # (N, 1, L)  # noqa: ERA001
    mask_n = level_mask.unsqueeze(1)
    # (1, N, L)  # noqa: ERA001
    mask_m = level_mask.unsqueeze(0)
    # (N, N, L) - Remove self-matches
    identity_mask = torch.eye(n_nodes, dtype=torch.bool,
                              device=store_device).unsqueeze(-1)
    # (N, N, L) - H_l \ {n}
    valid_level_mask = mask_n & mask_m & (~identity_mask)
    # (N, N, L)  # noqa: ERA001
    positives = same_parent & valid_level_mask

    # Distances
    # (N, N) - Pairwise hyperbolic distances between all prototype vectors
    dists = dist_matrix(store_mean, store_mean)
    # (N, N, L) - Expand across levels to align with positives
    dists = dists.unsqueeze(-1).expand(-1, -1, n_levels)
    # (N, N, L) - Scale dists by the corresponding temperature
    logits = -dists / temperature
    # (N, N, L) - Mask logits to exclude invalid entries from the denominator
    logits = logits.masked_fill(~valid_level_mask, float("-inf"))

    return weighted_levelwise_loss(
        logits=logits,
        positives=positives,
        lambda_coef=lambda_coef,
        n_levels=n_levels,
        hierarchical_constraint=False,
        level_mask=None
    )


def get_lambda_coef(
        strategy: Literal["uniform", "exp", "pow2", "inverse",
                          "exp_inverse", "exp_inverse2", "pow2_inverse"],
        n_levels: int
        ) -> torch.Tensor:
    r"""
    Generate per-level penalties.

    Parameters
    ----------
    strategy : Literal["uniform", "exp", "pow2", "inverse",
                       "exp_inverse", "pow2_inverse"]
        Strategy to compute level penalties:
        - "uniform":     \lambda_l = 1
        - "exp":         \lambda_l = exp(l)
        - "pow2":        \lambda_l_ = 2^l
        - "inverse":     \lambda_l = 1 / (|L| - l)
        - "exp_inverse": \lambda_l = exp(1 / (|L| - l))
        - "exp_inverse2": \lambda_l = exp(-(|L| - 1 - l)/(|L| - 1))
        - "pow2_inverse":\lambda_l = 2^(1 / (|L| - l))
    n_levels : int
        Total number of hierarchy levels |L|.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_levels,) containing the level penalties.
    """
    levels = torch.arange(n_levels, dtype=torch.float32)
    match strategy:
        case "uniform":
            return torch.ones((n_levels,))
        case "exp":
            return torch.exp(levels)
        case "pow2":
            return torch.pow(2, levels)
        case "inverse":
            return 1 / (n_levels - levels)
        case "exp_inverse":
            return torch.exp(1 / (n_levels - levels))
        case "exp_inverse2":
            return torch.exp(-(n_levels - 1 - levels) / (n_levels - 1))
        case "pow2_inverse":
            return torch.pow(2, 1 / (n_levels - levels))
        case _:
            msg = f"{strategy} strategy not supported."
            raise ValueError(msg)
