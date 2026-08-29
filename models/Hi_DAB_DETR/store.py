# ------------------------------------------------------------------------
# Hi-DAB-DETR
# Copyright (c) 2025 Emanuele Malagoli. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

from typing import Literal

import torch
from torch import nn
from typing_extensions import Self

from models.Hi_DAB_DETR import pmath


class Store(nn.Module):
    """Prototypes store class."""

    def __init__(
            self,
            layers_idx: list[int],
            hierarchy: dict,
            dim: int,
            memory_size: int,
            mode: Literal["shared", "per_layer"],
            ) -> None:
        """
        Initialize.

        Parameters
        ----------
        layers_idx : list[int]
            List of layer indices to be associated with this memory store.
            Used to map each index to the corresponding decoder layer.
        hierarchy : dict
            hierarchy : dict of {int: int}
            Dictionary representing the hierarchy structure.
            Each key represents a node ID, and its corresponding
            value indicates the parent node ID.
            If a node has no parent (i.e., it is the root), its
            value must be set to -1.
            Only tree structures are supported (i.e., each node must
            have exactly one parent, and no node can have multiple parents).
            The node IDs must form a contiguous set of integers starting
            from 0.

            Example:
            >>> hierarchy = {
            ...     0: 2,
            ...     1: 2,
            ...     2: 3,
            ...     3: 4,
            ...     4: -1
            ... }
        dim : int
            Embedding dimension.
        memory_size : int
            Sample memory size.
        mode : Literal["shared", "per_layer"]
            Mode of memory usage.
            If "shared", the same memory is used across
            all decoder layers; if "per_layer", each decoder layer maintains
            its own independent memory.
        """
        super().__init__()

        # Check hierarchy
        _hierarchy = {int(key): int(value) for key, value in hierarchy.items()}
        self._check_hierarchy(_hierarchy)

        self.hierarchy = _hierarchy
        if mode not in ["per_layer", "shared"]:
            msg = f"Mode can be 'per_layer' or 'shared'. Found: {mode}"
            raise ValueError(msg)
        self.mode = mode
        self.layers_idx = layers_idx
        self.layers_map = {
            v : k if self.mode == "per_layer" else 0
            for k, v in dict(enumerate(self.layers_idx)).items()
            }
        # Number of layers (L)
        self.n_layers = (len(self.layers_idx) if self.mode == "per_layer"
                         else 1)
        # Embedding dimension (D)
        self.dim = dim
        # Memory size per node (M)
        self.memory_size = memory_size
        # Node IDs (N)
        self.n_nodes = len(self.hierarchy)

        # Prototype memory tensor (L, N, M, D)
        self.memory = torch.zeros((self.n_layers, self.n_nodes,
                                   self.memory_size, self.dim))
        # Current mean embedding for each node (L, N, D)
        self.mean = torch.zeros((self.n_layers, self.n_nodes, self.dim))
        # Write pointer for each node (L, N)
        self.ptr = torch.zeros(self.n_layers, self.n_nodes, dtype=torch.long)
        # Binary mask indicating valid entries in memory (L, N, M)
        self.mask_count = torch.zeros(self.n_layers, self.n_nodes,
                                      self.memory_size, dtype=torch.bool)
        # Count of valid entries per node (L, N)
        self.count = torch.zeros(self.n_layers, self.n_nodes, dtype=torch.long)
        # Full hierarchical path per node (N, max_depth)
        self.full_visit_path = self._build_full_visit_path()

    def _check_hierarchy(self, hierarchy: dict) -> None:
        """
        Validate the structure of the input hierarchy.

        Ensures that:
        - All parent node IDs (excluding -1) are present in the keys.
        - Node IDs start from 0.
        - Node IDs form a contiguous range with no gaps.

        Parameters
        ----------
        hierarchy : dict
            Dictionary representing the tree hierarchy as {child: parent}.
            The root node must have parent -1.

        Raises
        ------
        ValueError
            If the hierarchy violates any of the structural constraints.
        """
        if not {node for node in hierarchy.values() if node != -1}.issubset(
                set(hierarchy.keys())):
            msg = "`hierarchy` keys must contain all node ids."
            raise ValueError(msg)
        if 0 not in hierarchy:
            msg = "Nodes in hierarchy must start from 0"
            raise ValueError(msg)
        if (
            sorted(hierarchy.keys())
            != list(range(max(hierarchy.keys()) +1))
        ):
            msg = "Nodes in hierarchy are not consecutive"
            raise ValueError(msg)

    def _get_node_path(self, node_id: int) -> list[int]:
        """
        Return the path from the given node up to the root (single path).

        Parameters
        ----------
        node_id : int
            Starting node ID.

        Returns
        -------
        list[int]
            List of visited node IDs from the node up to the root.
        """
        path = [node_id]
        while self.hierarchy[node_id] != -1:
            node_id = self.hierarchy[node_id]
            path.append(node_id)
        return path

    def _build_full_visit_path(self) -> torch.Tensor:
        """
        Build a tensor of ancestor paths for each node.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, max_depth), where N is the number of nodes
            and max_depth is the maximum path length in the hierarchy.
            Each row contains the sequence of ancestor node IDs for the
            corresponding node, padded with -1.
        """
        _paths = [
            torch.tensor(self._get_node_path(node_id)[::-1],
                         dtype=torch.long)
            for node_id in range(self.n_nodes)
            ]
        _max_path = max(len(p) for p in _paths)
        # (N, max_path)  # noqa: ERA001
        full_visit_path = torch.full((self.n_nodes, _max_path), -1,
                                      dtype=torch.long)
        for i, p in enumerate(_paths):
            full_visit_path[i, :len(p)] = p.clone().detach()
        return full_visit_path

    def _update_memory(
            self,
            node_ids: torch.Tensor,
            values: torch.Tensor,
            ) -> torch.Tensor:
        """
        Update the memory buffers for the specified nodes.

        Parameters
        ----------
        node_ids : torch.Tensor
            Tensor of shape (L, B) with node IDs for each decoder layer.
        values : torch.Tensor
            Tensor of shape (L, B, D) with associated embeddings.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (L, N) indicating which nodes were updated.
        """
        local_update_mask = torch.zeros((self.n_layers, self.n_nodes),
                                        dtype=torch.bool, device=values.device)
        for layer_idx in range(self.n_layers):
            for node_id, value in zip(node_ids[layer_idx], values[layer_idx]):
                paths = self.full_visit_path[node_id]
                paths = paths[paths != -1]
                ptrs = self.ptr[layer_idx, paths] % self.memory_size
                self.memory[layer_idx, paths, ptrs] = value
                self.mask_count[layer_idx, paths, ptrs] = True
                self.ptr[layer_idx, paths] += 1
                local_update_mask[layer_idx, paths] = 1
        # Update count
        self.count = self.mask_count.sum(dim=-1)
        return local_update_mask

    def _get_global_nodes_to_update(
            self,
            local_update_mask: torch.Tensor,
            sync: bool,  # noqa: FBT001
            ) -> torch.Tensor:
        """
        Compute the set of nodes to update.

        Parameters
        ----------
        local_update_mask : torch.Tensor
            Boolean tensor of shape (L, N), where L is the number of layers
            and N is the number of nodes.
        sync : bool
            If `True`, updates are aggregated across processes (DDP).

        Returns
        -------
        torch.Tensor
            Tensor of shape (L, MAX), where MAX is the maximum number
            of updated nodes across layers.
        """
        if torch.distributed.is_initialized() and sync:
            torch.distributed.all_reduce(local_update_mask,
                                         op=torch.distributed.ReduceOp.SUM)
        else:
            msg = "Currently not supported"
            raise ValueError(msg)
        updated_mask = local_update_mask > 0
        global_node_ids_to_update = [
            torch.where(updated_mask[layer_idx])[0]
            for layer_idx in range(updated_mask.shape[0])
        ]
        # Pad each row to the maximum length by repeating the first element
        _max = max(x.shape[0] for x in global_node_ids_to_update)
        # (L, MAX)  # noqa: ERA001
        return torch.stack([
            x if x.shape[0] == _max else
            torch.cat([x, x[0].repeat(_max - x.shape[0])])
            for x in global_node_ids_to_update
        ])

    def _compute_mean(
            self,
            global_node_ids_to_update: torch.Tensor,
            hyperbolic_c: torch.Tensor,
            sync: bool,  # noqa: FBT001
            geodesic_alpha: torch.Tensor,
            ) -> None:
        """
        Compute the hyperbolic mean.

        Parameters
        ----------
        global_node_ids_to_update : torch.Tensor
            Tensor of shape (L, MAX), where L is the number of layers and
            MAX is the (padded) number of nodes to update per layer.
        hyperbolic_c : torch.Tensor
            Hyperbolic curvature used in mean and geodesic computations.
        sync : bool
            If `True`, the memory and mask buffers are gathered across
            all processes via DDP.
        geodesic_alpha : torch.Tensor
            Geodesic interpolation factor between the current mean and
            the new mean.

        Raises
        ------
        ValueError
            If called in non-distributed or non-sync mode
            (currently unsupported).
        """
        if not torch.distributed.is_initialized() or not sync:
            msg = "Currently not supported"
            raise ValueError(msg)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # (L, MAX, M, D)  # noqa: ERA001
        mem_selected = torch.stack([
            self.memory[layer_idx, global_node_ids_to_update[layer_idx]]
            for layer_idx in range(self.n_layers)
        ])
        gathered_memories = [torch.zeros_like(mem_selected)
                             for _ in range(world_size)]
        torch.distributed.all_gather(gathered_memories, mem_selected)

        # (L, MAX, M)  # noqa: ERA001
        mask_selected = torch.stack([
            self.mask_count[layer_idx, global_node_ids_to_update[layer_idx]]
            for layer_idx in range(self.n_layers)
        ])
        gathered_masks = [torch.zeros_like(mask_selected)
                          for _ in range(world_size)]
        torch.distributed.all_gather(gathered_masks, mask_selected)

        # Master concatenates valid values
        if rank == 0:
            # (P, L, MAX, M)  # noqa: ERA001
            gathered_masks = torch.stack(gathered_masks)

            full = gathered_masks.all().item()

            # Full mode
            if full:
                valid_data = torch.cat(gathered_memories, dim=2)
                new_mean = pmath.poincare_mean(
                    x=valid_data,
                    dim=2,
                    c=hyperbolic_c
                    )

                _shape = global_node_ids_to_update.shape
                layer_idx = torch.arange(
                    _shape[0], device=self.mean.device).unsqueeze(1).expand(
                        _shape)

                mean = pmath.geodesic(
                    t=geodesic_alpha,
                    x=self.mean[layer_idx, global_node_ids_to_update],
                    y=new_mean,
                    c=hyperbolic_c,
                )

                self.mean[layer_idx, global_node_ids_to_update] = mean

            # Not full mode
            else:
                # (P, L, MAX, M, D)  # noqa: ERA001
                gathered_memories = torch.stack(gathered_memories)

                updated_pairs = set()
                for layer_idx in range(self.n_layers):
                    for local_idx, node_id in enumerate(
                        global_node_ids_to_update[layer_idx]):
                        key = (layer_idx, node_id.item())
                        if key in updated_pairs:
                            continue
                        updated_pairs.add(key)

                        # (P, M, D)  # noqa: ERA001
                        mem_per_node = gathered_memories[:, layer_idx,
                                                        local_idx]
                        # (P, M)  # noqa: ERA001
                        mask_per_node = gathered_masks[:, layer_idx, local_idx]
                        valid_data = mem_per_node[mask_per_node]

                        if valid_data.numel() != 0:
                            new_mean = pmath.poincare_mean(
                                x=valid_data,
                                dim=0,
                                c=hyperbolic_c
                            )
                            mean = pmath.geodesic(
                                t=geodesic_alpha,
                                x=self.mean[layer_idx, node_id],
                                y=new_mean,
                                c=hyperbolic_c,
                            )
                            self.mean[layer_idx, node_id] = mean

        # Broadcast
        torch.distributed.broadcast(self.mean, src=0)

    def _update_mean(
            self,
            local_update_mask: torch.Tensor,
            hyperbolic_c: torch.Tensor,
            sync: bool,  # noqa: FBT001
            geodesic_alpha: torch.Tensor,
            ) -> None:
        """
        Update mean.

        Parameters
        ----------
        local_update_mask : torch.Tensor
            Boolean tensor of shape (L, N), where L is the number of layers
            and N is the number of nodes.
        hyperbolic_c : torch.Tensor
            Hyperbolic curvature.
        sync : bool
            If `True`, updates are aggregated across processes (DDP).
        geodesic_alpha : torch.Tensor
            Step size for the geodesic interpolation between old and new mean.
        """
        global_node_ids_to_update = self._get_global_nodes_to_update(
            local_update_mask=local_update_mask, sync=sync)
        self._compute_mean(
            global_node_ids_to_update=global_node_ids_to_update,
            hyperbolic_c=hyperbolic_c,
            sync=sync,
            geodesic_alpha=geodesic_alpha,
            )

    @torch.no_grad()
    def update(
            self,
            node_ids: torch.Tensor,
            hyperbolic_c: torch.Tensor,
            values: torch.Tensor,
            sync: bool,  # noqa: FBT001
            geodesic_alpha: torch.Tensor,
            ) -> None:
        """
        Update prototypes.

        Parameters
        ----------
        node_ids : torch.Tensor
            Tensor of shape (L, B), where L is the number of decoder layers
            and B is the number of updated nodes per layer in the batch.
        hyperbolic_c : torch.Tensor
            Hyperbolic curvature.
        values : torch.Tensor
            Tensor of shape (L, B, D), where D is the embedding dimension.
            Each `values[l, b]` corresponds to `node_ids[l, b]`.
        sync : bool
            If `True`, the mean will be synchronized across all
            processes (DDP).
            If `False`, the mean is updated locally using only the current
            process memory.
        geodesic_alpha : torch.Tensor
            Step size for geodesic interpolation between old and new mean.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or incompatible with the store's
            embedding dimension.
        """
        # Check shapes
        if node_ids.ndim != 2:  # noqa: PLR2004
            msg = f"`node_ids` must be 2D (L, B), got shape {node_ids.shape}"
            raise ValueError(msg)
        if values.ndim != 3 or values.shape[2] != self.dim:  # noqa: PLR2004
            msg = (f"`values` must have shape (L, B, {self.dim}), "
                   f"got {values.shape}")
            raise ValueError(msg)
        if node_ids.shape[0] != values.shape[0]:
            msg = (f"Mismatched number of layers: "
                   f"{node_ids.shape[0]} != {values.shape[0]}")
            raise ValueError(msg)
        if node_ids.shape[1] != values.shape[1]:
            msg = (f"Mismatched number of items: "
                   f"{node_ids.shape[1]} != {values.shape[1]}")
            raise ValueError(msg)

        if self.mode == "shared":
            values = values.flatten(0, 1).unsqueeze(0)
            node_ids = node_ids.flatten(0, 1).unsqueeze(0)

        node_ids = node_ids.to(dtype=torch.long)

        # Update memory
        local_update_mask = self._update_memory(
            node_ids=node_ids,
            values=values,
            )

        # Update mean
        self._update_mean(
            local_update_mask=local_update_mask,
            hyperbolic_c=hyperbolic_c,
            sync=sync,
            geodesic_alpha=geodesic_alpha,
            )

    def get_memory(self, layer_idx: int) -> torch.Tensor:
        """
        Return the memory buffer for a specific decoder layer.

        Parameters
        ----------
        layer_idx : int
            Index of the decoder layer.

        Returns
        -------
        torch.Tensor
            Memory tensor of shape (N, M, D), where N is the number of nodes,
            M is the memory size, and D is the embedding dimension.
        """
        return self.memory[self.layers_map[layer_idx]]

    def get_mean(self, layer_idx: int) -> torch.Tensor:
        """
        Return the current mean embeddings for a given decoder layer.

        Parameters
        ----------
        layer_idx : int
            Decoder layer index.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, D) containing the mean embedding for each
            node, where N is the number of nodes and D is the embedding
            dimension.
        """
        return self.mean[self.layers_map[layer_idx]]

    def get_count(self, layer_idx: int) -> torch.Tensor:
        """
        Return the number of valid memory entries per node for a given layer.

        Parameters
        ----------
        layer_idx : int
            Index of the decoder layer.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the number of stored samples
            for each node.
        """
        return self.count[self.layers_map[layer_idx]]

    def to(self, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
        """
        Override `nn.Module.to()` method.

        Returns
        -------
        Self
            The module with all tensors moved to the specified device/dtype.
        """
        super().to(*args, **kwargs)

        self.memory = self.memory.to(*args, **kwargs)
        self.mean = self.mean.to(*args, **kwargs)
        self.ptr = self.ptr.to(*args, **kwargs)
        self.mask_count = self.mask_count.to(*args, **kwargs)
        self.count = self.count.to(*args, **kwargs)
        self.full_visit_path = self.full_visit_path.to(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        """Return the current device of the store tensors."""
        if not hasattr(self, "memory"):
            msg = "The 'memory' tensor is not initialized."
            raise RuntimeError(msg)
        return self.memory.device
