# ------------------------------------------------------------------------
# Hi-DAB-DETR
# Copyright (c) 2025 Emanuele Malagoli. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Based on DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------




import os
from typing_extensions import Self
import json
from pathlib import Path

import math
from typing import Dict, Literal
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .transformer import build_transformer
from . import pmath
from .store import Store
from .contrastive_loss import loss_sts, loss_stp, loss_ptp, get_lambda_coef
from .attention import MultiheadAttention


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        self.c = c
        self.train_x = train_x
        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = self.c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
            
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space

    This version is designed to reuse the curvature `c` and base point `xp`
    defined elsewhere (e.g., in a shared ToPoincare module).
    """

    def __init__(self, c, train_x, xp):
        super(FromPoincare, self).__init__()
        self.c = c
        self.train_x = train_x
        self.xp = xp


    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_x={}".format(self.train_x)


def build_hierarchy_mask(
        full_visit_path: torch.Tensor,
        use_distance: bool,  # noqa: FBT001
        normalize_distance: bool,  # noqa: FBT001
        ) -> torch.Tensor:
    n_nodes = full_visit_path.shape[0]
    max_distance = full_visit_path.shape[1] -1
    mask = torch.full((n_nodes, n_nodes), float("-inf"),
                      device=full_visit_path.device)

    for node in range(n_nodes):
        path = full_visit_path[node,:]
        node_lvl = (path == node).nonzero(as_tuple=True)[0]
        for parent in path:
            if parent != -1:
                parent_lvl = (path == parent).nonzero(as_tuple=True)[0]
                value = -(node_lvl - parent_lvl) if use_distance else 0
                if normalize_distance:
                    value = value / max_distance
                mask[node, parent] = value

    return mask


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class HypAct(nn.Module):
    def __init__(
            self,
            from_poincare: FromPoincare,
            to_poincare: ToPoincare,
            act: nn.Module,
            ) -> None:
        super().__init__()
        self.from_poincare = from_poincare
        self.to_poincare = to_poincare
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.from_poincare(x)
        x = self.act(x)
        return self.to_poincare(x)


class HypMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            from_poincare: FromPoincare,
            to_poincare: ToPoincare,
            act: nn.Module,
            c: float | nn.Parameter,
            bias: bool,
        ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(HypLinear(
            in_features=n,
            out_features=k,
            c=c,
            bias=bias
            ) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = HypAct(
            from_poincare=from_poincare,
            to_poincare=to_poincare,
            act=act,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HyperbolicRefiner(nn.Module):
    """Module for hyperbolic projection and prototype refinement."""

    def __init__(
            self,
            use_hyperbolic: bool,  # noqa: FBT001
            use_bbox_proto: bool,  # noqa: FBT001
            use_bbox_proto_refinement: bool,  # noqa: FBT001
            use_distance_proto_attn_mask: bool,  # noqa: FBT001
            normalize_distance_proto_attn_mask: bool,  # noqa: FBT001
            use_proto_refiner: bool,  # noqa: FBT001
            use_bbox_proto_iter_update: bool,  # noqa: FBT001
            hyperbolic_c: float,
            hyperbolic_train_c: bool,  # noqa: FBT001
            hyperbolic_train_x: bool,  # noqa: FBT001
            hyperbolic_riemannian: bool,  # noqa: FBT001
            hyperbolic_clip_r: float | None,
            hierarchy: dict | None,
            store_layers_idx: list[int],
            store_mode: Literal["shared", "per_layer"],
            store_memory_size: int,
            store_sync: bool,  # noqa: FBT001
            store_geodesic_alpha: float,
            store_topk_per_sample: int,
            store_match_update: bool,  # noqa: FBT001
            bbox_proto_sync: bool,  # noqa: FBT001
            hidden_dim: int,
            nhead: int,
            dropout: float,
            ) -> None:
        """
        Initialize.

        Parameters
        ----------
        use_hyperbolic : bool
            Whether to use hyperbolic embeddings.
        use_bbox_proto : bool
            Whether to use prototype-based bounding box refinement.
        hyperbolic_c : float
            Initial curvature for Poincaré ball.
        hyperbolic_train_c : bool
            Whether to learn the curvature.
        hyperbolic_train_x : bool
            Whether to learn the base point.
        hyperbolic_riemannian : bool
            Whether to use Riemannian projection.
        hyperbolic_clip_r : float | None
            Optional clipping radius for embeddings.
        hierarchy : dict | None
            Hierarchy.
        store_memory_size : int
            Buffer dimension.
        store_sync : bool
            Whether to synchronize store across DDP processes.
        store_geodesic_alpha : float
            Geodesic interpolation factor.
        store_topk_per_sample : int
            Top-k selection per sample for memory update.
        store_match_update : bool
            Whether to apply match based update to prototypes.
        bbox_proto_sync : bool
            Whether to synchronize bbox prototypes in DDP.
        hidden_dim : int
            Hidden dimension of embeddings.
        nhead : int
            Number of attention heads.
        dropout : float
            Dropout.
        """
        super().__init__()

        self.use_hyperbolic = use_hyperbolic
        self.use_bbox_proto = use_bbox_proto
        if self.use_hyperbolic:
            if hyperbolic_train_c:
                self.hyperbolic_c = nn.Parameter(torch.tensor([hyperbolic_c]))
            else:
                self.register_buffer("hyperbolic_c",
                                     torch.tensor([hyperbolic_c]))
            self.to_poincare = ToPoincare(
                c=self.hyperbolic_c,
                train_x=hyperbolic_train_x,
                ball_dim=hidden_dim,
                riemannian=hyperbolic_riemannian,
                clip_r=hyperbolic_clip_r
            )
            # Share curvature and base point with ToPoincare
            self.from_poincare = FromPoincare(
                c=self.hyperbolic_c,
                train_x=hyperbolic_train_x,
                xp=self.to_poincare.xp,
            )

            self.store_sync=store_sync
            self.store_geodesic_alpha = store_geodesic_alpha
            self.store_topk_per_sample = store_topk_per_sample
            self.store_match_update = store_match_update
            self.bbox_proto_sync = bbox_proto_sync

            # Store
            self.store = Store(
                layers_idx=store_layers_idx,
                hierarchy=hierarchy,
                dim=hidden_dim,
                memory_size=store_memory_size,
                mode=store_mode,
            )

            self.hierarchy_mask = build_hierarchy_mask(
                full_visit_path=self.store.full_visit_path,
                use_distance=use_distance_proto_attn_mask,
                normalize_distance=normalize_distance_proto_attn_mask,
            )
            self.prototypes = nn.Parameter(
                torch.zeros((
                    len(store_layers_idx),
                    self.store.n_nodes,
                    hidden_dim,
                ))
            )
            self.proto_offset_scale = MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_layers=2,
            )
            self.use_bbox_proto_refinement = use_bbox_proto_refinement
            self.use_bbox_proto_iter_update = use_bbox_proto_iter_update
            self.use_proto_refiner = use_proto_refiner

            self.nhead = nhead
            self.proto_self_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.proto_self_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.proto_self_v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.proto_self_attn = MultiheadAttention(
                hidden_dim, nhead, dropout=dropout)
            self.proto_self_dropout = nn.Dropout(dropout)
            self.proto_self_norm = nn.LayerNorm(hidden_dim)
        else:
            self.store = None

    def match_based_update(
        self,
        aux_loss: bool,  # noqa: FBT001
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        outputs: dict[str, torch.Tensor],
        criterion: nn.Module,
        dec_layers: int,
        ) -> None:
        """
        Match-based memory update for all decoder layers.

        Parameters
        ----------
        aux_loss : bool
            Whether auxiliary outputs are enabled.
        indices : list of tuple of Tensors
            Matching indices from Hungarian algorithm.
        targets : list of dict
            Ground-truth labels per image.
        outputs : dict
            Model outputs including 'hyperbolic_emb'.
        criterion : nn.Module
            Loss criterion providing permutation indices.
        dec_layers : int
            Total number of decoder layers.
        """
        if self.use_hyperbolic and self.store_match_update:
            node_ids = []
            values = []
            for layer_idx in self.store.layers_idx:
                if layer_idx == dec_layers -1:
                    _indices = indices[layer_idx] if aux_loss else indices
                    _hyperbolic_emb = outputs["hyperbolic_emb"]
                else:
                    if not aux_loss:
                        msg = ("`aux_loss` must be `True` for intermediate "
                               "layers.")
                        raise ValueError(msg)
                    _indices = indices[layer_idx]
                    _hyperbolic_emb = outputs["aux_outputs"][layer_idx][
                        "hyperbolic_emb"]

                idx_layer = criterion._get_src_permutation_idx(_indices)  # noqa: SLF001
                node_ids_layer = torch.cat(
                    [t["labels"][J] for t, (_, J) in zip(targets, _indices)])
                values_layer = _hyperbolic_emb[idx_layer].reshape(
                    -1, _hyperbolic_emb.shape[-1])
                node_ids.append(node_ids_layer)
                values.append(values_layer)

            node_ids = torch.stack(node_ids, dim=0)
            values = torch.stack(values, dim=0)
            self.store.update(
                node_ids=node_ids,
                hyperbolic_c=self.hyperbolic_c,
                values=values,
                sync=self.store_sync,
                geodesic_alpha=torch.tensor(self.store_geodesic_alpha),
            )


    def get_hyperbolic_embeddings(self, hs: torch.Tensor) -> list[torch.Tensor]:
        if self.use_hyperbolic:
            hyperbolic_embeddings = self.to_poincare(hs).unbind(0)
            hyperbolic_embeddings = [
                None if i not in self.store.layers_idx else x
                for i, x in enumerate(hyperbolic_embeddings)
                ]
        else:
            hyperbolic_embeddings = [None]*len(hs)
        return hyperbolic_embeddings

    def get_prototypes(self, num_decoder_layers: int, device: torch.device):
        prototypes = None
        valid_mask = None

        if self.use_hyperbolic and self.use_bbox_proto:
            valid_mask = torch.tensor(
                [i in self.store.layers_idx
                 for i in range(num_decoder_layers)],
                dtype=torch.bool, device=device
            )
            shape_ = self.store.get_mean(self.store.layers_idx[0]).shape
            dummy = torch.zeros(shape_, device=device)
            means = torch.stack([
                self.store.get_mean(i) if valid_mask[i] else dummy
                for i in range(num_decoder_layers)])
            stored_prototypes = self.from_poincare(means)

            n_layers = len(stored_prototypes)
            stored_prototypes = stored_prototypes.flatten(0,1)
            proto_shape = self.prototypes.shape
            prototypes = self.prototypes.flatten(0,1)
            attn_mask = self.hierarchy_mask.repeat(
                1, n_layers).repeat(n_layers, 1)
            attn_mask = attn_mask.unsqueeze(0).repeat(self.nhead,1,1)
            q_proto = self.proto_self_q_proj(prototypes.unsqueeze(1))
            k_proto = self.proto_self_k_proj(stored_prototypes.unsqueeze(1))
            v_proto = self.proto_self_v_proj(stored_prototypes.unsqueeze(1))
            prototypes2 = self.proto_self_attn(
                    q_proto, k_proto, v_proto, attn_mask = attn_mask)[0]
            prototypes2 = prototypes2.squeeze(1)
            prototypes = prototypes.squeeze(1)
            prototypes = prototypes + self.proto_self_dropout(prototypes2)
            prototypes = self.proto_self_norm(prototypes)

            prototypes = prototypes.squeeze(1).reshape(proto_shape)
        return prototypes, valid_mask


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


class HiDABDETR(nn.Module):
    """ This is the Hi-DAB-DETR module that performs object detection """
    def __init__(
            self,
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_dec_layers,
            use_hyperbolic: bool,
            use_bbox_proto: bool,
            use_bbox_proto_refinement: bool,
            use_distance_proto_attn_mask: bool,
            normalize_distance_proto_attn_mask: bool,
            use_bbox_proto_iter_update: bool,
            hyperbolic_c: float,
            hyperbolic_train_c: bool,
            hyperbolic_train_x: bool,
            hyperbolic_riemannian: bool,
            hyperbolic_clip_r: float | None,
            hierarchy: dict | None,
            store_layers_idx: list[int],
            store_mode: Literal["shared", "per_layer"],
            store_memory_size: int,
            store_sync: bool,
            store_geodesic_alpha: float,
            store_topk_per_sample: int,
            store_match_update: bool,
            bbox_proto_sync: bool,
            use_proto_refiner: bool,
            use_proto_loss_ce: bool,
            nhead: int,
            dropout: float,
            aux_loss=False,
            iter_update=True,
            query_dim=4, 
            bbox_embed_diff_each_layer=False,
            random_refpoints_xy=False,
            ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            use_hyperbolic : bool
                If `True`, activates the hyperbolic embedding pipeline,
                mapping object queries into the Poincaré ball.
            use_bbox_proto : bool
                Whether to use bbox prototypes.
            use_bbox_proto_iter_update : bool
                Whether to apply prototype-based offsets during iterative
                refinement.
            hyperbolic_c : float
                Curvature of the Poincaré ball.
            hyperbolic_train_c : bool
                If `True~, makes the curvature `hyperbolic_c` a learnable
                parameter during training.
            hyperbolic_train_x : bool
                If `True`, enables learning of a base point for the
                exponential map.
            hyperbolic_riemannian : bool
                If `True~, applies Riemannian gradient correction to ensure
                updates remain within the geometry of the hyperbolic manifold.
            hyperbolic_clip_r : float | None
                Optional maximum norm for input vectors before projection
                into the Poincaré ball.
                If `None`, no clipping is applied.
            hierarchy: dict | None
                Hierarchy.
            store_memory_size : int
                Buffer dimension.
            store_sync: bool
                If `True`, the mean will be synchronized across all
                processes (DDP).
                If `False`, the mean is updated locally using only the current
                process memory.
            store_sync : float
                Step size for geodesic interpolation between old and new mean.
            store_topk_per_sample : int
                Number of top (query, class) pairs per sample used to update
                the prototype store.
            store_match_update : bool
                Whether to apply match based update to prototypes.
            bbox_proto_sync : bool
                If `True`, synchronizes bbox prototypes across
                all DDP processes.
            use_proto_refiner : bool
                Whether to use proto refiner.
            nhead : int
                Number of attention heads.
            dropout : float
                Dropout.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False





        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.iter_update = iter_update

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)


        self.hyperbolic_refiner = HyperbolicRefiner(
            use_hyperbolic=use_hyperbolic,
            use_bbox_proto=use_bbox_proto,
            use_bbox_proto_refinement=use_bbox_proto_refinement,
            use_bbox_proto_iter_update=use_bbox_proto_iter_update,
            use_distance_proto_attn_mask=use_distance_proto_attn_mask,
            normalize_distance_proto_attn_mask=normalize_distance_proto_attn_mask,
            use_proto_refiner=use_proto_refiner,
            hyperbolic_c=hyperbolic_c,
            hyperbolic_train_c=hyperbolic_train_c,
            hyperbolic_train_x=hyperbolic_train_x,
            hyperbolic_riemannian=hyperbolic_riemannian,
            hyperbolic_clip_r=hyperbolic_clip_r,
            hierarchy=hierarchy,
            store_layers_idx=store_layers_idx,
            store_mode=store_mode,
            store_memory_size=store_memory_size,
            store_sync=store_sync,
            store_geodesic_alpha=store_geodesic_alpha,
            store_topk_per_sample=store_topk_per_sample,
            store_match_update=store_match_update,
            bbox_proto_sync=bbox_proto_sync,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )
        self.use_proto_loss_ce = use_proto_loss_ce
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.hyperbolic_refiner = (
                self.hyperbolic_refiner)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        # Prototypes
        prototypes, mask_prototypes = self.hyperbolic_refiner.get_prototypes(
            num_decoder_layers=self.transformer.dec_layers,
            device=src.device
            )
        if self.hyperbolic_refiner.use_proto_refiner:
            prototypes = self.transformer.decoder.proto_refiner(prototypes)
        if self.bbox_embed_diff_each_layer:
            msg = "The case with `bbox_embed_diff_each_layer` not supported."
            raise ValueError(msg)
        prototypes_diff = prototypes.diff(
            dim=0, prepend=torch.zeros_like(prototypes[0][None]))
        prototypes_offset = torch.where(
            mask_prototypes[:, None, None],
            self.bbox_embed(prototypes_diff)[..., 2:],
            0)
        if self.use_proto_loss_ce:
            proto_logits = self.class_embed(prototypes[:, :self.num_classes,:])
            proto_logits = [None if not mask_prototypes[i] else x for
                            i,x in enumerate(proto_logits)]
        else:
            proto_logits = [None]*self.transformer.dec_layers

        # default pipeline
        embedweight = self.refpoint_embed.weight
        hs, reference = self.transformer(
            self.input_proj(src), mask, embedweight, pos[-1],
            prototypes, mask_prototypes, prototypes_offset
            )

        outputs_class = self.class_embed(hs)

        # Hyperbolic embeddings
        hyperbolic_emb = self.hyperbolic_refiner.get_hyperbolic_embeddings(hs)

        hs_diff = hs.diff(dim=0, prepend=torch.zeros_like(hs[0][None]))
        if not self.bbox_embed_diff_each_layer:
            if self.hyperbolic_refiner.use_bbox_proto_refinement:
                # gumbel-softmax trick
                class_weights = F.gumbel_softmax(
                    outputs_class, tau=1, hard=True, dim=-1)
                class_weights_padded = F.pad(
                    class_weights,
                    (0, prototypes_offset.shape[1] - class_weights.shape[-1]),
                    value=0.0
                    )
                prototypes_offset_selected = (class_weights_padded
                                              @ prototypes_offset.unsqueeze(1))
                scale = self.hyperbolic_refiner.proto_offset_scale(
                    hs_diff).sigmoid()
                prototypes_offset_selected = (
                    prototypes_offset_selected * scale)
            else:
                prototypes_offset_selected = 0
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs_diff)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            tmp[..., 2:] += prototypes_offset_selected
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'hyperbolic_emb': hyperbolic_emb[-1],
            'proto_logits': proto_logits[-1],
            }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, hyperbolic_emb, proto_logits)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, hyperbolic_emb,
                      proto_logits):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_boxes': b, 'hyperbolic_emb': c,
             'proto_logits': d,
             }
            for a, b, c, d in zip(
                outputs_class[:-1], outputs_coord[:-1], hyperbolic_emb[:-1],
                proto_logits[:-1],)
            ]

    def to(self, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
        super().to(*args, **kwargs)

        store = getattr(self.hyperbolic_refiner, "store", None)
        if store:
            store.to(*args, **kwargs)
        self.hyperbolic_refiner.hierarchy_mask = (
            self.hyperbolic_refiner.hierarchy_mask.to(*args, **kwargs))

        return self


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
            self,
            num_classes: int,
            matcher: nn.Module,
            weight_dict: dict,
            focal_alpha: float,
            losses: list[str],
            loss_contrastive_lambda_coef_strategy: Literal[
                "uniform", "exp", "pow2", "inverse", "exp_inverse",
                "pow2_inverse"
                ],
            loss_contrastive_temperature: list[float] | float,
            loss_contrastive_sampling_from_store: bool,  # noqa: FBT001
            loss_contrastive_extra_samples: int,
            loss_contrastive_k_pos: int,
            loss_contrastive_hierarchical_constraint: bool,  # noqa: FBT001
            loss_contrastive_use_sts: bool,  # noqa: FBT001
            loss_contrastive_use_stp: bool,  # noqa: FBT001
            loss_contrastive_use_ptp: bool,  # noqa: FBT001
            dec_layers: int,
            ):
        """
        Create the criterion.

        Parameters
        ----------
        num_classes : int
            Number of object categories, omitting the special no-object
            category.
        matcher : nn.Module
            Module able to compute a matching between targets and proposals
        weight_dict : dict
            Dictionary containing as key the names of the losses and as
            values their relative weight.
        losses : list[str]
            List of all the losses to be applied.
            See get_loss for list of available losses.
        focal_alpha : float
            Alpha in Focal Loss
        loss_contrastive_lambda_coef_strategy : str
            Strategy for computing level-specific weights
            for contrastive losses.
            Must be one of:
            "uniform", "exp", "pow2", "inverse", "exp_inverse", "pow2_inverse".
        loss_contrastive_temperature : list[float] | float
            Per-level or scalar temperature used to scale contrastive logits.
        loss_contrastive_sampling_from_store : bool
            Whether to sample additional positive pairs from the buffer store.
        loss_contrastive_extra_samples : int
            Number of additional random samples to draw from the store
            and append to the current batch before computing the contrastive
            loss.
            Set to 0 to disable.
        loss_contrastive_k_pos : int
            Minimum number of positive samples per anchor
            per level in STS loss.
        loss_contrastive_hierarchical_constraint : bool
            Whether to enforce hierarchical constraints across levels
            in STP loss.
        loss_contrastive_use_sts : bool
            Whether to use STS loss.
        loss_contrastive_use_stp : bool
            Whether to use STP loss.
        loss_contrastive_use_ptp : bool
            Whether to use PTP loss.
        dec_layers : int
            Number of decoder layers.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.loss_contrastive_lambda_coef_strategy = (
            loss_contrastive_lambda_coef_strategy)
        self.loss_contrastive_temperature=loss_contrastive_temperature
        self.loss_contrastive_sampling_from_store=(
            loss_contrastive_sampling_from_store)
        self.loss_contrastive_extra_samples = loss_contrastive_extra_samples
        self.loss_contrastive_k_pos=loss_contrastive_k_pos
        self.loss_contrastive_hierarchical_constraint=(
            loss_contrastive_hierarchical_constraint)
        self.loss_contrastive_use_sts=loss_contrastive_use_sts
        self.loss_contrastive_use_stp=loss_contrastive_use_stp
        self.loss_contrastive_use_ptp=loss_contrastive_use_ptp
        self.dec_layers = dec_layers

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_contrastive(self, outputs, targets, indices, num_boxes, store,
                         layer_idx):
        store_full_visit_path = store.full_visit_path
        store_count = store.get_count(layer_idx)
        store_memory = store.get_memory(layer_idx)
        store_mean = store.get_mean(layer_idx)
        store_device = store.device

        idx = self._get_src_permutation_idx(indices)
        node_ids = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        values = outputs["hyperbolic_emb"][idx].reshape(
            -1, outputs["hyperbolic_emb"].shape[-1])

        # Add extra samples
        if self.loss_contrastive_extra_samples > 0:
            valid_nodes = [i for i in range(store_memory.shape[0])
                           if store_count[i].item() > 0]
            if valid_nodes:
                k = min(self.loss_contrastive_extra_samples, len(valid_nodes))
                selected_nodes = torch.randperm(
                    len(valid_nodes), device=store_device)[:k]
                sampled_pairs = [
                    (valid_nodes[i], torch.randint(
                        0, store_count[valid_nodes[i]], (1,),
                        device=store_device).item())
                    for i in selected_nodes.tolist()
                ]
                extra_values = torch.stack(
                    [store_memory[i, j] for (i, j) in sampled_pairs], dim=0)
                extra_node_ids = torch.tensor([i for (i, _) in sampled_pairs],
                                              device=store_device)
                values = torch.cat([values, extra_values], dim=0)
                node_ids = torch.cat([node_ids, extra_node_ids], dim=0)

        lambda_coef = get_lambda_coef(
            strategy=self.loss_contrastive_lambda_coef_strategy,
            n_levels=store_full_visit_path.shape[1],
        ).to(device=store_device)
        temperature = torch.tensor(self.loss_contrastive_temperature,
                                   device=store_device)
        output = {}

        if self.loss_contrastive_use_sts:
            output["loss_contrastive_sts"] = loss_sts(
                node_ids=node_ids,
                values=values,
                store_full_visit_path=store_full_visit_path,
                store_count=store_count,
                store_memory=store_memory,
                store_device=store_device,
                lambda_coef=lambda_coef,
                temperature=temperature,
                sampling_from_store=self.loss_contrastive_sampling_from_store,
                k_pos=self.loss_contrastive_k_pos
            )

        if self.loss_contrastive_use_stp:
            output["loss_contrastive_stp"]  = loss_stp(
                node_ids=node_ids,
                values=values,
                store_full_visit_path=store_full_visit_path,
                store_mean=store_mean,
                store_device=store_device,
                lambda_coef=lambda_coef,
                temperature=temperature,
                hierarchical_constraint=(
                    self.loss_contrastive_hierarchical_constraint),
                )

        if self.loss_contrastive_use_ptp:
            output["loss_contrastive_ptp"]  = loss_ptp(
                store_full_visit_path=store_full_visit_path,
                store_mean=store_mean,
                store_device=store_device,
                lambda_coef=lambda_coef,
                temperature=temperature,
            )

        return output

    def loss_proto_ce(self, outputs, targets, indices, num_boxes, log=True):
        proto_logits = outputs['proto_logits']
        target_proto_onehot = torch.eye(proto_logits.size(0),
                                        device=proto_logits.device)
        proto_loss_ce = F.binary_cross_entropy_with_logits(proto_logits, target_proto_onehot, reduction="none").mean(1).sum() / proto_logits.shape[0]

        return {'proto_loss_ce': proto_loss_ce}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive,
            'proto_loss_ce': self.loss_proto_ce,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
            self,
            outputs: dict,
            targets: list[dict],
            store: nn.Module | None,
            return_indices: bool=False,
            ):
        """
        Perform the loss computation.

        Parameters
        ----------
        outputs : dict
            Dict of tensors, see the output specification of the model
            for the format
        targets : list[dict]
            List of dicts, such that len(targets) == batch_size.
            The expected keys in each dict depends on the losses applied,
            see each loss' doc,
        store : nn.Module | None
            Store.
        return_indices : bool
            Used for vis. if True, the layer0-5 indices will be returned as
            well.
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == "contrastive":
                kwargs = {
                    "store": store,
                    "layer_idx": self.dec_layers - 1,
                    }
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss == "contrastive":
                        if i not in store.layers_idx:
                            continue
                        kwargs = {
                            "store": store,
                            "layer_idx": i,
                            }
                    if loss == "proto_loss_ce" and i not in store.layers_idx:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_HiDABDETR(args, hierarchy: dict | None = None):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    elif args.dataset_file == "custom":
        num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(
        args=args,
        n_nodes=len(hierarchy),
        )

    model = HiDABDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_dec_layers=args.dec_layers,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        use_hyperbolic=args.use_hyperbolic,
        use_bbox_proto=args.use_bbox_proto,
        use_bbox_proto_refinement=args.use_bbox_proto_refinement,
        use_distance_proto_attn_mask=args.use_distance_proto_attn_mask,
        normalize_distance_proto_attn_mask=args.normalize_distance_proto_attn_mask,
        use_bbox_proto_iter_update=args.use_bbox_proto_iter_update,
        hyperbolic_c=args.hyperbolic_c,
        hyperbolic_train_c=args.hyperbolic_train_c,
        hyperbolic_train_x=args.hyperbolic_train_x,
        hyperbolic_riemannian=args.hyperbolic_riemannian,
        hyperbolic_clip_r=args.hyperbolic_clip_r,
        hierarchy=hierarchy,
        store_layers_idx=args.store_layers_idx,
        store_mode=args.store_mode,
        store_memory_size=args.store_memory_size,
        store_sync=args.store_sync,
        store_geodesic_alpha=args.store_geodesic_alpha,
        store_topk_per_sample=args.store_topk_per_sample,
        store_match_update=args.store_match_update,
        bbox_proto_sync=args.bbox_proto_sync,
        use_proto_refiner=args.use_proto_refiner,
        use_proto_loss_ce=args.use_proto_loss_ce,
        nhead=args.nheads,
        dropout=args.dropout,
        )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.use_hyperbolic:
        losses += ["contrastive"]
        contrastive_weight_dict = {}
        contrastive_weight_dict["loss_contrastive_sts"] = (
            args.loss_contrastive_sts_coef)
        contrastive_weight_dict["loss_contrastive_stp"] = (
            args.loss_contrastive_stp_coef)
        contrastive_weight_dict["loss_contrastive_ptp"] = (
            args.loss_contrastive_ptp_coef)
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            if i in args.store_layers_idx:
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v
                     in contrastive_weight_dict.items()}
                    )
        contrastive_weight_dict.update(aux_weight_dict)
        weight_dict.update(contrastive_weight_dict)

    if args.use_proto_loss_ce:
        losses += ["proto_loss_ce"]
        proto_loss_ce_dict = {}
        proto_loss_ce_dict["proto_loss_ce"] = args.proto_loss_ce_coef
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            if i in args.store_layers_idx:
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v
                     in proto_loss_ce_dict.items()}
                    )
        proto_loss_ce_dict.update(aux_weight_dict)
        weight_dict.update(proto_loss_ce_dict)

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        loss_contrastive_lambda_coef_strategy=(
            args.loss_contrastive_lambda_coef_strategy),
        loss_contrastive_temperature=(args.loss_contrastive_temperature),
        loss_contrastive_sampling_from_store=(
            args.loss_contrastive_sampling_from_store),
        loss_contrastive_extra_samples=(args.loss_contrastive_extra_samples),
        loss_contrastive_k_pos=args.loss_contrastive_k_pos,
        loss_contrastive_hierarchical_constraint=(
            args.loss_contrastive_hierarchical_constraint),
        loss_contrastive_use_sts=args.loss_contrastive_use_sts,
        loss_contrastive_use_stp=args.loss_contrastive_use_stp,
        loss_contrastive_use_ptp=args.loss_contrastive_use_ptp,
        dec_layers=args.dec_layers,
        )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
