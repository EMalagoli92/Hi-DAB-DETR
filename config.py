import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from typing_extensions import Self


class Config(BaseModel):
    """Configuration."""

    lr: float = 1e-4
    lr_backbone: float = 1e-5
    batch_size: int = 4
    weight_decay: float = 1e-4
    epochs: int = 300
    lr_drop: int = 200
    save_checkpoint_interval: int = 100
    clip_max_norm: float = 0.1

    # Model parameters
    modelname: Literal['hi_dab_detr', 'dab_deformable_detr'] = "hi_dab_detr"
    frozen_weights: str | None = None

    # Backbone
    backbone: str = "resnet50"
    dilation: bool = False
    position_embedding: Literal['sine', 'learned'] = "sine"
    pe_temperatureH: int = 20  # noqa: N815
    pe_temperatureW: int = 20  # noqa: N815
    batch_norm_type: Literal["SyncBatchNorm", "FrozenBatchNorm2d",
                             "BatchNorm2d"] = 'FrozenBatchNorm2d'

    # Transformer
    return_interm_layers: bool = False
    backbone_freeze_keywords: list[str] | None = None
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.0
    nheads: int = 8
    num_queries: int = 300
    pre_norm: bool = False
    num_select: int = 300
    transformer_activation: str = "prelu"
    num_patterns: int = 0
    random_refpoints_xy: bool = False

    # DAB-Deformable-DETR
    two_stage: bool = False
    num_feature_levels: int = 4
    dec_n_points: int = 4
    enc_n_points: int = 4

    # Segmentation
    masks: bool = False

    # Loss
    aux_loss: bool = True

    # Matcher
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0

    # Loss coefficients
    cls_loss_coef: float = 1.0
    mask_loss_coef: float = 1.0
    dice_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1
    focal_alpha: float = 0.25

    # dataset parameters
    dataset_file: str = "coco"
    coco_path: str | None = str(Path("data") / "coco2017")
    use_coco_minitrain: bool = False
    coco_panoptic_path: str | None = None
    remove_difficult: bool = False
    fix_size: bool = False
    # custom
    custom_path: str | None = None
    # Supposing label id [0, num_classes -1]
    num_classes: int | None = None

    # Traing utils
    output_dir: str = ""
    note: str = ""
    device: str = "cuda"
    seed: int = 42
    resume: str = ""
    pretrain_model_path: str | None = None
    finetune_ignore: list[str] | None = None
    start_epoch: int = 0
    eval: bool = False
    num_workers: int = 10
    debug: bool = False
    find_unused_params: bool = True

    save_results: bool = False
    save_log: bool = False

    # distributed training parameters
    world_size: int = 1
    dist_url: str = 'env://'
    rank: int = 0
    gpu: int | None = None
    gpu_name: str | None = None
    local_rank: int | None = None
    amp: bool = False
    distributed: bool | None = None
    dist_backend: str | None = None

    # cuDNN
    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None

    # Hyperbolic and hierarchy
    use_hyperbolic: bool = True
    use_bbox_proto: bool = True
    use_proto_cross_attn: bool = True
    use_proto_attn_mask: bool = True
    use_proto_self_attn: bool = False
    use_distance_proto_attn_mask: bool = False
    normalize_distance_proto_attn_mask: bool = False
    use_proto_refiner: bool = False
    use_bbox_proto_refinement: bool = True
    use_bbox_proto_iter_update: bool = True
    hyperbolic_c: float = 0.1
    hyperbolic_train_c: bool = False
    hyperbolic_train_x: bool = False
    hyperbolic_riemannian: bool = False
    hyperbolic_clip_r: float | None = None
    hierarchy_path: str | None = None
    coco_wordtree_dir: str = str(Path("datasets") / "coco_wordtree")
    store_layers_idx: list[int] = [0,1,2,3,4,5]
    store_mode: Literal["shared", "per_layer"] = "per_layer"
    store_memory_size: int = 40
    store_sync: bool = True
    store_geodesic_alpha: float = 0.1
    store_topk_per_sample: int = 15
    store_match_update: bool = True
    bbox_proto_sync: bool = True
    loss_contrastive_lambda_coef_strategy: Literal[
        "uniform", "exp", "pow2", "inverse",
        "exp_inverse", "exp_inverse2", "pow2_inverse"] = "inverse"
    loss_contrastive_temperature: list[float] | float = 0.1
    loss_contrastive_sampling_from_store: bool = True
    loss_contrastive_extra_samples: int = 0
    loss_contrastive_k_pos: int = 2
    loss_contrastive_hierarchical_constraint: bool = True
    loss_contrastive_use_sts: bool = True
    loss_contrastive_use_stp: bool = False
    # Debug only (does not propagate gradients)
    loss_contrastive_use_ptp: bool = False
    loss_contrastive_sts_coef: float = 0.05
    loss_contrastive_stp_coef: float = 0.5
    # Debug-only loss
    loss_contrastive_ptp_coef: float = 0.5
    save_store: bool = True
    # Proto
    use_proto_loss_ce: bool = False
    proto_loss_ce_coef: float = 1.0

    @classmethod
    def parse(
        cls,
        config: str,
        ) -> Self:
        """Parse configuration."""
        if isinstance(config, (str, Path)):
            _config = Path(config)
            # JSON
            if _config.suffix == ".json":
                with _config.open(encoding="utf8") as handle:
                    load_config = json.load(handle)
            # YAML
            elif _config.suffix in (".yaml", ".yml"):
                with _config.open(encoding="utf8") as handle:
                    load_config = yaml.safe_load(handle)
            else:
                _msg = ("`config` should be the path to a JSON or YAML file.")
                raise ValueError(_msg)
            parsed_config = cls.model_validate(load_config)
        else:
            _msg = (f"`config` must be a `str`. Found: {type(config)}")
            raise TypeError(_msg)

        return parsed_config

    @classmethod
    def get_base(
        cls,
        backbone: Literal["resnet50", "resnet101"],
        dilation: bool,  # noqa: FBT001
        num_patterns: Literal[0, 3],
        random_refpoints_xy: bool,  # noqa: FBT001
        **kwargs,  # noqa: ANN003
        ) -> Self:
        """Return a base configuration for Hi-DAB-DETR experiments."""
        base = {
            "lr": 0.0001,
            "lr_backbone": 1e-05,
            "batch_size": 4,
            "weight_decay": 0.0001,
            "epochs": 50,
            "lr_drop": 40,
            "save_checkpoint_interval": 100,
            "clip_max_norm": 0.1,
            "modelname": "hi_dab_detr",
            "frozen_weights": None,
            "position_embedding": "sine",
            "pe_temperatureH": 20,
            "pe_temperatureW": 20,
            "batch_norm_type": "FrozenBatchNorm2d",
            "return_interm_layers": False,
            "enc_layers": 6,
            "dec_layers": 6,
            "dim_feedforward": 2048,
            "hidden_dim": 256,
            "dropout": 0.1,
            "nheads": 8,
            "num_queries": 300,
            "pre_norm": False,
            "num_select": 300,
            "transformer_activation": "prelu",
            "two_stage": False,
            "num_feature_levels": 1,
            "dec_n_points": 0,
            "enc_n_points": 0,
            "masks": False,
            "aux_loss": True,
            "set_cost_class": 2.0,
            "set_cost_bbox": 5.0,
            "set_cost_giou": 2.0,
            "cls_loss_coef": 1.0,
            "mask_loss_coef": 1.0,
            "dice_loss_coef": 1.0,
            "bbox_loss_coef": 5.0,
            "giou_loss_coef": 2.0,
            "eos_coef": 0.1,
            "focal_alpha": 0.25,
            "remove_difficult": False,
            "seed": 42,
            "finetune_ignore": None,
            "start_epoch": 0,
            "num_workers": 2,
            "find_unused_params": True,
            "amp": False,
            "backbone_freeze_keywords": None,
            "fix_size": False,
            "debug": False
        }

        base |= {**kwargs}

        return cls.model_validate(base | {
            "backbone": backbone,
            "dilation": dilation,
            "num_patterns": num_patterns,
            "random_refpoints_xy": random_refpoints_xy,
        })
