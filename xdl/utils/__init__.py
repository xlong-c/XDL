"""
Utilities module for deep learning project.
"""

from .checkpoint import (
    detect_and_load_checkpoint,
    flatten_state_dict,
    format_value_for_dirname,
    generate_checkpoint_dirname,
    save_checkpoint,
    unflatten_state_dict,
)
from .tools import (
    enable_tensor_debug_info,
    path_win2wsl,
    resolve_dtype,
    save_yaml,
    seed_everything,
)
from .tiling import tile_inference
from .registry import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    Registry,
    build_dataset,
    build_loss,
    build_metric,
    build_model,
    build_optimizer,
    build_scheduler,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_scheduler,
)

# ProgressManager功能已合并到TqdmLoggerCallback中

__all__ = [
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "register_model",
    "register_dataset",
    "register_optimizer",
    "register_scheduler",
    "register_loss",
    "register_metric",
    "build_model",
    "build_dataset",
    "build_optimizer",
    "build_scheduler",
    "build_loss",
    "build_metric",
    # Checkpoint
    "format_value_for_dirname",
    "generate_checkpoint_dirname",
    "flatten_state_dict",
    "unflatten_state_dict",
    "save_checkpoint",
    "detect_and_load_checkpoint",
    # Tools
    "enable_tensor_debug_info",
    "path_win2wsl",
    "resolve_dtype",
    "save_yaml",
    "seed_everything",
    # Tiling
    "tile_inference",
]
