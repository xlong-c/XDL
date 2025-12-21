"""
Utilities module for deep learning project.
"""

from .registry import (
    Registry,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    register_model,
    register_dataset,
    register_optimizer,
    register_scheduler,
    register_loss,
    register_metric,
    build_model,
    build_dataset,
    build_optimizer,
    build_scheduler,
    build_loss,
    build_metric
)

from .checkpoint import (
    format_value_for_dirname,
    generate_checkpoint_dirname,
    flatten_state_dict,
    unflatten_state_dict,
    save_checkpoint,
    detect_and_load_checkpoint,
)

# ProgressManager功能已合并到TqdmLoggerCallback中

__all__ = [
    # Registry
    'Registry',
    'MODEL_REGISTRY',
    'DATASET_REGISTRY',
    'OPTIMIZER_REGISTRY',
    'SCHEDULER_REGISTRY',
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
    'register_model',
    'register_dataset',
    'register_optimizer',
    'register_scheduler',
    'register_loss',
    'register_metric',
    'build_model',
    'build_dataset',
    'build_optimizer',
    'build_scheduler',
    'build_loss',
    'build_metric',
    # Checkpoint
    'format_value_for_dirname',
    'generate_checkpoint_dirname',
    'flatten_state_dict',
    'unflatten_state_dict',
    'save_checkpoint',
    'detect_and_load_checkpoint',
]
