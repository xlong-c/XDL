"""
setup_from_yaml 函数.

只支持当前官方 schema v1:
- `runtime`
- `trainer`
- `model`
- `train_transforms / val_transforms / test_transforms`
- `train_dataset / val_dataset / test_dataset`
- `train_dataloader / val_dataloader / test_dataloader`
- `dataloader_defaults`
- `optimization / loss / metrics`
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union
from collections.abc import Mapping

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .builder import (
    build_callbacks,
    build_collate_fn,
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task,
    build_transform,
)
from .accelerate_config import AccelerateConfig, DeepSpeedConfig
from .dataclass import TrainSetup
from .errors import ConfigValidationError
from .resolver import load_config_with_schema, to_plain_dict
from .schema import (
    CONFIG_SCHEMA_VERSION,
    CheckpointConfig,
    LoggingConfig,
    RuntimeConfig,
    TrainerConfig,
)

T = TypeVar("T")


def _to_structured(node: Any, expected_type: Type[T]) -> T:
    """把 OmegaConf 节点转换为声明好的结构化配置类型."""

    value = OmegaConf.to_object(node)
    if not isinstance(value, expected_type):
        raise ConfigValidationError(
            f"Expected structured config {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _resolve_transform_value(
    transform_value: Any,
    built_transforms: Dict[str, Any],
) -> Optional[Any]:
    if transform_value is None:
        return None
    if isinstance(transform_value, str) and transform_value in built_transforms:
        return built_transforms[transform_value]
    if isinstance(transform_value, list):
        return build_transform(transform_value)
    if isinstance(transform_value, dict):
        return build_transform(transform_value)
    return transform_value


def _resolve_dataset_transform_value(
    dataset_config: Mapping[str, Any],
    built_transforms: Dict[str, Any],
) -> Optional[Any]:
    params = dataset_config.get("params")
    if isinstance(params, dict) and "transform" in params:
        return _resolve_transform_value(params.get("transform"), built_transforms)
    return None


def _resolve_dataset_value(
    dataset_name: str,
    dataloader_cfg: Dict[str, Any],
    built_datasets: Dict[str, Any],
    built_transforms: Dict[str, Any],
) -> Any:
    if dataset_name not in built_datasets:
        dataset_value = dataloader_cfg.get("dataset")
        if isinstance(dataset_value, str) and dataset_value in built_datasets:
            return built_datasets[dataset_value]
        if isinstance(dataset_value, Mapping) and dataset_value.get("target"):
            transform_value = _resolve_dataset_transform_value(dataset_value, built_transforms)
            return build_dataset(dict(dataset_value), transform=transform_value)
        if dataset_value is not None and not isinstance(dataset_value, str):
            return dataset_value
        raise ConfigValidationError(f"No pre-built dataset found for dataloader '{dataset_name}'")
    return built_datasets[dataset_name]


def _collect_configs(
    root_config: Dict[str, Any],
    alias_mapping: tuple,
) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    for alias_key, short_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[short_name] = alias_value
    return collected


def _collect_dataloader_defaults(root_config: Dict[str, Any]) -> Dict[str, Any]:
    top_level_defaults = root_config.get("dataloader_defaults")
    if top_level_defaults is None:
        return {}
    if not isinstance(top_level_defaults, dict):
        raise ConfigValidationError("dataloader_defaults must be a mapping")
    return dict(top_level_defaults)


def _merge_dataloader_params(
    dataloader_cfg: Dict[str, Any],
    *,
    dataloader_defaults: Dict[str, Any],
    trainer_batch_size: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(dataloader_cfg, dict):
        raise ConfigValidationError("dataloader config must be a mapping")

    merged_config = dict(dataloader_cfg)
    merged_params = {
        key: value
        for key, value in dataloader_defaults.items()
        if value is not None
    }
    if merged_params.get("batch_size") is None and trainer_batch_size is not None:
        merged_params["batch_size"] = int(trainer_batch_size)

    loader_params = dataloader_cfg.get("params") or {}
    if not isinstance(loader_params, dict):
        raise ConfigValidationError("dataloader params must be a mapping")
    merged_params.update(loader_params)
    merged_config["params"] = merged_params
    return merged_config


def setup_from_yaml(
    config_path: Union[str, Path],
    device: Optional[str] = None,
) -> TrainSetup:
    """
    从 YAML 配置文件构建完整的训练配置.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    merged_config = load_config_with_schema(path, resolve=True)
    resolved_config = to_plain_dict(merged_config, resolve=False)

    declared_version = resolved_config.get("config_version")
    if declared_version is not None and declared_version != CONFIG_SCHEMA_VERSION:
        raise ConfigValidationError(
            f"Unsupported config version {declared_version}. "
            f"Expected {CONFIG_SCHEMA_VERSION}."
        )

    task_config = resolved_config.get("task")
    model_config = resolved_config.get("model")
    if task_config:
        if not isinstance(task_config, dict):
            raise ConfigValidationError("task config must be a mapping")
        model = build_task(task_config)
    elif not isinstance(model_config, dict) or not model_config:
        raise ConfigValidationError("model config cannot be empty")
    else:
        model = build_model(model_config)

    transform_config = _collect_configs(resolved_config, (
        ("train_transforms", "train"),
        ("val_transforms", "val"),
        ("test_transforms", "test"),
    ))
    dataset_config = _collect_configs(resolved_config, (
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "test"),
    ))
    dataloader_config = _collect_configs(resolved_config, (
        ("train_dataloader", "train"),
        ("val_dataloader", "val"),
        ("test_dataloader", "test"),
    ))
    dataloader_defaults = _collect_dataloader_defaults(resolved_config)
    trainer_cfg = _to_structured(merged_config.trainer, TrainerConfig)
    trainer_batch_size = trainer_cfg.batch_size

    built_transforms: Dict[str, Any] = {}
    for transform_name, transform_cfg in transform_config.items():
        built_transforms[transform_name] = build_transform(transform_cfg)

    built_datasets: Dict[str, Any] = {}
    for dataset_name, dataset_cfg in dataset_config.items():
        transform_value = _resolve_dataset_transform_value(dataset_cfg, built_transforms)
        built_datasets[dataset_name] = build_dataset(dataset_cfg, transform=transform_value)

    built_dataloaders: Dict[str, DataLoader] = {}
    for dataloader_name, dataloader_cfg in dataloader_config.items():
        merged_dataloader_cfg = _merge_dataloader_params(
            dataloader_cfg,
            dataloader_defaults=dataloader_defaults,
            trainer_batch_size=trainer_batch_size,
        )
        dataset = _resolve_dataset_value(
            dataloader_name,
            dataloader_cfg,
            built_datasets,
            built_transforms,
        )
        collate_fn = build_collate_fn(
            dataloader_cfg.get("collate_fn") or dataloader_defaults.get("collate_fn")
        )
        built_dataloaders[dataloader_name] = build_dataloader(
            dataset, merged_dataloader_cfg, collate_fn=collate_fn
        )

    optimization_config = resolved_config.get("optimization") or {}
    optimizer_config = optimization_config.get("optimizer")
    if not optimizer_config and not task_config:
        raise ConfigValidationError("optimizer config cannot be empty")
    optimizer = build_optimizer(model, optimizer_config) if optimizer_config else None

    scheduler_config = optimization_config.get("scheduler") or {}
    scheduler = build_scheduler(optimizer, scheduler_config) if optimizer is not None else None

    loss_config = resolved_config.get("loss", [])
    if loss_config:
        loss_fn = build_loss(loss_config)
    elif task_config:
        loss_fn = None
    else:
        raise ConfigValidationError("loss config cannot be empty")

    metrics_config = resolved_config.get("metrics", [])
    metrics = build_metrics(metrics_config)
    callbacks = build_callbacks(resolved_config.get("callbacks", []))

    runtime_cfg = _to_structured(merged_config.runtime, RuntimeConfig)
    if device is not None:
        runtime_cfg.device = str(device)

    logging_cfg = _to_structured(merged_config.logging, LoggingConfig)
    checkpoint_cfg = _to_structured(merged_config.checkpoint, CheckpointConfig)
    accelerate_cfg = (
        _to_structured(merged_config.accelerate, AccelerateConfig)
        if merged_config.accelerate is not None
        else None
    )
    deepspeed_cfg = (
        _to_structured(merged_config.deepspeed, DeepSpeedConfig)
        if merged_config.deepspeed is not None
        else None
    )

    selected_batch_size = trainer_batch_size
    if selected_batch_size is None:
        selected_batch_size = dataloader_defaults.get("batch_size")
    if selected_batch_size is None and "train" in dataloader_config:
        selected_batch_size = dataloader_config["train"].get("params", {}).get("batch_size")
    if selected_batch_size is None and built_dataloaders.get("train") is not None:
        selected_batch_size = built_dataloaders["train"].batch_size
    if selected_batch_size is None:
        raise ConfigValidationError(
            "batch_size must be specified in at least one of: "
            "trainer.batch_size, dataloader_defaults.batch_size, "
            "or train_dataloader.params.batch_size"
        )

    train_loader = built_dataloaders.get("train")
    if train_loader is None:
        raise ConfigValidationError("train_dataloader config is required")
    trainer_cfg.batch_size = int(selected_batch_size)

    return TrainSetup(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_loader=built_dataloaders.get("val"),
        test_loader=built_dataloaders.get("test"),
        scheduler=scheduler,
        metrics=metrics,
        callbacks=callbacks,
        full_config=resolved_config,
        trainer=trainer_cfg,
        runtime=runtime_cfg,
        logging=logging_cfg,
        checkpoint=checkpoint_cfg,
        accelerate=accelerate_cfg,
        deepspeed=deepspeed_cfg,
    )


__all__ = ["setup_from_yaml"]
