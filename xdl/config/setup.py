"""
setup_from_yaml 函数。

只支持当前官方 schema v1：
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
from typing import Any, Dict, Optional, Union

import torch
import yaml
from torch.utils.data import DataLoader

from .builder import (
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transform,
)
from .dataclass import TrainSetup
from .errors import ConfigValidationError
from .resolver import load_config_with_schema, to_plain_dict

def _load_raw_yaml(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ConfigValidationError("Top-level config must be a mapping", field_path=str(config_path))
    return data


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
    dataset_config: Dict[str, Any],
    built_transforms: Dict[str, Any],
) -> Optional[Any]:
    params = dataset_config.get("params")
    if isinstance(params, dict) and "transform" in params:
        return _resolve_transform_value(params.get("transform"), built_transforms)
    return None


def _resolve_dataset_value(
    dataset_name: str,
    dataset_value: Any,
    built_datasets: Dict[str, Any],
    built_transforms: Dict[str, Any],
) -> Any:
    if dataset_name in built_datasets:
        return built_datasets[dataset_name]
    if isinstance(dataset_value, str) and dataset_value in built_datasets:
        return built_datasets[dataset_value]
    if isinstance(dataset_value, dict):
        transform_value = _resolve_dataset_transform_value(dataset_value, built_transforms)
        return build_dataset(dataset_value, transform=transform_value)
    raise ConfigValidationError(f"Unable to resolve dataset for dataloader '{dataset_name}'")


def _collect_transform_configs(root_config: Dict[str, Any]) -> Dict[str, Any]:
    alias_mapping = (
        ("train_transforms", "train"),
        ("val_transforms", "val"),
        ("test_transforms", "test"),
    )
    collected: Dict[str, Any] = {}

    for alias_key, transform_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[transform_name] = alias_value

    return collected


def _collect_dataset_configs(root_config: Dict[str, Any]) -> Dict[str, Any]:
    alias_mapping = (
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "test"),
    )
    collected: Dict[str, Any] = {}

    for alias_key, dataset_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[dataset_name] = alias_value

    return collected


def _collect_dataloader_configs(root_config: Dict[str, Any]) -> Dict[str, Any]:
    alias_mapping = (
        ("train_dataloader", "train"),
        ("val_dataloader", "val"),
        ("test_dataloader", "test"),
    )
    collected: Dict[str, Any] = {}

    for alias_key, dataloader_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[dataloader_name] = alias_value

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
    从 YAML 配置文件构建完整的训练配置。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_config = _load_raw_yaml(path)

    merged_config = load_config_with_schema(raw_config)
    resolved_config = to_plain_dict(merged_config, resolve=True)

    model_config = resolved_config.get("model")
    if not isinstance(model_config, dict) or not model_config:
        raise ConfigValidationError("model config cannot be empty")
    model = build_model(model_config)

    transform_config = _collect_transform_configs(resolved_config)
    dataset_config = _collect_dataset_configs(resolved_config)
    dataloader_config = _collect_dataloader_configs(resolved_config)
    dataloader_defaults = _collect_dataloader_defaults(resolved_config)
    trainer_config = resolved_config.get("trainer", {})
    trainer_batch_size = trainer_config.get("batch_size")

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
        dataset_value = dataloader_cfg.get("dataset")
        dataset = _resolve_dataset_value(
            dataloader_name,
            dataset_value,
            built_datasets,
            built_transforms,
        )
        built_dataloaders[dataloader_name] = build_dataloader(dataset, merged_dataloader_cfg)

    optimization_config = resolved_config.get("optimization") or {}
    optimizer_config = optimization_config.get("optimizer")
    if not optimizer_config:
        raise ConfigValidationError("optimizer config cannot be empty")
    optimizer = build_optimizer(model, optimizer_config)

    scheduler_config = optimization_config.get("scheduler") or {}
    scheduler = build_scheduler(optimizer, scheduler_config)

    loss_config = resolved_config.get("loss", [])
    loss_fn = build_loss(loss_config)

    metrics_config = resolved_config.get("metrics", [])
    metrics = build_metrics(metrics_config)

    runtime_config = resolved_config.get("runtime", {})
    runtime_device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    selected_device = str(device) if device is not None else str(runtime_device)

    selected_batch_size = trainer_batch_size
    if selected_batch_size is None:
        selected_batch_size = dataloader_defaults.get("batch_size")
    if selected_batch_size is None and "train" in dataloader_config:
        selected_batch_size = dataloader_config["train"].get("params", {}).get("batch_size")
    if selected_batch_size is None and built_dataloaders.get("train") is not None:
        selected_batch_size = built_dataloaders["train"].batch_size

    return TrainSetup(
        model=model,
        train_loader=built_dataloaders.get("train"),
        optimizer=optimizer,
        loss_fn=loss_fn,
        val_loader=built_dataloaders.get("val"),
        test_loader=built_dataloaders.get("test"),
        scheduler=scheduler,
        metrics=metrics,
        full_config=resolved_config,
        device=selected_device,
        num_epochs=int(trainer_config.get("max_epochs", 100)),
        batch_size=int(selected_batch_size or 128),
    )


__all__ = ["setup_from_yaml"]
