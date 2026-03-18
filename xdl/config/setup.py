"""
setup_from_yaml 函数。

支持两类配置：
- 新 schema：`runtime / trainer / model / data / optimization / loss / metrics`
- 旧 schema：`training / core_config / data_config / save_config / logger_config`
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

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

LEGACY_TOP_LEVEL_KEYS = {
    "training",
    "core_config",
    "data_config",
    "save_config",
    "logger_config",
    "unified_logger",
}


def _load_raw_yaml(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ConfigValidationError("Top-level config must be a mapping", field_path=str(config_path))
    return data


def _is_legacy_config(config: Dict[str, Any]) -> bool:
    return any(key in config for key in LEGACY_TOP_LEVEL_KEYS)


def _legacy_component_to_schema(
    config: Dict[str, Any],
    *,
    wrapper_key: Optional[str] = None,
    default_source: str,
) -> Optional[Dict[str, Any]]:
    if not config:
        return None

    component_cfg = config.get(wrapper_key, config) if wrapper_key else config
    if not isinstance(component_cfg, dict) or not component_cfg:
        return None

    component_type = component_cfg.get("type") or component_cfg.get("name")
    if not component_type:
        return None

    normalized = {
        "type": component_type,
        "source": component_cfg.get("source") or component_cfg.get("from_library") or default_source,
        "params": dict(component_cfg.get("params") or {}),
    }

    for key, value in component_cfg.items():
        if key not in {"type", "name", "source", "from_library", "params"}:
            normalized[key] = value
    return normalized


def _legacy_transform_pipeline_to_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for item in config.get("transforms", []):
        normalized_item = _legacy_component_to_schema(
            item,
            default_source="torchvision.transforms",
        )
        if normalized_item is not None:
            items.append(normalized_item)

    return {
        "type": config.get("type") or config.get("combination_strategy") or "compose",
        "items": items,
    }


def _normalize_legacy_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    training_config = raw_config.get("training", {})
    core_config = raw_config.get("core_config", {})
    data_config = raw_config.get("data_config", {})
    save_config = raw_config.get("save_config", {})
    logger_config = raw_config.get("unified_logger") or raw_config.get("logger_config", {})

    transforms_cfg: Dict[str, Any] = {}
    datasets_cfg: Dict[str, Any] = {}
    dataloaders_cfg: Dict[str, Any] = {}

    legacy_transform_config = data_config.get("transform", {})
    for legacy_name, schema_name in (
        ("train_transform", "train"),
        ("val_transform", "val"),
        ("test_transform", "test"),
    ):
        if legacy_name in legacy_transform_config:
            transforms_cfg[schema_name] = _legacy_transform_pipeline_to_schema(
                legacy_transform_config[legacy_name]
            )

    legacy_dataset_config = data_config.get("dataset", {})
    for legacy_name, schema_name in (
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "test"),
    ):
        dataset_cfg = legacy_dataset_config.get(legacy_name)
        normalized_dataset = _legacy_component_to_schema(
            dataset_cfg or {},
            default_source="torchvision.datasets",
        )
        if normalized_dataset is None:
            continue

        transform_name = dataset_cfg.get("transform") if isinstance(dataset_cfg, dict) else None
        if isinstance(transform_name, str):
            if transform_name.endswith("_transform"):
                normalized_dataset["transform"] = f"${{data.transforms.{transform_name[:-10]}}}"
            else:
                normalized_dataset["transform"] = f"${{data.transforms.{transform_name}}}"
        datasets_cfg[schema_name] = normalized_dataset

    legacy_dataloader_config = data_config.get("dataloader", {})
    for legacy_name, schema_name in (
        ("train_loader", "train"),
        ("val_loader", "val"),
        ("test_loader", "test"),
    ):
        loader_cfg = legacy_dataloader_config.get(legacy_name)
        if not isinstance(loader_cfg, dict):
            continue
        dataloaders_cfg[schema_name] = {
            "dataset": f"${{data.datasets.{schema_name}}}",
            "params": dict(loader_cfg.get("params") or {}),
        }

    loss_items = []
    for loss_cfg in core_config.get("loss", []):
        normalized_loss = _legacy_component_to_schema(loss_cfg, default_source="torch.nn")
        if normalized_loss is not None:
            loss_items.append(normalized_loss)

    metric_items = []
    for metric_cfg in core_config.get("metrics", []):
        normalized_metric = _legacy_component_to_schema(metric_cfg, default_source="registry")
        if normalized_metric is not None:
            metric_items.append(normalized_metric)

    normalized_config: Dict[str, Any] = {
        "config_version": 1,
        "runtime": {
            "device": training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            "output_dir": save_config.get("base_dir", "./others"),
            "experiment_name": save_config.get("exp_name")
            or logger_config.get("experiment_name")
            or "default_exp",
        },
        "trainer": {
            "max_epochs": training_config.get("max_epochs", training_config.get("num_epochs", 100)),
            "batch_size": training_config.get("batch_size"),
            "gradient_accumulation_steps": training_config.get(
                "gradient_accumulation_steps",
                training_config.get("grad_steps", 1),
            ),
            "grad_clip_max_norm": training_config.get("grad_clip_max_norm"),
        },
        "model": _legacy_component_to_schema(
            core_config.get("model", {}),
            wrapper_key="backbone",
            default_source="registry",
        ),
        "data": {
            "transforms": transforms_cfg,
            "datasets": datasets_cfg,
            "dataloaders": dataloaders_cfg,
        },
        "optimization": {
            "optimizer": _legacy_component_to_schema(
                core_config.get("optimizer", {}),
                wrapper_key="main_optimizer",
                default_source="torch.optim",
            ),
            "scheduler": _legacy_component_to_schema(
                core_config.get("scheduler", {}),
                wrapper_key="main_scheduler",
                default_source="torch.optim.lr_scheduler",
            ),
        },
        "loss": loss_items,
        "metrics": metric_items,
        "logging": {
            "log_dir": logger_config.get("log_dir", "./others/logs"),
            "enable_console": logger_config.get("enable_console", True),
            "enable_tensorboard": logger_config.get("enable_tensorboard", False),
            "enable_wandb": logger_config.get("enable_wandb", False),
            "wandb_project": logger_config.get("wandb_project"),
            "wandb_entity": logger_config.get("wandb_entity"),
        },
        "checkpoint": {
            "dirpath": save_config.get("base_dir"),
            "save_last": True,
            "save_top_k": save_config.get("max_best_checkpoints", 1),
            "monitor": save_config.get("best_metric"),
            "mode": "max" if save_config.get("maximize_best_metric", False) else "min",
            "every_n_epochs": save_config.get("save_every_n_epochs", 1),
        },
    }

    if normalized_config["optimization"]["optimizer"] is None and normalized_config["optimization"]["scheduler"] is None:
        normalized_config["optimization"] = None

    return normalized_config


def _resolve_transform_value(
    transform_value: Any,
    built_transforms: Dict[str, Any],
) -> Optional[Any]:
    if transform_value is None:
        return None
    if isinstance(transform_value, str) and transform_value in built_transforms:
        return built_transforms[transform_value]
    if isinstance(transform_value, dict):
        return build_transform(transform_value)
    return transform_value


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
        transform_value = _resolve_transform_value(dataset_value.get("transform"), built_transforms)
        return build_dataset(dataset_value, transform=transform_value)
    raise ConfigValidationError(f"Unable to resolve dataset for dataloader '{dataset_name}'")


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
    normalized_config = _normalize_legacy_config(raw_config) if _is_legacy_config(raw_config) else raw_config

    merged_config = load_config_with_schema(normalized_config)
    resolved_config = to_plain_dict(merged_config, resolve=True)

    model_config = resolved_config.get("model")
    if not isinstance(model_config, dict) or not model_config:
        raise ConfigValidationError("model config cannot be empty")
    model = build_model(model_config)

    data_config = resolved_config.get("data", {})
    transform_config = data_config.get("transforms", {})
    dataset_config = data_config.get("datasets", {})
    dataloader_config = data_config.get("dataloaders", {})

    built_transforms: Dict[str, Any] = {}
    for transform_name, transform_cfg in transform_config.items():
        built_transforms[transform_name] = build_transform(transform_cfg)

    built_datasets: Dict[str, Any] = {}
    for dataset_name, dataset_cfg in dataset_config.items():
        transform_value = _resolve_transform_value(dataset_cfg.get("transform"), built_transforms)
        built_datasets[dataset_name] = build_dataset(dataset_cfg, transform=transform_value)

    built_dataloaders: Dict[str, DataLoader] = {}
    for dataloader_name, dataloader_cfg in dataloader_config.items():
        dataset_value = dataloader_cfg.get("dataset")
        dataset = _resolve_dataset_value(
            dataloader_name,
            dataset_value,
            built_datasets,
            built_transforms,
        )
        built_dataloaders[dataloader_name] = build_dataloader(dataset, dataloader_cfg)

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
    trainer_config = resolved_config.get("trainer", {})

    runtime_device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    selected_device = str(device) if device is not None else str(runtime_device)
    selected_batch_size = trainer_config.get("batch_size")
    if selected_batch_size is None and "train" in dataloader_config:
        selected_batch_size = dataloader_config["train"].get("params", {}).get("batch_size", 128)

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
