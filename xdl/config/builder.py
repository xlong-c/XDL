"""
组件构建器。

职责：
- 解析统一组件配置格式：`type + source + params`
- 兼容旧格式：`name + from_library + params`
- 通过 registry 或 import path 定位组件
- 实例化模型、数据集、优化器、scheduler、loss、metrics 等对象
"""

import importlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .errors import ComponentResolutionError, ConfigValidationError

LEGACY_SOURCE_ALIASES = {
    ("model", "local"): "registry",
    ("model", "registry"): "registry",
    ("model", "torch"): "torch.nn",
    ("model", "torchvision"): "torchvision.models",
    ("dataset", "local"): "registry",
    ("dataset", "registry"): "registry",
    ("dataset", "torchvision"): "torchvision.datasets",
    ("optimizer", "local"): "registry",
    ("optimizer", "registry"): "registry",
    ("optimizer", "torch"): "torch.optim",
    ("scheduler", "local"): "registry",
    ("scheduler", "registry"): "registry",
    ("scheduler", "torch"): "torch.optim.lr_scheduler",
    ("loss", "local"): "registry",
    ("loss", "registry"): "registry",
    ("loss", "torch"): "torch.nn",
    ("metric", "local"): "registry",
    ("metric", "registry"): "registry",
    ("transform", "local"): "registry",
    ("transform", "registry"): "registry",
    ("transform", "torchvision"): "torchvision.transforms",
}

REGISTRY_IMPORTS = {
    "model": "xdl.model",
    "dataset": "xdl.dataset",
    "loss": "xdl.loss",
    "metric": "xdl.metric",
    "optimizer": "xdl.optimizer",
    "scheduler": "xdl.scheduler",
}


def _normalize_source(kind: str, source: Optional[str], default_source: str) -> str:
    raw_source = source or default_source
    return LEGACY_SOURCE_ALIASES.get((kind, raw_source), raw_source)


def _ensure_registry_populated(kind: str) -> None:
    module_path = REGISTRY_IMPORTS.get(kind)
    if module_path is not None:
        importlib.import_module(module_path)


def _get_registry(kind: str) -> Any:
    _ensure_registry_populated(kind)
    from xdl.utils.registry import (
        DATASET_REGISTRY,
        LOSS_REGISTRY,
        METRIC_REGISTRY,
        MODEL_REGISTRY,
        OPTIMIZER_REGISTRY,
        SCHEDULER_REGISTRY,
        TRANSFORM_REGISTRY,
    )

    registries = {
        "model": MODEL_REGISTRY,
        "dataset": DATASET_REGISTRY,
        "loss": LOSS_REGISTRY,
        "metric": METRIC_REGISTRY,
        "optimizer": OPTIMIZER_REGISTRY,
        "scheduler": SCHEDULER_REGISTRY,
        "transform": TRANSFORM_REGISTRY,
    }
    if kind not in registries:
        raise ConfigValidationError(f"Unsupported registry kind: {kind}")
    return registries[kind]


def _resolve_component(kind: str, component_type: str, source: str) -> Any:
    if source == "registry":
        registry = _get_registry(kind)
        try:
            return registry.get(component_type)
        except KeyError as exc:
            raise ComponentResolutionError(kind, component_type, source=source) from exc

    try:
        module = importlib.import_module(source)
    except ImportError as exc:
        raise ComponentResolutionError(kind, component_type, source=source) from exc

    try:
        return getattr(module, component_type)
    except AttributeError as exc:
        raise ComponentResolutionError(kind, component_type, source=source) from exc


def _extract_component_config(
    config: Dict[str, Any],
    *,
    kind: str,
    wrapper_keys: Sequence[str] = (),
    default_source: str = "registry",
) -> Dict[str, Any]:
    if not isinstance(config, dict) or not config:
        raise ConfigValidationError(f"{kind} config cannot be empty")

    component_cfg = config
    for key in wrapper_keys:
        wrapped = component_cfg.get(key)
        if isinstance(wrapped, dict):
            component_cfg = wrapped
            break

    component_type = component_cfg.get("type") or component_cfg.get("name")
    if not component_type:
        raise ConfigValidationError(f"{kind} config requires 'type' or 'name'")

    source = _normalize_source(
        kind,
        component_cfg.get("source") or component_cfg.get("from_library"),
        default_source,
    )
    params = dict(component_cfg.get("params") or {})
    ignored_keys = {"type", "name", "source", "from_library", "params"}
    extras = {key: value for key, value in component_cfg.items() if key not in ignored_keys}

    return {
        "type": component_type,
        "source": source,
        "params": params,
        **extras,
    }


def _build_single_transform(config: Dict[str, Any]) -> Any:
    transform_cfg = _extract_component_config(
        config,
        kind="transform",
        default_source="torchvision.transforms",
    )
    transform_cls = _resolve_component(
        "transform",
        transform_cfg["type"],
        transform_cfg["source"],
    )
    return transform_cls(**transform_cfg["params"])


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """从配置构建模型。"""

    model_cfg = _extract_component_config(
        config,
        kind="model",
        wrapper_keys=("backbone",),
        default_source="registry",
    )
    model_cls = _resolve_component("model", model_cfg["type"], model_cfg["source"])
    return model_cls(**model_cfg["params"])


def build_transform(config: Dict[str, Any]) -> Any:
    """从配置构建 transform 或 transform pipeline。"""

    if not config:
        return None
    if not isinstance(config, dict):
        raise ConfigValidationError("transform config must be a mapping")

    if "items" not in config and "transforms" not in config and config.get("type") not in {
        "compose",
        "list",
        "raw",
    } and "combination_strategy" not in config:
        return _build_single_transform(config)

    import torchvision.transforms as transforms

    pipeline_type = (config.get("type") or config.get("combination_strategy") or "compose").lower()
    raw_items = config.get("items") or config.get("transforms") or []
    transforms_list = [_build_single_transform(item) for item in raw_items]

    if pipeline_type == "compose":
        return transforms.Compose(transforms_list)
    if pipeline_type in {"list", "raw"}:
        return transforms_list

    raise ConfigValidationError(f"Unsupported transform pipeline type: {pipeline_type}")


def build_dataset(config: Dict[str, Any], transform: Optional[Any] = None) -> Any:
    """从配置构建数据集。"""

    dataset_cfg = _extract_component_config(
        config,
        kind="dataset",
        default_source="torchvision.datasets",
    )
    params = dataset_cfg["params"].copy()

    transform_value = transform
    if transform_value is None and config.get("transform") is not None:
        raw_transform = config.get("transform")
        if isinstance(raw_transform, dict):
            transform_value = build_transform(raw_transform)
        elif not isinstance(raw_transform, str):
            transform_value = raw_transform

    if transform_value is not None:
        params["transform"] = transform_value

    dataset_cls = _resolve_component("dataset", dataset_cfg["type"], dataset_cfg["source"])
    return dataset_cls(**params)


def build_dataloader(dataset: Any, config: Dict[str, Any]) -> DataLoader:
    """从配置构建 DataLoader。"""

    if not isinstance(config, dict):
        raise ConfigValidationError("dataloader config must be a mapping")
    params = dict(config.get("params") or {})
    return DataLoader(dataset, **params)


def _module_parameters(module: Any) -> List[torch.nn.Parameter]:
    if not hasattr(module, "parameters"):
        raise ConfigValidationError(f"Target '{type(module).__name__}' does not expose parameters()")
    return list(module.parameters())


def _select_model_parameters(
    model: torch.nn.Module,
    target_modules: Optional[Any],
) -> Iterable[torch.nn.Parameter]:
    if target_modules is None:
        return model.parameters()

    if isinstance(target_modules, str):
        target_names: List[str] = [target_modules]
    elif isinstance(target_modules, Sequence):
        target_names = [str(name) for name in target_modules]
    else:
        raise ConfigValidationError("optimizer target modules must be a string or list of strings")

    aliases = {"model", "all", "backbone"}
    collected: List[torch.nn.Parameter] = []
    seen_ids = set()

    for name in target_names:
        if name in aliases:
            modules = [model]
        else:
            if not hasattr(model, name):
                raise ConfigValidationError(f"Model has no submodule or attribute named '{name}'")
            modules = [getattr(model, name)]

        for module in modules:
            for parameter in _module_parameters(module):
                if id(parameter) not in seen_ids:
                    seen_ids.add(id(parameter))
                    collected.append(parameter)

    if not collected:
        raise ConfigValidationError("No parameters selected for optimizer")
    return collected


def _build_param_groups(
    model: torch.nn.Module,
    param_group_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(param_group_configs, dict):
        raise ConfigValidationError("param_groups must be a mapping")

    groups: List[Dict[str, Any]] = []
    for group_name, group_params in param_group_configs.items():
        if not isinstance(group_params, dict):
            raise ConfigValidationError(f"param group '{group_name}' must be a mapping")
        groups.append(
            {
                "params": list(_select_model_parameters(model, group_name)),
                **group_params,
            }
        )

    if not groups:
        raise ConfigValidationError("param_groups cannot be empty")
    return groups


def build_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """从配置构建优化器。"""

    optimizer_cfg = _extract_component_config(
        config,
        kind="optimizer",
        wrapper_keys=("main_optimizer",),
        default_source="torch.optim",
    )
    optimizer_cls = _resolve_component(
        "optimizer",
        optimizer_cfg["type"],
        optimizer_cfg["source"],
    )
    params = optimizer_cfg["params"].copy()

    param_group_configs = optimizer_cfg.get("param_groups")
    if param_group_configs:
        groups = _build_param_groups(model, param_group_configs)
        return optimizer_cls(groups, **params)

    target_modules = optimizer_cfg.get("target_modules")
    if target_modules is None:
        target_modules = optimizer_cfg.get("model")

    return optimizer_cls(_select_model_parameters(model, target_modules), **params)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[Any]:
    """从配置构建学习率调度器。"""

    if not config:
        return None

    scheduler_cfg = _extract_component_config(
        config,
        kind="scheduler",
        wrapper_keys=("main_scheduler",),
        default_source="torch.optim.lr_scheduler",
    )
    scheduler_cls = _resolve_component(
        "scheduler",
        scheduler_cfg["type"],
        scheduler_cfg["source"],
    )
    return scheduler_cls(optimizer, **scheduler_cfg["params"])


def build_loss(config: Any) -> torch.nn.Module:
    """从配置构建损失函数。"""

    if not config:
        raise ConfigValidationError("loss config cannot be empty")

    loss_items = [config] if isinstance(config, dict) else list(config)
    if not loss_items:
        raise ConfigValidationError("loss config cannot be empty")

    if len(loss_items) == 1:
        loss_cfg = _extract_component_config(
            loss_items[0],
            kind="loss",
            default_source="torch.nn",
        )
        loss_cls = _resolve_component("loss", loss_cfg["type"], loss_cfg["source"])
        return loss_cls(**loss_cfg["params"])

    from .loss_weighted import WeightedLoss

    losses = []
    weights = []
    for loss_item in loss_items:
        loss_cfg = _extract_component_config(
            loss_item,
            kind="loss",
            default_source="torch.nn",
        )
        loss_cls = _resolve_component("loss", loss_cfg["type"], loss_cfg["source"])
        losses.append(loss_cls(**loss_cfg["params"]))
        weights.append(float(loss_cfg.get("weight", 1.0)))

    return WeightedLoss(losses, weights)


def build_metrics(config: List[Dict[str, Any]]) -> List[Any]:
    """从配置构建指标列表。"""

    metrics = []
    for metric_item in config or []:
        metric_cfg = _extract_component_config(
            metric_item,
            kind="metric",
            default_source="registry",
        )
        metric_cls = _resolve_component("metric", metric_cfg["type"], metric_cfg["source"])
        metrics.append(metric_cls(**metric_cfg["params"]))
    return metrics
