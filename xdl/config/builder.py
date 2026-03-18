"""
组件构建器。

职责：
- 官方主格式：`target + params`
- 兼容旧格式：`type + source + params`、`name + from_library + params`
- transform 支持紧凑语法：列表即 `Compose`，并支持 inline 参数
- 通过 registry 或 import path 定位组件
- 实例化模型、数据集、优化器、scheduler、loss、metrics 等对象
"""

import importlib
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .errors import ComponentResolutionError, ConfigValidationError

DEFAULT_KIND_SOURCES = {
    "model": "registry",
    "dataset": "torchvision.datasets",
    "optimizer": "torch.optim",
    "scheduler": "torch.optim.lr_scheduler",
    "loss": "torch.nn",
    "metric": "registry",
    "transform": "torchvision.transforms",
}

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

TRANSFORM_CONFIG_META_KEYS = {
    "target",
    "type",
    "name",
    "source",
    "from_library",
    "params",
}


def _default_source(kind: str) -> str:
    try:
        return DEFAULT_KIND_SOURCES[kind]
    except KeyError as exc:
        raise ConfigValidationError(f"Unsupported component kind: {kind}") from exc


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


def _split_target(kind: str, target: str, default_source: str) -> tuple[str, str]:
    if not isinstance(target, str) or not target.strip():
        raise ConfigValidationError(f"{kind} target must be a non-empty string")
    if ":" not in target:
        raise ConfigValidationError(
            f"{kind} target must use 'source:name' format, got '{target}'"
        )

    raw_source, component_type = target.rsplit(":", 1)
    if not raw_source or not component_type:
        raise ConfigValidationError(
            f"{kind} target must use 'source:name' format, got '{target}'"
        )

    return _normalize_source(kind, raw_source, default_source), component_type


def _looks_like_component_config(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(key in value for key in ("target", "type", "name", "source", "from_library"))


def _extract_component_config(
    config: Dict[str, Any],
    *,
    kind: str,
    wrapper_keys: Sequence[str] = (),
    default_source: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(config, Mapping) or not config:
        raise ConfigValidationError(f"{kind} config cannot be empty")

    component_cfg: Mapping[str, Any] = config
    for key in wrapper_keys:
        wrapped = component_cfg.get(key)
        if isinstance(wrapped, Mapping):
            component_cfg = wrapped
            break

    normalized_default_source = _default_source(kind) if default_source is None else default_source
    raw_target = component_cfg.get("target")
    if raw_target:
        source, component_type = _split_target(kind, str(raw_target), normalized_default_source)
    else:
        component_type = component_cfg.get("type") or component_cfg.get("name")
        if not component_type:
            raise ConfigValidationError(
                f"{kind} config requires 'target' or legacy fields 'type'/'name'"
            )
        source = _normalize_source(
            kind,
            component_cfg.get("source") or component_cfg.get("from_library"),
            normalized_default_source,
        )

    params = dict(component_cfg.get("params") or {})
    ignored_keys = {"target", "type", "name", "source", "from_library", "params"}
    extras = {key: value for key, value in component_cfg.items() if key not in ignored_keys}

    return {
        "target": f"{source}:{component_type}",
        "type": component_type,
        "source": source,
        "params": params,
        **extras,
    }


def _build_nested_component_value(value: Any, *, nested_kind: str) -> Any:
    if isinstance(value, Mapping):
        if _looks_like_component_config(value):
            return _build_component(dict(value), kind=nested_kind)
        return {
            key: _build_nested_component_value(item, nested_kind=nested_kind)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_build_nested_component_value(item, nested_kind=nested_kind) for item in value]
    return value


def _build_component(
    config: Dict[str, Any],
    *,
    kind: str,
    wrapper_keys: Sequence[str] = (),
    default_source: Optional[str] = None,
) -> Any:
    component_cfg = _extract_component_config(
        config,
        kind=kind,
        wrapper_keys=wrapper_keys,
        default_source=default_source,
    )
    component_cls = _resolve_component(kind, component_cfg["type"], component_cfg["source"])
    params = dict(component_cfg["params"])

    if kind == "transform":
        params = _build_nested_component_value(params, nested_kind="transform")

    return component_cls(**params)


def _is_legacy_transform_pipeline(config: Dict[str, Any]) -> bool:
    pipeline_type = config.get("type") or config.get("combination_strategy")
    if "items" in config or "transforms" in config or "combination_strategy" in config:
        return True
    return (
        config.get("target") in {None, ""}
        and pipeline_type in {"compose", "list", "raw"}
        and "params" not in config
    )


def _build_legacy_transform_pipeline(config: Dict[str, Any]) -> Any:
    import torchvision.transforms as transforms

    pipeline_type = str(config.get("type") or config.get("combination_strategy") or "compose").lower()
    raw_items = config.get("items") or config.get("transforms") or []
    transforms_list = [build_transform(item) for item in raw_items]

    if pipeline_type == "compose":
        return transforms.Compose(transforms_list)
    if pipeline_type in {"list", "raw"}:
        return transforms_list

    raise ConfigValidationError(f"Unsupported transform pipeline type: {pipeline_type}")


def _normalize_transform_shorthand(config: Any) -> Any:
    if isinstance(config, str):
        return {
            "target": config,
            "params": {},
        }

    if isinstance(config, list):
        return {
            "target": "torchvision.transforms:Compose",
            "params": {
                "transforms": [_normalize_transform_shorthand(item) for item in config],
            },
        }

    if not isinstance(config, Mapping):
        raise ConfigValidationError("transform config must be a mapping, list, or target string")

    config_dict = dict(config)
    if _is_legacy_transform_pipeline(config_dict):
        return config_dict

    params = dict(config_dict.get("params") or {})
    inline_params = {
        key: value
        for key, value in config_dict.items()
        if key not in TRANSFORM_CONFIG_META_KEYS
    }
    for key, value in inline_params.items():
        params.setdefault(key, value)

    if "transforms" in params and isinstance(params["transforms"], list):
        params["transforms"] = [
            _normalize_transform_shorthand(item)
            for item in params["transforms"]
        ]

    normalized = {
        key: value
        for key, value in config_dict.items()
        if key in TRANSFORM_CONFIG_META_KEYS and key != "params"
    }
    normalized["params"] = params
    return normalized


def _resolve_optional_transform(raw_transform: Any) -> Any:
    if isinstance(raw_transform, str):
        return build_transform(raw_transform)
    if isinstance(raw_transform, list):
        return build_transform(raw_transform)
    if isinstance(raw_transform, Mapping):
        return build_transform(dict(raw_transform))
    return raw_transform


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """从配置构建模型。"""

    return _build_component(
        config,
        kind="model",
        wrapper_keys=("backbone",),
        default_source="registry",
    )


def build_transform(config: Any) -> Any:
    """从配置构建 transform，支持紧凑 list / string 写法。"""

    if not config:
        return None
    if isinstance(config, list):
        config_dict = _normalize_transform_shorthand(config)
    elif isinstance(config, str):
        config_dict = _normalize_transform_shorthand(config)
    elif isinstance(config, Mapping):
        config_dict = dict(config)
    else:
        raise ConfigValidationError("transform config must be a mapping, list, or target string")

    if _is_legacy_transform_pipeline(config_dict):
        return _build_legacy_transform_pipeline(config_dict)

    return _build_component(
        _normalize_transform_shorthand(config_dict),
        kind="transform",
        default_source="torchvision.transforms",
    )


def build_dataset(config: Dict[str, Any], transform: Optional[Any] = None) -> Any:
    """从配置构建数据集。"""

    dataset_cfg = _extract_component_config(
        config,
        kind="dataset",
        default_source="torchvision.datasets",
    )
    params = dict(dataset_cfg["params"])

    transform_value = transform
    if transform_value is None and params.get("transform") is not None:
        transform_value = _resolve_optional_transform(params.get("transform"))
    if transform_value is None and config.get("transform") is not None:
        transform_value = _resolve_optional_transform(config.get("transform"))

    if transform_value is not None:
        params["transform"] = transform_value

    for optional_key in ("target_transform", "transforms"):
        if optional_key in params:
            params[optional_key] = _resolve_optional_transform(params[optional_key])

    dataset_cls = _resolve_component("dataset", dataset_cfg["type"], dataset_cfg["source"])
    return dataset_cls(**params)


def build_dataloader(dataset: Any, config: Dict[str, Any]) -> DataLoader:
    """从配置构建 DataLoader。"""

    if not isinstance(config, Mapping):
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
    if not isinstance(param_group_configs, Mapping):
        raise ConfigValidationError("param_groups must be a mapping")

    groups: List[Dict[str, Any]] = []
    for group_name, group_params in param_group_configs.items():
        if not isinstance(group_params, Mapping):
            raise ConfigValidationError(f"param group '{group_name}' must be a mapping")
        groups.append(
            {
                "params": list(_select_model_parameters(model, group_name)),
                **dict(group_params),
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
    params = dict(optimizer_cfg["params"])

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

    loss_items = [config] if isinstance(config, Mapping) else list(config)
    if not loss_items:
        raise ConfigValidationError("loss config cannot be empty")

    if len(loss_items) == 1:
        loss_cfg = _extract_component_config(
            dict(loss_items[0]),
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
            dict(loss_item),
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
