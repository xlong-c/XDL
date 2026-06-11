"""
组件构建器。

职责：
- 官方主格式：`target + params`
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
from xdl.errors import RegistryError

REGISTRY_IMPORTS = {
    "model": "xdl.model",
    "dataset": "xdl.dataset",
    "collate": "xdl.dataset",
    "loss": "xdl.loss",
    "metric": "xdl.metric",
    "optimizer": "xdl.optimizer",
    "scheduler": "xdl.scheduler",
    "transform": "xdl.dataset",
}

TRANSFORM_CONFIG_META_KEYS = {
    "target",
    "params",
}
COMPONENT_ALLOWED_EXTRA_KEYS = {
    "model": set(),
    "dataset": set(),
    "collate": set(),
    "optimizer": {"target_modules", "param_groups"},
    "scheduler": set(),
    "loss": {"weight"},
    "metric": set(),
    "transform": set(),
    "callback": set(),
    "task": set(),
}
DISALLOWED_TRANSFORM_INLINE_KEYS = {"combination_strategy", "items"}


def _ensure_registry_populated(kind: str) -> None:
    module_path = REGISTRY_IMPORTS.get(kind)
    if module_path is not None:
        importlib.import_module(module_path)


def _get_registry(kind: str) -> Any:
    _ensure_registry_populated(kind)
    from xdl.utils.registry import (
        COLLATE_REGISTRY,
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
        "collate": COLLATE_REGISTRY,
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
        except (RegistryError, KeyError) as exc:
            raise ComponentResolutionError(kind, component_type, source=source) from exc

    try:
        module = importlib.import_module(source)
    except ImportError as exc:
        raise ComponentResolutionError(kind, component_type, source=source) from exc

    try:
        return getattr(module, component_type)
    except AttributeError as exc:
        raise ComponentResolutionError(kind, component_type, source=source) from exc


def _split_target(kind: str, target: str) -> tuple[str, str]:
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

    return raw_source, component_type


def _looks_like_component_config(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return "target" in value


def _extract_component_config(
    config: Dict[str, Any],
    *,
    kind: str,
) -> Dict[str, Any]:
    if not isinstance(config, Mapping) or not config:
        raise ConfigValidationError(f"{kind} config cannot be empty")

    component_cfg = dict(config)

    raw_target = component_cfg.get("target")
    if not raw_target:
        raise ConfigValidationError(f"{kind} config requires 'target'")

    source, component_type = _split_target(kind, str(raw_target))
    params = dict(component_cfg.get("params") or {})
    extras = {
        key: value
        for key, value in component_cfg.items()
        if key not in {"target", "params"} and value not in (None, {}, [])
    }
    allowed_keys = COMPONENT_ALLOWED_EXTRA_KEYS.get(kind)
    if allowed_keys is None:
        raise ConfigValidationError(f"Unsupported component kind: {kind}")
    invalid_keys = sorted(key for key in extras if key not in allowed_keys)
    if invalid_keys:
        raise ConfigValidationError(
            f"{kind} config contains unsupported fields {invalid_keys}; "
            "use 'target' + 'params'"
        )

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
) -> Any:
    component_cfg = _extract_component_config(
        config,
        kind=kind,
    )
    component_cls = _resolve_component(kind, component_cfg["type"], component_cfg["source"])
    params = dict(component_cfg["params"])

    if kind == "transform":
        params = _build_nested_component_value(params, nested_kind="transform")

    return component_cls(**params)


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
    if "target" not in config_dict:
        raise ConfigValidationError("transform config requires 'target' when using mapping syntax")

    invalid_inline_keys = sorted(
        key
        for key in DISALLOWED_TRANSFORM_INLINE_KEYS.intersection(config_dict)
        if config_dict.get(key) not in (None, "")
    )
    if invalid_inline_keys:
        raise ConfigValidationError(
            f"transform config contains unsupported fields {invalid_inline_keys}; "
            "use 'target' + 'params'"
        )

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

    return _build_component(config, kind="model")


def build_task(config: Dict[str, Any]) -> Any:
    """从配置构建 CoreModel 任务对象."""

    return _build_component(config, kind="task")


def build_transform(config: Any) -> Any:
    """从配置构建 transform，支持紧凑 list / string 写法。"""

    if not config:
        return None
    if isinstance(config, (list, str)):
        normalized = _normalize_transform_shorthand(config)
    elif isinstance(config, Mapping):
        normalized = _normalize_transform_shorthand(dict(config))
    else:
        raise ConfigValidationError("transform config must be a mapping, list, or target string")

    return _build_component(normalized, kind="transform")


def build_dataset(config: Dict[str, Any], transform: Optional[Any] = None) -> Any:
    """从配置构建数据集。"""

    dataset_cfg = _extract_component_config(
        config,
        kind="dataset",
    )
    params = dict(dataset_cfg["params"])

    transform_value = transform
    if transform_value is None and params.get("transform") is not None:
        transform_value = _resolve_optional_transform(params.get("transform"))

    if transform_value is not None:
        params["transform"] = transform_value

    for optional_key in (
        "target_transform",
        "transforms",
        "text_transform",
        "target_text_transform",
    ):
        if optional_key in params:
            params[optional_key] = _resolve_optional_transform(params[optional_key])

    dataset_cls = _resolve_component("dataset", dataset_cfg["type"], dataset_cfg["source"])
    return dataset_cls(**params)


def build_dataloader(
    dataset: Any, config: Dict[str, Any], collate_fn: Optional[Any] = None
) -> DataLoader:
    """从配置构建 DataLoader。"""

    if not isinstance(config, Mapping):
        raise ConfigValidationError("dataloader config must be a mapping")
    params = dict(config.get("params") or {})
    if collate_fn is not None:
        params["collate_fn"] = collate_fn
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

    aliases = {"model", "all"}
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
        )
        loss_cls = _resolve_component("loss", loss_cfg["type"], loss_cfg["source"])
        losses.append(loss_cls(**loss_cfg["params"]))
        weights.append(float(loss_cfg.get("weight", 1.0)))

    return WeightedLoss(losses, weights)


def build_collate_fn(config: Any) -> Optional[Any]:
    """从配置解析 collate_fn。支持三种形式：None / 可调用对象 / target+params 配置。"""

    if config is None:
        return None
    if callable(config):
        return config
    if isinstance(config, str):
        source, component_type = _split_target("collate", config)
        return _resolve_component("collate", component_type, source)
    if isinstance(config, Mapping) and "target" in config:
        return _build_component(dict(config), kind="collate")
    raise ConfigValidationError(f"Invalid collate_fn config: {config}")


def build_callback(config: Dict[str, Any]) -> Any:
    """从配置构建 callback."""

    return _build_component(config, kind="callback")


def build_callbacks(config: Any) -> List[Any]:
    """从配置构建 callback 列表."""

    if not config:
        return []
    callback_items = [config] if isinstance(config, Mapping) else list(config)
    return [build_callback(dict(item)) for item in callback_items]


def build_metrics(config: List[Dict[str, Any]]) -> List[Any]:
    """从配置构建指标列表。"""

    metrics = []
    for metric_item in config or []:
        metric_cfg = _extract_component_config(
            metric_item,
            kind="metric",
        )
        metric_cls = _resolve_component("metric", metric_cfg["type"], metric_cfg["source"])
        metrics.append(metric_cls(**metric_cfg["params"]))
    return metrics
