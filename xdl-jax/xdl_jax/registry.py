"""独立于 Torch XDL 的 xdl-jax registry, 保持自包含与契约对齐."""

from __future__ import annotations

import difflib
import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from .errors import ConfigurationError


class Registry:
    """通用对象注册表, 负责组件映射,查询与模糊纠错."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        """注册装饰器."""

        def decorator(value: Any) -> Any:
            key = name or getattr(value, "__name__", None)
            if not key:
                raise ConfigurationError(f"{self.name} registration requires a name")
            if key in self._values and self._values[key] is not value:
                raise ConfigurationError(f"duplicate {self.name} registration: {key}")
            self._values[key] = value
            return value

        return decorator

    def get(self, name: str) -> Any:
        """根据名称获取已注册对象, 找不到时提供候选建议."""
        try:
            return self._values[name]
        except KeyError as exc:
            matches = difflib.get_close_matches(name, list(self._values.keys()), n=3, cutoff=0.6)
            suggestion = f"; did you mean {matches}?" if matches else ""
            raise ConfigurationError(f"{name!r} is not registered in {self.name}{suggestion}") from exc

    def list_available(self) -> list[str]:
        """获取所有可用注册组件名."""
        return sorted(self._values)

    def get_signature(self, name: str) -> str:
        """获取并格式化组件的参数签名."""
        obj = self.get(name)
        try:
            sig_obj = obj.__init__ if hasattr(obj, "__init__") else obj
            sig = inspect.signature(sig_obj)
            params = []
            for p_name, param in sig.parameters.items():
                if p_name == "self":
                    continue
                p_str = f"{p_name}"
                if param.annotation is not inspect.Parameter.empty:
                    p_str += f": {getattr(param.annotation, '__name__', str(param.annotation))}"
                if param.default is not inspect.Parameter.empty:
                    p_str += f" = {repr(param.default)}"
                params.append(p_str)
            return f"{name}({', '.join(params)})"
        except Exception as exc:
            return f"Could not get signature for {name}: {exc}"


MODEL_REGISTRY = Registry("MODEL")
DATASET_REGISTRY = Registry("DATASET")
OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
SCHEDULER_REGISTRY = Registry("SCHEDULER")
LOSS_REGISTRY = Registry("LOSS")
METRIC_REGISTRY = Registry("METRIC")
TRANSFORM_REGISTRY = Registry("TRANSFORM")
COLLATE_REGISTRY = Registry("COLLATE")
CALLBACK_REGISTRY = Registry("CALLBACK")
TASK_REGISTRY = Registry("TASK")

register_model = MODEL_REGISTRY.register
register_dataset = DATASET_REGISTRY.register
register_optimizer = OPTIMIZER_REGISTRY.register
register_scheduler = SCHEDULER_REGISTRY.register
register_loss = LOSS_REGISTRY.register
register_metric = METRIC_REGISTRY.register
register_transform = TRANSFORM_REGISTRY.register
register_collate = COLLATE_REGISTRY.register
register_callback = CALLBACK_REGISTRY.register
register_task = TASK_REGISTRY.register

_REGISTRIES: Mapping[str, Registry] = {
    "model": MODEL_REGISTRY,
    "dataset": DATASET_REGISTRY,
    "task": TASK_REGISTRY,
    "callback": CALLBACK_REGISTRY,
}


def resolve_target(target: str, *, kind: str) -> Any:
    """通过 'module:object' 或 'registry:name' 解析目标类或函数."""
    if not isinstance(target, str) or ":" not in target:
        raise ConfigurationError(
            f"{kind} target must use 'source:name' format, got {target!r}"
        )
    source, name = target.rsplit(":", 1)
    if source in {"registry", "xdl_jax"}:
        if kind not in _REGISTRIES:
            raise ConfigurationError(f"registry target is not supported for {kind}")
        return _REGISTRIES[kind].get(name)

    try:
        module = importlib.import_module(source)
        return getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(
            f"cannot resolve {kind} target {target!r}"
        ) from exc


def build_component(config: dict[str, Any], *, kind: str) -> Any:
    """基于 target 和 params 配置字典动态实例化组件."""
    if not isinstance(config, dict):
        raise ConfigurationError(f"{kind} config must be a mapping")
    target = config.get("target")
    if not target:
        raise ConfigurationError(f"{kind} config requires 'target'")
    params = dict(config.get("params") or {})
    extras = set(config) - {"target", "params"}
    if extras:
        raise ConfigurationError(
            f"{kind} config contains unsupported fields: {sorted(extras)}"
        )
    component = resolve_target(str(target), kind=kind)
    try:
        return component(**params)
    except Exception as exc:
        raise ConfigurationError(
            f"failed to build {kind} from {target!r}: {exc}"
        ) from exc
