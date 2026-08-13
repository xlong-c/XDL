"""独立于 Torch XDL 的 xdl-jax registry."""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable
from typing import Any

from .errors import ConfigurationError


class Registry:
    """名称到对象的最小映射."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
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
        try:
            return self._values[name]
        except KeyError as exc:
            matches = difflib.get_close_matches(name, self._values, n=3)
            suggestion = f"; did you mean {matches}?" if matches else ""
            raise ConfigurationError(
                f"{name!r} is not registered in {self.name}{suggestion}"
            ) from exc

    def list_available(self) -> list[str]:
        return sorted(self._values)


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


def resolve_target(target: str, *, kind: str) -> Any:
    if not isinstance(target, str) or ":" not in target:
        raise ConfigurationError(
            f"{kind} target must use 'source:name' format, got {target!r}"
        )
    source, name = target.rsplit(":", 1)
    if source in {"registry", "xdl_jax"}:
        registries = {
            "model": MODEL_REGISTRY,
            "dataset": DATASET_REGISTRY,
            "task": TASK_REGISTRY,
            "callback": CALLBACK_REGISTRY,
        }
        if kind not in registries:
            raise ConfigurationError(
                f"registry target is not supported for {kind}"
            )
        return registries[kind].get(name)
    try:
        module = importlib.import_module(source)
        return getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(
            f"cannot resolve {kind} target {target!r}"
        ) from exc


def build_component(config: dict[str, Any], *, kind: str) -> Any:
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
