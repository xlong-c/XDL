"""Optax optimizer 构建器."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optax

from .errors import ConfigurationError


def _build_schedule(value: Any, *, total_steps: int) -> Any:
    if not isinstance(value, Mapping):
        return value
    target = str(value.get("target", ""))
    params = dict(value.get("params") or {})
    name = target.rsplit(":", 1)[-1] if target else str(value.get("name", ""))
    if name in {"constant", "constant_schedule"}:
        return optax.constant_schedule(float(params.get("value", params.get("learning_rate", 0.001))))
    if name in {"cosine_decay", "cosine_decay_schedule"}:
        return optax.cosine_decay_schedule(
            init_value=float(params.get("init_value", 0.001)),
            decay_steps=int(params.get("decay_steps", total_steps)),
            alpha=float(params.get("alpha", 0.0)),
        )
    if name in {"linear_schedule"}:
        return optax.linear_schedule(
            init_value=float(params.get("init_value", 0.001)),
            end_value=float(params.get("end_value", 0.0)),
            transition_steps=int(params.get("transition_steps", total_steps)),
        )
    raise ConfigurationError(f"Unsupported learning-rate schedule: {target or name}")


def build_optimizer(
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> optax.GradientTransformation:
    """从独立的 `target + params` 配置构建 Optax transformation."""

    if not isinstance(config, Mapping) or not config:
        raise ConfigurationError("optimizer config cannot be empty")

    target = str(config.get("target") or config.get("name") or "")
    if ":" in target:
        source, name = target.rsplit(":", 1)
        if source not in {"optax", "xdl_jax.optimizer"}:
            raise ConfigurationError(
                f"xdl-jax optimizer target must use optax, got '{target}'"
            )
    else:
        name = target
    name = name.lower()
    aliases = {"adam_w": "adamw", "momentum": "sgd"}
    name = aliases.get(name, name)
    if name not in {"adam", "adamw", "sgd", "adafactor"}:
        raise ConfigurationError(f"Unsupported xdl-jax optimizer: {name}")

    params = dict(config.get("params") or {})
    for key, value in config.items():
        if key not in {"target", "name", "params", "clip_norm"}:
            params.setdefault(key, value)
    if "learning_rate" not in params and "lr" in params:
        params["learning_rate"] = params.pop("lr")
    if "learning_rate" not in params:
        params["learning_rate"] = 1e-3
    params["learning_rate"] = _build_schedule(
        params["learning_rate"],
        total_steps=total_steps,
    )

    clip_norm = config.get("clip_norm")
    if clip_norm is None:
        clip_norm = params.pop("clip_norm", None)
    optimizer_factory = getattr(optax, name)
    transformation = optimizer_factory(**params)
    if clip_norm is not None:
        transformation = optax.chain(
            optax.clip_by_global_norm(float(clip_norm)),
            transformation,
        )
    return transformation
