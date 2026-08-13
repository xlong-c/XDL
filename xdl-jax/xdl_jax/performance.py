"""JAX 编译和稳态性能测量."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np


def _block(value: Any) -> Any:
    def block_leaf(leaf: Any) -> Any:
        return leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf

    return jax.tree_util.tree_map(block_leaf, value)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


@dataclass(frozen=True)
class BenchmarkReport:
    """一次 JAX callable benchmark 的完整上下文."""

    platform: str
    devices: tuple[str, ...]
    warmup_steps: int
    measured_steps: int
    first_compile_time_s: float
    steady_state_p50_s: float
    steady_state_p90_s: float
    steady_state_mean_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "devices": list(self.devices),
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "first_compile_time_s": self.first_compile_time_s,
            "steady_state_p50_s": self.steady_state_p50_s,
            "steady_state_p90_s": self.steady_state_p90_s,
            "steady_state_mean_s": self.steady_state_mean_s,
        }


def benchmark_callable(
    fn: Callable[..., Any],
    *args: Any,
    warmup_steps: int = 1,
    measured_steps: int = 10,
    **kwargs: Any,
) -> BenchmarkReport:
    """测量 callable, 将首次执行单独计为 compile/first-execution 时间."""

    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError("warmup_steps must be >= 0 and measured_steps must be positive")
    started = time.perf_counter()
    result = _block(fn(*args, **kwargs))
    first_compile_time_s = time.perf_counter() - started
    for _ in range(warmup_steps):
        result = _block(fn(*args, **kwargs))
    del result
    durations: list[float] = []
    for _ in range(measured_steps):
        started = time.perf_counter()
        result = _block(fn(*args, **kwargs))
        del result
        durations.append(time.perf_counter() - started)
    devices = tuple(f"{device.platform}:{device.id}" for device in jax.devices())
    return BenchmarkReport(
        platform=jax.default_backend(),
        devices=devices,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        first_compile_time_s=first_compile_time_s,
        steady_state_p50_s=_percentile(durations, 50),
        steady_state_p90_s=_percentile(durations, 90),
        steady_state_mean_s=float(np.mean(np.asarray(durations))),
    )
