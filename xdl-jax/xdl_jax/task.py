"""训练任务协议."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax
import optax

from .model import ModelAdapter
from .types import ModelState


@runtime_checkable
class JaxTask(Protocol):
    """描述模型, loss, metrics 和 optimizer 的任务对象."""

    def build_model(self) -> ModelAdapter:
        """返回模型 adapter."""
        ...

    def loss_and_metrics(
        self,
        model: ModelAdapter,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[jax.Array, Mapping[str, jax.Array], ModelState]:
        """返回 loss, 指标和可能更新 mutable state 的模型状态."""
        ...

    def configure_optimizer(
        self,
        *,
        total_steps: int,
    ) -> optax.GradientTransformation:
        """构建 Optax transformation."""
        ...
