"""训练任务协议与基类, 对齐 PyTorch CoreModel 扩展接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
import optax

from .model import ModelAdapter
from .types import ModelState

if TYPE_CHECKING:
    from .callbacks import Callback


@runtime_checkable
class JaxTask(Protocol):
    """描述模型, loss, metrics 和 optimizer 的任务对象协议."""

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


class BaseJaxTask(ABC):
    """JAX 任务基础抽象类, 提供类似 PyTorch CoreModel 的开箱即用扩展能力."""

    @abstractmethod
    def build_model(self) -> ModelAdapter:
        """构建并返回模型 adapter."""

    @abstractmethod
    def loss_and_metrics(
        self,
        model: ModelAdapter,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[jax.Array, Mapping[str, jax.Array], ModelState]:
        """计算单步损失与度量."""

    @abstractmethod
    def configure_optimizer(
        self,
        *,
        total_steps: int,
    ) -> optax.GradientTransformation:
        """配置并返回 Optax 优化器转换."""

    def validation_loss_and_metrics(
        self,
        model: ModelAdapter,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array,
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        """可选的独立验证步逻辑, 默认复用 loss_and_metrics(..., training=False)."""
        loss, metrics, _ = self.loss_and_metrics(
            model,
            model_state,
            batch,
            rng,
            training=False,
        )
        return loss, metrics

    def predict_step(
        self,
        model: ModelAdapter,
        model_state: ModelState,
        batch: Any,
        rng: jax.Array | None = None,
    ) -> Any:
        """单批次推理前向传播, 默认直接调用 model.apply 并返回首个输出."""
        if rng is None:
            rng = jax.random.key(0)
        preds, _ = model.apply(model_state, batch, rng, training=False)
        return preds

    def configure_callbacks(self) -> list[Callback]:
        """任务自带的专属生命周期回调列表(对齐 PyTorch CoreModel.configure_callbacks)."""
        return []
