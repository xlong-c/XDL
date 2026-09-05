"""xdl-jax 的公共 PyTree 类型."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax

PyTree = Any


@dataclass(frozen=True)
class ModelState:
    """模型参数和非参数变量.

    `params` 是唯一允许被 optimizer 更新的 PyTree. `mutable` 保存 BatchNorm
    statistics 或其他由模型前向过程更新的变量.
    """

    params: PyTree
    mutable: PyTree = None

    def replace(self, **changes: Any) -> "ModelState":
        """返回替换字段后的 immutable model state."""

        return replace(self, **changes)


jax.tree_util.register_dataclass(
    ModelState,
    data_fields=("params", "mutable"),
    meta_fields=(),
)


@dataclass(frozen=True)
class JaxTrainState:
    """可传入 compiled step 的完整训练状态."""

    model_state: ModelState
    optimizer_state: PyTree
    rng_key: jax.Array
    micro_step: int
    optimizer_step: int
    epoch: int
    accumulation_grads: PyTree
    accumulation_count: int

    def replace(self, **changes: Any) -> "JaxTrainState":
        """返回替换字段后的 immutable state."""

        return replace(self, **changes)

    def weights_only(self) -> "JaxTrainState":
        """返回只保留模型权重语义的状态副本.

        该方法不负责重建 optimizer state. 训练器或 checkpoint facade 必须
        使用目标运行时的 optimizer/RNG/loop state, 避免把 weights-only restore
        错误地当成 exact restore.
        """

        return self.replace(
            optimizer_state=None,
            rng_key=None,
            micro_step=0,
            optimizer_step=0,
            epoch=0,
            accumulation_grads=None,
            accumulation_count=0,
        )


jax.tree_util.register_dataclass(
    JaxTrainState,
    data_fields=(
        "model_state",
        "optimizer_state",
        "rng_key",
        "micro_step",
        "optimizer_step",
        "epoch",
        "accumulation_grads",
        "accumulation_count",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class StepOutput:
    """compiled train step 返回的设备侧结果."""

    loss: jax.Array
    metrics: Mapping[str, jax.Array]
    did_optimizer_step: jax.Array
    loss_finite: jax.Array
    gradients_finite: jax.Array
    gradient_norm: jax.Array


jax.tree_util.register_dataclass(
    StepOutput,
    data_fields=(
        "loss",
        "metrics",
        "did_optimizer_step",
        "loss_finite",
        "gradients_finite",
        "gradient_norm",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class MetricSnapshot:
    """已经从设备取回的 host-side 指标快照."""

    epoch: int
    micro_step: int
    optimizer_step: int
    metrics: dict[str, float]
    loss: float
    did_optimizer_step: bool
    loss_finite: bool
    gradients_finite: bool
    gradient_norm: float
    step_time_s: float | None = None
    compile_time_s: float | None = None

    @property
    def step(self) -> int:
        return self.optimizer_step

    @property
    def train_loss(self) -> float:
        return self.loss


@dataclass(frozen=True)
class JaxTrainerState:
    """host control plane 的轻量运行状态."""

    epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    stop_requested: bool = False


@dataclass(frozen=True)
class TrainResult:
    """一次 `fit` 的结果."""

    state: JaxTrainState
    trainer_state: JaxTrainerState
    history: tuple[MetricSnapshot, ...]
    validation_history: tuple[dict[str, float], ...]
    first_compile_time_s: float | None
