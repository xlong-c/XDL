"""JAX 训练 callback, 保持自包含且与框架生命周期契约对齐."""

from __future__ import annotations

import logging
import time
from typing import Any

from .errors import CallbackError
from .types import MetricSnapshot

logger = logging.getLogger(__name__)


class Callback:
    """JAX 训练 callback 基类."""

    priority: int = 999
    fast_fail: bool = False

    def on_fit_start(self, trainer: Any, state: Any | None = None) -> None:
        """完整训练 fit 开始前触发."""

    def on_fit_end(self, trainer: Any, result: Any | None = None) -> None:
        """完整训练 fit 结束时触发."""

    def on_train_start(self, trainer: Any) -> None:
        """训练过程(所有 Epoch 前)开始前触发."""

    def on_train_end(self, trainer: Any) -> None:
        """训练过程(所有 Epoch 结束后)触发."""

    def on_train_epoch_start(self, trainer: Any, state: Any | None = None) -> None:
        """每个训练 Epoch 开始时触发."""

    def on_train_epoch_end(self, trainer: Any, state: Any | None = None) -> None:
        """每个训练 Epoch 结束时触发."""

    def on_train_batch_start(
        self,
        trainer: Any,
        batch_idx: int,
    ) -> None:
        """每个训练 Batch 执行前触发."""

    def on_train_batch_end(
        self,
        trainer: Any,
        snapshot: MetricSnapshot,
    ) -> None:
        """每个训练 Batch 结束时触发."""

    def on_validation_start(self, trainer: Any) -> None:
        """验证流程启动时触发."""

    def on_validation_end(self, trainer: Any) -> None:
        """验证流程结束时触发."""

    def on_validation_epoch_start(self, trainer: Any) -> None:
        """验证 Epoch 开始时触发."""

    def on_validation_epoch_end(
        self,
        trainer: Any,
        metrics: dict[str, float],
    ) -> None:
        """验证 Epoch 结束时触发."""

    def on_validation_batch_start(
        self,
        trainer: Any,
        batch_idx: int,
    ) -> None:
        """验证 Batch 开始前触发."""

    def on_validation_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        """验证 Batch 结束后触发."""

    def on_predict_start(self, trainer: Any) -> None:
        """推理预测流程开始时触发."""

    def on_predict_end(self, trainer: Any) -> None:
        """推理预测流程结束时触发."""

    def on_checkpoint_start(self, trainer: Any) -> None:
        """检查点保存前触发."""

    def on_checkpoint_end(self, trainer: Any, report: Any) -> None:
        """检查点保存完成后触发."""

    def on_compile_end(self, trainer: Any, report: dict[str, Any]) -> None:
        """JIT 编译完成后触发."""

    def state_dict(self) -> dict[str, Any]:
        """导出回调状态."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复回调状态."""
        del state


class CallbackList:
    """按优先级排序并驱动回调链, 支持 fast-fail 与异常隔离."""

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        *,
        fail_on_error: bool = False,
    ) -> None:
        self.callbacks = sorted(
            list(callbacks or []),
            key=lambda callback: int(getattr(callback, "priority", 999)),
        )
        self.fail_on_error = fail_on_error
        self.errors: list[CallbackError] = []

    def invoke(self, hook: str, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """按顺序调用指定 hook 方法."""
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is None:
                continue
            try:
                method(trainer, *args, **kwargs)
            except Exception as exc:
                error = CallbackError(
                    f"Callback {type(callback).__name__}.{hook} failed: {exc}"
                )
                self.errors.append(error)
                callback_fast_fail = bool(getattr(callback, "fast_fail", False))
                if self.fail_on_error or callback_fast_fail:
                    raise error from exc
                logger.exception("%s", error)

    def state_dict(self) -> dict[str, Any]:
        """按类名持久化每个 callback 的状态."""
        return {
            type(callback).__name__: callback.state_dict()
            for callback in self.callbacks
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复各 callback 状态."""
        for callback in self.callbacks:
            key = type(callback).__name__
            if key in state:
                callback.load_state_dict(dict(state[key]))


class TimerCallback(Callback):
    """统计训练与各 Epoch 耗时."""

    priority = 100

    def __init__(self) -> None:
        self.fit_start_time: float | None = None
        self.fit_elapsed_seconds: float = 0.0
        self.epoch_start_time: float | None = None
        self.epoch_durations: list[float] = []

    def on_fit_start(self, trainer: Any, state: Any | None = None) -> None:
        del trainer, state
        self.fit_start_time = time.perf_counter()

    def on_train_epoch_start(self, trainer: Any, state: Any | None = None) -> None:
        del trainer, state
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: Any, state: Any | None = None) -> None:
        del trainer, state
        if self.epoch_start_time is not None:
            self.epoch_durations.append(time.perf_counter() - self.epoch_start_time)

    def on_fit_end(self, trainer: Any, result: Any | None = None) -> None:
        del trainer, result
        if self.fit_start_time is not None:
            self.fit_elapsed_seconds = time.perf_counter() - self.fit_start_time


class ConsoleLoggerCallback(Callback):
    """定时打印控制台日志."""

    priority = 900

    def __init__(self, *, log_every_n_steps: int = 1) -> None:
        self.log_every_n_steps = max(1, log_every_n_steps)

    def on_train_batch_end(
        self,
        trainer: Any,
        snapshot: MetricSnapshot,
    ) -> None:
        del trainer
        if snapshot.step % self.log_every_n_steps != 0:
            return
        parts = [f"epoch={snapshot.epoch}", f"step={snapshot.step}"]
        if snapshot.train_loss is not None:
            parts.append(f"train_loss={snapshot.train_loss:.4f}")
        for k, v in sorted(snapshot.metrics.items()):
            parts.append(f"{k}={v:.4f}")
        logger.info("[train] %s", " ".join(parts))

    def on_validation_epoch_end(
        self,
        trainer: Any,
        metrics: dict[str, float],
    ) -> None:
        del trainer
        parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
        logger.info("[eval] %s", " ".join(parts))


class EarlyStoppingCallback(Callback):
    """基于监控标量指标的早停回调."""

    priority = 800

    def __init__(
        self,
        monitor: str = "loss",
        *,
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"unsupported early stopping mode: {mode}")
        self.monitor = monitor
        self.mode = mode
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.best_value: float | None = None
        self.wait_count = 0
        self.stopped_epoch: int | None = None

    def on_validation_epoch_end(
        self,
        trainer: Any,
        metrics: dict[str, float],
    ) -> None:
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor])
        if self._is_better(current):
            self.best_value = current
            self.wait_count = 0
            return
        self.wait_count += 1
        if self.wait_count > self.patience:
            self.stopped_epoch = getattr(trainer, "current_epoch", None)
            if hasattr(trainer, "request_stop"):
                trainer.request_stop()

    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        return current > (self.best_value + self.min_delta)

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_value = state.get("best_value")
        self.wait_count = int(state.get("wait_count", 0))
        self.stopped_epoch = state.get("stopped_epoch")


class CompileReportCallback(Callback):
    """保存首次 compiled executable 的 host-side 报告."""

    priority = 150

    def __init__(self) -> None:
        self.report: dict[str, Any] | None = None

    def on_compile_end(self, trainer: Any, report: dict[str, Any]) -> None:
        del trainer
        self.report = dict(report)

    def state_dict(self) -> dict[str, Any]:
        return {"report": self.report}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        report = state.get("report")
        self.report = None if report is None else dict(report)


class CheckpointCallback(Callback):
    """在 epoch boundary 保存完整 train state."""

    priority = 100
    fast_fail = True

    def __init__(
        self,
        directory: str,
        *,
        every_n_epochs: int = 1,
        max_to_keep: int | None = None,
        asynchronous: bool = True,
    ) -> None:
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        self.directory = directory
        self.every_n_epochs = int(every_n_epochs)
        self.max_to_keep = max_to_keep
        self.asynchronous = asynchronous
        self._manager: Any = None
        self.last_report: Any = None

    def on_fit_start(self, trainer: Any, state: Any) -> None:
        del state
        from .checkpoint import JaxCheckpointManager

        self._manager = JaxCheckpointManager(
            self.directory,
            max_to_keep=self.max_to_keep,
            asynchronous=self.asynchronous,
        )

    def on_train_epoch_end(self, trainer: Any, state: Any) -> None:
        if self._manager is None:
            raise RuntimeError("checkpoint callback was not initialized")
        epoch = int(state.epoch)
        if epoch % self.every_n_epochs != 0:
            return
        data_state: dict[str, Any] = {}
        data_source = getattr(trainer, "train_data", None)
        if data_source is not None and hasattr(data_source, "state_dict"):
            data_state = data_source.state_dict()
        from .checkpoint import CheckpointMetadata

        self.last_report = self._manager.save(
            int(state.optimizer_step),
            state,
            data_state=data_state,
            callback_state=trainer.callbacks.state_dict(),
            metadata=CheckpointMetadata(
                loop={
                    "epoch": int(state.epoch),
                    "micro_step": int(state.micro_step),
                    "optimizer_step": int(state.optimizer_step),
                },
            ),
        )
        trainer.callbacks.invoke("on_checkpoint_end", trainer, self.last_report)

    def on_fit_end(self, trainer: Any, result: Any) -> None:
        del trainer, result
        if self._manager is not None:
            self._manager.close()
            self._manager = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": None
            if self.last_report is None
            else int(self.last_report.step),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state


__all__ = [
    "Callback",
    "CallbackError",
    "CallbackList",
    "CheckpointCallback",
    "CompileReportCallback",
    "ConsoleLoggerCallback",
    "EarlyStoppingCallback",
    "TimerCallback",
]
