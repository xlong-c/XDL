"""JAX 训练 callback."""

from __future__ import annotations

import logging
import time
from typing import Any

from .errors import CallbackError
from .types import MetricSnapshot


class Callback:
    """无 Torch 依赖的 callback 基类."""

    priority: int = 999
    fast_fail: bool = False

    def on_fit_start(self, trainer: Any, state: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: Any, state: Any) -> None:
        pass

    def on_train_batch_end(
        self,
        trainer: Any,
        snapshot: MetricSnapshot,
    ) -> None:
        pass

    def on_validation_epoch_end(
        self,
        trainer: Any,
        metrics: dict[str, float],
    ) -> None:
        pass

    def on_train_epoch_end(self, trainer: Any, state: Any) -> None:
        pass

    def on_checkpoint_end(self, trainer: Any, report: Any) -> None:
        pass

    def on_compile_end(self, trainer: Any, report: dict[str, Any]) -> None:
        pass

    def on_fit_end(self, trainer: Any, result: Any) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state


class CallbackList:
    """按 priority 排序并执行 callback, 支持 fast-fail 和错误隔离."""

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
                logging.getLogger(__name__).exception("%s", error)

    def state_dict(self) -> dict[str, Any]:
        return {
            type(callback).__name__: callback.state_dict()
            for callback in self.callbacks
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for callback in self.callbacks:
            key = type(callback).__name__
            if key in state:
                callback.load_state_dict(dict(state[key]))


class ConsoleLoggerCallback(Callback):
    """将训练指标输出到日志."""

    priority = 900

    def __init__(self, every_n_steps: int = 1) -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.logger = logging.getLogger("xdl_jax.train")

    def on_train_batch_end(
        self,
        trainer: Any,
        snapshot: MetricSnapshot,
    ) -> None:
        if snapshot.micro_step % self.every_n_steps == 0:
            self.logger.info(
                "epoch=%d micro_step=%d optimizer_step=%d loss=%.6f metrics=%s",
                snapshot.epoch,
                snapshot.micro_step,
                snapshot.optimizer_step,
                snapshot.loss,
                snapshot.metrics,
            )


class TimerCallback(Callback):
    """记录 fit 的 host 运行时间."""

    priority = 100

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.elapsed_s: float | None = None

    def on_fit_start(self, trainer: Any, state: Any) -> None:
        del trainer, state
        self.started_at = time.perf_counter()

    def on_fit_end(self, trainer: Any, result: Any) -> None:
        del trainer, result
        if self.started_at is not None:
            self.elapsed_s = time.perf_counter() - self.started_at

    def state_dict(self) -> dict[str, Any]:
        return {"elapsed_s": self.elapsed_s}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        value = state.get("elapsed_s")
        self.elapsed_s = None if value is None else float(value)


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


class EarlyStoppingCallback(Callback):
    """验证指标连续无改善时请求停止."""

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
            raise ValueError("mode must be 'min' or 'max'")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_validation_epoch_end(
        self,
        trainer: Any,
        metrics: dict[str, float],
    ) -> None:
        if self.monitor not in metrics:
            raise ValueError(f"early-stopping metric not found: {self.monitor}")
        value = float(metrics[self.monitor])
        if self._improved(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                trainer.request_stop()

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        best = state.get("best")
        self.best = None if best is None else float(best)
        self.bad_epochs = int(state.get("bad_epochs", 0))


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
