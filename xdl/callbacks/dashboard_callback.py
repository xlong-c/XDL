"""Low-overhead scalar and lifecycle event callback for dashboard workers."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, Mapping

from .base import Callback

logger = logging.getLogger(__name__)
EventSink = Callable[[str, Dict[str, Any], bool], None]


class DashboardCallback(Callback):
    """Send scalar metrics and lifecycle changes to a non-blocking event sink.

    The sink must enqueue events quickly; it must not perform network or large
    file I/O inline. Tensor values are intentionally not materialized here,
    because doing so can synchronize an accelerator device.
    """

    def __init__(
        self,
        emit: EventSink,
        *,
        every_n_steps: int = 1,
        interval_s: float = 2.0,
        priority: int = 950,
    ) -> None:
        super().__init__(priority=priority)
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        self.emit = emit
        self.every_n_steps = int(every_n_steps)
        self.interval_s = max(0.0, float(interval_s))
        self._last_metric_emit = 0.0
        self._enabled = True
        self._sink_failed = False

    def setup(self, trainer: Any, core_module: Any, stage: str) -> None:
        del core_module, stage
        is_main = getattr(trainer, "is_main_process", None)
        self._enabled = bool(is_main() if callable(is_main) else True)
        self._last_metric_emit = 0.0

    def on_fit_start(self, trainer: Any, core_module: Any) -> None:
        self._emit(
            "run_state",
            {"state": "starting", "epoch": trainer.current_epoch},
            reliable=True,
        )

    def on_train_start(self, trainer: Any, core_module: Any) -> None:
        del core_module
        self._emit(
            "run_state",
            {
                "state": "running",
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
            },
            reliable=True,
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        core_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps:
            return
        now = time.monotonic()
        if self._last_metric_emit and now - self._last_metric_emit < self.interval_s:
            return
        self._last_metric_emit = now
        metrics = self._numeric_metrics(getattr(core_module, "current_metrics", {}))
        self._emit(
            "metrics",
            {
                "global_step": step,
                "epoch": int(trainer.current_epoch),
                "metrics": metrics,
            },
            reliable=False,
        )

    def on_train_epoch_end(self, trainer: Any, core_module: Any) -> None:
        self._emit(
            "train_epoch_end",
            {
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "metrics": self._numeric_metrics(
                    getattr(core_module, "last_epoch_avg", {})
                ),
            },
            reliable=True,
        )

    def on_validation_epoch_end(
        self,
        trainer: Any,
        core_module: Any,
        outputs: Any = None,
        **kwargs: Any,
    ) -> None:
        del outputs, kwargs
        self._emit(
            "validation_epoch_end",
            {
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "metrics": self._numeric_metrics(
                    getattr(core_module, "last_epoch_avg", {})
                ),
            },
            reliable=True,
        )

    def on_train_end(self, trainer: Any, core_module: Any) -> None:
        del core_module
        stopping = bool(getattr(trainer, "should_stop", False))
        control = getattr(trainer, "control", None)
        checkpoint_requested = bool(
            getattr(control, "save_checkpoint_requested", False)
        )
        self._emit(
            "run_state",
            {
                "state": "stopping" if stopping else "succeeded",
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "checkpoint_requested": checkpoint_requested,
            },
            reliable=True,
        )

    def on_fit_end(self, trainer: Any, core_module: Any) -> None:
        del core_module
        self._emit(
            "run_finished",
            {
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "stopped": bool(getattr(trainer, "should_stop", False)),
            },
            reliable=True,
        )

    def on_exception(
        self, trainer: Any, core_module: Any, exception: Exception
    ) -> None:
        del core_module
        self._emit(
            "run_error",
            {
                "global_step": int(getattr(trainer, "global_step", 0)),
                "epoch": int(getattr(trainer, "current_epoch", 0)),
                "error_type": type(exception).__name__,
                "message": str(exception),
            },
            reliable=True,
        )

    def _emit(
        self, event_type: str, payload: Dict[str, Any], *, reliable: bool
    ) -> None:
        if not self._enabled or self._sink_failed:
            return
        try:
            self.emit(event_type, payload, reliable)
        except Exception:
            self._sink_failed = True
            logger.warning(
                "Dashboard event sink failed; disabling callback", exc_info=True
            )

    @staticmethod
    def _numeric_metrics(metrics: Any) -> Dict[str, float]:
        if not isinstance(metrics, Mapping):
            return {}
        result: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                result[str(key)] = numeric
        return result


__all__ = ["DashboardCallback", "EventSink"]
