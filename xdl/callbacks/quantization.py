"""Quantization-related training callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import Callback


@dataclass
class QATLifecycleState:
    """Summary of QAT lifecycle transitions triggered by a callback."""

    observer_disabled: int = 0
    fake_quant_disabled: int = 0
    bn_frozen: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "observer_disabled": self.observer_disabled,
            "fake_quant_disabled": self.fake_quant_disabled,
            "bn_frozen": self.bn_frozen,
        }


class QATLifecycleCallback(Callback):
    """Toggle QAT observer, fake-quant and BN state by epoch."""

    def __init__(
        self,
        disable_observer_epoch: int | None = None,
        freeze_bn_epoch: int | None = None,
        disable_fake_quant_epoch: int | None = None,
        priority: int = 999,
    ) -> None:
        super().__init__(priority=priority)
        self.disable_observer_epoch = disable_observer_epoch
        self.freeze_bn_epoch = freeze_bn_epoch
        self.disable_fake_quant_epoch = disable_fake_quant_epoch

    def on_train_epoch_start(self, trainer: Any, core_module: Any) -> None:
        epoch = self._current_epoch(trainer)
        model = self._resolve_model(core_module)
        state = QATLifecycleState()

        if self.disable_observer_epoch is not None and epoch >= self.disable_observer_epoch:
            state.observer_disabled = self._call_on_modules(
                model.modules(),
                "disable_observer",
            )

        if self.disable_fake_quant_epoch is not None and epoch >= self.disable_fake_quant_epoch:
            state.fake_quant_disabled = self._call_on_modules(
                model.modules(),
                "disable_fake_quant",
            )

        if self.freeze_bn_epoch is not None and epoch >= self.freeze_bn_epoch:
            state.bn_frozen = self._call_on_modules(model.modules(), "freeze_bn_stats")

        self._state["last_epoch"] = epoch
        self._state["qat_lifecycle"] = state.to_dict()

    @staticmethod
    def _current_epoch(trainer: Any) -> int:
        epoch = getattr(trainer, "current_epoch", 0)
        try:
            return int(epoch)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_model(core_module: Any) -> Any:
        model = getattr(core_module, "model", None)
        return model if model is not None else core_module

    @staticmethod
    def _call_on_modules(modules: Iterable[Any], method_name: str) -> int:
        count = 0
        for module in modules:
            fn = getattr(module, method_name, None)
            if callable(fn):
                fn()
                count += 1
        return count


__all__ = [
    "QATLifecycleCallback",
    "QATLifecycleState",
]
