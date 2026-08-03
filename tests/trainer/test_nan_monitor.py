"""Trainer NaN/Inf 监控测试."""

from typing import Any

import pytest
import torch

from xdl.errors import TrainingError
from xdl.trainer import Trainer
from xdl.trainer.core_model import CoreModel


class NanLoggingModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(1, 1)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self.log("train_loss", float("nan"))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_nan_metrics_abort_training_after_patience() -> None:
    model = NanLoggingModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        nan_monitor=True,
        nan_patience=2,
    )

    with pytest.raises(TrainingError, match="NaN/Inf 指标"):
        trainer.fit(model=model, train_dataloader=[1.0, 2.0, 3.0])


def test_nan_monitor_can_be_disabled() -> None:
    model = NanLoggingModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        nan_monitor=False,
    )

    trainer.fit(model=model, train_dataloader=[1.0, 2.0])
    assert trainer._nan_steps == 0


def test_finite_metrics_reset_nan_counter() -> None:
    class FlakyModel(CoreModel):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(1, 1)
            self.calls = 0

        def training_step(self, batch: Any, batch_idx: int) -> None:
            del batch, batch_idx
            self.calls += 1
            value = float("nan") if self.calls == 1 else 0.5
            self.log("train_loss", value)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = FlakyModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        nan_monitor=True,
        nan_patience=3,
    )

    trainer.fit(model=model, train_dataloader=[1.0, 2.0, 3.0])
    assert model.calls == 3
    assert trainer._nan_steps == 0
