"""Trainer 性能诊断埋点测试 (CPU-only)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xdl.callbacks import Callback
from xdl.diagnostics.phase_timer import PhaseTimer
from xdl.diagnostics.records import SyncCounters
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer


class TensorLoggingModel(CoreModel):
    """每步记录一个 Tensor 指标, 触发 CoreModel.log 的 log_item 计数."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        inputs = batch[0]
        loss = self.net(inputs).sum()
        self.log("loss", loss.detach(), prefix="train")
        self.manual_optimization_step(loss)


class DiagnosticsInstaller(Callback):
    """模拟 DiagnosticsCallback 在 on_train_start 安装计时器与计数器."""

    def __init__(self) -> None:
        super().__init__()
        self.phase_timer = PhaseTimer()
        self.counters = SyncCounters()

    def on_train_start(self, trainer: Trainer, core_module: CoreModel) -> None:
        trainer._diag_enabled = True
        trainer._phase_timer = self.phase_timer
        trainer._diag_counters = self.counters
        core_module._diag_counters = self.counters


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)


def test_phase_marks_and_counters_recorded() -> None:
    installer = DiagnosticsInstaller()
    model = TensorLoggingModel()
    trainer = Trainer(max_epochs=2, device="cpu", callbacks=[installer])

    trainer.fit(model=model, train_dataloader=_loader())

    timer = installer.phase_timer
    assert len(timer) == 4
    aggregate = timer.snapshot()
    assert aggregate.steps == 4
    assert aggregate.means["total"] > 0.0

    for sample in timer._samples:
        assert sample.total > 0.0
        phase_sum = (
            sample.data_wait
            + sample.h2d
            + sample.train_step
            + sample.nan_monitor
            + sample.callbacks
        )
        assert math.isclose(phase_sum, sample.total, rel_tol=1e-6, abs_tol=1e-9)

    assert installer.counters.log_item > 0
    assert installer.counters.grad_norm > 0


def test_disabled_diagnostics_records_nothing() -> None:
    model = TensorLoggingModel()
    trainer = Trainer(max_epochs=1, device="cpu")

    trainer.fit(model=model, train_dataloader=_loader())

    assert trainer._diag_enabled is False
    assert trainer._phase_timer is None
    assert trainer._diag_counters is None
    assert model._diag_counters is None
