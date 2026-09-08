"""Trainer / CoreModel 契约回归测试."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xdl.callbacks import Callback
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer


class TinyModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self.manual_optimization_step(self.fc(batch).sum())

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)


class RecordingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epoch_starts = 0
        self.epoch_ends = 0

    def on_train_epoch_start(self, trainer, core_module) -> None:
        del trainer, core_module
        self.epoch_starts += 1

    def on_train_epoch_end(self, trainer, core_module) -> None:
        del trainer, core_module
        self.epoch_ends += 1


class RecordingAccelerator:
    """记录 prepare 入参的 Accelerator 替身, 避免依赖全局单例状态."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.device = torch.device("cpu")
        self.prepared: list[Any] = []

    def prepare(self, *items: Any) -> tuple[Any, ...]:
        self.prepared.extend(items)
        return items

    def prepare_optimizer(self, optimizer: Any) -> Any:
        return optimizer

    def prepare_scheduler(self, scheduler: Any) -> Any:
        return scheduler


def test_legacy_accumulation_attribute_conflict_warns(caplog) -> None:
    """模型自带属性与 Trainer 注入值不一致时必须显式告警."""

    model = TinyModel()
    model.gradient_accumulation_steps = 8

    with caplog.at_level(logging.WARNING):
        model._set_gradient_accumulation_steps(2)

    assert model.accumulation_steps == 8
    assert "gradient_accumulation_steps" in caplog.text


def test_train_epoch_early_exit_keeps_epoch_hooks_balanced() -> None:
    """虚拟 epoch 尾部无剩余 step 时, epoch 结束钩子仍必须执行."""

    model = TinyModel()
    callback = RecordingCallback()
    trainer = Trainer(max_epochs=1, device="cpu", callbacks=[callback])
    trainer._model = model
    trainer._train_dataloader = _loader()
    trainer._target_total_train_steps = 0

    trainer._train_epoch()

    assert callback.epoch_starts == 1
    assert callback.epoch_ends == 1


def test_test_loader_is_prepared_with_accelerate(monkeypatch) -> None:
    """Accelerate 路径下 test loader 必须进入 prepare 列表."""

    created: list[RecordingAccelerator] = []

    def factory(**kwargs: Any) -> RecordingAccelerator:
        accelerator = RecordingAccelerator(**kwargs)
        created.append(accelerator)
        return accelerator

    monkeypatch.setattr("xdl.trainer.trainer.Accelerator", factory)

    model = TinyModel()
    loader = _loader()
    trainer = Trainer(max_epochs=1, accelerate_config={"cpu": True})

    trainer.test(model, loader)

    assert trainer._test_loader_prepared is True
    assert any(item is loader for item in created[0].prepared)
