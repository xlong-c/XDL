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


class StepTrackingModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.recorded_trainer: Any = None
        self.recorded_total_train_steps: int | None = None
        self.recorded_total_optimizer_steps: int | None = None
        self.recorded_accumulation_steps: int | None = None

    def configure_optimizers(self):
        trainer = getattr(self, "trainer", None)
        self.recorded_trainer = trainer
        if trainer is not None:
            self.recorded_total_train_steps = trainer.total_train_steps
            self.recorded_total_optimizer_steps = trainer.total_optimizer_steps
        self.recorded_accumulation_steps = self.accumulation_steps
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        self.manual_optimization_step(self.fc(data).sum())

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx


def test_trainer_backfills_model_trainer_and_injects_steps() -> None:
    """Trainer.fit 必须回填 model.trainer，且在 configure_optimizers 时步数和累积已正确注入."""

    model = StepTrackingModel()
    # 4 samples, batch_size=2 -> 2 steps per epoch. 5 epochs -> 10 micro steps.
    loader = _loader()
    trainer = Trainer(
        max_epochs=5,
        device="cpu",
        gradient_accumulation_steps=2,
    )

    trainer.fit(model, loader)

    assert model.trainer is trainer
    assert getattr(model, "trainer", None) is trainer
    assert model.recorded_trainer is trainer
    # 10 micro steps
    assert model.recorded_total_train_steps == 10
    # 10 micro steps // 2 accum = 5 optimizer steps
    assert model.recorded_total_optimizer_steps == 5
    assert model.recorded_accumulation_steps == 2
    assert model.accumulation_steps == 2


def test_trainer_attach_model_before_setup() -> None:
    """在 fit 之前通过 attach_model 或 model.trainer 挂载，能保证 setup 读到正确的累积步数."""

    model = StepTrackingModel()
    trainer = Trainer(
        max_epochs=3,
        device="cpu",
        gradient_accumulation_steps=4,
    )

    trainer.attach_model(model)
    assert model.trainer is trainer
    assert model.accumulation_steps == 4

    # 另一个独立模型通过属性设置
    model2 = StepTrackingModel()
    model2.trainer = trainer
    assert model2.accumulation_steps == 4


def test_tqdm_progress_bar_lr_formatting() -> None:
    """验证 TqdmCallback 对微小学习率采用科学计数法，避免 0.0000 误判."""

    from xdl.callbacks.tqdm_callback import TqdmCallback

    # 小于 0.001 的非零学习率使用科学计数法
    assert TqdmCallback._format_metric_entry("lr", 5e-5) == "lr: 5.00e-05"
    assert TqdmCallback._format_metric_entry("train_lr", 1e-4) == "train_lr: 1.00e-04"
    # 普通指标仍然使用 4 位小数
    assert TqdmCallback._format_metric_entry("loss", 0.123456) == "loss: 0.1235"
    # 0.0 保持 0.0000
    assert TqdmCallback._format_metric_entry("lr", 0.0) == "lr: 0.0000"
