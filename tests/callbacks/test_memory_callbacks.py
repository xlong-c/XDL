"""显存管理回调测试."""

from typing import Any

import torch

from xdl.callbacks import ActivationOffloadCallback, GradientCheckpointingCallback
from xdl.trainer import Trainer
from xdl.trainer.core_model import CoreModel


class FakeTrainer:
    def __init__(self) -> None:
        self.global_step = 1


class FakeCore:
    pass


def test_gradient_checkpointing_callback_hf_style() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def gradient_checkpointing_enable(self) -> None:
            self.called = True

    core = Model()
    GradientCheckpointingCallback().on_fit_start(FakeTrainer(), core)
    assert core.called is True


def test_gradient_checkpointing_callback_fallback_setter() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = False

        def set_gradient_checkpointing(self, value: bool) -> None:
            self.value = value

    core = Model()
    GradientCheckpointingCallback().on_fit_start(FakeTrainer(), core)
    assert core.value is True


def test_activation_offload_callback_enter_and_exit() -> None:
    trainer, core = FakeTrainer(), FakeCore()
    callback = ActivationOffloadCallback(min_bytes=1)

    callback.on_train_batch_start(trainer, core, batch={}, batch_idx=0)
    assert callback._active_ctx is not None

    callback.on_train_batch_end(trainer, core, outputs={}, batch={}, batch_idx=0)
    assert callback._active_ctx is None


def test_activation_offload_callback_cleans_up_on_exception() -> None:
    trainer, core = FakeTrainer(), FakeCore()
    callback = ActivationOffloadCallback(min_bytes=1)

    callback.on_train_batch_start(trainer, core, batch={}, batch_idx=0)
    callback.on_exception(trainer, core, ValueError("boom"))

    assert callback._active_ctx is None


def test_activation_offload_callback_teardown_cleans_up() -> None:
    trainer, core = FakeTrainer(), FakeCore()
    callback = ActivationOffloadCallback(min_bytes=1)

    callback.on_train_batch_start(trainer, core, batch={}, batch_idx=0)
    callback.teardown(trainer, core, stage="fit")

    assert callback._active_ctx is None


def test_activation_offload_callback_end_to_end_fit() -> None:
    class SimpleModel(CoreModel):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

        def training_step(self, batch: Any, batch_idx: int) -> None:
            del batch, batch_idx

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = SimpleModel()
    trainer = Trainer(
        max_epochs=1,
        device="cpu",
        callbacks=[ActivationOffloadCallback(min_bytes=1)],
    )

    trainer.fit(model=model, train_dataloader=[1.0, 2.0])
    assert model.total_train_steps == 2
