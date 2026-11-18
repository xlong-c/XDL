"""TrainerControl lifecycle tests."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from xdl.callbacks import Callback
from xdl.trainer import CoreModel, Trainer, TrainerControl


class CountingModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.seen_batches = 0

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        inputs = batch[0]
        loss = self.layer(inputs).square().mean()
        self.manual_optimization_step(loss)
        self.seen_batches += 1

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx


class EndCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.ended = False

    def on_train_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        del trainer, core_module
        self.ended = True


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=2)


def test_pause_waits_until_resume() -> None:
    control = TrainerControl()
    control.request_pause()
    resumed = threading.Event()

    def wait_for_resume() -> None:
        control.wait_if_paused()
        resumed.set()

    thread = threading.Thread(target=wait_for_resume)
    thread.start()
    time.sleep(0.03)
    assert not resumed.is_set()

    control.resume()
    thread.join(timeout=1)
    assert resumed.is_set()


def test_stop_is_honored_after_current_batch_and_runs_end_hooks() -> None:
    model = CountingModel()
    callback = EndCallback()
    control = TrainerControl()
    trainer = Trainer(max_epochs=10, device="cpu", callbacks=[callback])

    class StopAfterTwo(Callback):
        def on_train_batch_end(
            self,
            trainer: Trainer,
            core_module: CoreModel,
            **kwargs: Any,
        ) -> None:
            del core_module, kwargs
            if trainer.global_step == 2:
                control.request_stop(save_checkpoint=True)

    trainer.callback_list.add_callback(StopAfterTwo())
    trainer.fit(model, _loader(), control=control)

    assert trainer.global_step == 2
    assert model.seen_batches == 2
    assert callback.ended
    assert control.save_checkpoint_requested


def test_fit_accepts_resume_checkpoint_argument(tmp_path) -> None:
    model = CountingModel()
    trainer = Trainer(max_epochs=1, device="cpu")

    try:
        trainer.fit(model, _loader(), resume_from_checkpoint=tmp_path / "missing")
    except ValueError as exc:
        assert "目录路径" in str(exc)
    else:
        raise AssertionError("missing checkpoint must fail before the first batch")

    assert trainer.global_step == 0


def test_resume_restores_weights_optimizer_and_remaining_steps(tmp_path) -> None:
    source = CountingModel()
    source._optimizers = [source.configure_optimizers()]
    source._current_epoch = 1
    source._total_train_steps = 4
    checkpoint = source.save_checkpoint(
        str(tmp_path),
        naming_keys=["step"],
        save_optimizer=True,
        save_scheduler=True,
    )
    expected_weight = source.layer.weight.detach().clone()

    model = CountingModel()
    trainer = Trainer(max_epochs=2, device="cpu")
    trainer.fit(model, _loader(), resume_from_checkpoint=checkpoint)

    assert torch.equal(model.layer.weight.detach(), expected_weight) is False
    assert trainer.global_step == 8
    assert trainer.current_epoch == 2
