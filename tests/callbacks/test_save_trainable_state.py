from pathlib import Path
from typing import List

from xdl.post_training import SaveTrainableStateCallback


class FakeTrainer:
    def __init__(self) -> None:
        self.current_epoch = 2
        self.global_step = 4

    def is_main_process(self) -> bool:
        return True


class FakeCore:
    def __init__(self) -> None:
        self.paths: List[Path] = []

    def is_main_process(self) -> bool:
        return True

    def save_trainable_state(self, path: Path) -> None:
        self.paths.append(path)


def test_save_trainable_state_callback_saves_step_epoch_and_final(tmp_path) -> None:
    trainer = FakeTrainer()
    core = FakeCore()
    callback = SaveTrainableStateCallback(
        tmp_path,
        every_n_epochs=2,
        every_n_train_steps=2,
        save_final=True,
    )

    callback.on_train_batch_end(trainer, core, outputs={}, batch={}, batch_idx=0)
    callback.on_train_epoch_end(trainer, core)
    callback.on_train_end(trainer, core)

    assert core.paths == [
        tmp_path / "step_0000004",
        tmp_path / "epoch_0002_step_0000004",
        tmp_path / "final",
    ]


def test_save_trainable_state_callback_ignores_missing_method(tmp_path) -> None:
    callback = SaveTrainableStateCallback(tmp_path)

    callback.on_train_end(FakeTrainer(), object())
