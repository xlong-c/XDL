from pathlib import Path
from typing import Any, List, Tuple

from xdl.callbacks import PreviewCallback


class FakeTrainer:
    def __init__(self) -> None:
        self.current_epoch = 2
        self.global_step = 4

    def is_main_process(self) -> bool:
        return True


class FakePreviewCore:
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, Path]] = []

    def is_main_process(self) -> bool:
        return True

    def save_preview(self, batch: Any, output_dir: Path) -> None:
        self.calls.append((batch, output_dir))


def test_preview_callback_calls_first_validation_batch_only(tmp_path) -> None:
    trainer = FakeTrainer()
    core = FakePreviewCore()
    callback = PreviewCallback(output_dir=tmp_path, every_n_epochs=2)

    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch={"idx": 0},
        batch_idx=0,
    )
    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch={"idx": 1},
        batch_idx=1,
    )

    assert core.calls == [({"idx": 0}, tmp_path)]


def test_preview_callback_can_run_on_train_step_interval(tmp_path) -> None:
    trainer = FakeTrainer()
    core = FakePreviewCore()
    callback = PreviewCallback(
        output_dir=tmp_path,
        every_n_epochs=None,
        every_n_train_steps=2,
    )

    callback.on_train_batch_end(
        trainer,
        core,
        outputs={},
        batch={"train": True},
        batch_idx=0,
    )

    assert core.calls == [({"train": True}, tmp_path)]
