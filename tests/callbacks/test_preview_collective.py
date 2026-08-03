"""PreviewCallback 集体采样模式测试."""

from pathlib import Path
from typing import Any, List

from xdl.callbacks import PreviewCallback


class FakeTrainer:
    def __init__(self) -> None:
        self.current_epoch = 2
        self.global_step = 4
        self.wait_calls = 0

    def is_main_process(self) -> bool:
        return True

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1


class FakeNonMainTrainer(FakeTrainer):
    def is_main_process(self) -> bool:
        return False


class FakePreviewCore:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def is_main_process(self) -> bool:
        return False

    def save_preview(self, batch: Any, output_dir: Path) -> None:
        self.calls.append((batch, output_dir))


def test_preview_callback_collective_runs_on_non_main_rank(tmp_path) -> None:
    trainer = FakeNonMainTrainer()
    core = FakePreviewCore()
    callback = PreviewCallback(
        output_dir=tmp_path,
        every_n_epochs=1,
        collective=True,
    )

    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch={"idx": 0},
        batch_idx=0,
    )

    assert core.calls == [({"idx": 0}, tmp_path)]
    assert trainer.wait_calls == 2


def test_preview_callback_default_skips_non_main_rank(tmp_path) -> None:
    trainer = FakeNonMainTrainer()
    core = FakePreviewCore()
    callback = PreviewCallback(output_dir=tmp_path, every_n_epochs=1)

    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch={"idx": 0},
        batch_idx=0,
    )

    assert core.calls == []
