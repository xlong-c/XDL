from pathlib import Path

import torch
from torch import nn

from xdl.callbacks import FeatureCaptureCallback


class FakeTrainer:
    def __init__(self) -> None:
        self.current_epoch = 2
        self.global_step = 7

    def is_main_process(self) -> bool:
        return True


class FakeCoreModel:
    def __init__(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )

    def is_main_process(self) -> bool:
        return True


def test_feature_capture_callback_writes_first_validation_batch(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    core = FakeCoreModel()
    callback = FeatureCaptureCallback(
        ["0"],
        output_dir=tmp_path,
        every_n_epochs=1,
        first_val_batch_only=True,
    )

    callback.on_validation_epoch_start(trainer, core)
    batch = torch.randn(2, 4)
    core.model(batch)
    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch=batch,
        batch_idx=0,
    )
    callback.on_validation_epoch_end(trainer, core)

    files = sorted(tmp_path.glob("features_epoch*.pt"))
    assert len(files) == 1
    payload = torch.load(files[0])
    assert payload["epoch"] == 2
    assert payload["global_step"] == 7
    assert payload["modules"]["0"]["shape"] == (2, 3)


def test_feature_capture_callback_skips_non_matching_epoch(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    trainer.current_epoch = 1
    core = FakeCoreModel()
    callback = FeatureCaptureCallback(
        ["0"],
        output_dir=tmp_path,
        every_n_epochs=2,
    )

    callback.on_validation_epoch_start(trainer, core)
    core.model(torch.randn(2, 4))
    callback.on_validation_batch_end(
        trainer,
        core,
        outputs={},
        batch={},
        batch_idx=0,
    )

    assert list(tmp_path.glob("*.pt")) == []


def test_feature_capture_callback_respects_max_batches_per_epoch(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    core = FakeCoreModel()
    callback = FeatureCaptureCallback(
        ["0"],
        output_dir=tmp_path,
        every_n_epochs=1,
        first_val_batch_only=False,
        max_batches_per_epoch=1,
    )

    callback.on_validation_epoch_start(trainer, core)
    core.model(torch.randn(2, 4))
    callback.on_validation_batch_end(trainer, core, outputs={}, batch={}, batch_idx=0)
    core.model(torch.randn(2, 4))
    callback.on_validation_batch_end(trainer, core, outputs={}, batch={}, batch_idx=1)
    callback.on_validation_epoch_end(trainer, core)

    files = sorted(tmp_path.glob("features_epoch*.pt"))
    assert len(files) == 1


def test_feature_capture_callback_can_write_safetensors(tmp_path: Path) -> None:
    safetensors = __import__("safetensors")
    assert safetensors is not None

    trainer = FakeTrainer()
    core = FakeCoreModel()
    callback = FeatureCaptureCallback(
        ["0"],
        output_dir=tmp_path,
        every_n_epochs=1,
        save_format="safetensors",
    )

    callback.on_validation_epoch_start(trainer, core)
    core.model(torch.randn(2, 4))
    callback.on_validation_batch_end(trainer, core, outputs={}, batch={}, batch_idx=0)
    callback.on_validation_epoch_end(trainer, core)

    tensor_files = sorted(tmp_path.glob("features_epoch*.safetensors"))
    manifest_files = sorted(tmp_path.glob("features_epoch*.json"))
    assert len(tensor_files) == 1
    assert len(manifest_files) == 1
