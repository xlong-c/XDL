from pathlib import Path

import torch

from xdl.callbacks import AttentionRolloutCallback
from xdl.model.vit import VisionTransformer


class FakeTrainer:
    def __init__(self) -> None:
        self.current_epoch = 2
        self.global_step = 5

    def is_main_process(self) -> bool:
        return True


class FakeCoreModel:
    def __init__(self) -> None:
        self.model = VisionTransformer(
            img_size=8,
            patch_size=4,
            in_channels=3,
            num_classes=2,
            embed_dim=16,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            attn_dropout=0.0,
        )

    def is_main_process(self) -> bool:
        return True


def test_attention_rollout_callback_writes_rollout_file(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    core = FakeCoreModel()
    callback = AttentionRolloutCallback(output_dir=tmp_path)

    batch = {"x": torch.randn(1, 3, 8, 8)}
    callback.on_validation_epoch_start(trainer, core)
    callback.on_validation_batch_end(trainer, core, outputs={}, batch=batch, batch_idx=0)

    files = sorted(tmp_path.glob("attention_rollout_epoch*.pt"))
    assert len(files) == 1
    payload = torch.load(files[0])
    assert payload["rollout"].shape == (1, 5, 5)


def test_attention_rollout_callback_respects_max_batches_per_epoch(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    core = FakeCoreModel()
    callback = AttentionRolloutCallback(
        output_dir=tmp_path,
        first_val_batch_only=False,
        max_batches_per_epoch=1,
    )

    batch = {"x": torch.randn(1, 3, 8, 8)}
    callback.on_validation_epoch_start(trainer, core)
    callback.on_validation_batch_end(trainer, core, outputs={}, batch=batch, batch_idx=0)
    callback.on_validation_batch_end(trainer, core, outputs={}, batch=batch, batch_idx=1)

    files = sorted(tmp_path.glob("attention_rollout_epoch*.pt"))
    assert len(files) == 1
