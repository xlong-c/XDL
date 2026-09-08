"""checkpoint 工具行为回归测试."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xdl.utils.checkpoint import save_checkpoint

pytest.importorskip("safetensors")


def test_save_checkpoint_does_not_mutate_input(tmp_path: Path) -> None:
    """safetensors 分支不应从调用方字典中删掉 state_dict."""

    checkpoint = {
        "epoch": 1,
        "state_dict": {"model": {"weight": torch.zeros(2)}},
        "optimizer_states": None,
    }
    original_keys = set(checkpoint)

    save_checkpoint(tmp_path / "ckpt", checkpoint, format="safetensors")

    assert set(checkpoint) == original_keys
    assert "state_dict" in checkpoint
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    assert (tmp_path / "ckpt" / "meta.pt").exists()
