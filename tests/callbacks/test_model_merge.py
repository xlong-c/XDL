"""Tests for ModelMergeCallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from xdl.trainer.core_model import CoreModel
from xdl.post_training.model_merge import ModelMergeCallback, merge_checkpoints


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)


class MergeCoreModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleModel()

    def training_step(self, batch, batch_idx: int):
        return torch.tensor(0.0)


def _save_checkpoint(model: nn.Module, path: str, wrapped: bool = False) -> str:
    sd = model.state_dict()
    if wrapped:
        sd = {"model_state_dict": sd, "epoch": 0}
    torch.save(sd, path)
    return path


def test_linear_merge_two_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create two checkpoints with known weights
        m1 = SimpleModel()
        with torch.no_grad():
            m1.fc.weight.fill_(1.0)
            m1.fc.bias.fill_(0.0)
        p1 = _save_checkpoint(m1, str(base / "ckpt1.pt"))

        m2 = SimpleModel()
        with torch.no_grad():
            m2.fc.weight.fill_(3.0)
            m2.fc.bias.fill_(0.0)
        p2 = _save_checkpoint(m2, str(base / "ckpt2.pt"))

        callback = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            merge_weights=[1.0, 1.0],  # equal weight → avg
            output_path=str(base / "merged.pt"),
        )

        model = MergeCoreModel()
        callback.on_fit_end(None, model)

        merged = torch.load(str(base / "merged.pt"), weights_only=True)
        # (1.0 + 3.0) / 2 = 2.0
        assert torch.allclose(merged["fc.weight"], torch.full((4, 4), 2.0))


def test_linear_merge_weighted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        m1 = SimpleModel()
        with torch.no_grad():
            m1.fc.weight.fill_(0.0)
        p1 = _save_checkpoint(m1, str(base / "a.pt"))

        m2 = SimpleModel()
        with torch.no_grad():
            m2.fc.weight.fill_(10.0)
        p2 = _save_checkpoint(m2, str(base / "b.pt"))

        # weights: 0.2 for m1, 0.8 for m2 → 0.2*0 + 0.8*10 = 8.0
        callback = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            merge_weights=[0.2, 0.8],
            output_path=str(base / "merged.pt"),
        )

        model = MergeCoreModel()
        callback.on_fit_end(None, model)

        merged = torch.load(str(base / "merged.pt"), weights_only=True)
        assert torch.allclose(merged["fc.weight"], torch.full((4, 4), 8.0))


def test_wrapped_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        m1 = SimpleModel()
        with torch.no_grad():
            m1.fc.weight.fill_(5.0)
        p1 = _save_checkpoint(m1, str(base / "w1.pt"), wrapped=True)

        m2 = SimpleModel()
        with torch.no_grad():
            m2.fc.weight.fill_(5.0)
        p2 = _save_checkpoint(m2, str(base / "w2.pt"), wrapped=True)

        callback = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            output_path=str(base / "merged.pt"),
        )

        model = MergeCoreModel()
        callback.on_fit_end(None, model)

        merged = torch.load(str(base / "merged.pt"), weights_only=True)
        assert torch.allclose(merged["fc.weight"], torch.full((4, 4), 5.0))


def test_key_mismatch_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        m1 = SimpleModel()
        p1 = _save_checkpoint(m1, str(base / "k1.pt"))

        # Save a different model structure
        class OtherModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.other = nn.Linear(2, 2)

        m2 = OtherModel()
        p2 = _save_checkpoint(m2, str(base / "k2.pt"))

        callback = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            output_path=str(base / "merged.pt"),
        )

        try:
            callback._linear_merge()
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass


def test_merge_checkpoints_is_usable_without_trainer() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        m1 = SimpleModel()
        m2 = SimpleModel()
        with torch.no_grad():
            m1.fc.weight.fill_(2.0)
            m2.fc.weight.fill_(6.0)
        p1 = _save_checkpoint(m1, str(base / "plain1.pt"))
        p2 = _save_checkpoint(m2, str(base / "plain2.pt"))

        output = merge_checkpoints([p1, p2], merge_weights=[1.0, 3.0])

        merged = torch.load(output, weights_only=True)
        assert torch.allclose(merged["fc.weight"], torch.full((4, 4), 5.0))


def test_merge_at_start_vs_end() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        m1 = SimpleModel()
        p1 = _save_checkpoint(m1, str(base / "s1.pt"))
        m2 = SimpleModel()
        p2 = _save_checkpoint(m2, str(base / "s2.pt"))

        # merge_at_start=True
        cb_start = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            output_path=str(base / "start.pt"),
            merge_at_start=True,
        )
        model = MergeCoreModel()
        cb_start.on_fit_start(None, model)
        assert Path(base / "start.pt").exists()
        # on_fit_end should NOT merge again
        cb_start.on_fit_end(None, model)

        # merge_at_start=False (default)
        cb_end = ModelMergeCallback(
            checkpoint_paths=[p1, p2],
            output_path=str(base / "end.pt"),
            merge_at_start=False,
        )
        cb_end.on_fit_start(None, model)
        assert not Path(base / "end.pt").exists()
        cb_end.on_fit_end(None, model)
        assert Path(base / "end.pt").exists()


def test_requires_at_least_two_checkpoints() -> None:
    try:
        ModelMergeCallback(checkpoint_paths=["/tmp/one.pt"])
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_ties_dare_not_implemented() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        m = SimpleModel()
        p = _save_checkpoint(m, str(base / "t1.pt"))
        p2 = _save_checkpoint(m, str(base / "t2.pt"))

        cb = ModelMergeCallback(
            checkpoint_paths=[p, p2],
            output_path=str(base / "ties.pt"),
            method="ties",
        )
        try:
            cb.on_fit_end(None, MergeCoreModel())
            raise AssertionError("Expected NotImplementedError")
        except NotImplementedError:
            pass
