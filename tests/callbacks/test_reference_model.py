"""Tests for ReferenceModelCallback."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn

from xdl.trainer.coreModel import CoreModel
from xdl.callbacks.reference_model import ReferenceModelCallback


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DPOCoreModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.policy = SimpleMLP()
        self.ref_model = copy.deepcopy(self.policy)

    def training_step(self, batch, batch_idx: int):
        return torch.tensor(0.0)


def test_reference_model_callback_freezes_on_fit_start() -> None:
    model = DPOCoreModel()
    callback = ReferenceModelCallback()

    # Before callback: ref should be trainable
    for p in model.ref_model.parameters():
        assert p.requires_grad is True

    callback.on_fit_start(None, model)

    # After callback: ref should be frozen
    for p in model.ref_model.parameters():
        assert p.requires_grad is False


def test_reference_model_callback_preserves_policy_trainable() -> None:
    model = DPOCoreModel()
    callback = ReferenceModelCallback()

    callback.on_fit_start(None, model)

    for p in model.policy.parameters():
        assert p.requires_grad is True


def test_ema_sync_updates_ref_toward_policy() -> None:
    model = DPOCoreModel()
    callback = ReferenceModelCallback(ema_sync_every_n_steps=1, ema_decay=0.5)

    callback.on_fit_start(None, model)

    # Capture initial ref weights before modifying policy
    initial_ref_w = model.ref_model.fc.weight.clone()

    # Change policy weights to all ones
    with torch.no_grad():
        model.policy.fc.weight.copy_(torch.ones_like(model.policy.fc.weight))

    # EMA sync: ref = 0.5 * ref_init + 0.5 * policy
    callback.on_train_batch_end(None, model, None, None, 0)

    expected = 0.5 * initial_ref_w + 0.5 * torch.ones_like(initial_ref_w)
    assert torch.allclose(model.ref_model.fc.weight, expected, atol=1e-5)


def test_ema_skips_when_not_sync_step() -> None:
    model = DPOCoreModel()
    callback = ReferenceModelCallback(ema_sync_every_n_steps=5, ema_decay=0.5)

    callback.on_fit_start(None, model)

    orig_ref = model.ref_model.fc.weight.clone()

    with torch.no_grad():
        model.policy.fc.weight.copy_(torch.ones_like(model.policy.fc.weight))

    # Only 3 steps — should skip EMA sync
    for _ in range(3):
        callback.on_train_batch_end(None, model, None, None, 0)

    assert torch.allclose(model.ref_model.fc.weight, orig_ref)


def test_missing_ref_attr_raises() -> None:
    class BadModel(CoreModel):
        def training_step(self, batch, batch_idx: int):
            return torch.tensor(0.0)

    model = BadModel()
    callback = ReferenceModelCallback()

    try:
        callback.on_fit_start(None, model)
        raise AssertionError("Expected AttributeError")
    except AttributeError:
        pass
