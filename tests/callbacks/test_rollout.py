"""Tests for RolloutCallback."""

from __future__ import annotations

import torch
import torch.nn as nn

from xdl.trainer.core_model import CoreModel
from xdl.post_training.rollout import RolloutBatch, RolloutCallback


class DummyPolicy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyReward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * 0.5


class GRPOCoreModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.policy = DummyPolicy()

    def training_step(self, batch, batch_idx: int):
        return torch.tensor(0.0)


def dummy_rollout_fn(
    policy: nn.Module, prompts: list[str], num_rollouts: int
) -> torch.Tensor:
    """Generate dummy rollout latents: (B, K, 4, 4)."""
    B = len(prompts)
    return torch.randn(B, num_rollouts, 4, 4)


def test_rollout_callback_generates_augmented_batch() -> None:
    model = GRPOCoreModel()
    reward = DummyReward()
    callback = RolloutCallback(
        reward_models={"mock": reward},
        prompt_pool=["a cat", "a dog", "a sunset"],
        num_rollouts_per_prompt=2,
        rollout_fn=dummy_rollout_fn,
        ref_logp_fn=dummy_ref_logp_fn,
    )

    original_batch = [0, 1, 2]
    result = callback.on_train_batch_start(None, model, original_batch, 0)

    assert isinstance(result, RolloutBatch)
    assert result.original_batch == original_batch
    assert result.rollout_latents is not None
    assert result.rollout_latents.shape == (3, 2, 4, 4)
    assert result.rollout_rewards is not None
    assert result.rollout_rewards.shape == (3, 2)


def test_rollout_callback_respects_every_n_steps() -> None:
    model = GRPOCoreModel()
    reward = DummyReward()
    callback = RolloutCallback(
        reward_models={"mock": reward},
        prompt_pool=["prompt"],
        num_rollouts_per_prompt=1,
        rollout_fn=dummy_rollout_fn,
        ref_logp_fn=dummy_ref_logp_fn,
        rollout_every_n_steps=3,
    )

    batch = [1]
    # Steps 1 and 2: no rollout (returns original batch)
    r1 = callback.on_train_batch_start(None, model, batch, 0)
    assert isinstance(r1, list)

    r2 = callback.on_train_batch_start(None, model, batch, 0)
    assert isinstance(r2, list)

    # Step 3: rollout triggered
    r3 = callback.on_train_batch_start(None, model, batch, 0)
    assert isinstance(r3, RolloutBatch)


def test_rollout_callback_empty_pool_no_op() -> None:
    model = GRPOCoreModel()
    callback = RolloutCallback(
        reward_models={},
        prompt_pool=[],
        rollout_fn=dummy_rollout_fn,
    )

    batch = [1, 2]
    result = callback.on_train_batch_start(None, model, batch, 0)
    assert result == batch


def test_rollout_callback_no_rollout_fn_raises() -> None:
    model = GRPOCoreModel()
    callback = RolloutCallback(
        reward_models={},
        prompt_pool=["prompt"],
    )

    try:
        callback._generate_rollouts(model, ["prompt"])
        raise AssertionError("Expected RuntimeError")
    except RuntimeError:
        pass


def test_uniform_prompt_selection() -> None:
    callback = RolloutCallback(
        reward_models={},
        prompt_pool=["a", "b", "c", "d", "e"],
        rollout_fn=dummy_rollout_fn,
    )
    selected = callback._select_prompts(3)
    assert len(selected) == 3
    assert all(p in callback.prompt_pool for p in selected)


def dummy_ref_logp_fn(ref: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Compute dummy reference logps: (B, K) of -1.5."""
    return torch.full(latents.shape[:2], -1.5)


def test_rollout_callback_fills_ref_logps() -> None:
    model = GRPOCoreModel()
    callback = RolloutCallback(
        reward_models={"mock": DummyReward()},
        prompt_pool=["a cat"],
        num_rollouts_per_prompt=2,
        rollout_fn=dummy_rollout_fn,
        ref_logp_fn=dummy_ref_logp_fn,
    )

    result = callback.on_train_batch_start(None, model, [0], 0)

    assert isinstance(result, RolloutBatch)
    assert result.rollout_ref_logps is not None
    assert result.rollout_ref_logps.shape == (1, 2)
    assert torch.allclose(result.rollout_ref_logps, torch.full((1, 2), -1.5))


def test_rollout_callback_requires_ref_logp_fn() -> None:
    """有 rollout_fn 但缺 ref_logp_fn 时应显式报错, 不能静默写零占位."""
    model = GRPOCoreModel()
    callback = RolloutCallback(
        reward_models={"mock": DummyReward()},
        prompt_pool=["a cat"],
        num_rollouts_per_prompt=1,
        rollout_fn=dummy_rollout_fn,
    )

    try:
        callback.on_train_batch_start(None, model, [0], 0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "ref_logp_fn" in str(exc)
