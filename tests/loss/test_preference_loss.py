"""Tests for preference optimization losses: DPO, STPO, GRPO."""

from __future__ import annotations

import torch

from xdl.loss.preference_loss import (
    GRPOLossBreakdown,
    STPOLossBreakdown,
    dpo_loss,
    grpo_loss,
    stpo_loss,
)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def test_dpo_loss_scalar_and_differentiable() -> None:
    """DPO loss returns scalar and supports backward."""
    N = 8
    pw = torch.randn(N, requires_grad=True)
    pl = torch.randn(N, requires_grad=True)
    rw = torch.randn(N)
    rl = torch.randn(N)

    loss = dpo_loss(pw, pl, rw, rl, beta=0.1)
    loss.backward()

    assert loss.ndim == 0
    assert loss.item() > 0
    assert pw.grad is not None
    assert pl.grad is not None


def test_dpo_loss_reduction_none() -> None:
    N = 8
    pw = torch.randn(N)
    pl = torch.randn(N)
    rw = torch.randn(N)
    rl = torch.randn(N)

    loss = dpo_loss(pw, pl, rw, rl, reduction="none")
    assert loss.shape == (N,)


def test_dpo_loss_penalises_worse_policy() -> None:
    """A policy identical to reference should have lower loss than one that
    prefers the losing sample."""
    # Use controlled values so the test is deterministic:
    # ref says: win > lose (logps: 2.0 > 0.0)
    rw = torch.tensor([2.0, 2.0, 2.0, 2.0])
    rl = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # Good policy: roughly matches reference
    pw_good = torch.tensor([2.2, 1.9, 2.1, 1.8])
    pl_good = torch.tensor([0.1, -0.1, 0.0, 0.2])

    # Bad policy: higher logp for lose than win
    pw_bad = torch.tensor([0.1, 0.0, -0.1, 0.2])
    pl_bad = torch.tensor([2.2, 1.9, 2.1, 1.8])

    loss_good = dpo_loss(pw_good, pl_good, rw, rl, beta=1.0)
    loss_bad = dpo_loss(pw_bad, pl_bad, rw, rl, beta=1.0)

    assert loss_bad > loss_good


def test_dpo_shape_mismatch_raises() -> None:
    pw = torch.randn(8)
    pl = torch.randn(4)

    try:
        dpo_loss(pw, pl, pw, pl)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# STPO
# ---------------------------------------------------------------------------


def test_stpo_loss_returns_breakdown() -> None:
    N = 8
    pw = torch.randn(N, requires_grad=True)
    pl = torch.randn(N, requires_grad=True)
    rw = torch.randn(N)
    rl = torch.randn(N)

    result = stpo_loss(pw, pl, rw, rl, beta=0.1, auxiliary_weight=0.01)
    result.total.backward()

    assert isinstance(result, STPOLossBreakdown)
    assert result.total.ndim == 0
    assert result.dpo.ndim == 0
    assert result.auxiliary.ndim == 0
    assert pw.grad is not None


def test_stpo_auxiliary_penalises_divergence() -> None:
    """Higher auxiliary weight should increase loss when policy diverges."""
    rw = torch.tensor([2.0, 2.0, 2.0, 2.0])
    rl = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # Policy far from reference: lower win, higher lose
    pw_far = rw - 5.0
    pl_far = rl + 2.0  # asymmetric so deviations don't cancel

    r1 = stpo_loss(pw_far, pl_far, rw, rl, auxiliary_weight=0.0)
    r2 = stpo_loss(pw_far, pl_far, rw, rl, auxiliary_weight=0.1)

    assert r2.total.item() > r1.total.item()


def test_stpo_to_dict() -> None:
    N = 4
    x = torch.randn(N, requires_grad=True)
    r = stpo_loss(x, torch.randn(N), torch.randn(N), torch.randn(N))
    d = r.to_dict()
    assert isinstance(d["total"], float)
    assert isinstance(d["dpo"], float)
    assert isinstance(d["auxiliary"], float)


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


def test_grpo_loss_scalar_and_differentiable() -> None:
    B, K = 4, 4
    policy_logps = torch.randn(B, K, requires_grad=True)
    advantages = torch.randn(B, K)
    ref_logps = torch.randn(B, K)

    result = grpo_loss(policy_logps, advantages, ref_logps)
    result.total.backward()

    assert result.total.ndim == 0
    assert policy_logps.grad is not None


def test_grpo_loss_returns_breakdown() -> None:
    B, K = 4, 4
    policy_logps = torch.randn(B, K, requires_grad=True)
    advantages = torch.randn(B, K)
    ref_logps = torch.randn(B, K)

    result = grpo_loss(policy_logps, advantages, ref_logps)
    result.total.backward()

    assert isinstance(result, GRPOLossBreakdown)
    assert result.advantage.ndim == 0
    assert result.kl_penalty.ndim == 0


def test_grpo_loss_kl_penalty_increases_with_distance() -> None:
    """Absolute KL should increase when policy moves away from reference."""
    ref = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=False)
    advantages = torch.tensor([0.5, 0.5, 0.5, 0.5])

    # Close to reference
    r_near = grpo_loss(ref.clone() + 0.1, advantages, ref, kl_beta=1.0)
    # Far from reference
    r_far = grpo_loss(ref.clone() - 10.0, advantages, ref, kl_beta=1.0)

    # KL penalty magnitude should be larger
    assert abs(r_far.kl_penalty.item()) > abs(r_near.kl_penalty.item())


def test_grpo_shape_mismatch_raises() -> None:
    p = torch.randn(4, 4)
    a = torch.randn(4, 3)
    r = torch.randn(4, 4)

    try:
        grpo_loss(p, a, r)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_grpo_positive_advantage_decreases_loss() -> None:
    """Policy that scores higher on positive-advantage samples should
    see lower loss."""
    B = 4
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    ref = torch.zeros(B)

    # Policy aligns with advantages: positive for first two
    policy_good = advantages.clone().requires_grad_(True)
    # Policy opposes advantages
    policy_bad = (-advantages).clone().requires_grad_(True)

    result_good = grpo_loss(policy_good, advantages, ref, kl_beta=0.0)
    result_bad = grpo_loss(policy_bad, advantages, ref, kl_beta=0.0)

    assert result_bad.advantage.item() > result_good.advantage.item()
