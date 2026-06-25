"""Tests for diffusion distillation losses: TDM."""

from __future__ import annotations

import torch

from xdl.loss.diffusion_distillation_loss import tdm_loss, tdm_loss_weighted


def test_tdm_loss_scalar_and_differentiable() -> None:
    N, C, H, W = 4, 3, 32, 32
    student = torch.randn(N, C, H, W, requires_grad=True)
    teacher = torch.randn(N, C, H, W)
    t = torch.rand(N)

    loss = tdm_loss(student, teacher, t)
    loss.backward()

    assert loss.ndim == 0
    assert student.grad is not None


def test_tdm_loss_zero_when_identical() -> None:
    N, C, H, W = 4, 3, 16, 16
    x = torch.randn(N, C, H, W)
    t = torch.rand(N)

    loss = tdm_loss(x, x, t)
    assert loss.item() < 1e-6


def test_tdm_loss_increases_with_larger_error() -> None:
    N, C, H, W = 4, 3, 16, 16
    teacher = torch.randn(N, C, H, W)
    t = torch.rand(N)

    student_close = teacher + 0.1 * torch.randn(N, C, H, W)
    student_far = teacher + 10.0 * torch.randn(N, C, H, W)

    loss_close = tdm_loss(student_close, teacher, t)
    loss_far = tdm_loss(student_far, teacher, t)

    assert loss_far.item() > loss_close.item()


def test_tdm_loss_reduction_none() -> None:
    N, C, H, W = 8, 3, 16, 16
    student = torch.randn(N, C, H, W)
    teacher = torch.randn(N, C, H, W)
    t = torch.rand(N)

    loss = tdm_loss(student, teacher, t, reduction="none")
    assert loss.shape == (N,)


def test_tdm_loss_shape_mismatch_raises() -> None:
    s = torch.randn(4, 3, 16, 16)
    t = torch.randn(4, 3, 32, 32)
    ts = torch.rand(4)

    try:
        tdm_loss(s, t, ts)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_tdm_loss_weighted_smoke() -> None:
    N, C, H, W = 4, 3, 16, 16
    student = torch.randn(N, C, H, W, requires_grad=True)
    teacher = torch.randn(N, C, H, W)
    t = torch.rand(N)

    loss = tdm_loss_weighted(student, teacher, t, weight_power=0.5)
    loss.backward()

    assert loss.ndim == 0
    assert student.grad is not None
