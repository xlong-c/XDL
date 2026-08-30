"""Diffusion-specific distillation losses.

Provides loss functions for timestep / trajectory-level distillation of
diffusion models, used after the RL stage to produce few-step samplers
(ref: Krea 2 Technical Report, 2026 - TDM section).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tdm_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Trajectory Distribution Matching loss for diffusion step distillation.

    Unlike standard KD which matches only at the clean-image level
    (DMD-style), TDM applies distribution matching across **multiple
    sampled timesteps**, effectively matching the teacher's *trajectory*
    rather than just its endpoint.

    The basic form is weighted MSE between student and teacher noise (or
    velocity) predictions at each sampled timestep.  The caller is
    responsible for:

    * sampling the timesteps from an appropriate schedule,
    * forwarding both student and teacher at those timesteps,
    * calling this function with the resulting predictions.

    Args:
        student_pred: ``(N, C, H, W)`` - student model prediction
            (noise / v-prediction / flow velocity) at the given timesteps.
        teacher_pred: Same shape as ``student_pred`` - teacher prediction.
        timesteps: ``(N,)`` - scalar timestep values in [0, 1] used for
            optional per-sample weighting.  Not used for the loss itself,
            but passed for future extensions (e.g. SNR weighting).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss when ``reduction`` is ``"mean"`` or ``"sum"``;
        per-sample tensor when ``"none"``.

    References:
        * Trajectory Distribution Matching (TDM): https://arxiv.org/abs/2503.06674
        * Krea 2 Technical Report (2026), Timestep Distillation section.
    """
    _ = timesteps  # reserved for future weighting strategies

    if student_pred.shape != teacher_pred.shape:
        raise ValueError(
            f"student_pred and teacher_pred must have the same shape, "
            f"got {student_pred.shape} vs {teacher_pred.shape}"
        )

    loss = F.mse_loss(student_pred, teacher_pred, reduction="none")

    # Flatten all non-batch dims for per-sample aggregation
    per_sample = loss.reshape(loss.shape[0], -1).mean(dim=-1)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"Unknown reduction '{reduction}'; expected mean/sum/none")


def tdm_loss_weighted(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    weight_power: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """TDM loss with SNR-style per-timestep weighting.

    Weights each timestep by ``(1 - t)^weight_power``, giving more weight
    to later (cleaner) timesteps.  This is a common heuristic in
    diffusion distillation.

    Args:
        student_pred: ``(N, C, H, W)`` - student prediction.
        teacher_pred: Same shape - teacher prediction.
        timesteps: ``(N,)`` - scalar timestep values in [0, 1].
        weight_power: Exponent for the ``(1 - t)`` weighting.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Weighted scalar or per-sample loss.
    """
    if student_pred.shape != teacher_pred.shape:
        raise ValueError(
            f"student_pred and teacher_pred must have the same shape, "
            f"got {student_pred.shape} vs {teacher_pred.shape}"
        )

    weights = (1.0 - timesteps) ** weight_power  # (N,)
    weights = weights / weights.sum()  # normalise so mean is comparable

    loss = F.mse_loss(student_pred, teacher_pred, reduction="none")
    per_sample = loss.reshape(loss.shape[0], -1).mean(dim=-1)

    weighted = per_sample * weights.to(per_sample.device)

    if reduction == "mean":
        return weighted.sum()
    if reduction == "sum":
        return weighted.sum()
    if reduction == "none":
        return weighted
    raise ValueError(f"Unknown reduction '{reduction}'; expected mean/sum/none")


__all__ = [
    "tdm_loss",
    "tdm_loss_weighted",
]
