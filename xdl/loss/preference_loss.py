"""Preference optimization losses for diffusion model post-training.

Provides DPO, STPO, and GRPO loss functions used in the preference
optimization and reinforcement learning stages of diffusion model
post-training pipelines (ref: Krea 2 Technical Report, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_win_logps: torch.Tensor,
    policy_lose_logps: torch.Tensor,
    ref_win_logps: torch.Tensor,
    ref_lose_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    .. math::

        L_{DPO} = -\\log \\sigma\\bigl(
            \\beta \\cdot
            [(\\log \\pi_\\theta(y_w) - \\log \\pi_{\\text{ref}}(y_w))
             - (\\log \\pi_\\theta(y_l) - \\log \\pi_{\\text{ref}}(y_l))]
        \\bigr)

    All inputs are assumed to be **scalar** log-probabilities per sample
    (shape ``(N,)`` or ``(N, 1)`` after squeezing).  The caller is
    responsible for computing these log-probabilities by integrating
    over the model output (e.g. diffusion noise prediction or
    autoregressive token sequence).

    Args:
        policy_win_logps: Log-probabilities of the winning samples under
            the policy model.
        policy_lose_logps: Log-probabilities of the losing samples under
            the policy model.
        ref_win_logps: Log-probabilities of the winning samples under
            the frozen reference model.
        ref_lose_logps: Log-probabilities of the losing samples under
            the frozen reference model.
        beta: Temperature / scaling factor for the implicit reward.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss tensor when ``reduction`` is ``"mean"`` or ``"sum"``;
        per-sample tensor when ``"none"``.
    """
    _validate_preference_shapes(
        policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps
    )

    policy_ratio = policy_win_logps - policy_lose_logps
    ref_ratio = ref_win_logps - ref_lose_logps
    logits = beta * (policy_ratio - ref_ratio)

    loss = -F.logsigmoid(logits)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# STPO Loss (DPO + auxiliary divergence penalty)
# ---------------------------------------------------------------------------


@dataclass
class STPOLossBreakdown:
    """Loss components returned by :func:`stpo_loss`."""

    total: torch.Tensor
    dpo: torch.Tensor
    auxiliary: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "dpo": float(self.dpo.detach().item()),
            "auxiliary": float(self.auxiliary.detach().item()),
        }


def stpo_loss(
    policy_win_logps: torch.Tensor,
    policy_lose_logps: torch.Tensor,
    ref_win_logps: torch.Tensor,
    ref_lose_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    auxiliary_weight: float = 0.01,
    reduction: str = "mean",
) -> STPOLossBreakdown:
    """Stabilized Preference Optimization loss (Krea 2 variant of DPO).

    Extends standard DPO with an **auxiliary KL-divergence penalty** that
    discourages the policy model from drifting too far from the reference
    distribution on *both* winning and losing samples.  This mitigates
    policy divergence (the phenomenon where DPO decreases likelihood of
    both samples at different rates) which otherwise manifests as
    high-frequency artifacts in later training stages.

    The auxiliary term is the reverse-KL proxy:

    .. math::

        L_{\\text{aux}} = \\bigl(
            \\log \\pi_{\\text{ref}}(y_w) - \\log \\pi_\\theta(y_w)
        \\bigr) + \\bigl(
            \\log \\pi_{\\text{ref}}(y_l) - \\log \\pi_\\theta(y_l)
        \\bigr)

    Args:
        policy_win_logps: Policy log-probabilities for winning samples.
        policy_lose_logps: Policy log-probabilities for losing samples.
        ref_win_logps: Reference log-probabilities for winning samples.
        ref_lose_logps: Reference log-probabilities for losing samples.
        beta: DPO temperature / scaling factor.
        auxiliary_weight: Weight of the auxiliary KL penalty term.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        :class:`STPOLossBreakdown` with ``total``, ``dpo``, and ``auxiliary``.
    """
    _validate_preference_shapes(
        policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps
    )

    dpo = dpo_loss(
        policy_win_logps,
        policy_lose_logps,
        ref_win_logps,
        ref_lose_logps,
        beta=beta,
        reduction="none",
    )

    # Reverse-KL proxy: penalize the policy for pulling away from the
    # reference on BOTH win and lose samples.
    kl_win = ref_win_logps - policy_win_logps
    kl_lose = ref_lose_logps - policy_lose_logps
    auxiliary = (kl_win + kl_lose).mean() if reduction != "none" else kl_win + kl_lose

    per_sample = dpo + auxiliary_weight * auxiliary

    return STPOLossBreakdown(
        total=_reduce(per_sample, reduction),
        dpo=_reduce(dpo, reduction),
        auxiliary=_reduce(auxiliary, reduction),
    )


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------


@dataclass
class GRPOLossBreakdown:
    """Loss components returned by :func:`grpo_loss`."""

    total: torch.Tensor
    advantage: torch.Tensor
    kl_penalty: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "advantage": float(self.advantage.detach().item()),
            "kl_penalty": float(self.kl_penalty.detach().item()),
        }


def grpo_loss(
    policy_logps: torch.Tensor,
    advantages: torch.Tensor,
    ref_logps: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.04,
    reduction: str = "mean",
) -> GRPOLossBreakdown:
    """Group Relative Policy Optimization loss (Shao et al., 2024).

    Designed for the reinforcement-learning stage of diffusion
    post-training (Krea 2 style).  The caller generates *K* rollouts per
    prompt, scores them with reward models, and normalises advantages
    within each group.  This function then computes the clipped surrogate
    objective with a KL penalty.

    .. math::

        r_i = \\frac{\\pi_\\theta(x_i)}{\\pi_{\\text{ref}}(x_i)}

        L = -\\min\\bigl(r_i A_i,\\ \\text{clip}(r_i, 1-\\epsilon, 1+\\epsilon) A_i\\bigr)

    plus a KL penalty ``β * KL(π_ref || π_θ)`` approximated as
    ``ref_logps - policy_logps``.

    Args:
        policy_logps: ``(N,)`` or ``(N, K)`` — policy log-probabilities
            for each rollout sample.
        advantages: Same shape as ``policy_logps`` — group-normalised
            advantages computed from reward-model scores.
        ref_logps: Same shape as ``policy_logps`` — reference model
            log-probabilities for the same rollouts.
        clip_epsilon: PPO-style clipping threshold.
        kl_beta: Weight of the KL penalty term.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        :class:`GRPOLossBreakdown` with ``total``, ``advantage``, and
        ``kl_penalty``.
    """
    _validate_grpo_shapes(policy_logps, advantages, ref_logps)

    # PPO clipped surrogate
    ratio = torch.exp(policy_logps - ref_logps)  # (N,) or (N, K)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    advantage_loss = -torch.min(ratio * advantages, clipped * advantages)

    # Approximate reverse KL
    kl = ref_logps - policy_logps

    per_sample = advantage_loss + kl_beta * kl

    return GRPOLossBreakdown(
        total=_reduce(per_sample, reduction),
        advantage=_reduce(advantage_loss, reduction),
        kl_penalty=_reduce(kl, reduction),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_preference_shapes(
    pw: torch.Tensor,
    pl: torch.Tensor,
    rw: torch.Tensor,
    rl: torch.Tensor,
) -> None:
    if pw.shape != pl.shape:
        raise ValueError(
            f"policy_win and policy_lose must have same shape, "
            f"got {pw.shape} vs {pl.shape}"
        )
    if rw.shape != rl.shape:
        raise ValueError(
            f"ref_win and ref_lose must have same shape, "
            f"got {rw.shape} vs {rl.shape}"
        )
    if pw.shape != rw.shape:
        raise ValueError(
            f"policy and ref shapes must match, "
            f"got {pw.shape} vs {rw.shape}"
        )


def _validate_grpo_shapes(
    policy: torch.Tensor,
    advantages: torch.Tensor,
    ref: torch.Tensor,
) -> None:
    if policy.shape != advantages.shape:
        raise ValueError(
            f"policy_logps and advantages shapes must match, "
            f"got {policy.shape} vs {advantages.shape}"
        )
    if policy.shape != ref.shape:
        raise ValueError(
            f"policy_logps and ref_logps shapes must match, "
            f"got {policy.shape} vs {ref.shape}"
        )


def _reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return tensor.mean()
    if reduction == "sum":
        return tensor.sum()
    if reduction == "none":
        return tensor
    raise ValueError(f"Unknown reduction '{reduction}'; expected mean/sum/none")


__all__ = [
    "GRPOLossBreakdown",
    "STPOLossBreakdown",
    "dpo_loss",
    "grpo_loss",
    "stpo_loss",
]
