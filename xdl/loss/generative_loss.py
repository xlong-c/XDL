"""Generative modeling losses for VAE, GAN and diffusion-style training."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_loss import _validate_reduction


class KLDivergenceLoss(nn.Module):
    """KL divergence from a diagonal Gaussian posterior to ``N(0, I)``.

    Args:
        reduction: ``'none'``, ``'mean'`` or ``'sum'`` over the batch.

    Shape:
        ``mu`` and ``logvar`` share shape ``(N, latent_dim, ...)``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if mu.shape != logvar.shape:
            raise ValueError(
                f"mu and logvar shape mismatch: {mu.shape} vs {logvar.shape}"
            )
        logvar_clamped = torch.clamp(logvar, max=20.0)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar_clamped.exp())
        kl = kl.flatten(1).sum(dim=1)
        if self.reduction == "mean":
            return kl.mean()
        if self.reduction == "sum":
            return kl.sum()
        return kl


class VAELoss(nn.Module):
    """VAE objective combining reconstruction loss and KL divergence.

    Args:
        reconstruction: ``'mse'``, ``'l1'`` or ``'bce'``.
        beta: KL weight for beta-VAE style training.
        reduction: ``'mean'`` or ``'sum'``.
    """

    def __init__(
        self,
        reconstruction: str = "mse",
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reconstruction not in {"mse", "l1", "bce"}:
            raise ValueError("reconstruction must be one of: 'mse', 'l1', 'bce'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reconstruction = reconstruction
        self.beta = beta
        self.reduction = reduction
        self.kl = KLDivergenceLoss(reduction=reduction)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"reconstruction and target shape mismatch: {reconstruction.shape} vs {target.shape}"
            )
        if self.reconstruction == "mse":
            recon_loss = F.mse_loss(reconstruction, target, reduction=self.reduction)
        elif self.reconstruction == "l1":
            recon_loss = F.l1_loss(reconstruction, target, reduction=self.reduction)
        else:
            recon_loss = F.binary_cross_entropy(
                reconstruction,
                target,
                reduction=self.reduction,
            )
        return recon_loss + self.beta * self.kl(mu, logvar)


class GANLoss(nn.Module):
    """Common adversarial loss wrapper.

    Args:
        mode: ``'vanilla'`` BCE, ``'lsgan'``, ``'hinge'`` or ``'wgan'``.
        real_label: Target label for real samples in BCE/LSGAN modes.
        fake_label: Target label for fake samples in BCE/LSGAN modes.

    ``forward(logits, target_is_real, is_discriminator)`` can be called for each
    discriminator branch or for the generator. For hinge/WGAN generator loss,
    ``target_is_real`` is ignored and ``logits`` should be fake logits.
    """

    def __init__(
        self,
        mode: str = "hinge",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"vanilla", "lsgan", "hinge", "wgan"}:
            raise ValueError("mode must be one of: 'vanilla', 'lsgan', 'hinge', 'wgan'")
        self.mode = mode
        self.real_label = real_label
        self.fake_label = fake_label

    def forward(
        self,
        logits: torch.Tensor,
        target_is_real: bool = True,
        is_discriminator: bool = True,
    ) -> torch.Tensor:
        if self.mode == "vanilla":
            target = torch.full_like(
                logits, self.real_label if target_is_real else self.fake_label
            )
            return F.binary_cross_entropy_with_logits(logits, target)
        if self.mode == "lsgan":
            target = torch.full_like(
                logits, self.real_label if target_is_real else self.fake_label
            )
            return F.mse_loss(logits, target)
        if self.mode == "hinge":
            if not is_discriminator:
                return -logits.mean()
            if target_is_real:
                return F.relu(1.0 - logits).mean()
            return F.relu(1.0 + logits).mean()
        if not is_discriminator or target_is_real:
            return -logits.mean()
        return logits.mean()


class HingeDiscriminatorLoss(nn.Module):
    """Hinge discriminator loss for real and fake logits."""

    def forward(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


class HingeGeneratorLoss(nn.Module):
    """Hinge generator loss for fake logits."""

    def forward(
        self, fake_logits: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return -fake_logits.mean()


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN and perceptual-style training.

    Args:
        loss: ``'l1'`` or ``'l2'``.
        layer_weights: Optional per-feature weights.

    Inputs can be tensors or equally-sized sequences of tensors, e.g.
    discriminator intermediate activations.
    """

    def __init__(
        self,
        loss: str = "l1",
        layer_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if loss not in {"l1", "l2"}:
            raise ValueError("loss must be 'l1' or 'l2'")
        self.loss = loss
        self.layer_weights = list(layer_weights) if layer_weights is not None else None

    def forward(
        self,
        pred_features: torch.Tensor | Sequence[torch.Tensor],
        target_features: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        pred_list = _as_feature_list(pred_features)
        target_list = _as_feature_list(target_features)
        if len(pred_list) != len(target_list):
            raise ValueError(
                "pred_features and target_features must have the same length"
            )
        if self.layer_weights is not None and len(self.layer_weights) != len(pred_list):
            raise ValueError("layer_weights length must match feature list length")

        weights = self.layer_weights or [1.0] * len(pred_list)
        losses: list[torch.Tensor] = []
        for pred, target, weight in zip(pred_list, target_list, weights):
            if pred.shape != target.shape:
                raise ValueError(
                    f"feature shape mismatch: {pred.shape} vs {target.shape}"
                )
            if self.loss == "l1":
                losses.append(weight * F.l1_loss(pred, target))
            else:
                losses.append(weight * F.mse_loss(pred, target))
        return torch.stack(losses).sum() / sum(weights)


class DiffusionPredictionLoss(nn.Module):
    """Prediction loss for diffusion noise/sample/velocity targets.

    Args:
        loss: ``'mse'``, ``'l1'`` or ``'huber'``.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        huber_delta: Delta used when ``loss='huber'``.
    """

    def __init__(
        self,
        loss: str = "mse",
        reduction: str = "mean",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if loss not in {"mse", "l1", "huber"}:
            raise ValueError("loss must be one of: 'mse', 'l1', 'huber'")
        _validate_reduction(reduction)
        self.loss = loss
        self.reduction = reduction
        self.huber_delta = huber_delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
            )
        if self.loss == "mse":
            loss = (pred - target).pow(2)
        elif self.loss == "l1":
            loss = (pred - target).abs()
        else:
            loss = F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")

        if mask is not None:
            if (
                mask.shape != pred.shape
                and mask.shape != pred.shape[:1] + pred.shape[2:]
            ):
                raise ValueError(
                    "mask must match pred shape or omit the channel dimension"
                )
            if mask.dim() == pred.dim() - 1:
                mask = mask.unsqueeze(1)
            mask = mask.to(dtype=loss.dtype, device=loss.device)
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.expand_as(loss).sum().clamp_min(1.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _as_feature_list(
    features: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(features, torch.Tensor):
        return [features]
    return list(features)
