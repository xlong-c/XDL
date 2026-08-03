"""Image reconstruction losses for restoration, autoencoding and generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_loss import _reduce_loss, _validate_reduction


def _check_image_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.dim() != 4:
        raise ValueError("image losses expect tensors shaped (N, C, H, W)")


class CharbonnierLoss(nn.Module):
    """Differentiable robust L1 loss used in image restoration.

    Args:
        epsilon: Small value inside the square root.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt((pred - target).pow(2) + self.epsilon**2)
        return _reduce_loss(loss, self.reduction)


class TotalVariationLoss(nn.Module):
    """Total variation regularization for generated or reconstructed images.

    Args:
        reduction: ``'mean'`` or ``'sum'``.
        normalize: Divide by the number of finite differences when true.
    """

    def __init__(self, reduction: str = "mean", normalize: bool = True) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.normalize = normalize

    def forward(
        self, image: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("TotalVariationLoss expects image shaped (N, C, H, W)")
        horizontal = (image[..., 1:, :] - image[..., :-1, :]).abs()
        vertical = (image[..., :, 1:] - image[..., :, :-1]).abs()
        loss = horizontal.sum() + vertical.sum()
        if self.normalize:
            count = horizontal.numel() + vertical.numel()
            loss = loss / max(count, 1)
        if self.reduction == "mean":
            return loss
        return loss * image.shape[0]


class GradientDifferenceLoss(nn.Module):
    """L1/L2 difference between image gradients.

    Args:
        penalty: ``'l1'`` or ``'l2'`` gradient difference.
        reduction: ``'mean'`` or ``'sum'``.
    """

    def __init__(self, penalty: str = "l1", reduction: str = "mean") -> None:
        super().__init__()
        if penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be 'l1' or 'l2'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.penalty = penalty
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_image_pair(pred, target)
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        diff_x = pred_dx - target_dx
        diff_y = pred_dy - target_dy
        if self.penalty == "l1":
            loss_x = diff_x.abs()
            loss_y = diff_y.abs()
        else:
            loss_x = diff_x.pow(2)
            loss_y = diff_y.pow(2)
        if self.reduction == "sum":
            return loss_x.sum() + loss_y.sum()
        return (loss_x.mean() + loss_y.mean()) / 2.0


class SSIMLoss(nn.Module):
    """Structural similarity loss, ``1 - SSIM``.

    This implementation is fully differentiable and uses average pooling as the
    local window, so it has no optional dependency on image libraries.

    Args:
        data_range: Value range of the input image, usually ``1.0`` or ``255.0``.
        window_size: Local pooling window size.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        k1: SSIM constant multiplier.
        k2: SSIM constant multiplier.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        reduction: str = "mean",
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.data_range = data_range
        self.window_size = window_size
        self.reduction = reduction
        self.k1 = k1
        self.k2 = k2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_image_pair(pred, target)
        score = _ssim_map(
            pred,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            k1=self.k1,
            k2=self.k2,
        )
        loss = 1.0 - score.flatten(1).mean(dim=1)
        return _reduce_loss(loss, self.reduction)


class ReconstructionLoss(nn.Module):
    """Weighted image reconstruction loss combining pixel, SSIM, gradient and TV terms.

    Args:
        pixel: ``'l1'``, ``'l2'`` or ``'charbonnier'``.
        pixel_weight: Weight for the pixel reconstruction term.
        ssim_weight: Weight for ``SSIMLoss``.
        gradient_weight: Weight for ``GradientDifferenceLoss``.
        tv_weight: Weight for ``TotalVariationLoss`` on predictions.
        data_range: Value range used by SSIM.
    """

    def __init__(
        self,
        pixel: str = "l1",
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.0,
        gradient_weight: float = 0.0,
        tv_weight: float = 0.0,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if pixel not in {"l1", "l2", "charbonnier"}:
            raise ValueError("pixel must be 'l1', 'l2' or 'charbonnier'")
        self.pixel = pixel
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.tv_weight = tv_weight
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(data_range=data_range)
        self.gradient = GradientDifferenceLoss()
        self.tv = TotalVariationLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_image_pair(pred, target)
        if self.pixel == "l1":
            total = self.pixel_weight * F.l1_loss(pred, target)
        elif self.pixel == "l2":
            total = self.pixel_weight * F.mse_loss(pred, target)
        else:
            total = self.pixel_weight * self.charbonnier(pred, target)

        if self.ssim_weight:
            total = total + self.ssim_weight * self.ssim(pred, target)
        if self.gradient_weight:
            total = total + self.gradient_weight * self.gradient(pred, target)
        if self.tv_weight:
            total = total + self.tv_weight * self.tv(pred)
        return total


def _ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    window_size: int,
    k1: float,
    k2: float,
) -> torch.Tensor:
    padding = window_size // 2
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = (
        F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x_sq
    )
    sigma_y = (
        F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y_sq
    )
    sigma_xy = (
        F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_xy
    )

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(torch.finfo(pred.dtype).eps)
