"""Metrics for image reconstruction and image generation."""

import math

import torch
import torch.nn.functional as F


def _check_image_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.dim() != 4:
        raise ValueError("image metrics expect tensors shaped (N, C, H, W)")


class PeakSignalNoiseRatio:
    """PSNR for image reconstruction.

    Args:
        data_range: Value range of the input image, usually ``1.0`` or ``255.0``.
        reduction: ``'mean'`` or ``'none'`` over the batch.
        epsilon: Numerical stability epsilon.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        reduction: str = "mean",
        epsilon: float = 1e-10,
    ) -> None:
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.data_range = data_range
        self.reduction = reduction
        self.epsilon = epsilon

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float | torch.Tensor:
        _check_image_pair(pred, target)
        mse = (pred - target).pow(2).flatten(1).mean(dim=1)
        psnr = 20.0 * math.log10(self.data_range) - 10.0 * torch.log10(
            mse.clamp_min(self.epsilon)
        )
        if self.reduction == "none":
            return psnr
        return psnr.mean().item()


class StructuralSimilarity:
    """SSIM for image reconstruction.

    Args:
        data_range: Value range of the input image.
        window_size: Local average pooling window size.
        reduction: ``'mean'`` or ``'none'`` over the batch.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        reduction: str = "mean",
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.data_range = data_range
        self.window_size = window_size
        self.reduction = reduction
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float | torch.Tensor:
        _check_image_pair(pred, target)
        score_map = _ssim_map(
            pred,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            k1=self.k1,
            k2=self.k2,
        )
        score = score_map.flatten(1).mean(dim=1)
        if self.reduction == "none":
            return score
        return score.mean().item()


class MultiScaleStructuralSimilarity:
    """MS-SSIM using repeated average pooling downsampling."""

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        levels: int = 3,
        weights: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.data_range = data_range
        self.window_size = window_size
        self.levels = levels
        self.weights = weights or [1.0 / levels] * levels
        if len(self.weights) != levels:
            raise ValueError("weights length must match levels")
        self.reduction = reduction

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float | torch.Tensor:
        _check_image_pair(pred, target)
        scores = []
        current_pred = pred
        current_target = target
        for level in range(self.levels):
            score = _ssim_map(
                current_pred,
                current_target,
                data_range=self.data_range,
                window_size=self.window_size,
                k1=0.01,
                k2=0.03,
            )
            scores.append(score.flatten(1).mean(dim=1).clamp_min(1e-6))
            if level != self.levels - 1:
                if min(current_pred.shape[-2:]) < 2:
                    raise ValueError(
                        "image is too small for the requested MS-SSIM levels"
                    )
                current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        stacked = torch.stack(
            [score.pow(weight) for score, weight in zip(scores, self.weights)],
            dim=0,
        )
        result = stacked.prod(dim=0)
        if self.reduction == "none":
            return result
        return result.mean().item()


class FrechetInceptionDistance:
    """FID from precomputed image features.

    This metric expects feature tensors, not raw images. Use features from the
    same encoder for generated and reference images.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def __call__(
        self, pred_features: torch.Tensor, target_features: torch.Tensor
    ) -> float:
        if pred_features.dim() != 2 or target_features.dim() != 2:
            raise ValueError("FID expects feature tensors shaped (N, D)")
        if pred_features.shape[1] != target_features.shape[1]:
            raise ValueError("feature dimensions must match")
        mu_pred, cov_pred = _feature_mean_cov(pred_features, self.epsilon)
        mu_target, cov_target = _feature_mean_cov(target_features, self.epsilon)
        diff = mu_pred - mu_target
        cov_pred_sqrt = _symmetric_matrix_sqrt(cov_pred, self.epsilon)
        cov_prod_sqrt = _symmetric_matrix_sqrt(
            cov_pred_sqrt @ cov_target @ cov_pred_sqrt,
            self.epsilon,
        )
        fid = diff.dot(diff) + torch.trace(cov_pred + cov_target - 2.0 * cov_prod_sqrt)
        return fid.clamp_min(0.0).item()


class KernelInceptionDistance:
    """Polynomial-kernel KID from precomputed image features."""

    def __init__(
        self,
        degree: int = 3,
        gamma: float | None = None,
        coef: float = 1.0,
    ) -> None:
        self.degree = degree
        self.gamma = gamma
        self.coef = coef

    def __call__(
        self, pred_features: torch.Tensor, target_features: torch.Tensor
    ) -> float:
        if pred_features.dim() != 2 or target_features.dim() != 2:
            raise ValueError("KID expects feature tensors shaped (N, D)")
        if pred_features.shape[1] != target_features.shape[1]:
            raise ValueError("feature dimensions must match")
        if pred_features.shape[0] < 2 or target_features.shape[0] < 2:
            raise ValueError("KID requires at least two samples per feature set")
        gamma = self.gamma or 1.0 / pred_features.shape[1]
        k_xx = (gamma * pred_features @ pred_features.T + self.coef).pow(self.degree)
        k_yy = (gamma * target_features @ target_features.T + self.coef).pow(
            self.degree
        )
        k_xy = (gamma * pred_features @ target_features.T + self.coef).pow(self.degree)
        m = pred_features.shape[0]
        n = target_features.shape[0]
        xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
        yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
        return (xx + yy - 2.0 * k_xy.mean()).item()


class InceptionScore:
    """Inception Score from class probabilities or logits.

    Args:
        from_logits: Apply softmax before scoring when true.
        splits: Number of splits used for the final mean score.
        epsilon: Numerical stability epsilon.
    """

    def __init__(
        self,
        from_logits: bool = True,
        splits: int = 1,
        epsilon: float = 1e-8,
    ) -> None:
        if splits <= 0:
            raise ValueError("splits must be positive")
        self.from_logits = from_logits
        self.splits = splits
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> float:
        if pred.dim() != 2:
            raise ValueError("InceptionScore expects tensor shaped (N, C)")
        probs = torch.softmax(pred, dim=1) if self.from_logits else pred
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        if self.splits > probs.shape[0]:
            raise ValueError("splits cannot exceed the number of samples")

        scores = []
        for split in torch.chunk(probs, self.splits, dim=0):
            marginal = split.mean(dim=0, keepdim=True)
            kl = split * (
                torch.log(split.clamp_min(self.epsilon))
                - torch.log(marginal.clamp_min(self.epsilon))
            )
            scores.append(torch.exp(kl.sum(dim=1).mean()))
        return torch.stack(scores).mean().item()


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


def _feature_mean_cov(
    features: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.to(dtype=torch.float64)
    mean = features.mean(dim=0)
    centered = features - mean
    denom = max(features.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return mean, cov + epsilon * eye


def _symmetric_matrix_sqrt(matrix: torch.Tensor, epsilon: float) -> torch.Tensor:
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = torch.linalg.eigh(sym)
    sqrt_vals = eigvals.clamp_min(epsilon).sqrt()
    return (eigvecs * sqrt_vals.unsqueeze(0)) @ eigvecs.T
