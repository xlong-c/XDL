"""Reusable model-side components for TBSM training.

The implementation follows the Apache-2.0 licensed TBSM reference workflow
from https://github.com/sp12138/TBSM and keeps the training loop in
``xdl.trainer.tbsm_model``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


SCATTERING_EPS = 1e-6
_TRACKER_OUTPUT_MODES = {"field", "potential_grad", "potential"}


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    """Return a world-size averaged tensor when distributed is initialized."""

    if not (dist.is_available() and dist.is_initialized()):
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced.div_(dist.get_world_size())
    return reduced


def _self_normalized(loss: torch.Tensor) -> torch.Tensor:
    """Rescale a loss by its detached value while preserving its gradient."""

    return loss / loss.detach().clamp_min(1e-8)


def _feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert common feature-map layouts to ``[batch, tokens, channels]``."""

    if feature_map.ndim == 4:
        return feature_map.flatten(2).transpose(1, 2)
    if feature_map.ndim == 3:
        return feature_map
    if feature_map.ndim == 2:
        return feature_map.unsqueeze(1)
    raise ValueError(
        "TBSM representor output must have rank 2, 3, or 4; "
        f"got shape {tuple(feature_map.shape)}"
    )


class TBSMGenerator(nn.Module):
    """Adapt a conditional backbone and an optional decoder to TBSM's interface.

    The backbone is called as ``backbone(x, labels, t=t)``.  A decoder may
    convert the backbone output to the image space used by representation
    maps.  Without a decoder, the convention is the TBSM pixel-space mapping
    ``x * 0.5 + 0.5``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # The decoder is frozen infrastructure and should not enter the
        # generator optimizer or module state.
        object.__setattr__(self, "decoder", decoder)
        if isinstance(decoder, nn.Module):
            decoder.requires_grad_(False)
            decoder.eval()

    def __deepcopy__(self, memo: dict[int, Any]) -> "TBSMGenerator":
        """Copy trainable backbone state while sharing the frozen decoder."""

        copied = type(self)(
            copy.deepcopy(self.backbone, memo),
            decoder=self.decoder,
        )
        memo[id(self)] = copied
        return copied

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "TBSMGenerator":
        """Move a non-registered decoder together with the generator."""

        super()._apply(fn)
        if isinstance(self.decoder, nn.Module):
            self.decoder._apply(fn)
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the conditional backbone."""

        timesteps = (
            torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
            if t is None
            else t
        )
        try:
            return self.backbone(inputs, labels, t=timesteps)
        except TypeError:
            return self.backbone(inputs, labels, timesteps)

    def unwrapped_forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the raw backbone when it is wrapped by DDP."""

        backbone = self.backbone
        if hasattr(backbone, "module") and isinstance(backbone.module, nn.Module):
            backbone = backbone.module
        timesteps = (
            torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
            if t is None
            else t
        )
        try:
            return backbone(inputs, labels, t=timesteps)
        except TypeError:
            return backbone(inputs, labels, timesteps)

    def to_image(self, value: torch.Tensor) -> torch.Tensor:
        """Map backbone output into the representation-map image space."""

        if self.decoder is None:
            return value * 0.5 + 0.5
        return self.decoder(value)


class IdentityRepresentor(nn.Module):
    """Identity representation map for pixel-space scattering."""

    feat_dim = 3

    def extract_featmap(self, images: torch.Tensor) -> torch.Tensor:
        """Return images as the representation feature map."""

        return images


def _sincos_embedding_1d(
    values: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create a sinusoidal embedding for a one-dimensional position."""

    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(
            0,
            half_dim,
            dtype=torch.float32,
            device=values.device,
        )
        / max(half_dim, 1)
    )
    args = values.reshape(-1, 1).float() * frequencies.reshape(1, -1)
    embedding = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal embeddings for scalar conditions."""

    return _sincos_embedding_1d(values, dim, max_period=max_period)


class _ParameterFreeRMSNorm(nn.Module):
    """RMS normalization without learned affine parameters."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``value``."""

        rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value * rms


class _SpatialTransformerBlock(nn.Module):
    """Self-attention and MLP block for scattering tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = _ParameterFreeRMSNorm()
        self.qkv = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        )
        self.proj = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.norm2 = _ParameterFreeRMSNorm()
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim, mlp_dim, bias=False)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(mlp_dim, hidden_dim, bias=False)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply one token transformer block."""

        hidden = self.norm1(value)
        batch_size, token_count, channels = hidden.shape
        query_key_value = self.qkv(hidden).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, val = query_key_value.permute(2, 0, 3, 1, 4)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        hidden = (attention @ val).transpose(1, 2).reshape(
            batch_size,
            token_count,
            channels,
        )
        value = value + self.proj(hidden)
        return value + self.mlp(self.norm2(value))


class ScatteringTracker(nn.Module):
    """Predict a condition- and time-dependent scattering vector."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int = 736,
        time_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        output_mode: str = "field",
        use_potential_grad: bool = False,
    ) -> None:
        super().__init__()
        if use_potential_grad:
            output_mode = "potential_grad"
        if output_mode not in _TRACKER_OUTPUT_MODES:
            raise ValueError(
                f"Unknown tracker output_mode: {output_mode}; "
                f"expected one of {sorted(_TRACKER_OUTPUT_MODES)}"
            )
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.output_mode = output_mode
        self.register_buffer("_pos_embed_cache", torch.empty(0), persistent=False)
        self._pos_embed_cache_key: Optional[tuple[Any, ...]] = None

        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(time_dim, hidden_dim, bias=False)),
            _ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            ),
            _ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.x_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
            _ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList(
            [
                _SpatialTransformerBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.potential_head = (
            nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Linear(feat_dim, hidden_dim, bias=False)
                ),
                _ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                ),
                _ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1, bias=False)),
            )
            if output_mode != "field"
            else None
        )

    def _position_embedding(
        self,
        token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build and cache a 2D or 1D token position embedding."""

        key = (token_count, device.type, device.index, dtype)
        if self._pos_embed_cache_key == key and self._pos_embed_cache.numel() > 0:
            return self._pos_embed_cache
        side = math.isqrt(token_count)
        if side * side == token_count:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            )
            dim_y = self.hidden_dim // 2
            dim_x = self.hidden_dim - dim_y
            embedding = torch.cat(
                [
                    _sincos_embedding_1d(grid_y.reshape(-1), dim_y),
                    _sincos_embedding_1d(grid_x.reshape(-1), dim_x),
                ],
                dim=-1,
            )
        else:
            position = torch.arange(token_count, device=device)
            embedding = _sincos_embedding_1d(position, self.hidden_dim)
        embedding = embedding.to(dtype).unsqueeze(0)
        self._pos_embed_cache = embedding
        self._pos_embed_cache_key = key
        return self._pos_embed_cache

    def forward(
        self,
        condition: torch.Tensor,
        query: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a scattering field, potential, or potential gradient."""

        differentiable_query = query
        if (
            self.output_mode == "potential_grad"
            and not differentiable_query.requires_grad
        ):
            differentiable_query = query.detach().clone().requires_grad_(True)
        _, token_count, _ = differentiable_query.shape
        class_embedding = self.label_emb(condition.reshape(-1).long())
        time_embedding = _sinusoidal_embedding(time.reshape(-1) * 1000, self.time_dim)
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )
        hidden = (
            self.x_proj(differentiable_query)
            + self._position_embedding(
                token_count,
                differentiable_query.device,
                differentiable_query.dtype,
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden = block(hidden)
        predicted_field = self.out_proj(hidden)
        if self.output_mode == "field":
            return predicted_field
        potential = self.potential_head(predicted_field)
        if self.output_mode == "potential":
            return potential.mean(dim=1).squeeze(-1)
        return torch.autograd.grad(
            outputs=potential.sum(),
            inputs=differentiable_query,
            create_graph=True,
            retain_graph=True,
        )[0]


class RepresentationScatteringField(nn.Module):
    """Frozen-target three-body scattering regression in one feature space."""

    def __init__(
        self,
        representor: nn.Module,
        lambda_weight: float = 1.0,
        rho: float = 0.0,
        num_classes: int = 1000,
        tracker_config: Optional[Mapping[str, Any]] = None,
        ema_beta: float = 0.999,
        loss_weight: float = 1.0,
        feature_norm: Sequence[str] = ("mu", "std", "rms"),
    ) -> None:
        super().__init__()
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be in [0, 1]")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if not hasattr(representor, "feat_dim"):
            raise AttributeError("TBSM representor must define feat_dim")
        if not isinstance(feature_norm, (list, tuple)):
            raise TypeError("feature_norm must be a list or tuple")
        supported_norms = {"mu", "std", "rms"}
        unknown_norms = set(feature_norm) - supported_norms
        if unknown_norms:
            raise ValueError(
                f"Unknown feature_norm values: {sorted(unknown_norms)}; "
                f"expected a subset of {sorted(supported_norms)}"
            )
        if len(set(feature_norm)) != len(feature_norm):
            raise ValueError("feature_norm values must be unique")

        self.representor = representor
        self.lambda_weight = float(lambda_weight)
        self.rho = float(rho)
        self.loss_weight = float(loss_weight)
        self.feature_norm = tuple(feature_norm)
        self.beta = float(ema_beta)
        tracker_config_dict = (
            {} if tracker_config is None else dict(tracker_config)
        )
        self.tracker = (
            ScatteringTracker(
                num_classes,
                feat_dim=int(representor.feat_dim),
                **tracker_config_dict,
            )
            if self.rho > 0
            else None
        )
        self.register_buffer(
            "mu_ema",
            torch.zeros(int(representor.feat_dim)),
        )
        self.register_buffer(
            "x2_ema",
            torch.zeros(int(representor.feat_dim)),
        )
        self.register_buffer(
            "_ema_inited",
            torch.zeros(1, dtype=torch.bool),
        )
        self.representor.requires_grad_(False)
        self.representor.eval()

    @property
    def representation_map(self) -> nn.Module:
        """Return the frozen representation map."""

        return self.representor

    @property
    def intra_source_weight(self) -> float:
        """Return the paper's lambda parameter."""

        return self.lambda_weight

    @property
    def tracked_supervision_weight(self) -> float:
        """Return the paper's rho parameter."""

        return self.rho

    def train(self, mode: bool = True) -> "RepresentationScatteringField":
        """Keep the frozen representation map in evaluation mode."""

        super().train(mode)
        self.representor.eval()
        return self

    def _extract_tokens(
        self,
        images: torch.Tensor,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Extract representation tokens while optionally retaining image grads."""

        with torch.set_grad_enabled(requires_grad):
            if hasattr(self.representor, "extract_featmap"):
                feature_map = self.representor.extract_featmap(images)
            elif hasattr(self.representor, "extract"):
                try:
                    feature_map = self.representor.extract(
                        images,
                        no_grad=not requires_grad,
                    )
                except TypeError:
                    feature_map = self.representor.extract(images)
            else:
                feature_map = self.representor(images)
        if not isinstance(feature_map, torch.Tensor):
            raise TypeError("TBSM representor must return a torch.Tensor")
        return _feature_map_to_tokens(feature_map)

    def _normalize_features(
        self,
        real_source: torch.Tensor,
        projectile: torch.Tensor,
        generated_source: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Normalize feature tokens using EMA statistics from real images."""

        channels = real_source.shape[-1]
        use_mu = "mu" in self.feature_norm
        use_std = "std" in self.feature_norm
        use_rms = "rms" in self.feature_norm
        if use_mu or use_std:
            mu_new = _distributed_mean(
                real_source.mean(dim=(0, 1)).detach().float()
            )
            x2_new = None
            if use_std:
                x2_new = _distributed_mean(
                    real_source.float().square().mean(dim=(0, 1)).detach()
                )
            if not bool(self._ema_inited.item()):
                self.mu_ema.copy_(mu_new)
                if x2_new is not None:
                    self.x2_ema.copy_(x2_new)
                self._ema_inited.fill_(True)
            else:
                amount = 1.0 - self.beta
                self.mu_ema.lerp_(mu_new, amount)
                if x2_new is not None:
                    self.x2_ema.lerp_(x2_new, amount)

        mu = self.mu_ema.view(1, 1, channels) if use_mu else None
        std = None
        if use_std:
            variance = self.x2_ema - self.mu_ema.square()
            std = variance.clamp_min(1e-5).sqrt().view(1, 1, channels)

        def normalize_one(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if mu is not None:
                value = value - mu
            if std is not None:
                value = value / std
            if use_rms:
                value = F.normalize(value, p=2, dim=-1) * math.sqrt(channels)
            return value

        return (
            normalize_one(real_source),
            normalize_one(projectile),
            normalize_one(generated_source),
        )

    @torch.no_grad()
    def _frozen_target(
        self,
        real_source: torch.Tensor,
        projectile: torch.Tensor,
        generated_source: torch.Tensor,
        tracked_vector: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the detached TBSM target and lambda-weighted vector."""

        projectile_detached = projectile.detach()
        inter_offset = (real_source - projectile_detached).flatten(1)
        intra_offset = (generated_source - projectile_detached).flatten(1)
        radius_scale = math.sqrt(projectile.shape[1] * projectile.shape[2])
        inter_radius = torch.linalg.vector_norm(
            inter_offset,
            ord=2,
            dim=-1,
            keepdim=True,
        )
        intra_radius = torch.linalg.vector_norm(
            intra_offset,
            ord=2,
            dim=-1,
            keepdim=True,
        )
        inter_bearing = inter_offset / (inter_radius + SCATTERING_EPS)
        intra_bearing = intra_offset / (intra_radius + SCATTERING_EPS)
        lambda_vector = (
            inter_bearing - self.lambda_weight * intra_bearing
        ).view_as(projectile) * radius_scale
        mixed_vector = (
            lambda_vector
            if tracked_vector is None
            else (1.0 - self.rho) * lambda_vector
            + self.rho * tracked_vector
        )
        return (projectile + mixed_vector).detach(), lambda_vector

    def regression_losses(
        self,
        real_image: torch.Tensor,
        projectile_image: torch.Tensor,
        generated_source_image: Optional[torch.Tensor],
        condition: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute generator and tracker losses for one representation field."""

        real_source = self._extract_tokens(real_image, requires_grad=False).detach()
        projectile = self._extract_tokens(projectile_image, requires_grad=True)
        generated_source = None
        if self.lambda_weight > 0 and generated_source_image is not None:
            generated_source = self._extract_tokens(
                generated_source_image,
                requires_grad=False,
            ).detach()
        real_source, projectile, generated_source = self._normalize_features(
            real_source,
            projectile,
            generated_source,
        )
        if generated_source is None:
            generated_source = projectile.detach()

        alpha = torch.rand(
            projectile.shape[0],
            1,
            1,
            device=projectile.device,
            dtype=projectile.dtype,
        ) * (1.0 - self.lambda_weight)
        query = alpha * real_source + (1.0 - alpha) * projectile.detach()
        tracked_vector = (
            None
            if self.tracker is None
            else self.tracker(condition, query, time)
        )
        target, lambda_vector = self._frozen_target(
            real_source,
            projectile,
            generated_source,
            tracked_vector,
        )
        tracker_prediction = (
            lambda_vector if tracked_vector is None else tracked_vector
        )
        raw_projectile_loss = F.mse_loss(projectile, target)
        raw_tracker_loss = F.mse_loss(tracker_prediction, lambda_vector)
        projectile_loss = self.loss_weight * _self_normalized(raw_projectile_loss)
        tracker_loss = self.loss_weight * _self_normalized(raw_tracker_loss)
        with torch.no_grad():
            similarity = F.cosine_similarity(
                projectile.mean(dim=1),
                real_source.mean(dim=1),
                dim=-1,
            ).mean()
        return (
            projectile_loss,
            tracker_loss,
            {
                "generator": raw_projectile_loss.detach(),
                "tracker": raw_tracker_loss.detach(),
                "similarity": similarity.detach(),
            },
        )


DistributionMatcher = RepresentationScatteringField
RepresentationScattering = RepresentationScatteringField


__all__ = [
    "DistributionMatcher",
    "IdentityRepresentor",
    "RepresentationScattering",
    "RepresentationScatteringField",
    "ScatteringTracker",
    "TBSMGenerator",
]
