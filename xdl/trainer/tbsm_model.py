"""XDL CoreModel adapter for TBSM one-step training."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn

from xdl.model.generate.tbsm import RepresentationScatteringField

from .core_model import CoreModel


class TBSMCoreModel(CoreModel):
    """Run TBSM scattering training through the standard XDL Trainer lifecycle.

    The model owns the online generator, representation fields, tracker
    optimizer, and an EMA generator used by :meth:`sample`.  Optimizer
    construction is deferred to ``configure_optimizers`` so Trainer can first
    move and prepare modules.

    A training batch is either ``(images, labels)`` or a mapping containing
    ``images``/``image``/``x`` and ``labels``/``label``/``y``.
    """

    def __init__(
        self,
        generator: nn.Module,
        representation_fields: Optional[Sequence[RepresentationScatteringField]] = None,
        gen_lr: float = 1e-5,
        tracker_lr: float = 1e-3,
        ema_decay: float = 0.999,
        t_sampling: Optional[Sequence[float]] = None,
        noise_scale: float = 1.0,
        gen_betas: tuple[float, float] = (0.9, 0.999),
        tracker_betas: tuple[float, float] = (0.9, 0.999),
        ema_generator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("ema_decay must be in [0, 1]")
        if not isinstance(t_sampling, (list, tuple, type(None))):
            raise TypeError("t_sampling must be a list, tuple, or None")
        if t_sampling is not None and any(
            not 0.0 <= float(value) <= 1.0 for value in t_sampling
        ):
            raise ValueError("t_sampling values must be in [0, 1]")
        if len(gen_betas) != 2 or len(tracker_betas) != 2:
            raise ValueError("optimizer betas must contain exactly two values")

        self.generator = generator
        self.representation_fields = nn.ModuleList(
            list(representation_fields or [])
        )
        self._ema_generator = (
            copy.deepcopy(generator)
            if ema_generator is None
            else ema_generator
        )
        self._ema_generator.requires_grad_(False)
        self._ema_generator.eval()

        self.gen_lr = float(gen_lr)
        self.tracker_lr = float(tracker_lr)
        self.ema_decay = float(ema_decay)
        self.t_sampling = tuple(float(value) for value in (t_sampling or ()))
        self.noise_scale = float(noise_scale)
        self.gen_betas = tuple(float(value) for value in gen_betas)
        self.tracker_betas = tuple(float(value) for value in tracker_betas)

    @property
    def ema_generator(self) -> nn.Module:
        """Return the frozen EMA generator used for evaluation."""

        return self._ema_generator

    def configure_unwrapped_modules(self) -> list[str]:
        """Keep the evaluation-only EMA generator outside FSDP wrapping."""

        return ["_ema_generator"]

    @staticmethod
    def _unwrap_parallel(module: nn.Module) -> nn.Module:
        """Return the original module behind common distributed wrappers."""

        wrapped = getattr(module, "module", None)
        return wrapped if isinstance(wrapped, nn.Module) else module

    def _fields(self) -> list[RepresentationScatteringField]:
        """Return representation fields whether or not Accelerate wrapped them."""

        fields = self._unwrap_parallel(self.representation_fields)
        if not isinstance(fields, nn.ModuleList):
            raise TypeError("TBSM representation_fields must be an nn.ModuleList")
        return list(fields)

    def _generator_parameters(self) -> list[nn.Parameter]:
        """Return trainable generator parameters, preferring ``backbone``."""

        generator = self._unwrap_parallel(self.generator)
        backbone = getattr(generator, "backbone", None)
        module = backbone if isinstance(backbone, nn.Module) else generator
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("TBSM generator has no trainable parameters")
        return parameters

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Build the generator and optional tracker optimizers."""

        generator_optimizer = torch.optim.AdamW(
            self._generator_parameters(),
            lr=self.gen_lr,
            betas=self.gen_betas,
        )
        tracker_parameters = [
            parameter
            for field in self._fields()
            if field.tracker is not None
            for parameter in field.tracker.parameters()
            if parameter.requires_grad
        ]
        optimizers = [generator_optimizer]
        if tracker_parameters:
            optimizers.append(
                torch.optim.AdamW(
                    tracker_parameters,
                    lr=self.tracker_lr,
                    betas=self.tracker_betas,
                    weight_decay=0.0,
                )
            )
        return optimizers

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract image tensors and class labels from a supported batch."""

        if isinstance(batch, Mapping):
            image_key = next(
                (
                    key
                    for key in ("images", "image", "x")
                    if key in batch
                ),
                None,
            )
            label_key = next(
                (
                    key
                    for key in ("labels", "label", "y")
                    if key in batch
                ),
                None,
            )
            if image_key is None or label_key is None:
                raise ValueError(
                    "TBSM mapping batches require image(s)/x and label(s)/y keys"
                )
            images = batch[image_key]
            labels = batch[label_key]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise TypeError(
                "TBSM batches must be (images, labels) or a supported mapping"
            )
        if not torch.is_tensor(images) or not torch.is_tensor(labels):
            raise TypeError("TBSM images and labels must be torch.Tensor objects")
        return images, labels

    def _sample_time(self, images: torch.Tensor) -> torch.Tensor:
        """Sample continuous or configured discrete projectile times."""

        batch_size = images.shape[0]
        if not self.t_sampling:
            return torch.rand(
                batch_size,
                device=images.device,
                dtype=images.dtype,
            )
        values = images.new_tensor(self.t_sampling)
        indices = torch.randint(
            values.numel(),
            (batch_size,),
            device=images.device,
        )
        return values[indices]

    def _to_image(self, value: torch.Tensor) -> torch.Tensor:
        """Use the generator's image conversion hook when available."""

        converter = getattr(self._unwrap_parallel(self.generator), "to_image", None)
        return converter(value) if callable(converter) else value * 0.5 + 0.5

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        time: torch.Tensor,
        *,
        evaluation: bool = False,
    ) -> torch.Tensor:
        """Run either the online or EMA generator with common signatures."""

        model = self._ema_generator if evaluation else self.generator
        try:
            return model(inputs, labels, t=time)
        except TypeError:
            return model(inputs, labels, time)

    @torch.no_grad()
    def _generated_source_image(
        self,
        real_images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Generate the independent source required by intra-source scattering."""

        if not any(
            field.lambda_weight > 0
            for field in self._fields()
        ):
            return None
        generator = self._unwrap_parallel(self.generator)
        was_training = generator.training
        generator.eval()
        time = torch.ones(
            real_images.shape[0],
            device=real_images.device,
            dtype=real_images.dtype,
        )
        noise = torch.randn_like(real_images) * self.noise_scale
        unwrapped = getattr(generator, "unwrapped_forward", None)
        if callable(unwrapped):
            generated = unwrapped(noise, labels, time)
        else:
            generated = self._forward_generator(noise, labels, time)
        if was_training:
            generator.train()
        return self._to_image(generated)

    def _scattering_losses(
        self,
        real_images: torch.Tensor,
        projectile: torch.Tensor,
        generated_source_image: Optional[torch.Tensor],
        labels: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
        """Aggregate all configured representation-field objectives."""

        with torch.no_grad():
            real_image = self._to_image(real_images)
        projectile_image = self._to_image(projectile)
        generator_loss = real_images.new_zeros(())
        tracker_loss = real_images.new_zeros(())
        logs: dict[str, torch.Tensor | float] = {}
        fields = self._fields()
        multiple_fields = len(fields) > 1
        for index, field in enumerate(fields):
            generator_term, tracker_term, field_logs = field.regression_losses(
                real_image,
                projectile_image,
                generated_source_image,
                labels,
                time,
            )
            generator_loss = generator_loss + generator_term
            tracker_loss = tracker_loss + tracker_term
            suffix = f"_{index}" if multiple_fields else ""
            logs.update(
                {name + suffix: value for name, value in field_logs.items()}
            )
        if not fields:
            raise RuntimeError(
                "TBSMCoreModel requires at least one representation field"
            )
        return generator_loss, tracker_loss, logs

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update EMA parameters after a completed optimizer step."""

        online_generator = self._unwrap_parallel(self.generator)
        for ema_parameter, parameter in zip(
            self._ema_generator.parameters(),
            online_generator.parameters(),
            strict=True,
        ):
            if parameter.requires_grad:
                ema_parameter.mul_(self.ema_decay).add_(
                    parameter.detach(),
                    alpha=1.0 - self.ema_decay,
                )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Perform one manually optimized TBSM micro-step."""

        del batch_idx
        real_images, labels = self._unpack_batch(batch)
        self.generator.train()
        noise = torch.randn_like(real_images) * self.noise_scale
        time = self._sample_time(real_images)
        time_image = time.view(real_images.shape[0], *([1] * (real_images.ndim - 1)))
        projectile = self._forward_generator(
            real_images * (1.0 - time_image) + noise * time_image,
            labels,
            time,
        )
        generated_source_image = self._generated_source_image(real_images, labels)
        generator_loss, tracker_loss, logs = self._scattering_losses(
            real_images,
            projectile,
            generated_source_image,
            labels,
            time,
        )

        optimizers = self.optimizers
        generator_optimizer = optimizers[0]
        tracker_optimizer = optimizers[1] if len(optimizers) > 1 else None
        if self.is_accumulation_start:
            generator_optimizer.zero_grad(set_to_none=True)
            if tracker_optimizer is not None:
                tracker_optimizer.zero_grad(set_to_none=True)

        scale = 1.0 / self.accumulation_steps
        self.manual_backward(generator_loss * scale)
        if tracker_optimizer is not None and tracker_loss.requires_grad:
            self.manual_backward(tracker_loss * scale)

        if self.should_optimizer_step:
            generator_optimizer.step()
            if tracker_optimizer is not None:
                tracker_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            if tracker_optimizer is not None:
                tracker_optimizer.zero_grad(set_to_none=True)
            self._update_ema()

        self.log("loss", generator_loss.detach(), prefix="train")
        self.log("tracker_loss", tracker_loss.detach(), prefix="train")
        self.log_metrics(logs, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate the scattering objective without updating parameters."""

        del batch_idx
        real_images, labels = self._unpack_batch(batch)
        time = self._sample_time(real_images)
        time_image = time.view(real_images.shape[0], *([1] * (real_images.ndim - 1)))
        noise = torch.randn_like(real_images) * self.noise_scale
        with torch.enable_grad():
            projectile = self._forward_generator(
                real_images * (1.0 - time_image) + noise * time_image,
                labels,
                time,
            )
            generated_source_image = self._generated_source_image(
                real_images,
                labels,
            )
            generator_loss, tracker_loss, logs = self._scattering_losses(
                real_images,
                projectile,
                generated_source_image,
                labels,
                time,
            )
        self.log("loss", generator_loss.detach(), prefix="val")
        self.log("tracker_loss", tracker_loss.detach(), prefix="val")
        self.log_metrics(logs, prefix="val")

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the online generator."""

        return self._forward_generator(
            inputs,
            labels,
            time
            if time is not None
            else torch.zeros(
                inputs.shape[0],
                device=inputs.device,
                dtype=inputs.dtype,
            ),
        )

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one-step samples with the EMA generator."""

        self._ema_generator.eval()
        device = next(self._ema_generator.parameters()).device
        noise = noise.to(device) * self.noise_scale
        labels = labels.to(device)
        time = torch.ones(
            noise.shape[0],
            device=device,
            dtype=noise.dtype,
        )
        generated = self._forward_generator(
            noise,
            labels,
            time,
            evaluation=True,
        )
        converter = getattr(self._ema_generator, "to_image", None)
        images = converter(generated) if callable(converter) else generated * 0.5 + 0.5
        return images.float().clamp(0, 1)

    @torch.no_grad()
    def inference(self, data: Any) -> Optional[torch.Tensor]:
        """Generate a preview from ``{"noise": ..., "labels": ...}`` data."""

        if isinstance(data, Mapping):
            noise = data.get("noise")
            labels = data.get("labels", data.get("y"))
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            noise, labels = data
        else:
            raise TypeError(
                "TBSM inference data must be a mapping or (noise, labels) pair"
            )
        if not torch.is_tensor(noise) or not torch.is_tensor(labels):
            raise TypeError("TBSM inference data requires tensor noise and labels")
        return self.sample(noise, labels)

    def reset_ema(self) -> None:
        """Reset EMA weights to the current online generator."""

        online_generator = self._unwrap_parallel(self.generator)
        self._ema_generator.load_state_dict(online_generator.state_dict())
        self._ema_generator.requires_grad_(False)
        self._ema_generator.eval()


TBSMModel = TBSMCoreModel


__all__ = ["TBSMCoreModel", "TBSMModel"]
