"""
使用 XDL Trainer 结合 third_party/APEX 方法训练 SD3.5 Medium LoRA.

默认读取同目录的 sd35m_apex_lora.yaml.需要切换配置时设置环境变量:

XDL_SD35_APEX_CONFIG=train/posttrain/sd35m_apex_lora.yaml python train/posttrain/train_sd35m_apex_xdl.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from PIL import Image, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from xdl.callbacks import Callback
from xdl.config import load_structured_dataclass_config
from xdl.post_training import SaveTrainableStateCallback
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.utils import save_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APEX_ROOT = PROJECT_ROOT / "third_party" / "APEX"
APEX_SRC = APEX_ROOT / "src"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("sd35m_apex_lora.yaml")
CONFIG_ENV = "XDL_SD35_APEX_CONFIG"
PROMPT_KEYS: Tuple[str, ...] = ("text", "prompt", "caption")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class RuntimeConfig:
    config_path: Path
    raw: Dict[str, Any]


@dataclass
class SD35APEXConfig:
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    method: Dict[str, Any] = field(default_factory=dict)
    sample: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)


class ImageTextPairDataset(Dataset[List[Dict[str, Any]]]):
    """APEX 风格的 image/txt 目录数据集."""

    def __init__(
        self,
        data_dirs: Sequence[Path],
        height: int,
        width: int,
        center_crop: bool = True,
        random_flip: bool = False,
        datasets_repeat: int = 1,
    ) -> None:
        self.height = height
        self.width = width
        self.datasets_repeat = max(1, int(datasets_repeat))
        self.image_paths: List[Path] = []
        self.text_paths: List[Path] = []

        for data_dir in data_dirs:
            if not data_dir.exists():
                print(f"[Data] 跳过不存在的数据目录: {data_dir}")
                continue
            for image_path in sorted(data_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                text_path = image_path.with_suffix(".txt")
                if not text_path.exists():
                    print(f"[Data] 缺少同名 txt,跳过: {image_path}")
                    continue
                self.image_paths.append(image_path)
                self.text_paths.append(text_path)

        if not self.image_paths:
            raise ValueError("没有找到有效的 image/txt 数据对")

        crop_transform = (
            transforms.CenterCrop((height, width))
            if center_crop
            else transforms.RandomCrop((height, width))
        )
        flip_transform = transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(min(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                crop_transform,
                flip_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self.datasets_repeat

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        data_index = index % len(self.image_paths)
        image = Image.open(self.image_paths[data_index]).convert("RGB")
        image = self.image_processor(image)

        prompts = [
            line.strip()
            for line in self.text_paths[data_index].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        text = random.choice(prompts) if prompts else ""
        return [{"text": text, "image": image, "z": torch.empty(0)}]


class ManifestImageTextDataset(Dataset[List[Dict[str, Any]]]):
    """JSONL/JSON/CSV manifest 数据集,字段支持 image + text/prompt/caption."""

    def __init__(
        self,
        manifest_path: Path,
        height: int,
        width: int,
        center_crop: bool = True,
        random_flip: bool = False,
        datasets_repeat: int = 1,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.records = load_manifest_records(self.manifest_path)
        self.datasets_repeat = max(1, int(datasets_repeat))
        if not self.records:
            raise ValueError(f"manifest 为空: {manifest_path}")

        flip_transform = transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)
        self.center_crop = center_crop
        self.random_flip = flip_transform
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.records) * self.datasets_repeat

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        record = self.records[index % len(self.records)]
        image_path = resolve_data_path(str(record["image"]), self.manifest_path.parent)
        text = pick_prompt(record)
        image = Image.open(image_path).convert("RGB")

        if self.center_crop:
            image = ImageOps.fit(
                image,
                (self.width, self.height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            tensor = transforms.functional.to_tensor(image)
        else:
            processor = transforms.Compose(
                [
                    transforms.Resize(min(self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop((self.height, self.width)),
                    self.random_flip,
                    transforms.ToTensor(),
                ]
            )
            tensor = processor(image)

        tensor = tensor * 2.0 - 1.0
        return [{"text": text, "image": tensor, "z": torch.empty(0)}]


class SD3APEXTransformer(torch.nn.Module):
    """把 diffusers SD3 transformer 包装成 APEX 期望的 forward(x_t, t, c, tt) 形式."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        num_train_timesteps: int,
        aux_time_embed: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = int(transformer.config.in_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        self.aux_time_embed = aux_time_embed

    def enable_gradient_checkpointing(self) -> None:
        target = self.transformer
        if hasattr(target, "enable_gradient_checkpointing"):
            target.enable_gradient_checkpointing()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Sequence[torch.Tensor],
        tt: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        cfg_interval: Optional[Sequence[float]] = None,
        **_: Any,
    ) -> torch.Tensor:
        if cfg_scale > 0.0:
            return self.forward_with_cfg(x_t, t, c, cfg_scale, cfg_interval, tt)
        return self.forward_no_cfg(x_t, t, c, tt)

    def forward_no_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Sequence[torch.Tensor],
        tt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, pooled_prompt_embeds = c[0], c[1]
        timestep = t.flatten().to(device=x_t.device, dtype=x_t.dtype)
        target_timestep = tt.flatten().to(device=x_t.device, dtype=x_t.dtype) if tt is not None else None

        adv_idx = timestep < 0
        if adv_idx.any():
            prompt_embeds = prompt_embeds.clone()
            pooled_prompt_embeds = pooled_prompt_embeds.clone()
            prompt_embeds[adv_idx] = -prompt_embeds[adv_idx]
            pooled_prompt_embeds[adv_idx] = -pooled_prompt_embeds[adv_idx]

        transformer_kwargs: Dict[str, Any] = {
            "hidden_states": x_t,
            "timestep": timestep * self.num_train_timesteps,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": False,
        }
        if self.aux_time_embed and target_timestep is not None:
            transformer_kwargs["target_timestep"] = target_timestep * self.num_train_timesteps

        prediction = self.transformer(**transformer_kwargs)[0]
        return prediction

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Sequence[torch.Tensor],
        cfg_scale: float,
        cfg_interval: Optional[Sequence[float]],
        tt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cfg_interval is None:
            cfg_interval = (0.0, 1.0)
        t_flat = t.flatten()
        in_interval = cfg_interval[0] <= float(t_flat[0].detach().cpu()) <= cfg_interval[1]
        if in_interval:
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward_no_cfg(combined, t, c, tt)
            cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
            guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        else:
            half = x_t[: len(x_t) // 2]
            half_t = t_flat[: len(half)]
            half_tt = tt.flatten()[: len(half)] if tt is not None else None
            half_c = [item[: len(half)] for item in c]
            guided = self.forward_no_cfg(half, half_t, half_c, half_tt)
        return torch.cat([guided, guided], dim=0)


class SD35APEXPipelineModel(torch.nn.Module):
    """包含 SD3.5 pipeline 组件,并提供 APEX 需要的模型契约."""

    def __init__(
        self,
        pipe: StableDiffusion3Pipeline,
        transformer: SD3APEXTransformer,
        device_name: str,
        imgs_dtype: torch.dtype,
        text_dtype: torch.dtype,
        max_sequence_length: int,
        vae_sample_mode: str,
    ) -> None:
        super().__init__()
        self.model = pipe
        self.transformer = transformer
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        self.device_name = device_name
        self.imgs_dtype = imgs_dtype
        self.text_dtype = text_dtype
        self.max_sequence_length = max_sequence_length
        self.vae_sample_mode = vae_sample_mode

    @property
    def device(self) -> torch.device:
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return torch.device(self.device_name)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Sequence[torch.Tensor],
        tt: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.transformer(x_t, t, c, tt=tt, **kwargs)

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Sequence[str],
        do_cfg: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        prompt_list = list(prompt)
        negative_prompt = [""] * len(prompt_list) if do_cfg else None
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt_list,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            max_sequence_length=self.max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.imgs_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.imgs_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.imgs_dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=self.device, dtype=self.imgs_dtype)
        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        pixel_values = pixels.to(device=self.device, dtype=self.vae.dtype)
        latent_dist = self.vae.encode(pixel_values).latent_dist
        latents = latent_dist.mean if self.vae_sample_mode == "mean" else latent_dist.sample()
        shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))
        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        return ((latents - shift_factor) * scaling_factor).to(dtype=self.imgs_dtype)

    def latents_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.vae.dtype)
        shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0))
        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        latents = (latents / scaling_factor) + shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        sampler: Any,
        height: int,
        width: int,
        seed: int,
        cfg_scale: float = 0.0,
        cfg_interval: Optional[Sequence[float]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        return_traj: bool = False,
    ) -> torch.Tensor:
        sampling_kwargs = sampling_kwargs or {}
        do_cfg = cfg_scale > 0.0
        prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = self.encode_prompt(
            prompts,
            do_cfg=do_cfg,
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            [
                len(prompts),
                self.transformer.in_channels,
                height // int(self.model.vae_scale_factor),
                width // int(self.model.vae_scale_factor),
            ],
            dtype=self.imgs_dtype,
            generator=generator,
        ).to(self.device)

        if do_cfg:
            if neg_prompt_embeds is None or neg_pooled_prompt_embeds is None:
                raise RuntimeError("CFG 采样缺少 negative prompt embedding")
            latents = torch.cat([noise, noise], dim=0)
            c = [
                torch.cat([prompt_embeds, neg_prompt_embeds], dim=0),
                torch.cat([pooled_prompt_embeds, neg_pooled_prompt_embeds], dim=0),
            ]
            model_kwargs = {"c": c, "cfg_scale": cfg_scale, "cfg_interval": cfg_interval or (0.0, 1.0)}
        else:
            latents = noise
            model_kwargs = {"c": [prompt_embeds, pooled_prompt_embeds]}

        sampled = sampler(latents, self.transformer, **model_kwargs, **sampling_kwargs)
        if do_cfg:
            sampled, _ = sampled.chunk(2, dim=1)

        sampled = sampled if return_traj else sampled[-1:]
        flat_latents = sampled.reshape(-1, *sampled.shape[2:])
        decoded = self.latents_to_pixels(flat_latents)
        return decoded


class SD35LoRACheckpointCallback(Callback):
    def __init__(
        self,
        output_dir: Path,
        every_n_epochs: int = 1,
        every_n_train_steps: Optional[int] = None,
    ) -> None:
        super().__init__(priority=200)
        self.output_dir = output_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.every_n_train_steps = every_n_train_steps

    def on_train_batch_end(
        self,
        trainer: Trainer,
        core_module: CoreModel,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.every_n_train_steps:
            return
        if trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0:
            self._save(core_module, self.output_dir / "checkpoints" / f"step_{trainer.global_step:07d}")

    def on_train_epoch_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            path = self.output_dir / "checkpoints" / f"epoch_{trainer.current_epoch:04d}_step_{trainer.global_step:07d}"
            self._save(core_module, path)

    def on_train_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        self._save(core_module, self.output_dir / "final")

    def _save(self, core_module: CoreModel, path: Path) -> None:
        if hasattr(core_module, "save_lora_adapter"):
            core_module.save_lora_adapter(path)


class SD35APEXCoreModel(CoreModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model_cfg = get_mapping(config, "model")
        self.train_cfg = get_mapping(config, "train")
        self.method_cfg = get_mapping(config, "method")
        self.sample_cfg = get_mapping(config, "sample")
        self.output_dir = Path(self.train_cfg.get("output_dir", "./outputs/sd35m_apex_lora"))
        self.device_name = str(self.train_cfg.get("device", "cuda"))
        self.imgs_dtype = resolve_dtype(str(self.model_cfg.get("imgs_dtype", "bf16")), self.device_name)
        self.text_dtype = resolve_dtype(str(self.model_cfg.get("text_dtype", "bf16")), self.device_name)
        self.mixed_precision_dtype = resolve_dtype(str(self.train_cfg.get("mixed_precision", "bf16")), self.device_name)
        self.enable_amp = self.device_name.startswith("cuda") and self.mixed_precision_dtype != torch.float32
        self.gradient_accumulation_steps = max(1, int(self.train_cfg.get("grad_accumulation_steps", 1)))
        self.max_grad_norm = float(self.train_cfg.get("max_grad_norm", 1.0))

        self.sd_model: Optional[SD35APEXPipelineModel] = None
        self.method: Optional[torch.nn.Module] = None
        self._built = False

    def setup(self, stage: str) -> None:
        if self._built:
            return
        if not torch.cuda.is_available() and self.device_name.startswith("cuda"):
            raise RuntimeError("APEX 的训练步骤依赖 torch.cuda RNG;当前未检测到 CUDA")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(self.config, self.output_dir / "run_config.yaml")
        set_seed(int(self.train_cfg.get("seed", 42)))

        model_id = str(self.model_cfg.get("model_path", "stabilityai/stable-diffusion-3.5-medium"))
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.imgs_dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(torch.device(self.device_name))

        base_transformer = pipe.transformer
        if bool(self.model_cfg.get("aux_time_embed", False)):
            enable_sd3_aux_time_embed(base_transformer)

        freeze_pipeline_modules(pipe)
        transformer_module: torch.nn.Module = base_transformer
        if bool(self.model_cfg.get("enable_lora", True)):
            transformer_module = self._build_lora_transformer(base_transformer)
            pipe.transformer = transformer_module

        wrapped_transformer = SD3APEXTransformer(
            transformer=transformer_module,
            num_train_timesteps=int(pipe.scheduler.config.num_train_timesteps),
            aux_time_embed=bool(self.model_cfg.get("aux_time_embed", False)),
        )
        if bool(self.model_cfg.get("enable_gradient_checkpointing", True)):
            wrapped_transformer.enable_gradient_checkpointing()

        self.sd_model = SD35APEXPipelineModel(
            pipe=pipe,
            transformer=wrapped_transformer,
            device_name=self.device_name,
            imgs_dtype=self.imgs_dtype,
            text_dtype=self.text_dtype,
            max_sequence_length=int(self.model_cfg.get("max_sequence_length", 256)),
            vae_sample_mode=str(self.model_cfg.get("vae_sample_mode", "sample")),
        )
        self.method = build_apex_method(self.method_cfg)
        self._built = True

    def _build_lora_transformer(self, base_transformer: SD3Transformer2DModel) -> torch.nn.Module:
        disable_peft_torchao_dispatch_if_needed()
        target_modules = parse_string_list(self.model_cfg.get("lora_target_modules", []))
        if not target_modules:
            raise ValueError("启用 LoRA 时必须配置 model.lora_target_modules")

        lora_config = LoraConfig(
            r=int(self.model_cfg.get("lora_rank", 16)),
            lora_alpha=int(self.model_cfg.get("lora_alpha", 16)),
            lora_dropout=float(self.model_cfg.get("lora_dropout", 0.0)),
            init_lora_weights=str(self.model_cfg.get("init_lora_weights", "gaussian")),
            target_modules=target_modules,
            bias="none",
        )
        base_transformer.requires_grad_(False)
        peft_transformer = get_peft_model(base_transformer, lora_config)
        if bool(self.model_cfg.get("add_old_adapter", False)):
            peft_transformer.add_adapter("old", lora_config)
            peft_transformer.set_adapter("default")

        for name, param in peft_transformer.named_parameters():
            if "lora" not in name or "old" in name:
                param.requires_grad = False
            if "lora" in name and "default" in name:
                param.requires_grad = True
                param.data = param.data.to(torch.float32)
        return peft_transformer

    def configure_optimizers(self) -> AdamW:
        if self.sd_model is None:
            raise RuntimeError("模型尚未 setup")
        params = [param for param in self.sd_model.transformer.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数,请检查 LoRA target_modules")
        return AdamW(
            params,
            lr=float(self.train_cfg.get("lr", 1.0e-4)),
            betas=tuple(self.train_cfg.get("betas", [0.9, 0.99])),
            weight_decay=float(self.train_cfg.get("weight_decay", 0.0)),
            foreach=bool(self.train_cfg.get("foreach", True)),
        )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._compute_apex_loss(batch)
        optimizer = self.optimizers[0]
        should_step = self.manual_optimization_step(
            loss,
            optimizer=optimizer,
            model=self.sd_model.transformer if self.sd_model is not None else self,
            max_grad_norm=self.max_grad_norm if self.max_grad_norm > 0 else None,
        )

        self.log("train_loss", loss.detach())
        self.log("lr", self.get_lr())
        if should_step:
            self.log("optimizer_step", self.optimizer_step)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._compute_apex_loss(batch)
        self.log("val_loss", loss.detach())

    def _compute_apex_loss(self, batch: Any) -> torch.Tensor:
        if self.sd_model is None or self.method is None:
            raise RuntimeError("模型或 APEX method 尚未初始化")
        if not self.device.type == "cuda":
            raise RuntimeError("当前 APEX 实现训练时需要 CUDA")

        data = normalize_apex_batch(batch)
        text = list(data["text"])
        image = data["image"].to(device=self.device, dtype=self.imgs_dtype)
        z = data.get("z")
        v = z.to(device=self.device, dtype=self.imgs_dtype) if torch.is_tensor(z) and z.numel() > 0 else None

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = self.sd_model.encode_prompt(
                text,
                do_cfg=True,
            )
            if neg_prompt_embeds is None or neg_pooled_prompt_embeds is None:
                raise RuntimeError("训练需要 unconditional prompt embedding")
            latents = self.sd_model.pixels_to_latents(image).to(torch.float32)
            prompt_embeds = prompt_embeds.to(torch.float32)
            pooled_prompt_embeds = pooled_prompt_embeds.to(torch.float32)
            neg_prompt_embeds = neg_prompt_embeds.to(torch.float32)
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(torch.float32)

        with torch.autocast(
            device_type="cuda",
            dtype=self.mixed_precision_dtype,
            enabled=self.enable_amp,
            cache_enabled=False,
        ):
            loss = self.method.training_step(
                self.sd_model,
                latents,
                c=[prompt_embeds, pooled_prompt_embeds],
                e=[neg_prompt_embeds, neg_pooled_prompt_embeds],
                step=self.micro_step - 1,
                v=v,
            )
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss), device=self.device)
        return loss

    def inference(self, data: Any) -> None:
        if not self.sample_cfg.get("enabled", False):
            return
        if self.sd_model is None or self.method is None or not self.is_main_process():
            return
        prompts = list(data) if isinstance(data, Iterable) and not isinstance(data, str) else [str(data)]
        if not prompts:
            return

        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sampling_kwargs = {
            "sampling_steps": int(self.sample_cfg.get("sampling_steps", 4)),
            "stochast_ratio": self.sample_cfg.get("stochast_ratio", 0.0),
            "extrapol_ratio": float(self.sample_cfg.get("extrapol_ratio", 0.0)),
            "sampling_order": int(self.sample_cfg.get("sampling_order", 1)),
            "time_dist_ctrl": list(self.sample_cfg.get("time_dist_ctrl", [1.0, 1.0, 1.0])),
            "rfba_gap_steps": list(self.sample_cfg.get("rfba_gap_steps", [0.001, 0.0])),
            "sampling_style": str(self.sample_cfg.get("sampling_style", "mul")),
        }
        images = self.sd_model.sample(
            prompts=prompts,
            sampler=self.method.sampling_loop,
            height=int(get_mapping(self.config, "data").get("height", 512)),
            width=int(get_mapping(self.config, "data").get("width", 512)),
            seed=int(self.train_cfg.get("seed", 42)) + self.current_epoch,
            cfg_scale=float(self.sample_cfg.get("cfg_scale", 0.0)),
            cfg_interval=self.sample_cfg.get("cfg_interval", [0.0, 1.0]),
            sampling_kwargs=sampling_kwargs,
            return_traj=bool(self.sample_cfg.get("return_traj", False)),
        )
        save_path = sample_dir / f"epoch_{self.current_epoch:04d}_step_{self.total_train_steps:07d}.png"
        save_image((images.clamp(-1, 1) + 1.0) / 2.0, save_path, nrow=max(1, len(prompts)))

    def on_train_start(self) -> None:
        if self.sd_model is None:
            return
        total_params = sum(param.numel() for param in self.sd_model.transformer.parameters())
        trainable_params = sum(param.numel() for param in self.sd_model.transformer.parameters() if param.requires_grad)
        print(f"[SD3.5 APEX] transformer 参数: total={total_params / 1e6:.2f}M, trainable={trainable_params / 1e6:.2f}M")

    def save_lora_adapter(self, save_dir: Path) -> None:
        if self.sd_model is None:
            return
        if not self.is_main_process():
            return
        inner_transformer = self.sd_model.transformer.transformer
        save_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(inner_transformer, "peft_config"):
            lora_state_dict = get_peft_model_state_dict(inner_transformer, adapter_name="default")
            adapter_metadata = inner_transformer.peft_config["default"].to_dict()
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=str(save_dir),
                transformer_lora_layers=lora_state_dict,
                transformer_lora_adapter_metadata=adapter_metadata,
                is_main_process=self.is_main_process(),
                safe_serialization=True,
            )
            inner_transformer.save_pretrained(
                str(save_dir / "peft"),
                selected_adapters=["default"],
                is_main_process=self.is_main_process(),
                safe_serialization=True,
            )
        else:
            torch.save(inner_transformer.state_dict(), save_dir / "transformer.pt")


def enable_sd3_aux_time_embed(transformer: SD3Transformer2DModel) -> None:
    if hasattr(transformer, "time_text_embed_2"):
        return
    transformer.time_text_embed_2 = deepcopy(transformer.time_text_embed)
    transformer.forward = MethodType(sd3_forward_with_aux_time, transformer)


def sd3_forward_with_aux_time(
    self: SD3Transformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    pooled_projections: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
    target_timestep: Optional[torch.Tensor] = None,
    block_controlnet_hidden_states: Optional[List[torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    skip_layers: Optional[List[int]] = None,
) -> Any:
    height, width = hidden_states.shape[-2:]
    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    if target_timestep is not None:
        target_timestep = target_timestep.to(device=timestep.device, dtype=timestep.dtype)
        temb_2 = self.time_text_embed_2(target_timestep, pooled_projections)
        delta_t = (timestep - target_timestep).to(dtype=temb.dtype).reshape(-1, 1)
        temb = temb + temb_2 * delta_t
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
        joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

    for index_block, block in enumerate(self.transformer_blocks):
        is_skip = skip_layers is not None and index_block in skip_layers
        if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                joint_attention_kwargs,
            )
        elif not is_skip:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size
    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def load_apex_class() -> Type[torch.nn.Module]:
    if str(APEX_SRC) not in sys.path:
        sys.path.insert(0, str(APEX_SRC))
    apex_path = APEX_SRC / "methodes" / "apex" / "apex.py"
    spec = importlib.util.spec_from_file_location("xdl_third_party_apex", apex_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 APEX method: {apex_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.APEX


def build_apex_method(method_cfg: Dict[str, Any]) -> torch.nn.Module:
    method_args = dict(method_cfg)
    method_type = str(method_args.pop("method_type", "APEX"))
    if method_type != "APEX":
        print(f"[APEX] 本地 third_party/APEX 只包含 APEX 类,已忽略 method_type={method_type}")
    apex_class = load_apex_class()
    return apex_class(**method_args)


def disable_peft_torchao_dispatch_if_needed() -> None:
    """当前环境的 torchao 版本过低时,让 PEFT 走普通 Linear LoRA 分支."""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao
    except Exception:
        return

    def _torchao_unavailable() -> bool:
        return False

    if hasattr(peft_import_utils.is_torchao_available, "cache_clear"):
        peft_import_utils.is_torchao_available.cache_clear()
    peft_import_utils.is_torchao_available = _torchao_unavailable
    peft_lora_torchao.is_torchao_available = _torchao_unavailable


def freeze_pipeline_modules(pipe: StableDiffusion3Pipeline) -> None:
    for module in [pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer]:
        if module is not None:
            module.requires_grad_(False)
            module.eval()


def normalize_apex_batch(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], dict):
        return batch[0]
    if isinstance(batch, dict):
        return batch
    raise TypeError(f"不支持的 batch 格式: {type(batch)!r}")


def load_manifest_records(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest 必须是对象数组")
        return payload
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
            return list(csv.DictReader(file_obj))
    raise ValueError(f"不支持的 manifest 格式: {manifest_path.suffix}")


def pick_prompt(record: Dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"记录缺少 prompt 字段,需要其一: {PROMPT_KEYS}")


def resolve_data_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_dataset(config: Dict[str, Any]) -> Dataset[Any]:
    data_cfg = get_mapping(config, "data")
    data_format = str(data_cfg.get("format", "image_txt"))
    height = int(data_cfg.get("height", 512))
    width = int(data_cfg.get("width", 512))
    repeat = int(data_cfg.get("datasets_repeat", 1))
    center_crop = bool(data_cfg.get("center_crop", True))
    random_flip = bool(data_cfg.get("random_flip", False))

    if data_format == "image_txt":
        data_dirs = [resolve_data_path(str(path), PROJECT_ROOT) for path in data_cfg.get("train_dirs", [])]
        return ImageTextPairDataset(data_dirs, height, width, center_crop, random_flip, repeat)
    if data_format == "manifest":
        manifest_path = resolve_data_path(str(data_cfg["manifest_path"]), PROJECT_ROOT)
        return ManifestImageTextDataset(manifest_path, height, width, center_crop, random_flip, repeat)
    if data_format == "apex_parquet":
        if str(APEX_SRC) not in sys.path:
            sys.path.insert(0, str(APEX_SRC))
        from data.parquet_datasets import Text2ImageParquetDataset

        return Text2ImageParquetDataset(
            data_dirs=data_cfg.get("train_dirs", []),
            height=height,
            width=width,
            center_crop=center_crop,
            random_flip=random_flip,
            datasets_repeat=repeat,
            cache_dir=data_cfg.get("cache_dir"),
        )
    raise ValueError(f"未知 data.format: {data_format}")


def build_dataloader(config: Dict[str, Any]) -> DataLoader[Any]:
    train_cfg = get_mapping(config, "train")
    batch_size = int(train_cfg.get("micro_batch_size", 4))
    if batch_size < 4:
        raise ValueError("third_party/APEX 当前 prepare_inputs 要求 micro_batch_size >= 4")
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train_cfg.get("drop_last", True)),
    )


def resolve_dtype(dtype_name: str, device_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"fp32", "float32", "32", "no"}:
        return torch.float32
    if name in {"fp16", "float16", "16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name == "auto":
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    raise ValueError(f"不支持的 dtype: {dtype_name}")


def parse_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"无法解析列表: {value!r}")


def get_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"配置项 {key} 必须是 dict")
    return value


def load_runtime_config() -> RuntimeConfig:
    raw_path = os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))
    config_path = Path(raw_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    config = load_structured_dataclass_config(SD35APEXConfig, config_path)
    return RuntimeConfig(config_path=config_path, raw=asdict(config))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    runtime = load_runtime_config()
    config = runtime.raw
    train_cfg = get_mapping(config, "train")

    # 无人工数据时建议走 teacher self-distillation:先用原始 SD3.5M 根据
    # prompt 生成伪训练图,再用这些 image/txt 数据训练 APEX LoRA.
    # 不需要先准备 10k 条 prompt;更实际的起点是 2k-3k 条高质量 prompt,
    # 每条用 3-5 个 seed 生成图片,得到约 10k 张训练图.建议分阶段:
    # 1) 50-100 prompts x 1-2 seeds 做 smoke test;
    # 2) 500 prompts x 2-4 seeds 做小实验;
    # 3) 2k prompts x 5 seeds 做第一版正式训练;
    # 4) 如果方向正确但泛化不稳,再扩到 5k prompts x 5-6 seeds.
    # 不建议用纯 random latent 做"真无数据"训练,它只能验证代码链路,
    # 基本不能学到可用的图像分布.

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("allow_tf32", True))

    train_loader = build_dataloader(config)
    output_dir = Path(train_cfg.get("output_dir", "./outputs/sd35m_apex_lora"))
    checkpoint_callback = SaveTrainableStateCallback(
        dirpath=output_dir / "adapters",
        method_name="save_lora_adapter",
        every_n_epochs=int(train_cfg.get("save_every_n_epochs", 1)),
        every_n_train_steps=train_cfg.get("save_every_n_train_steps"),
    )

    trainer = Trainer(
        max_epochs=int(train_cfg.get("num_train_epochs", 1)),
        device=str(train_cfg.get("device", "cuda")),
        callbacks=[checkpoint_callback],
    )
    trainer.setup_logger(
        experiment_name="sd35m_apex_lora",
        log_dir=str(output_dir / "logs"),
        checkpoint_dir=str(output_dir / "trainer_checkpoints"),
        enable_tensorboard=bool(train_cfg.get("enable_tensorboard", False)),
        enable_checkpoint=False,
        enable_console=bool(train_cfg.get("enable_console", False)),
        enable_tqdm=bool(train_cfg.get("enable_tqdm", True)),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 1)),
    )

    model = SD35APEXCoreModel(config)
    sample_cfg = get_mapping(config, "sample")
    inference_data = sample_cfg.get("prompts") if sample_cfg.get("enabled", False) else None
    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=None,
        check_val_every_n_epoch=int(sample_cfg.get("every_n_epochs", 1)),
        inference_data=inference_data,
    )


if __name__ == "__main__":
    main()
