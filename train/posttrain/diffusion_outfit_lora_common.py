"""共用的图像生成/编辑模型换装 LoRA 微调骨架.

本文件服务 train/posttrain/ 下的新模型薄入口. 训练口径参考
flux2_klein_9b_lora_finetune.py: 固定一个任务提示词, 冻结 VAE/text
encoder, 只给主 transformer 挂 LoRA, 用 flow matching velocity 目标训练.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, List, Optional, Sequence, Tuple, Type

import torch
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    set_seed,
)
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from PIL import Image, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from xdl.callbacks import Callback
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer


DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "to_qkv_mlp_proj",
    "proj_in",
    "proj_out",
)
SOURCE_IMAGE_KEYS: Tuple[str, ...] = (
    "source_image",
    "input_image",
    "reference_image",
    "condition_image",
)


@dataclass
class OutfitLoRAConfig:
    experiment_name: str
    model_id: str
    pipeline_class: str
    train_manifest: Path
    val_manifest: Optional[Path]
    output_dir: Path
    training_prompt: str
    image_height: int
    image_width: int
    source_image_height: int
    source_image_width: int
    batch_size: int
    num_workers: int
    dataset_repeat: int
    num_epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    device: str
    model_dtype: str
    max_sequence_length: int
    text_encoder_out_layers: Optional[Tuple[int, ...]]
    train_component: str
    target_modules: Tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    weighting_scheme: str
    logit_mean: float
    logit_std: float
    mode_scale: float
    save_every_n_epochs: int
    sample_every_n_epochs: int
    sample_steps: int
    sample_guidance: float
    sample_image_guidance: float
    sample_prompts: Tuple[str, ...]
    sample_input_image: Optional[Path]
    seed: int
    enable_gradient_checkpointing: bool
    from_pretrained_kwargs: dict[str, Any]


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _coerce_optional_layers(raw_value: Any) -> Optional[Tuple[int, ...]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        layers = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    else:
        layers = tuple(int(part) for part in raw_value)
    if not layers:
        return None
    return layers


def _coerce_string_tuple(raw_value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(raw_value, str):
        values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    else:
        values = tuple(str(item).strip() for item in raw_value if str(item).strip())
    if not values:
        raise ValueError(f"{field_name} 不能为空")
    return values


def load_outfit_config(
    default_config: dict[str, Any],
    config_env_var: str,
) -> OutfitLoRAConfig:
    config_data: dict[str, Any] = dict(default_config)
    config_path = os.environ.get(config_env_var)
    if config_path:
        file_cfg = OmegaConf.load(config_path)
        if not isinstance(file_cfg, DictConfig):
            raise TypeError("配置文件顶层必须是 mapping")
        merged = OmegaConf.merge(OmegaConf.create(config_data), file_cfg)
        container = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("解析后的配置必须是 dict")
        config_data = container

    image_height = int(config_data["image_height"])
    image_width = int(config_data["image_width"])
    if image_height % 16 != 0 or image_width % 16 != 0:
        raise ValueError("image_height 和 image_width 必须都能被 16 整除")

    source_image_height = int(config_data.get("source_image_height") or image_height)
    source_image_width = int(config_data.get("source_image_width") or image_width)
    if source_image_height % 16 != 0 or source_image_width % 16 != 0:
        raise ValueError("source_image_height 和 source_image_width 必须都能被 16 整除")

    training_prompt = str(config_data["training_prompt"]).strip()
    if not training_prompt:
        raise ValueError("training_prompt 不能为空")

    sample_prompts_raw = config_data.get("sample_prompts", [])
    if isinstance(sample_prompts_raw, str):
        sample_prompts = (sample_prompts_raw,)
    else:
        sample_prompts = tuple(str(item) for item in sample_prompts_raw)

    val_manifest_raw = config_data.get("val_manifest")
    sample_input_raw = config_data.get("sample_input_image")
    return OutfitLoRAConfig(
        experiment_name=str(config_data["experiment_name"]),
        model_id=str(config_data["model_id"]),
        pipeline_class=str(config_data["pipeline_class"]),
        train_manifest=Path(str(config_data["train_manifest"])),
        val_manifest=Path(str(val_manifest_raw)) if val_manifest_raw is not None else None,
        output_dir=Path(str(config_data["output_dir"])),
        training_prompt=training_prompt,
        image_height=image_height,
        image_width=image_width,
        source_image_height=source_image_height,
        source_image_width=source_image_width,
        batch_size=int(config_data["batch_size"]),
        num_workers=int(config_data["num_workers"]),
        dataset_repeat=int(config_data["dataset_repeat"]),
        num_epochs=int(config_data["num_epochs"]),
        gradient_accumulation_steps=int(config_data["gradient_accumulation_steps"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["weight_decay"]),
        max_grad_norm=float(config_data["max_grad_norm"]),
        device=str(config_data["device"]),
        model_dtype=str(config_data["model_dtype"]),
        max_sequence_length=int(config_data["max_sequence_length"]),
        text_encoder_out_layers=_coerce_optional_layers(
            config_data.get("text_encoder_out_layers")
        ),
        train_component=str(config_data.get("train_component", "transformer")),
        target_modules=_coerce_string_tuple(
            config_data.get("target_modules", DEFAULT_TARGET_MODULES),
            "target_modules",
        ),
        lora_rank=int(config_data["lora_rank"]),
        lora_alpha=int(config_data["lora_alpha"]),
        lora_dropout=float(config_data["lora_dropout"]),
        weighting_scheme=str(config_data["weighting_scheme"]),
        logit_mean=float(config_data["logit_mean"]),
        logit_std=float(config_data["logit_std"]),
        mode_scale=float(config_data["mode_scale"]),
        save_every_n_epochs=max(1, int(config_data["save_every_n_epochs"])),
        sample_every_n_epochs=max(1, int(config_data["sample_every_n_epochs"])),
        sample_steps=int(config_data["sample_steps"]),
        sample_guidance=float(config_data["sample_guidance"]),
        sample_image_guidance=float(config_data.get("sample_image_guidance", 1.6)),
        sample_prompts=sample_prompts,
        sample_input_image=Path(str(sample_input_raw)) if sample_input_raw is not None else None,
        seed=int(config_data["seed"]),
        enable_gradient_checkpointing=bool(config_data["enable_gradient_checkpointing"]),
        from_pretrained_kwargs=dict(config_data.get("from_pretrained_kwargs", {})),
    )


def config_to_dict(config: OutfitLoRAConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, tuple):
            config_dict[key] = list(value)
    return config_dict


def save_run_config(config: OutfitLoRAConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(config_to_dict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_object(qualified_name: str) -> Any:
    module_name, _, object_name = qualified_name.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(f"必须使用完整对象路径: {qualified_name}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"找不到对象: {qualified_name}") from exc


def load_manifest_records(manifest_path: Path) -> List[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest 须为对象数组")
        return payload

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
            return list(csv.DictReader(file_obj))

    raise ValueError(f"不支持的 manifest 格式: {manifest_path.suffix}")


def resolve_data_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def preprocess_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    pixel_values = TF.to_tensor(image)
    return pixel_values * 2.0 - 1.0


class OutfitImageDataset(Dataset[dict[str, Optional[torch.Tensor]]]):
    def __init__(
        self,
        manifest_path: Path,
        image_height: int,
        image_width: int,
        source_image_height: int,
        source_image_width: int,
        dataset_repeat: int,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.image_height = image_height
        self.image_width = image_width
        self.source_image_height = source_image_height
        self.source_image_width = source_image_width
        self.dataset_repeat = dataset_repeat
        self.records = load_manifest_records(self.manifest_path)

        if not self.records:
            raise ValueError(f"manifest 为空: {manifest_path}")
        if self.dataset_repeat < 1:
            raise ValueError("dataset_repeat 必须至少为 1")

    def __len__(self) -> int:
        return len(self.records) * self.dataset_repeat

    def __getitem__(self, index: int) -> dict[str, Optional[torch.Tensor]]:
        record = self.records[index % len(self.records)]
        image_path = resolve_data_path(str(record["image"]), self.manifest_path)
        sample: dict[str, Optional[torch.Tensor]] = {
            "pixel_values": preprocess_image(
                image_path,
                self.image_height,
                self.image_width,
            ),
            "source_pixel_values": None,
        }

        for key in SOURCE_IMAGE_KEYS:
            raw_source_path = record.get(key)
            if raw_source_path:
                source_path = resolve_data_path(str(raw_source_path), self.manifest_path)
                sample["source_pixel_values"] = preprocess_image(
                    source_path,
                    self.source_image_height,
                    self.source_image_width,
                )
                break

        return sample


def outfit_collate_fn(
    batch: Sequence[dict[str, Optional[torch.Tensor]]],
) -> dict[str, Optional[torch.Tensor]]:
    pixel_values = torch.stack(
        [item["pixel_values"] for item in batch if item["pixel_values"] is not None]
    )
    source_values = [item["source_pixel_values"] for item in batch]
    if all(value is not None for value in source_values):
        source_pixel_values = torch.stack(
            [value for value in source_values if value is not None]
        )
    else:
        source_pixel_values = None
    return {
        "pixel_values": pixel_values,
        "source_pixel_values": source_pixel_values,
    }


class LoRACheckpointCallback(Callback):
    def __init__(self, output_dir: Path, every_n_epochs: int = 1) -> None:
        super().__init__(priority=200)
        self.output_dir = output_dir
        self.every_n_epochs = max(1, every_n_epochs)

    def on_train_epoch_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if hasattr(core_module, "save_lora_adapter"):
            checkpoint_dir = (
                self.output_dir
                / "checkpoints"
                / f"epoch_{trainer.current_epoch:04d}_step_{trainer.global_step:06d}"
            )
            core_module.save_lora_adapter(checkpoint_dir)

    def on_train_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        if hasattr(core_module, "save_lora_adapter"):
            core_module.save_lora_adapter(self.output_dir / "final")


class BaseOutfitLoRAModel(CoreModel):
    def __init__(self, config: OutfitLoRAConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dir = config.output_dir
        self.model_dtype = resolve_dtype(config.model_dtype, config.device)
        self.use_cuda_autocast = config.device.startswith("cuda") and torch.cuda.is_available()

        self.pipe: Any = None
        self.train_module: Optional[torch.nn.Module] = None
        self.scheduler: Any = None
        self._is_built = False

    def setup(self, stage: str) -> None:
        if self._is_built:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pipeline_cls = import_object(self.config.pipeline_class)
        from_pretrained_kwargs = dict(self.config.from_pretrained_kwargs)
        from_pretrained_kwargs.setdefault("torch_dtype", self.model_dtype)
        try:
            self.pipe = pipeline_cls.from_pretrained(
                self.config.model_id,
                **from_pretrained_kwargs,
            )
        except ImportError as exc:
            raise ImportError(
                f"加载 {self.config.model_id} 失败. 请先安装该模型要求的 remote/package 依赖, "
                f"或把 pipeline_class 改成可导入的 pipeline 类: {self.config.pipeline_class}"
            ) from exc

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        self.scheduler = getattr(self.pipe, "scheduler", None)
        if self.scheduler is None:
            raise RuntimeError("pipeline 缺少 scheduler, 无法构造扩散训练目标")

        base_module = getattr(self.pipe, self.config.train_component, None)
        if base_module is None:
            raise RuntimeError(f"pipeline 缺少训练组件: {self.config.train_component}")
        if not isinstance(base_module, torch.nn.Module):
            raise TypeError(f"{self.config.train_component} 不是 torch.nn.Module")

        for _, component in self.pipe.components.items():
            if isinstance(component, torch.nn.Module):
                component.requires_grad_(False)
                component.eval()

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.train_module = get_peft_model(base_module, lora_config)
        cast_training_params(self.train_module, dtype=torch.float32)

        base_model = (
            self.train_module.get_base_model()
            if hasattr(self.train_module, "get_base_model")
            else self.train_module
        )
        if (
            self.config.enable_gradient_checkpointing
            and hasattr(base_model, "enable_gradient_checkpointing")
        ):
            base_model.enable_gradient_checkpointing()

        setattr(self, self.config.train_component, self.train_module)
        self._register_known_modules()
        self._bind_pipeline_modules()
        if hasattr(self.train_module, "print_trainable_parameters"):
            self.train_module.print_trainable_parameters()
        self._is_built = True

    def _register_known_modules(self) -> None:
        if self.pipe is None:
            return
        for name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            component = getattr(self.pipe, name, None)
            if isinstance(component, torch.nn.Module):
                setattr(self, name, component)

    def _bind_pipeline_modules(self) -> None:
        if self.pipe is None:
            return
        if self.train_module is not None:
            setattr(self.pipe, self.config.train_component, self.train_module)
        for name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            if hasattr(self, name):
                setattr(self.pipe, name, getattr(self, name))

    def on_after_device_setup(self) -> None:
        self._bind_pipeline_modules()
        self._cache_static_conditions()

    def on_train_start(self) -> None:
        self._bind_pipeline_modules()
        if self.pipe is not None and hasattr(self.pipe, "to"):
            self.pipe.to(self.device)
        if self.pipe is not None and hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        super().on_train_start()

    def _autocast_context(self) -> ContextManager[None]:
        if not self.use_cuda_autocast:
            return nullcontext()
        if self.model_dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    def _cache_static_conditions(self) -> None:
        return None

    def _prepare_scheduler_state(self) -> None:
        if self.scheduler is None:
            raise RuntimeError("scheduler 尚未初始化")
        if not hasattr(self.scheduler, "timesteps") or len(self.scheduler.timesteps) == 0:
            self.scheduler.set_timesteps(
                int(self.scheduler.config.num_train_timesteps),
                device=self.device,
            )
        if hasattr(self.scheduler, "sigmas"):
            self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

    def _sample_timesteps(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._prepare_scheduler_state()
        density = compute_density_for_timestep_sampling(
            weighting_scheme=self.config.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.config.logit_mean,
            logit_std=self.config.logit_std,
            mode_scale=self.config.mode_scale,
            device=self.device,
        )
        indices = (density * self.scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(max=self.scheduler.config.num_train_timesteps - 1)

        timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        sigmas = self.scheduler.sigmas[indices].to(device=self.device, dtype=torch.float32)
        loss_weights = compute_loss_weighting_for_sd3(
            self.config.weighting_scheme,
            sigmas=sigmas,
        )
        return timesteps, loss_weights

    def _compute_loss(self, batch: dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self,
        batch: dict[str, Optional[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        optimizer = self.optimizers[0]
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")

        self.train_module.train()
        for name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            component = getattr(self, name, None)
            if isinstance(component, torch.nn.Module):
                component.eval()

        loss = self._compute_loss(batch)
        should_step = self.manual_optimization_step(
            loss,
            optimizer=optimizer,
            model=self.train_module,
            max_grad_norm=self.config.max_grad_norm,
        )

        self.log("train_loss", loss.detach())
        self.log("lr", self.get_lr())
        if should_step:
            self.log("optimizer_step", self.optimizer_step)

    def validation_step(
        self,
        batch: dict[str, Optional[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")
        self.train_module.eval()
        loss = self._compute_loss(batch)
        self.log("val_loss", loss.detach())

    def configure_optimizers(self) -> AdamW:
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")

        params = [param for param in self.train_module.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数, 请检查 LoRA 配置")

        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

    def _call_pipeline_filtered(self, **kwargs: Any) -> Any:
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")
        signature = inspect.signature(self.pipe.__call__)
        filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return self.pipe(**filtered)

    def inference(self, data: Any) -> None:
        if not data or self.pipe is None or not self.is_main_process():
            return
        prompts = [data] if isinstance(data, str) else list(data)
        if not prompts:
            return

        self._bind_pipeline_modules()
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")
        self.train_module.eval()

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(
            self.config.seed + self.current_epoch
        )

        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_image = None
        input_images = None
        if self.config.sample_input_image is not None:
            input_image = Image.open(self.config.sample_input_image).convert("RGB")
            input_images = [input_image]

        with torch.inference_mode():
            output = self._call_pipeline_filtered(
                prompt=prompts,
                image=input_image,
                input_images=input_images,
                height=self.config.image_height,
                width=self.config.image_width,
                guidance_scale=self.config.sample_guidance,
                true_cfg_scale=self.config.sample_guidance,
                img_guidance_scale=self.config.sample_image_guidance,
                num_inference_steps=self.config.sample_steps,
                generator=generator,
            )

        images = getattr(output, "images", output[0] if isinstance(output, tuple) else output)
        for index, image in enumerate(images):
            image.save(sample_dir / f"epoch_{self.current_epoch:04d}_sample_{index:02d}.png")

        self.train_module.train()

    def save_lora_adapter(self, save_dir: Path) -> None:
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")
        save_dir.mkdir(parents=True, exist_ok=True)

        lora_state_dict = get_peft_model_state_dict(self.train_module)
        adapter_metadata = None
        if hasattr(self.train_module, "peft_config") and "default" in self.train_module.peft_config:
            adapter_metadata = self.train_module.peft_config["default"].to_dict()

        save_lora_weights = getattr(type(self.pipe), "save_lora_weights", None)
        if callable(save_lora_weights):
            save_lora_weights(
                save_directory=str(save_dir),
                transformer_lora_layers=lora_state_dict,
                transformer_lora_adapter_metadata=adapter_metadata,
                is_main_process=self.is_main_process(),
                safe_serialization=True,
            )
            return

        if hasattr(self.train_module, "save_pretrained"):
            self.train_module.save_pretrained(
                save_dir,
                safe_serialization=True,
                is_main_process=self.is_main_process(),
            )
            return

        torch.save(lora_state_dict, save_dir / "pytorch_lora_weights.bin")


class FluxLikeOutfitLoRAModel(BaseOutfitLoRAModel):
    def __init__(self, config: OutfitLoRAConfig) -> None:
        super().__init__(config)
        self.register_buffer("_cached_prompt_embeds", None, persistent=False)
        self.register_buffer("_cached_text_ids", None, persistent=False)

    def _cache_static_conditions(self) -> None:
        if self._cached_prompt_embeds is not None and self._cached_text_ids is not None:
            return
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")

        encode_kwargs: dict[str, Any] = {
            "prompt": self.config.training_prompt,
            "device": self.device,
            "num_images_per_prompt": 1,
            "max_sequence_length": self.config.max_sequence_length,
        }
        if self.config.text_encoder_out_layers is not None:
            encode_kwargs["text_encoder_out_layers"] = self.config.text_encoder_out_layers

        with torch.no_grad():
            prompt_embeds, text_ids = self.pipe.encode_prompt(**encode_kwargs)
        self._cached_prompt_embeds = prompt_embeds.detach().to(
            device=self.device,
            dtype=self.model_dtype,
        )
        self._cached_text_ids = text_ids.detach().to(device=self.device)

    def _cached_text_conditions(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_prompt_embeds is None or self._cached_text_ids is None:
            raise RuntimeError("训练提示词尚未缓存")
        return (
            self._cached_prompt_embeds.expand(batch_size, -1, -1),
            self._cached_text_ids.expand(batch_size, -1, -1),
        )

    def _encode_images(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")
        vae = getattr(self, "vae", None)
        if vae is None:
            raise RuntimeError("pipeline 缺少 vae")

        vae_inputs = pixel_values.to(device=self.device, dtype=vae.dtype)
        with torch.no_grad():
            image_latents = self.pipe._encode_vae_image(vae_inputs, generator=None)
            latent_ids = self.pipe._prepare_latent_ids(image_latents).to(self.device)
            latents = self.pipe._pack_latents(image_latents)
        return latents, latent_ids

    def _compute_loss(self, batch: dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        if pixel_values is None:
            raise RuntimeError("batch 缺少 pixel_values")

        latents, latent_ids = self._encode_images(pixel_values)
        prompt_embeds, text_ids = self._cached_text_conditions(latents.shape[0])

        timesteps, loss_weights = self._sample_timesteps(latents.shape[0])
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)
        target = noise - latents

        model_input = noisy_latents.to(dtype=self.model_dtype)
        timestep_input = timesteps.to(device=self.device, dtype=self.model_dtype) / 1000.0

        with self._autocast_context():
            model_pred = self.train_module(
                hidden_states=model_input,
                timestep=timestep_input,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

        model_pred = model_pred.float()
        target = target.float()
        weights = loss_weights[:, None, None].float()
        loss = (weights * (model_pred - target).pow(2)).reshape(model_pred.shape[0], -1).mean(dim=1)
        return loss.mean()


class QwenImageLikeOutfitLoRAModel(BaseOutfitLoRAModel):
    def __init__(self, config: OutfitLoRAConfig) -> None:
        super().__init__(config)
        self.register_buffer("_cached_prompt_embeds", None, persistent=False)
        self.register_buffer("_cached_prompt_mask", None, persistent=False)

    @property
    def latent_channels(self) -> int:
        pipe_channels = getattr(self.pipe, "latent_channels", None)
        if pipe_channels is not None:
            return int(pipe_channels)
        if self.train_module is None:
            raise RuntimeError("LoRA 训练组件尚未初始化")
        return int(self.train_module.config.in_channels) // 4

    @property
    def vae_scale_factor(self) -> int:
        return int(getattr(self.pipe, "vae_scale_factor", 8))

    def _cache_static_conditions(self) -> None:
        if self._cached_prompt_embeds is not None and self._cached_prompt_mask is not None:
            return
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")

        with torch.no_grad():
            prompt_embeds, prompt_mask = self.pipe.encode_prompt(
                prompt=self.config.training_prompt,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.config.max_sequence_length,
            )
        self._cached_prompt_embeds = prompt_embeds.detach().to(
            device=self.device,
            dtype=self.model_dtype,
        )
        self._cached_prompt_mask = prompt_mask.detach().to(device=self.device)

    def _text_conditions(
        self,
        batch_size: int,
        source_pixel_values: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_pixel_values is None:
            if self._cached_prompt_embeds is None or self._cached_prompt_mask is None:
                raise RuntimeError("训练提示词尚未缓存")
            return (
                self._cached_prompt_embeds.expand(batch_size, -1, -1),
                self._cached_prompt_mask.expand(batch_size, -1),
            )

        encode_signature = inspect.signature(self.pipe.encode_prompt)
        if "image" not in encode_signature.parameters:
            raise RuntimeError("当前 pipeline 的 encode_prompt 不支持 source image 条件")

        source_tensor = source_pixel_values.to(device=self.device, dtype=self.model_dtype)
        with torch.no_grad():
            prompt_embeds, prompt_mask = self.pipe.encode_prompt(
                prompt=[self.config.training_prompt] * batch_size,
                image=source_tensor,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.config.max_sequence_length,
            )
        return (
            prompt_embeds.detach().to(device=self.device, dtype=self.model_dtype),
            prompt_mask.detach().to(device=self.device),
        )

    def _encode_image_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")
        vae = getattr(self, "vae", None)
        if vae is None:
            raise RuntimeError("pipeline 缺少 vae")

        vae_inputs = pixel_values.to(device=self.device, dtype=vae.dtype)
        with torch.no_grad():
            if hasattr(self.pipe, "_encode_vae_image"):
                image_latents = self.pipe._encode_vae_image(vae_inputs, generator=None)
            else:
                latent_dist = vae.encode(vae_inputs).latent_dist
                image_latents = latent_dist.mode()
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, self.latent_channels, 1, 1, 1)
                    .to(image_latents.device, image_latents.dtype)
                )
                latents_std = (
                    torch.tensor(vae.config.latents_std)
                    .view(1, self.latent_channels, 1, 1, 1)
                    .to(image_latents.device, image_latents.dtype)
                )
                image_latents = (image_latents - latents_mean) / latents_std
        return image_latents

    def _pack_qwen_latents(self, image_latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if image_latents.ndim == 5:
            image_latents = image_latents[:, :, 0]
        batch_size, channels, latent_height, latent_width = image_latents.shape
        packed = self.pipe._pack_latents(
            image_latents,
            batch_size,
            channels,
            latent_height,
            latent_width,
        )
        img_shape = (
            1,
            latent_height // 2,
            latent_width // 2,
        )
        return packed, img_shape

    def _compute_loss(self, batch: dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        if pixel_values is None:
            raise RuntimeError("batch 缺少 pixel_values")

        source_pixel_values = batch.get("source_pixel_values")
        target_latents_5d = self._encode_image_latents(pixel_values)
        latents, target_shape = self._pack_qwen_latents(target_latents_5d)
        source_latents = None
        source_shape = None
        if source_pixel_values is not None:
            source_latents_5d = self._encode_image_latents(source_pixel_values)
            source_latents, source_shape = self._pack_qwen_latents(source_latents_5d)

        prompt_embeds, prompt_mask = self._text_conditions(
            latents.shape[0],
            source_pixel_values,
        )
        img_shapes = [[target_shape]] * latents.shape[0]
        if source_shape is not None:
            img_shapes = [[target_shape, source_shape]] * latents.shape[0]

        timesteps, loss_weights = self._sample_timesteps(latents.shape[0])
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)
        target = noise - latents

        model_input = noisy_latents.to(dtype=self.model_dtype)
        if source_latents is not None:
            model_input = torch.cat([model_input, source_latents.to(dtype=self.model_dtype)], dim=1)
        timestep_input = timesteps.to(device=self.device, dtype=self.model_dtype) / 1000.0

        guidance = None
        if getattr(self.train_module.config, "guidance_embeds", False):
            guidance = torch.full(
                (latents.shape[0],),
                self.config.sample_guidance,
                device=self.device,
                dtype=torch.float32,
            )

        with self._autocast_context():
            model_pred = self.train_module(
                hidden_states=model_input,
                timestep=timestep_input,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                attention_kwargs=getattr(self.pipe, "attention_kwargs", None),
                return_dict=False,
            )[0]
            model_pred = model_pred[:, : latents.size(1)]

        model_pred = model_pred.float()
        target = target.float()
        weights = loss_weights[:, None, None].float()
        loss = (weights * (model_pred - target).pow(2)).reshape(model_pred.shape[0], -1).mean(dim=1)
        return loss.mean()


class BooguOutfitLoRAModel(QwenImageLikeOutfitLoRAModel):
    """Boogu 优先走 QwenImageEdit-like 训练口径.

    Boogu 官方模型使用 custom pipeline. 如果其 remote pipeline 不兼容
    QwenImageEdit 的 encode_prompt/_encode_vae_image/_pack_latents 接口,
    脚本会在运行初期显式报错.
    """


def build_dataloader(
    manifest_path: Path,
    image_height: int,
    image_width: int,
    source_image_height: int,
    source_image_width: int,
    batch_size: int,
    num_workers: int,
    dataset_repeat: int,
    shuffle: bool,
) -> DataLoader[dict[str, Optional[torch.Tensor]]]:
    dataset = OutfitImageDataset(
        manifest_path=manifest_path,
        image_height=image_height,
        image_width=image_width,
        source_image_height=source_image_height,
        source_image_width=source_image_width,
        dataset_repeat=dataset_repeat,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=outfit_collate_fn,
    )


def run_outfit_lora_training(
    config: OutfitLoRAConfig,
    model_cls: Type[BaseOutfitLoRAModel],
) -> None:
    save_run_config(config)
    set_seed(config.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = build_dataloader(
        manifest_path=config.train_manifest,
        image_height=config.image_height,
        image_width=config.image_width,
        source_image_height=config.source_image_height,
        source_image_width=config.source_image_width,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_repeat=config.dataset_repeat,
        shuffle=True,
    )

    val_loader = None
    if config.val_manifest is not None:
        val_loader = build_dataloader(
            manifest_path=config.val_manifest,
            image_height=config.image_height,
            image_width=config.image_width,
            source_image_height=config.source_image_height,
            source_image_width=config.source_image_width,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            dataset_repeat=1,
            shuffle=False,
        )

    checkpoint_callback = LoRACheckpointCallback(
        output_dir=config.output_dir,
        every_n_epochs=config.save_every_n_epochs,
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        device=config.device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        callbacks=[checkpoint_callback],
    )
    trainer.setup_logger(
        experiment_name=config.experiment_name,
        log_dir=str(config.output_dir / "logs"),
        checkpoint_dir=str(config.output_dir / "trainer_checkpoints"),
        enable_tensorboard=False,
        enable_checkpoint=False,
        enable_console=False,
        enable_tqdm=True,
        log_every_n_steps=1,
    )

    model = model_cls(config)
    inference_data: Optional[Sequence[str]] = config.sample_prompts or None
    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        check_val_every_n_epoch=config.sample_every_n_epochs,
        inference_data=inference_data,
    )
