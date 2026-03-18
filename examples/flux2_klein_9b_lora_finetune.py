"""
FLUX.2 klein 9B 的 XDL LoRA 微调示例。

这个脚本默认做的是 transformer LoRA 微调, 不做全量 9B 参数更新。

数据集 manifest 支持三种格式:

1. JSONL
   {"image": "images/0001.png", "prompt": "a red dress on white background"}
   {"image": "images/0002.png", "prompt": "studio portrait, soft light"}

2. JSON
   [
     {"image": "images/0001.png", "prompt": "a red dress on white background"}
   ]

3. CSV
   image,prompt
   images/0001.png,a red dress on white background

最小使用示例:

python examples/flux2_klein_9b_lora_finetune.py \
  --train-manifest ./others/data/flux_train.jsonl \
  --output-dir ./others/flux2_klein_9b_lora \
  --device cuda \
  --batch-size 1 \
  --num-epochs 1 \
  --sample-prompt "a studio fashion photo"
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    set_seed,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from PIL import Image, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from xdl.callbacks import Callback
from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer


PROMPT_KEYS: Tuple[str, ...] = ("prompt", "caption", "text")
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
)


@dataclass
class Flux2FinetuneConfig:
    model_id: str
    train_manifest: Path
    val_manifest: Optional[Path]
    output_dir: Path
    image_height: int
    image_width: int
    batch_size: int
    num_workers: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    device: str
    model_dtype: str
    max_sequence_length: int
    text_encoder_out_layers: Tuple[int, ...]
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
    sample_prompts: Tuple[str, ...]
    seed: int
    enable_gradient_checkpointing: bool


def parse_text_encoder_layers(raw_value: str) -> Tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not layers:
        raise ValueError("`text_encoder_out_layers` 不能为空")
    return layers


def parse_args() -> Flux2FinetuneConfig:
    parser = argparse.ArgumentParser(
        description="使用 XDL 对 FLUX.2 klein 9B 做 transformer LoRA 微调"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9B",
        help=(
            "基础模型名称。若你的环境里使用的是 base 变体, 可以改成 "
            "`black-forest-labs/FLUX.2-klein-base-9B`。"
        ),
    )
    parser.add_argument("--train-manifest", type=Path, required=True, help="训练集 manifest 路径")
    parser.add_argument("--val-manifest", type=Path, default=None, help="验证集 manifest 路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--image-height", type=int, default=1024, help="训练图像高度, 需能被 16 整除")
    parser.add_argument("--image-width", type=int, default=1024, help="训练图像宽度, 需能被 16 整除")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--num-epochs", type=int, default=1, help="训练 epoch 数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备, 如 cuda / cuda:0 / cpu")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="auto",
        choices=("auto", "fp32", "fp16", "bf16"),
        help="模型加载与前向计算 dtype",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512, help="prompt 最大 token 长度")
    parser.add_argument(
        "--text-encoder-out-layers",
        type=str,
        default="9,18,27",
        help="Qwen3 hidden state 层号, 逗号分隔",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--weighting-scheme",
        type=str,
        default="none",
        choices=("none", "sigma_sqrt", "cosmap", "logit_normal", "mode"),
        help="SD3/Flow Matching 训练的 timestep 采样与 loss weighting 策略",
    )
    parser.add_argument("--logit-mean", type=float, default=0.0, help="logit_normal 采样均值")
    parser.add_argument("--logit-std", type=float, default=1.0, help="logit_normal 采样标准差")
    parser.add_argument("--mode-scale", type=float, default=1.29, help="mode 采样缩放因子")
    parser.add_argument("--save-every-n-epochs", type=int, default=1, help="每隔多少个 epoch 保存一次 LoRA")
    parser.add_argument(
        "--sample-every-n-epochs",
        type=int,
        default=1,
        help="每隔多少个 epoch 跑一次验证/预览生图",
    )
    parser.add_argument("--sample-steps", type=int, default=20, help="预览生图步数")
    parser.add_argument("--sample-guidance", type=float, default=4.0, help="预览生图 guidance scale")
    parser.add_argument(
        "--sample-prompt",
        action="append",
        default=[],
        help="可重复指定, 每次验证后会用这些 prompt 生成预览图",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--enable-gradient-checkpointing",
        action="store_true",
        help="开启 transformer gradient checkpointing, 用更慢的速度换显存",
    )

    args = parser.parse_args()

    if args.image_height % 16 != 0 or args.image_width % 16 != 0:
        raise ValueError("`image_height` 和 `image_width` 必须都能被 16 整除")

    return Flux2FinetuneConfig(
        model_id=args.model_id,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        model_dtype=args.model_dtype,
        max_sequence_length=args.max_sequence_length,
        text_encoder_out_layers=parse_text_encoder_layers(args.text_encoder_out_layers),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        save_every_n_epochs=max(1, args.save_every_n_epochs),
        sample_every_n_epochs=max(1, args.sample_every_n_epochs),
        sample_steps=args.sample_steps,
        sample_guidance=args.sample_guidance,
        sample_prompts=tuple(args.sample_prompt),
        seed=args.seed,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
    )


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


def config_to_dict(config: Flux2FinetuneConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, tuple):
            config_dict[key] = list(value)
    return config_dict


def save_run_config(config: Flux2FinetuneConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(config_to_dict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest_records(manifest_path: Path) -> List[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
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


def pick_prompt(record: dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"记录缺少 prompt 字段, 需要其一: {PROMPT_KEYS}")


def resolve_data_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


class Flux2ImagePromptDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, manifest_path: Path, image_height: int, image_width: int):
        self.manifest_path = manifest_path.resolve()
        self.image_height = image_height
        self.image_width = image_width
        self.records = load_manifest_records(self.manifest_path)

        if not self.records:
            raise ValueError(f"manifest 为空: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image_path = resolve_data_path(str(record["image"]), self.manifest_path)
        prompt = pick_prompt(record)

        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(
            image,
            (self.image_width, self.image_height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

        pixel_values = TF.to_tensor(image)
        pixel_values = pixel_values * 2.0 - 1.0
        return pixel_values, prompt


def flux2_collate_fn(batch: Sequence[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, List[str]]:
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    prompts = [item[1] for item in batch]
    return pixel_values, prompts


class Flux2LoRACheckpointCallback(Callback):
    def __init__(self, output_dir: Path, every_n_epochs: int = 1):
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


class Flux2KleinLoRAModel(CoreModel):
    def __init__(self, config: Flux2FinetuneConfig):
        super().__init__()
        self.config = config
        self.output_dir = config.output_dir
        self.model_dtype = resolve_dtype(config.model_dtype, config.device)
        self.use_cuda_autocast = config.device.startswith("cuda") and torch.cuda.is_available()

        self.pipe: Optional[Flux2KleinPipeline] = None
        self.vae: Optional[torch.nn.Module] = None
        self.text_encoder: Optional[torch.nn.Module] = None
        self.transformer: Optional[torch.nn.Module] = None
        self.scheduler: Any = None
        self.tokenizer: Any = None
        self._is_built = False

    def setup(self, stage: str) -> None:
        if self._is_built:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.model_dtype,
        )
        self.pipe.set_progress_bar_config(disable=True)

        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer

        base_transformer = self.pipe.transformer

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        base_transformer.requires_grad_(False)

        self.vae.eval()
        self.text_encoder.eval()

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(DEFAULT_TARGET_MODULES),
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.transformer = get_peft_model(base_transformer, lora_config)
        cast_training_params(self.transformer, dtype=torch.float32)

        base_model = self.transformer.get_base_model() if hasattr(self.transformer, "get_base_model") else self.transformer
        if (
            self.config.enable_gradient_checkpointing
            and hasattr(base_model, "enable_gradient_checkpointing")
        ):
            base_model.enable_gradient_checkpointing()

        self._bind_pipeline_modules()

        if hasattr(self.transformer, "print_trainable_parameters"):
            self.transformer.print_trainable_parameters()

        self._is_built = True

    def _bind_pipeline_modules(self) -> None:
        if self.pipe is None:
            return
        if self.vae is not None:
            self.pipe.vae = self.vae
        if self.text_encoder is not None:
            self.pipe.text_encoder = self.text_encoder
        if self.transformer is not None:
            self.pipe.transformer = self.transformer

    def on_train_start(self) -> None:
        self._bind_pipeline_modules()
        if self.pipe is not None:
            self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
        super().on_train_start()

    def _autocast_context(self):
        if not self.use_cuda_autocast:
            return nullcontext()
        if self.model_dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    def _encode_images(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")

        vae_inputs = pixel_values.to(device=self.device, dtype=self.vae.dtype)  # type: ignore[union-attr]
        with torch.no_grad():
            image_latents = self.pipe._encode_vae_image(vae_inputs, generator=None)
            latent_ids = self.pipe._prepare_latent_ids(image_latents).to(self.device)
            latents = self.pipe._pack_latents(image_latents)
        return latents, latent_ids

    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError("pipeline 尚未初始化")
        with torch.no_grad():
            prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=list(prompts),
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.config.max_sequence_length,
                text_encoder_out_layers=self.config.text_encoder_out_layers,
            )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.model_dtype)
        text_ids = text_ids.to(device=self.device)
        return prompt_embeds, text_ids

    def _sample_timesteps(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        loss_weights = compute_loss_weighting_for_sd3(self.config.weighting_scheme, sigmas=sigmas)
        return timesteps, loss_weights

    def _compute_loss(self, batch: Sequence[Any]) -> torch.Tensor:
        pixel_values, prompts = batch

        latents, latent_ids = self._encode_images(pixel_values)
        prompt_embeds, text_ids = self._encode_prompts(prompts)

        timesteps, loss_weights = self._sample_timesteps(latents.shape[0])
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)
        target = noise - latents

        model_input = noisy_latents.to(dtype=self.model_dtype)
        timestep_input = timesteps.to(device=self.device, dtype=self.model_dtype) / 1000.0

        with self._autocast_context():
            model_pred = self.transformer(
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

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> None:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)

        self.transformer.train()
        self.vae.eval()  # type: ignore[union-attr]
        self.text_encoder.eval()  # type: ignore[union-attr]

        loss = self._compute_loss(batch)
        self.manual_backward(loss)
        self.clip_gradients(self.transformer, gradient_clip_val=self.config.max_grad_norm)
        optimizer.step()

        self.log("train_loss", loss.detach())
        self.log("lr", self.get_lr())

    def validation_step(self, batch: Sequence[Any], batch_idx: int) -> None:
        self.transformer.eval()
        loss = self._compute_loss(batch)
        self.log("val_loss", loss.detach())

    def configure_optimizers(self) -> AdamW:
        if self.transformer is None:
            raise RuntimeError("transformer 尚未初始化")

        params = [param for param in self.transformer.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数, 请检查 LoRA 配置")

        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

    def inference(self, data: Any) -> None:
        if not data or self.pipe is None or not self.is_main_process():
            return

        prompts = [data] if isinstance(data, str) else list(data)
        if not prompts:
            return

        self._bind_pipeline_modules()
        self.transformer.eval()

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(
            self.config.seed + self.current_epoch
        )

        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            images = self.pipe(
                prompt=prompts,
                height=self.config.image_height,
                width=self.config.image_width,
                guidance_scale=self.config.sample_guidance,
                num_inference_steps=self.config.sample_steps,
                generator=generator,
            ).images

        for index, image in enumerate(images):
            image.save(sample_dir / f"epoch_{self.current_epoch:04d}_sample_{index:02d}.png")

        self.transformer.train()

    def save_lora_adapter(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        lora_state_dict = get_peft_model_state_dict(self.transformer)
        adapter_metadata = None
        if hasattr(self.transformer, "peft_config") and "default" in self.transformer.peft_config:
            adapter_metadata = self.transformer.peft_config["default"].to_dict()

        Flux2KleinPipeline.save_lora_weights(
            save_directory=str(save_dir),
            transformer_lora_layers=lora_state_dict,
            transformer_lora_adapter_metadata=adapter_metadata,
            is_main_process=self.is_main_process(),
            safe_serialization=True,
        )


def build_dataloader(
    manifest_path: Path,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = Flux2ImagePromptDataset(
        manifest_path=manifest_path,
        image_height=image_height,
        image_width=image_width,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=flux2_collate_fn,
    )


def main() -> None:
    config = parse_args()
    save_run_config(config)
    set_seed(config.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = build_dataloader(
        manifest_path=config.train_manifest,
        image_height=config.image_height,
        image_width=config.image_width,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    val_loader = None
    if config.val_manifest is not None:
        val_loader = build_dataloader(
            manifest_path=config.val_manifest,
            image_height=config.image_height,
            image_width=config.image_width,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    checkpoint_callback = Flux2LoRACheckpointCallback(
        output_dir=config.output_dir,
        every_n_epochs=config.save_every_n_epochs,
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        device=config.device,
        callbacks=[checkpoint_callback],
    )
    trainer.setup_logger(
        experiment_name="flux2_klein_9b_lora",
        log_dir=str(config.output_dir / "logs"),
        checkpoint_dir=str(config.output_dir / "trainer_checkpoints"),
        enable_tensorboard=False,
        enable_checkpoint=False,
        enable_console=False,
        enable_tqdm=True,
        log_every_n_steps=1,
    )

    model = Flux2KleinLoRAModel(config)
    inference_data: Optional[Iterable[str]] = config.sample_prompts or None

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        check_val_every_n_epoch=config.sample_every_n_epochs,
        inference_data=inference_data,
    )


if __name__ == "__main__":
    main()
