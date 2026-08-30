"""SeFi-Image 的 XDL 换装 LoRA 微调入口.

默认训练 SeFi-Image-5B-Base 的 transformer LoRA. 训练数据 manifest 只需要
image 字段, 所有图片共享 training_prompt.

使用示例:

XDL_SEFI_IMAGE_CONFIG=train/posttrain/sefi_image_outfit_lora.yaml \
  python train/posttrain/sefi_image_outfit_lora_finetune.py
"""

from __future__ import annotations

from typing import Any

from diffusion_outfit_lora_common import (
    DEFAULT_TARGET_MODULES,
    FluxLikeOutfitLoRAModel,
    load_outfit_config,
    run_outfit_lora_training,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "sefi_image_outfit_lora",
    "model_id": "SeFi-Image/SeFi-Image-5B-Base",
    "pipeline_class": "diffusers.Flux2Pipeline",
    "train_manifest": "./data/sefi_image_outfit/train.jsonl",
    "val_manifest": None,
    "output_dir": "./outputs/sefi_image_outfit_lora",
    "training_prompt": "outfit swap",
    "image_height": 1536,
    "image_width": 864,
    "source_image_height": 1536,
    "source_image_width": 864,
    "batch_size": 1,
    "num_workers": 4,
    "dataset_repeat": 10,
    "num_epochs": 100,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,
    "device": "cuda",
    "model_dtype": "auto",
    "max_sequence_length": 512,
    "text_encoder_out_layers": [10, 20, 30],
    "train_component": "transformer",
    "target_modules": list(DEFAULT_TARGET_MODULES),
    "lora_rank": 96,
    "lora_alpha": 192,
    "lora_dropout": 0.05,
    "weighting_scheme": "none",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "save_every_n_epochs": 10,
    "sample_every_n_epochs": 10,
    "sample_steps": 20,
    "sample_guidance": 4.0,
    "sample_image_guidance": 1.6,
    "sample_prompts": ["outfit swap"],
    "sample_input_image": None,
    "seed": 42,
    "enable_gradient_checkpointing": True,
    "from_pretrained_kwargs": {},
}
CONFIG_ENV_VAR = "XDL_SEFI_IMAGE_CONFIG"


def main() -> None:
    config = load_outfit_config(DEFAULT_CONFIG, CONFIG_ENV_VAR)
    run_outfit_lora_training(config, FluxLikeOutfitLoRAModel)


if __name__ == "__main__":
    main()
