"""Boogu-Image-0.1-Edit 的 XDL 换装 LoRA 微调入口.

Boogu 是 custom pipeline 模型. 默认通过 DiffusionPipeline 加载模型仓库中的
custom pipeline, 并按 QwenImageEdit-like 的条件编辑口径训练 transformer LoRA.
manifest 可只写 image 字段; 如需图像编辑条件, 额外加 source_image 字段.

使用示例:

XDL_BOOGU_IMAGE_EDIT_CONFIG=examples/boogu_image_edit_outfit_lora.yaml \
  python examples/boogu_image_edit_outfit_lora_finetune.py
"""

from __future__ import annotations

from typing import Any

from diffusion_outfit_lora_common import (
    DEFAULT_TARGET_MODULES,
    BooguOutfitLoRAModel,
    load_outfit_config,
    run_outfit_lora_training,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "boogu_image_edit_outfit_lora",
    "model_id": "Boogu/Boogu-Image-0.1-Edit",
    "pipeline_class": "diffusers.DiffusionPipeline",
    "train_manifest": "./data/boogu_image_edit_outfit/train.jsonl",
    "val_manifest": None,
    "output_dir": "./outputs/boogu_image_edit_outfit_lora",
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
    "text_encoder_out_layers": None,
    "train_component": "transformer",
    "target_modules": list(DEFAULT_TARGET_MODULES),
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "weighting_scheme": "none",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "save_every_n_epochs": 10,
    "sample_every_n_epochs": 10,
    "sample_steps": 25,
    "sample_guidance": 5.0,
    "sample_image_guidance": 1.6,
    "sample_prompts": ["outfit swap"],
    "sample_input_image": None,
    "seed": 42,
    "enable_gradient_checkpointing": True,
    "from_pretrained_kwargs": {
        "custom_pipeline": "Boogu/Boogu-Image-0.1-Edit",
    },
}
CONFIG_ENV_VAR = "XDL_BOOGU_IMAGE_EDIT_CONFIG"


def main() -> None:
    config = load_outfit_config(DEFAULT_CONFIG, CONFIG_ENV_VAR)
    run_outfit_lora_training(config, BooguOutfitLoRAModel)


if __name__ == "__main__":
    main()
