"""xdl-jax 结构化配置 schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class JaxRuntimeConfig:
    seed: int = 42
    platform: str = "cpu"


@dataclass
class JaxTrainerConfig:
    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    drop_incomplete_accumulation: bool = True
    precision: str = "32"
    jit: bool = True
    grad_clip_max_norm: float | None = None
    fail_on_callback_error: bool = False
    nan_patience: int = 0
    validate_every_n_epochs: int = 1
    max_train_steps: int | None = None


@dataclass
class JaxConfigSchema:
    config_version: int = 1
    backend: str = "jax"
    runtime: JaxRuntimeConfig = field(default_factory=JaxRuntimeConfig)
    trainer: JaxTrainerConfig = field(default_factory=JaxTrainerConfig)
    task: dict[str, Any] | None = None
    train_data: dict[str, Any] | None = None
    val_data: dict[str, Any] | None = None
    optimization: dict[str, Any] = field(default_factory=dict)
    callbacks: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JaxTrainConfig:
    """从 OmegaConf 读取后传给 runtime 的不可变配置摘要."""

    seed: int
    platform: str
    trainer: dict[str, Any]
    optimization: dict[str, Any]
    checkpoint: dict[str, Any]


def create_structured_config() -> DictConfig:
    return OmegaConf.structured(JaxConfigSchema())
