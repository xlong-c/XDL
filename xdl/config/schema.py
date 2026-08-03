"""
配置 schema 定义.

固定顶层结构,组件统一使用 `target + params`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from .accelerate_config import AccelerateConfig, DeepSpeedConfig
from .errors import ConfigError

CONFIG_SCHEMA_VERSION = 1


@dataclass
class RuntimeConfig:
    """运行时配置."""

    device: str = "cuda"
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./others"
    experiment_name: str = "default_exp"


@dataclass
class TrainerConfig:
    """训练主循环配置."""

    max_epochs: int = 100
    batch_size: Optional[int] = None
    precision: str = "32"
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: Optional[float] = None
    grad_clip_norm_type: float = 2.0
    fsdp: Optional[Any] = None
    nan_monitor: bool = True
    nan_patience: int = 3
    fail_on_callback_error: bool = False


@dataclass
class ComponentConfig:
    """通用组件配置."""

    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    target_modules: Optional[List[str]] = None
    param_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight: Optional[float] = None


@dataclass
class XDLContextConfig:
    """配置加载上下文, 由 resolver 注入."""

    config_path: str = ""
    config_dir: str = "."
    project_root: str = "."


@dataclass
class TransformPipelineConfig(ComponentConfig):
    """数据增强配置."""

    pass


@dataclass
class DatasetConfig(ComponentConfig):
    """数据集配置."""

    pass


@dataclass
class DataloaderConfig:
    """DataLoader 配置."""

    dataset: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    collate_fn: Any = None


@dataclass
class DataloaderDefaultsConfig:
    """DataLoader 默认参数配置."""

    batch_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    collate_fn: Any = None


@dataclass
class OptimizationConfig:
    """优化相关配置."""

    optimizer: Optional[ComponentConfig] = None
    scheduler: Optional[ComponentConfig] = None


@dataclass
class LoggingConfig:
    """日志配置."""

    log_dir: str = "./others/logs"
    enable_console: bool = True
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class CheckpointConfig:
    """检查点配置."""

    save_last: bool = True
    save_top_k: int = 1
    monitor: Optional[str] = None
    mode: str = "min"
    every_n_epochs: int = 1
    dirpath: Optional[str] = None


@dataclass
class ConfigSchemaV1:
    """XDL 配置 schema v1."""

    config_version: int = CONFIG_SCHEMA_VERSION
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    xdl: XDLContextConfig = field(default_factory=XDLContextConfig)
    task: Optional[ComponentConfig] = None
    model: Optional[ComponentConfig] = None

    train_transforms: Any = None
    val_transforms: Any = None
    test_transforms: Any = None
    train_dataset: Optional[DatasetConfig] = None
    val_dataset: Optional[DatasetConfig] = None
    test_dataset: Optional[DatasetConfig] = None
    dataloader_defaults: Optional[DataloaderDefaultsConfig] = None
    train_dataloader: Optional[DataloaderConfig] = None
    val_dataloader: Optional[DataloaderConfig] = None
    test_dataloader: Optional[DataloaderConfig] = None

    optimization: Optional[OptimizationConfig] = None
    loss: List[ComponentConfig] = field(default_factory=list)
    metrics: List[ComponentConfig] = field(default_factory=list)
    callbacks: List[ComponentConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    accelerate: Optional[AccelerateConfig] = None
    deepspeed: Optional[DeepSpeedConfig] = None


def create_config_schema() -> ConfigSchemaV1:
    """创建 schema 默认实例."""

    return ConfigSchemaV1()


def create_structured_config(schema: Optional[ConfigSchemaV1] = None) -> DictConfig:
    """基于 dataclass schema 创建 OmegaConf 结构化配置对象."""

    schema_obj = schema if schema is not None else create_config_schema()
    if OmegaConf is None:
        raise ConfigError("OmegaConf is required to create a structured config")

    return OmegaConf.structured(schema_obj)


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RuntimeConfig",
    "TrainerConfig",
    "XDLContextConfig",
    "ComponentConfig",
    "TransformPipelineConfig",
    "DatasetConfig",
    "DataloaderConfig",
    "DataloaderDefaultsConfig",
    "OptimizationConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "ConfigSchemaV1",
    "create_config_schema",
    "DeepSpeedConfig",
    "create_structured_config",
]
