"""
配置 schema 定义。

固定上层结构，底层组件参数保持在 `params` 中。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from .accelerate_config import AccelerateConfig
from .errors import ConfigError

CONFIG_SCHEMA_VERSION = 1
DEFAULT_COMPONENT_SOURCE = "registry"


@dataclass
class RuntimeConfig:
    """运行时配置。"""

    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./others"
    experiment_name: str = "default_exp"


@dataclass
class TrainerConfig:
    """训练主循环配置。"""

    max_epochs: int = 100
    batch_size: Optional[int] = None
    precision: str = "32"
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: Optional[float] = None


@dataclass
class ComponentConfig:
    """通用组件配置。"""

    type: str = ""
    source: str = DEFAULT_COMPONENT_SOURCE
    params: Dict[str, Any] = field(default_factory=dict)
    target_modules: Optional[List[str]] = None
    param_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight: Optional[float] = None


@dataclass
class TransformPipelineConfig:
    """数据增强流水线配置。"""

    type: str = "compose"
    items: List[ComponentConfig] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """数据集配置。"""

    type: str = ""
    source: str = DEFAULT_COMPONENT_SOURCE
    params: Dict[str, Any] = field(default_factory=dict)
    transform: Any = None


@dataclass
class DataloaderConfig:
    """DataLoader 配置。"""

    dataset: Any = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """数据相关配置。"""

    transforms: Dict[str, TransformPipelineConfig] = field(default_factory=dict)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    dataloaders: Dict[str, DataloaderConfig] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """优化相关配置。"""

    optimizer: Optional[ComponentConfig] = None
    scheduler: Optional[ComponentConfig] = None


@dataclass
class LoggingConfig:
    """日志配置。"""

    log_dir: str = "./others/logs"
    enable_console: bool = True
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class CheckpointConfig:
    """检查点配置。"""

    save_last: bool = True
    save_top_k: int = 1
    monitor: Optional[str] = None
    mode: str = "min"
    every_n_epochs: int = 1
    dirpath: Optional[str] = None


@dataclass
class ConfigSchemaV1:
    """XDL 配置 schema v1。"""

    config_version: int = CONFIG_SCHEMA_VERSION
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Optional[ComponentConfig] = None
    data: DataConfig = field(default_factory=DataConfig)
    optimization: Optional[OptimizationConfig] = None
    loss: List[ComponentConfig] = field(default_factory=list)
    metrics: List[ComponentConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    accelerate: Optional[AccelerateConfig] = None


def create_config_schema() -> ConfigSchemaV1:
    """创建 schema 默认实例。"""

    return ConfigSchemaV1()


def create_structured_config(schema: Optional[ConfigSchemaV1] = None) -> DictConfig:
    """基于 dataclass schema 创建 OmegaConf 结构化配置对象。"""

    schema_obj = schema if schema is not None else create_config_schema()
    if OmegaConf is None:
        raise ConfigError("OmegaConf is required to create a structured config")

    return OmegaConf.structured(schema_obj)


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_COMPONENT_SOURCE",
    "RuntimeConfig",
    "TrainerConfig",
    "ComponentConfig",
    "TransformPipelineConfig",
    "DatasetConfig",
    "DataloaderConfig",
    "DataConfig",
    "OptimizationConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "ConfigSchemaV1",
    "create_config_schema",
    "create_structured_config",
]
