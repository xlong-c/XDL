"""
XDL 配置解析模块
提供从 YAML 配置文件构建训练组件的功能
"""

# 训练组件装配结果容器,统一承载 model / dataloader / optimizer 等对象.
from .dataclass import TrainSetup

# 配置层统一异常定义,便于上层按错误类型做校验与捕获.
from .errors import (
    ComponentResolutionError,
    ConfigError,
    ConfigInterpolationError,
    ConfigValidationError,
    UnusedConfigWarning,
)

# 配置解析工具:负责 schema merge,插值解析和普通 dict 转换.
from .resolver import load_config_with_schema, merge_with_schema, resolve_config, to_plain_dict

# Accelerate/FSDP 配置辅助.
from .accelerate_config import FSDPConfig, build_fsdp_plugin

# 配置 schema 与相关 dataclass:固定顶层结构,默认值和公共配置骨架.
from .schema import (
    CONFIG_SCHEMA_VERSION,
    CheckpointConfig,
    ComponentConfig,
    ConfigSchemaV1,
    DataloaderConfig,
    DataloaderDefaultsConfig,
    DatasetConfig,
    LoggingConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainerConfig,
    TransformPipelineConfig,
    create_config_schema,
    create_structured_config,
)

# 高层入口:从 YAML 一次性完成配置解析和组件构建.
from .setup import setup_from_yaml
from .structured import load_structured_dataclass_config

# 配置流到 CoreModel 的适配器 (转化器).
from .train_setup_model import TrainSetupModel

# 对外暴露的稳定公共 API,避免把内部实现细节直接泄漏给调用方.
__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RuntimeConfig",
    "TrainerConfig",
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
    "create_structured_config",
    "ConfigError",
    "ConfigValidationError",
    "ConfigInterpolationError",
    "ComponentResolutionError",
    "UnusedConfigWarning",
    "merge_with_schema",
    "resolve_config",
    "to_plain_dict",
    "load_config_with_schema",
    "load_structured_dataclass_config",
    "setup_from_yaml",
    "TrainSetup",
    "TrainSetupModel",
    "FSDPConfig",
    "build_fsdp_plugin",
]
