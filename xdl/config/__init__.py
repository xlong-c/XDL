"""
XDL 配置解析模块
提供从 YAML 配置文件构建训练组件的功能
"""

from .dataclass import TrainSetup
from .errors import (
    ComponentResolutionError,
    ConfigError,
    ConfigInterpolationError,
    ConfigValidationError,
    UnusedConfigWarning,
)
from .resolver import load_config_with_schema, merge_with_schema, resolve_config, to_plain_dict
from .schema import (
    CONFIG_SCHEMA_VERSION,
    CheckpointConfig,
    ComponentConfig,
    ConfigSchemaV1,
    DataConfig,
    DataloaderConfig,
    DatasetConfig,
    LoggingConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainerConfig,
    TransformPipelineConfig,
    create_config_schema,
    create_structured_config,
)
from .setup import setup_from_yaml

__all__ = [
    "CONFIG_SCHEMA_VERSION",
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
    "ConfigError",
    "ConfigValidationError",
    "ConfigInterpolationError",
    "ComponentResolutionError",
    "UnusedConfigWarning",
    "merge_with_schema",
    "resolve_config",
    "to_plain_dict",
    "load_config_with_schema",
    "setup_from_yaml",
    "TrainSetup",
]
