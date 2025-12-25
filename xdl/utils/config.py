"""
基于 OmegaConf 的配置系统
支持 Dataclass 结构化校验、变量插值、点号访问及层级化管理
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from omegaconf import OmegaConf, DictConfig

__all__ = [
    "TransformConfig",
    "TransformPipelineConfig",
    "DatasetConfig",
    "DataloaderConfig",
    "DataConfig",
    "ModelConfig",
    "ModelGroupConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "MetricConfig",
    "CoreConfig",
    "SaveLoadConfig",
    "TrainingConfig",
    "LoggerConfig",
    "TrainingConfigManager",
]

# --- 基础配置类 (使用 OmegaConf 结构化配置) ---

@dataclass
class TransformConfig:
    name: str = "Resize"
    params: Dict[str, Any] = field(default_factory=dict)
    from_library: Optional[str] = "torchvision"

@dataclass
class TransformPipelineConfig:
    transforms: List[TransformConfig] = field(default_factory=list)
    from_library: Optional[str] = "torchvision"
    combination_strategy: str = "compose"

@dataclass
class DatasetConfig:
    name: str = "MNIST"
    from_library: Optional[str] = "torchvision"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataloaderConfig:
    params: Dict[str, Any] = field(default_factory=dict)
    combination_strategy: Optional[str] = None

@dataclass
class DataConfig:
    dataset: Dict[str, DatasetConfig] = field(default_factory=dict)
    transform: Dict[str, TransformPipelineConfig] = field(default_factory=dict)
    dataloader: Dict[str, DataloaderConfig] = field(default_factory=dict)

@dataclass
class ModelConfig:
    name: str = "resnet18"
    from_library: Optional[str] = "local"
    params: Dict[str, Any] = field(default_factory=dict)
    weight_init: str = "default"
    load_ckpt: Optional[str] = None

@dataclass
class ModelGroupConfig:
    backbone: Optional[ModelConfig] = None
    head: Optional[ModelConfig] = None
    neck: Optional[ModelConfig] = None
    extra: Dict[str, ModelConfig] = field(default_factory=dict)

@dataclass
class OptimizerConfig:
    name: str = "Adam"
    from_library: Optional[str] = "torch"
    params: Dict[str, Any] = field(default_factory=dict)
    model: List[str] = field(default_factory=lambda: ["backbone"])

@dataclass
class SchedulerConfig:
    name: str = "StepLR"
    optimizer: str = "main_optimizer"
    from_library: Optional[str] = "torch"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LossConfig:
    name: str = "CrossEntropyLoss"
    from_library: Optional[str] = "torch"
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

@dataclass
class MetricConfig:
    name: str = "Accuracy"
    from_library: Optional[str] = "local"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoreConfig:
    model: ModelGroupConfig = field(default_factory=ModelGroupConfig)
    optimizer: Dict[str, OptimizerConfig] = field(default_factory=dict)
    scheduler: Dict[str, SchedulerConfig] = field(default_factory=dict)
    loss: List[LossConfig] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)

@dataclass
class SaveLoadConfig:
    base_dir: str = "./others"
    exp_name: str = "experiment"
    onlymodel: bool = True
    save_models: List[str] = field(default_factory=list)
    load_map_device: Optional[str] = None
    single_file: bool = True
    common_save_folder: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best: bool = True
    # 其他默认值...

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    enable_train_metrics: bool = True
    enable_valid_loss: bool = True
    validate_every: int = 1
    main_metric: Optional[str] = None
    mixed_precision: Optional[str] = None
    gradient_accumulation_steps: int = 1

@dataclass
class LoggerConfig:
    experiment_name: str = "experiment"
    log_dir: str = "others"
    enable_console: bool = True
    enable_file: bool = True
    enable_tensorboard: bool = True

# --- 配置管理器 (OmegaConf 核心封装) ---

@dataclass
class TrainingConfigManager:
    """
    基于 OmegaConf 的通用配置管理器
    支持自动类型转换、默认值合并以及 YAML 变量插值
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    core_config: CoreConfig = field(default_factory=CoreConfig)
    data_config: Optional[DataConfig] = None
    save_config: SaveLoadConfig = field(default_factory=SaveLoadConfig)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> Any:
        """从 YAML 加载并合并默认配置"""
        # 1. 创建结构化基础配置 (包含默认值)
        base_cfg = OmegaConf.structured(cls)
        
        # 2. 加载用户 YAML 配置
        user_cfg = OmegaConf.load(filepath)
        
        # 3. 合并配置 (用户配置覆盖默认配置)
        cfg = OmegaConf.merge(base_cfg, user_cfg)
        
        # 4. 返回配置对象 (支持点号访问和插值)
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """从字典创建配置"""
        base_cfg = OmegaConf.structured(cls)
        user_cfg = OmegaConf.create(data)
        return OmegaConf.merge(base_cfg, user_cfg)

    @staticmethod
    def to_dict(cfg: DictConfig) -> Any:
        """将 DictConfig 转换为纯字典 (解析所有插值)"""
        return OmegaConf.to_container(cfg, resolve=True)

    @staticmethod
    def save_to_yaml(cfg: DictConfig, filepath: str):
        """保存配置到 YAML"""
        with open(filepath, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)

    @staticmethod
    def save_to_json(cfg: DictConfig, filepath: str):
        """保存配置到 JSON"""
        container = OmegaConf.to_container(cfg, resolve=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(container, f, ensure_ascii=False, indent=2)
