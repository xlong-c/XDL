"""
配置的Dataclass定义
用于规范化配置结构, 支持层级化配置管理
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
import torch
import yaml
import json

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


@dataclass
class TransformConfig:
    """数据变换配置"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    from_library: Optional[str] = "torchvision"


@dataclass
class TransformPipelineConfig:
    """数据变换流水线配置"""

    transforms: List[TransformConfig]
    from_library: Optional[str] = "torchvision"
    combination_strategy: str = "compose"


@dataclass
class DatasetConfig:
    """数据集配置"""

    name: str
    from_library: Optional[str] = "torchvision"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataloaderConfig:
    """数据加载器配置"""

    params: Dict[str, Any] = field(default_factory=dict)  # batch_size, shuffle等
    combination_strategy: Optional[str] = None  # 'concat'等


@dataclass
class DataConfig:
    """数据相关配置

    简化的对应关系：
    - dataset.train 直接使用 transform.train
    - dataloader.train 直接对应 dataset.train
    - 同理适用于 valid 和 test
    """

    dataset: Dict[str, DatasetConfig] = field(
        default_factory=dict
    )  # train, valid, test
    transform: Dict[str, TransformPipelineConfig] = field(
        default_factory=dict
    )  # train, valid, test,collect_fn
    dataloader: Dict[str, DataloaderConfig] = field(
        default_factory=dict
    )  # train, valid, test


@dataclass
class ModelConfig:
    """模型配置"""

    name: str
    from_library: Optional[str] = "local"  # 'local', 'torchvision', 'timm', 'torch'
    params: Dict[str, Any] = field(default_factory=dict)
    weight_init: str = "default"  # 'default', 'no', 'cnn', 'transformer', 或自定义初始化类型
    load_ckpt: Optional[str] = None


@dataclass
class ModelGroupConfig:
    """模型组配置"""

    backbone: Optional[ModelConfig] = None # 默认使用backbone进行模型forward
    head: Optional[ModelConfig] = None
    neck: Optional[ModelConfig] = None
    # 可以通过字典添加其他模型组件
    extra: Dict[str, ModelConfig] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """优化器配置"""

    name: str
    from_library: Optional[str] = "torch"  # 'torch', 'local'
    params: Dict[str, Any] = field(default_factory=dict)  # lr, weight_decay等
    model: List[str] = field(default_factory=lambda: ["backbone"])  # 要优化的模型名称


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""

    name: str
    optimizer: str  # 对应的优化器名称
    from_library: Optional[str] = "torch"  # 'torch', 'local'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """损失函数配置"""

    name: str
    from_library: Optional[str] = "torch"  # 'torch', 'local'
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class MetricConfig:
    """评估指标配置"""

    name: str
    from_library: Optional[str] = "local"  # 'local', 'torch'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreConfig:
    """核心组件配置"""

    model: ModelGroupConfig = field(default_factory=ModelGroupConfig)
    optimizer: Dict[str, OptimizerConfig] = field(default_factory=dict)
    scheduler: Dict[str, SchedulerConfig] = field(default_factory=dict)
    loss: List[LossConfig] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)


@dataclass
class SaveLoadConfig:
    """
    保存和加载配置
    """

    # 基本配置
    base_dir: str = "./result"
    exp_name: str = "v1"
    onlymodel: bool = True  # 是否只保存模型, 优先级高于内容配置
    save_models: List[str] = field(
        default_factory=list
    )  # 只保存指定名称的模型, 空则保存所有
    load_map_device: Optional[Union[str, torch.device]] = None
    single_file: bool = True  # 是否使用单个文件保存所有内容
    common_save_folder: str = "checkpoints"
    specific_save_folder: str = "specific_saves"
    main_metric: Optional[str] = None

    # 运行中保存配置
    save_every_n_steps: int = int(1e12)
    save_every_n_epochs: int = 1
    max_latest_checkpoints: int = -1  # 最大保留的最新检查点数量(时间序列), -1表示无限制
    max_best_checkpoints: int = -1  # 最大保留的最佳检查点数量(基于指标值), -1表示无限制
    save_best: bool = True
    save_best_only: bool = False
    best_metric: Optional[str] = None
    maximize_best_metric: bool = True

    # 保存内容配置
    save_models_state: bool = True
    save_optimizers_state: bool = True
    save_schedulers_state: bool = True
    save_training_state: bool = True  # step_count, epoch_count, loss_weights等
    save_stats: bool = False  # 是否保存其他的数据
    save_scaler_state: bool = True  # 是否保存GradScaler状态

    # 加载配置
    load_models_state: bool = True
    load_optimizers_state: bool = True
    load_schedulers_state: bool = True
    load_training_state: bool = True
    load_stats: bool = False

    # 文件配置
    compression: bool = False  # 是否压缩保存
    save_format: str = "pth"  # 保存格式: "pth", "safetensors"

    # 单独文件保存路径
    models_path: Optional[str] = None
    optimizers_path: Optional[str] = None
    schedulers_path: Optional[str] = None
    stats_path: Optional[str] = None
    load_models_path: Optional[str] = None
    load_optimizers_path: Optional[str] = None
    load_schedulers_path: Optional[str] = None
    load_stats_path: Optional[str] = None

    @property
    def save_dir(self) -> str:
        """获取完整的保存目录路径"""
        return f"{self.base_dir}/{self.exp_name}"


@dataclass
class TrainingConfig:
    """训练配置 - 使用 Accelerate"""

    num_epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    # 训练阶段是否计算并记录metric
    enable_train_metrics: bool = True
    # 验证阶段是否计算并记录loss
    enable_valid_loss: bool = True
    validate_every: int = 1  # 每几个epoch验证一次
    validate_epoch: int = 1  # 从第几个epoch开始验证
    main_metric: Optional[str] = None  # 用于保存最佳模型的主指标名称

    # Accelerate 核心配置
    mixed_precision: Optional[str] = None  # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: Optional[float] = None  # 梯度裁切的最大范数, None表示不裁切
    grad_clip_norm_type: float = 2.0  # 梯度裁切的范数类型, 默认L2


@dataclass
class LoggerConfig:
    """
    统一的日志系统配置类.

    Attributes:
        experiment_name (str): 实验名称, 用于生成日志文件和TensorBoard目录.
        log_dir (str): 日志文件和TensorBoard日志的根目录. 默认为 "others".
        log_step_to_file (bool): 是否在日志文件中记录每一步(step)的详细信息. 默认为 True.
        log_step_to_tensorboard (bool): 是否在TensorBoard中记录每一步(step)的详细信息. 默认为 True.
        enable_console (bool): 是否启用控制台日志. 默认为 True.
        enable_file (bool): 是否启用文件日志. 默认为 True.
        enable_tensorboard (bool): 是否启用TensorBoard日志. 默认为 True.
        console_level (str): 控制台日志级别 (e.g., "INFO", "DEBUG"). 默认为 "INFO".
        console_log_with_timestamp (bool): 是否在控制台日志的每行前添加时间戳. 默认为 True.
        log_file_name (Optional[str]): 指定日志文件名. 如果为None, 则自动生成. 默认为 None.
        log_file_with_timestamp (bool): 是否在文件日志的每行前添加时间戳. 默认为 False.

        # 新增多格式支持
        enable_csv (bool): False # 是否启用CSV格式日志记录
        enable_txt (bool): False  # 是否启用TXT格式日志记录
        enable_wandb (bool): False # 是否启用WandB云平台日志记录

        # CSV配置
        csv_log_dir: Optional[str] = None  # CSV日志目录, 默认为log_dir/csv
        csv_file_name: Optional[str] = None # CSV文件名, 默认自动生成
        csv_flush_interval: int = 10        # CSV写入刷新间隔(行数)

        # TXT配置
        txt_log_dir: Optional[str] = None   # TXT日志目录, 默认为log_dir/txt
        txt_file_name: Optional[str] = None  # TXT文件名, 默认自动生成
        txt_max_file_size: int = 10 * 1024 * 1024  # TXT文件最大大小(字节)

        # WandB配置
        wandb_project: Optional[str] = None     # WandB项目名称
        wandb_entity: Optional[str] = None      # WandB实体名称
        wandb_run_name: Optional[str] = None    # WandB运行名称
        wandb_tags: Optional[List[str]] = None  # WandB标签
        wandb_notes: Optional[str] = None       # WandB运行说明
    """

    experiment_name: str = "experiment"
    log_dir: str = "others"
    log_step_to_file: bool = True
    log_step_to_tensorboard: bool = True
    enable_console: bool = True
    enable_file: bool = True
    enable_tensorboard: bool = True
    console_level: str = "INFO"
    console_log_with_timestamp: bool = True
    log_file_name: Optional[str] = None
    log_file_with_timestamp: bool = False

    # 新增多格式支持
    enable_csv: bool = False
    enable_txt: bool = False
    enable_wandb: bool = False

    # CSV配置
    csv_log_dir: Optional[str] = None
    csv_file_name: Optional[str] = None
    csv_flush_interval: int = 10

    # TXT配置
    txt_log_dir: Optional[str] = None
    txt_file_name: Optional[str] = None
    txt_max_file_size: int = 10 * 1024 * 1024  # 10MB

    # WandB配置
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None


@dataclass
class TrainingConfigManager:
    """通用训练配置管理器"""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    data: Optional[DataConfig] = None
    save: SaveLoadConfig = field(default_factory=SaveLoadConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    def save_to_yaml(self, filepath: str):
        """保存配置到YAML文件"""
        config_dict = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def save_to_json(self, filepath: str):
        """保存配置到JSON文件"""
        config_dict = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_yaml(cls, filepath: str):
        """从YAML文件加载配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_from_json(cls, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """递归将配置对象转换为可序列化的字典结构

        Args:
            obj: 要转换的对象, 可以是dataclass、字典、列表或基本类型

        Returns:
            转换后的可序列化对象
        """
        # 处理 None 值
        if obj is None:
            return None

        # 处理基本类型(字符串、数字、布尔值)
        elif isinstance(obj, (str, int, float, bool)):
            return obj

        # 处理 dataclass 对象
        elif hasattr(obj, '__dataclass_fields__'):
            return {key: self._convert_to_serializable(value)
                   for key, value in asdict(obj).items()}

        # 处理字典
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}

        # 处理列表或元组
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]

        # 处理其他对象(尝试转换为字典)
        elif hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)

        # 其他类型转换为字符串
        else:
            return str(obj)

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        result = {
            "training": self.training,
            "data_config": self.data,
            "core_config": self.core,
            "save_config": self.save,
            "logger_config": self.logger,
        }

        # 移除None值并递归转换
        return {k: self._convert_to_serializable(v) for k, v in result.items() if v is not None}

    def _build_config_from_dict(self, config_class: type, config_dict: Dict[str, Any]) -> Any:
        """通用的配置对象构建方法

        Args:
            config_class: 目标配置类
            config_dict: 配置字典

        Returns:
            构建好的配置对象
        """
        if not config_dict:
            return None

        # 处理特殊情况的配置类
        if config_class == DataConfig:
            return self._build_data_config(config_dict)
        elif config_class == CoreConfig:
            return self._build_core_config(config_dict)
        else:
            # 直接构造简单的dataclass
            return config_class(**config_dict)

    def _build_data_config(self, data_dict: Dict[str, Any]) -> DataConfig:
        """构建数据配置"""
        # 构建数据集配置
        datasets = {}
        if "dataset" in data_dict:
            datasets = {
                name: DatasetConfig(**cfg) for name, cfg in data_dict["dataset"].items()
            }

        # 构建变换配置
        transforms = {}
        if "transform" in data_dict:
            transforms = {
                name: TransformPipelineConfig(
                    transforms=[TransformConfig(**t) for t in cfg["transforms"]],
                    from_library=cfg.get("from_library", "torchvision"),
                    combination_strategy=cfg.get("combination_strategy", "compose"),
                )
                for name, cfg in data_dict["transform"].items()
            }

        # 构建数据加载器配置
        dataloaders = {}
        if "dataloader" in data_dict:
            dataloaders = {
                name: DataloaderConfig(**cfg) for name, cfg in data_dict["dataloader"].items()
            }

        return DataConfig(
            dataset=datasets,
            transform=transforms,
            dataloader=dataloaders
        )

    def _build_core_config(self, core_dict: Dict[str, Any]) -> CoreConfig:
        """构建核心配置"""
        # 构建模型配置
        model_group = ModelGroupConfig()
        if "model" in core_dict:
            model_dict = core_dict["model"]

            # 处理主要模型组件
            for part in ["backbone", "head", "neck"]:
                if part in model_dict and model_dict[part] is not None:
                    setattr(model_group, part, ModelConfig(**model_dict[part]))

            # 处理额外模型
            if "extra" in model_dict:
                model_group.extra = {
                    name: ModelConfig(**cfg) for name, cfg in model_dict["extra"].items()
                }

        # 构建优化器配置
        optimizers = {}
        if "optimizer" in core_dict:
            optimizers = {
                name: OptimizerConfig(**cfg) for name, cfg in core_dict["optimizer"].items()
            }

        # 构建调度器配置
        schedulers = {}
        if "scheduler" in core_dict:
            schedulers = {
                name: SchedulerConfig(**cfg) for name, cfg in core_dict["scheduler"].items()
            }

        # 构建损失函数配置
        losses = []
        if "loss" in core_dict:
            losses = [LossConfig(**cfg) for cfg in core_dict["loss"]]

        # 构建指标配置
        metrics = []
        if "metrics" in core_dict:
            metrics = [MetricConfig(**cfg) for cfg in core_dict["metrics"]]

        return CoreConfig(
            model=model_group,
            optimizer=optimizers,
            scheduler=schedulers,
            loss=losses,
            metrics=metrics,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置"""
        # 使用字典映射配置类
        config_mapping = {
            "training": TrainingConfig,
            "data_config": DataConfig,
            "core_config": CoreConfig,
            "save_config": SaveLoadConfig,
            "logger_config": LoggerConfig,
        }

        # 创建实例以便调用辅助方法
        instance = cls.__new__(cls)

        # 构建各个配置
        configs = {}
        for key, config_class in config_mapping.items():
            config_data = config_dict.get(key)
            if config_data is not None:
                configs[key.replace("_config", "")] = instance._build_config_from_dict(config_class, config_data)

        return cls(**configs)
