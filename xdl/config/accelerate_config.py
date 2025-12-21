"""
Accelerate 配置管理
用于统一管理 Accelerate 相关的配置参数
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any


@dataclass
class AccelerateConfig:
    """
    Accelerate 配置类

    基于 Hugging Face Accelerate 的核心配置选项
    """
    # 基础配置
    mixed_precision: Optional[str] = None  # 'no', 'fp16', 'bf16', 'fp8'
    gradient_accumulation_steps: int = 1
    cpu: bool = False

    # 设备和分布式
    device_placement: bool = True
    split_batches: bool = False

    # 日志和实验跟踪
    log_with: Optional[Union[str, List[str]]] = None
    project_dir: Optional[str] = None
    project_config: Optional[Dict[str, Any]] = None

    # 优化和编译
    dynamo_backend: Optional[str] = None
    dynamo_plugin: Optional[str] = None

    # DeepSpeed 和 FSDP
    deepspeed_plugin: Optional[Union[str, Dict[str, Any]]] = None
    fsdp_plugin: Optional[str] = None

    # 梯度累积
    gradient_accumulation_plugin: Optional[str] = None
    step_scheduler_with_optimizer: bool = True

    # 随机数生成器同步
    rng_types: Optional[List[str]] = None

    # 其他高级选项
    kwargs_handlers: Optional[List[Dict[str, Any]]] = None
    dataloader_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后的验证和默认值设置"""
        # 验证 mixed_precision 参数
        valid_precision = ['no', 'fp16', 'bf16', 'fp8', None]
        if self.mixed_precision not in valid_precision:
            raise ValueError(f"mixed_precision 必须是 {valid_precision} 中的一个")

        # 验证 gradient_accumulation_steps
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 必须大于等于 1")

        # 设置默认的 rng_types
        if self.rng_types is None:
            self.rng_types = ["torch"]


@dataclass
class DistributedConfig:
    """
    分布式训练配置
    """
    # 基本分布式设置
    use_distributed: bool = False
    backend: str = "nccl"  # 'nccl', 'gloo', 'mpi'

    # 多 GPU 相关
    num_processes: int = 1
    process_index: int = 0

    # FSDP 相关
    fsdp_config: Optional[Dict[str, Any]] = None
    sharding_strategy: str = "FULL_SHARD"  # 'NO_SHARD', 'SHARD_GRAD_OP', 'FULL_SHARD'

    # DeepSpeed 相关
    deepspeed_config: Optional[Dict[str, Any]] = None

    # ZeRO 优化
    zero_optimization: bool = False
    zero_stage: int = 2


@dataclass
class LoggingConfig:
    """
    Accelerate 日志配置
    """
    # 实验跟踪
    log_with: Optional[Union[str, List[str]]] = None
    project_name: str = "accelerate-experiment"
    project_dir: Optional[str] = "./others/logs"

    # 日志频率
    log_every_n_steps: int = 50

    # TensorBoard
    enable_tensorboard: bool = True
    tensorboard_log_dir: Optional[str] = None

    # WandB
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # 其他跟踪器
    enable_aim: bool = False
    enable_comet_ml: bool = False
    enable_mlflow: bool = False

    def get_loggers(self) -> List[str]:
        """获取启用的日志器列表"""
        loggers = []

        if self.enable_tensorboard:
            loggers.append("tensorboard")
        if self.enable_wandb and self.wandb_project:
            loggers.append("wandb")
        if self.enable_aim:
            loggers.append("aim")
        if self.enable_comet_ml:
            loggers.append("comet_ml")
        if self.enable_mlflow:
            loggers.append("mlflow")

        return loggers