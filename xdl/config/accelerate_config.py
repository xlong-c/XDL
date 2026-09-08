"""
Accelerate 配置管理
用于统一管理 Accelerate 相关的配置参数
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from xdl.errors import TrainingError


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
    fsdp_plugin: Optional[Any] = None

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
        valid_precision = ["no", "fp16", "bf16", "fp8", None]
        if self.mixed_precision not in valid_precision:
            raise ValueError(f"mixed_precision 必须是 {valid_precision} 中的一个")

        # 验证 gradient_accumulation_steps
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 必须大于等于 1")

        # 设置默认的 rng_types
        if self.rng_types is None:
            self.rng_types = ["torch"]


@dataclass
class DeepSpeedConfig:
    """原生 DeepSpeed 配置 - 对应 deepspeed.initialize() 的 config 参数.

    支持两种使用方式:
    1. 指定 config_path 引用外部 JSON
    2. 直接在 YAML 中内联配置(zero_stage / fp16 / bf16 等)
    """

    enabled: bool = False
    config_path: Optional[str] = None
    zero_stage: int = 2
    fp16: bool = True
    bf16: bool = False
    gradient_accumulation_steps: Optional[int] = None
    train_batch_size: Optional[int] = None
    train_micro_batch_size_per_gpu: Optional[int] = None
    gradient_clipping: Optional[float] = None
    offload_optimizer: bool = False
    offload_param: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepSpeedConfig":
        """从 YAML 解析的字典构建 DeepSpeed 配置."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_initialize_kwargs(self) -> Dict[str, Any]:
        """生成 deepspeed.initialize(config=...) 所需的配置字典或路径.

        直接返回 config 的值(dict 或 str 路径),
        调用方用 deepspeed.initialize(model=..., optimizer=..., config=result).

        如果指定了 config_path,返回路径字符串;
        否则根据内联字段自动生成 DeepSpeed 配置字典.
        """
        if self.config_path:
            return {"config": self.config_path}

        # batch_size 三要素:指定两个,第三个自动推断
        # 至少需要 train_micro_batch_size_per_gpu
        ds_config: Dict[str, Any]
        if self.train_micro_batch_size_per_gpu is None:
            ds_config = {
                "train_micro_batch_size_per_gpu": 1,
            }
        else:
            ds_config = {
                "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            }

        if self.train_batch_size is not None:
            ds_config["train_batch_size"] = self.train_batch_size
        if self.gradient_accumulation_steps is not None:
            ds_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps

        zero_optimization: Dict[str, Any] = {"stage": self.zero_stage}
        ds_config["zero_optimization"] = zero_optimization

        if self.fp16:
            ds_config["fp16"] = {"enabled": True}
        elif self.bf16:
            ds_config["bf16"] = {"enabled": True}

        if self.gradient_clipping is not None:
            ds_config["gradient_clipping"] = self.gradient_clipping

        if self.zero_stage >= 2 and self.offload_optimizer:
            zero_optimization["offload_optimizer"] = {"device": "cpu"}
        if self.zero_stage == 3 and self.offload_param:
            zero_optimization["offload_param"] = {"device": "cpu"}

        return {"config": ds_config}

    def is_available(self) -> bool:
        """检查 deepspeed 是否可导入."""
        try:
            import deepspeed  # noqa: F401
            return True
        except ImportError:
            return False


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


@dataclass
class FSDPConfig:
    """结构化 FSDP 配置, 对应 ``FullyShardedDataParallelPlugin`` 的常用字段.

    ``sharding_strategy`` 只做语义映射 (FULL_SHARD -> reshard_after_forward=True),
    构造插件时统一转成 accelerate 1.14+ 推荐的 ``reshard_after_forward``.
    ``auto_wrap_policy`` 支持 ``None`` 或 ``"transformer_based_wrap"``;
    transformer 按 block 自动 wrap 时通过 ``transformer_cls_names_to_wrap``
    指定模块类名, 插件会按名称在模型内解析并生成 wrap policy.
    """

    fsdp_version: Optional[int] = None  # 1 或 2; None 表示由 accelerate 默认
    sharding_strategy: str = "FULL_SHARD"
    reshard_after_forward: Optional[bool] = None
    auto_wrap_policy: Optional[str] = None  # None 或 "transformer_based_wrap"
    transformer_cls_names_to_wrap: Optional[List[str]] = None
    min_num_params: Optional[int] = None
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    sync_module_states: Optional[bool] = None
    state_dict_type: Optional[str] = None  # "FULL_STATE_DICT"/"SHARDED_STATE_DICT"
    backward_prefetch: Optional[str] = None

    def to_plugin_kwargs(self) -> Dict[str, Any]:
        """转成 ``FullyShardedDataParallelPlugin`` 可接受的参数字典."""
        kwargs: Dict[str, Any] = {
            "limit_all_gathers": self.limit_all_gathers,
            "use_orig_params": self.use_orig_params,
            "cpu_offload": self.cpu_offload,
            "activation_checkpointing": self.activation_checkpointing,
            "auto_wrap_policy": self.auto_wrap_policy,
            "transformer_cls_names_to_wrap": self.transformer_cls_names_to_wrap,
        }
        # sharding_strategy 在 accelerate 1.14+ 已弃用, 统一映射成
        # reshard_after_forward: FSDP2 用 bool (FULL_SHARD -> True),
        # FSDP1 用字符串/ShardingStrategy.
        if self.reshard_after_forward is not None:
            kwargs["reshard_after_forward"] = self.reshard_after_forward
        elif self.sharding_strategy:
            kwargs["reshard_after_forward"] = (
                self.sharding_strategy == "FULL_SHARD"
                if self.fsdp_version == 2
                else self.sharding_strategy
            )
        if self.fsdp_version is not None:
            kwargs["fsdp_version"] = self.fsdp_version
        if self.fsdp_version == 2:
            # FSDP2 固定使用原始参数, 传 use_orig_params 只会触发弃用警告.
            kwargs.pop("use_orig_params", None)
        if self.reshard_after_forward is not None:
            kwargs["reshard_after_forward"] = self.reshard_after_forward
        if self.min_num_params is not None:
            kwargs["min_num_params"] = self.min_num_params
        if self.sync_module_states is not None:
            kwargs["sync_module_states"] = self.sync_module_states
        if self.state_dict_type is not None:
            kwargs["state_dict_type"] = self.state_dict_type
        if self.backward_prefetch is not None:
            kwargs["backward_prefetch"] = self.backward_prefetch
        return kwargs


_FSDP_PLUGIN_KEYS = {
    "fsdp_version",
    "sharding_strategy",
    "reshard_after_forward",
    "backward_prefetch",
    "mixed_precision_policy",
    "auto_wrap_policy",
    "cpu_offload",
    "ignored_modules",
    "state_dict_type",
    "state_dict_config",
    "optim_state_dict_config",
    "limit_all_gathers",
    "use_orig_params",
    "param_init_fn",
    "sync_module_states",
    "forward_prefetch",
    "activation_checkpointing",
    "cpu_ram_efficient_loading",
    "transformer_cls_names_to_wrap",
    "min_num_params",
}


def build_fsdp_plugin(fsdp: Union[int, Dict[str, Any], FSDPConfig, Any]) -> Any:
    """把 ``Trainer(fsdp=...)`` 的配置构建成 ``FullyShardedDataParallelPlugin``.

    支持:
    - ``fsdp=1`` / ``fsdp=2``: 快捷分支;
    - ``fsdp={...}``: 透传给插件构造函数的字段字典;
    - ``FSDPConfig``: 结构化配置;
    - 已经是插件实例: 原样返回.
    """
    from accelerate import FullyShardedDataParallelPlugin

    if isinstance(fsdp, FullyShardedDataParallelPlugin):
        return fsdp
    if isinstance(fsdp, FSDPConfig):
        return FullyShardedDataParallelPlugin(**fsdp.to_plugin_kwargs())
    if isinstance(fsdp, int):
        if fsdp not in (1, 2):
            raise TrainingError(f"fsdp 只支持 1 或 2, 收到 {fsdp}")
        plugin_kwargs: Dict[str, Any] = {
            "limit_all_gathers": True,
        }
        if fsdp == 2:
            plugin_kwargs.update(
                {
                    "fsdp_version": 2,
                    "reshard_after_forward": True,  # FULL_SHARD
                }
            )
        else:
            plugin_kwargs.update(
                {
                    "reshard_after_forward": "FULL_SHARD",
                    "use_orig_params": True,
                }
            )
        return FullyShardedDataParallelPlugin(**plugin_kwargs)
    if isinstance(fsdp, dict):
        plugin_dict = dict(fsdp)
        unknown = sorted(set(plugin_dict) - _FSDP_PLUGIN_KEYS)
        if unknown:
            raise TrainingError(
                "fsdp 配置包含插件不支持的字段: " + ", ".join(unknown)
            )
        # sharding_strategy 已弃用, 转换成 reshard_after_forward 再透传:
        # FSDP2 用 bool, FSDP1 保留字符串.
        if "sharding_strategy" in plugin_dict:
            strategy = str(plugin_dict.pop("sharding_strategy")).upper()
            if plugin_dict.get("fsdp_version") == 2:
                plugin_dict.setdefault("reshard_after_forward", strategy == "FULL_SHARD")
            else:
                plugin_dict.setdefault("reshard_after_forward", strategy)
        if plugin_dict.get("fsdp_version") == 2:
            plugin_dict.pop("use_orig_params", None)
        return FullyShardedDataParallelPlugin(**plugin_dict)
    raise TrainingError(
        "fsdp 必须是 int(1/2),dict,FSDPConfig 或 FullyShardedDataParallelPlugin"
    )
