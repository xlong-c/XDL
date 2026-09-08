"""
配置数据类定义
定义从 YAML 配置构建的训练组件容器
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .accelerate_config import AccelerateConfig, DeepSpeedConfig
from .schema import CheckpointConfig, LoggingConfig, RuntimeConfig, TrainerConfig


@dataclass
class TrainSetup:
    """
    训练配置数据类.

    封装从 YAML 配置构建的训练组件和结构化配置. 训练参数只保留一份事实源:

    - `trainer`: `TrainerConfig`, 含 `max_epochs`, `batch_size`, `precision`,
      `gradient_accumulation_steps`, 梯度裁剪, `fsdp`, `nan_monitor` 等
    - `runtime`: `RuntimeConfig`, 含 `device`, `seed`, `data_dir`, `output_dir`
    - `logging` / `checkpoint`: 日志与检查点配置
    - `accelerate` / `deepspeed`: 可选加速配置, 未配置时为 `None`

    `trainer.batch_size` 在 `setup_from_yaml()` 中会被替换为解析后的
    DataLoader batch size.

    Example:
        >>> setup = setup_from_yaml('config/vgg_cifar100.yaml')
        >>> model = setup.create_model()
        >>> trainer = Trainer.from_setup(setup)
        >>> trainer.fit(model, setup.train_loader, setup.val_loader)
    """

    # 核心组件
    model: torch.nn.Module
    train_loader: DataLoader
    optimizer: Optional[torch.optim.Optimizer] = None
    loss_fn: Optional[torch.nn.Module] = None

    # 可选组件
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    scheduler: Optional[Any] = None

    # 指标与回调
    metrics: List[Any] = field(default_factory=list)
    callbacks: List[Any] = field(default_factory=list)

    # 结构化配置(唯一事实源)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    accelerate: Optional[AccelerateConfig] = None
    deepspeed: Optional[DeepSpeedConfig] = None

    # 完整配置(用于调试或自定义逻辑)
    full_config: Dict[str, Any] = field(default_factory=dict)

    def create_model(self):
        """将外部组件包装为 CoreModel 子类,直接对接 Trainer.fit().

        Returns:
            TrainSetupModel: 包装后的 CoreModel 实例,内部已配置好
                optimizer / loss_fn / scheduler / metrics.
        """
        from xdl.trainer.core_model import CoreModel
        from xdl.config.train_setup_model import TrainSetupModel

        if isinstance(self.model, CoreModel):
            return self.model
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError(
                "TrainSetup.create_model() requires optimizer and loss_fn for "
                "plain torch.nn.Module models. Use task.target for CoreModel tasks."
            )
        return TrainSetupModel(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            scheduler=self.scheduler,
            metrics=self.metrics,
        )

    def __repr__(self) -> str:
        """友好的字符串表示"""
        return (
            f"TrainSetup(\n"
            f"  model={type(self.model).__name__},\n"
            f"  train_loader={type(self.train_loader).__name__},\n"
            f"  val_loader={type(self.val_loader).__name__ if self.val_loader else None},\n"
            f"  optimizer={type(self.optimizer).__name__},\n"
            f"  scheduler={type(self.scheduler).__name__ if self.scheduler else None},\n"
            f"  loss_fn={type(self.loss_fn).__name__},\n"
            f"  metrics={[type(m).__name__ for m in self.metrics]},\n"
            f")"
        )
