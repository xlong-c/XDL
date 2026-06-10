"""
配置数据类定义
定义从 YAML 配置构建的训练组件容器
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainSetup:
    """
    训练配置数据类
    
    封装从 YAML 配置构建的所有训练组件，
    提供类型安全的属性访问。
    
    Example:
        >>> setup = setup_from_yaml('config/vgg_cifar100.yaml')
        >>> model = setup.model
        >>> optimizer = setup.optimizer
        >>> trainer = Trainer(
        ...     model=setup.model,
        ...     train_dataloader=setup.train_loader,
        ...     val_dataloader=setup.val_loader,
        ...     optimizer=setup.optimizer,
        ...     loss_fn=setup.loss_fn,
        ...     metrics=setup.metrics,
        ...     max_epochs=100,
        ... )
        >>> trainer.fit()
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

    # 指标
    metrics: List[Any] = field(default_factory=list)
    callbacks: List[Any] = field(default_factory=list)
    
    # 完整配置（用于调试或自定义逻辑）
    full_config: Dict[str, Any] = field(default_factory=dict)
    
    # 日志 / 检查点 / 加速配置
    logging_config: Dict[str, Any] = field(default_factory=dict)
    checkpoint_config: Dict[str, Any] = field(default_factory=dict)
    accelerate_config: Optional[Dict[str, Any]] = None
    deepspeed_config: Optional[Dict[str, Any]] = None

    # 训练参数
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 128
    precision: Optional[str] = None
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: Optional[float] = None
    grad_clip_norm_type: float = 2.0
    trainer_config: Dict[str, Any] = field(default_factory=dict)
    
    def create_model(self):
        """将外部组件包装为 CoreModel 子类，直接对接 Trainer.fit()。

        Returns:
            TrainSetupModel: 包装后的 CoreModel 实例，内部已配置好
                optimizer / loss_fn / scheduler / metrics。
        """
        from xdl.trainer.coreModel import CoreModel
        from xdl.trainer.trainSetupModel import TrainSetupModel

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
