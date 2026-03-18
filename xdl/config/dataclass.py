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
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module

    # 可选组件
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    scheduler: Optional[Any] = None

    # 指标
    metrics: List[Any] = field(default_factory=list)
    
    # 完整配置（用于调试或自定义逻辑）
    full_config: Dict[str, Any] = field(default_factory=dict)
    
    # 训练参数
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 128
    
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
