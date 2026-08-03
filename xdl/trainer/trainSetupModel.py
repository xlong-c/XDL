"""
适配层：将 TrainSetup 的外部组件包装为 CoreModel 子类，
使 YAML 配置流可以直接对接 Trainer.fit()。
"""

from typing import Any, Dict, List, Optional

import torch

from .coreModel import CoreModel


class TrainSetupModel(CoreModel):
    """将外部 nn.Module + optimizer + loss_fn + scheduler + metrics 包装为 CoreModel。

    Trainer.fit() 期望一个 CoreModel 子类，其 training_step 内部自行处理
    前向/反向/优化器步进/日志。本适配器接收 YAML 配置流构建的外部组件，
    将它们桥接到 CoreModel 的手动优化接口。

    Usage:
        setup = setup_from_yaml('config/vgg_cifar100.yaml')
        model = setup.create_model()  # 返回 TrainSetupModel 实例
        trainer.fit(model, setup.train_loader, setup.val_loader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
    ):
        super().__init__()
        self.model = model
        self._external_optimizer = optimizer
        self.loss_fn = loss_fn
        self._external_scheduler = scheduler
        self._metrics: List[Any] = metrics or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __call__(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(batch, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        opt = self.optimizers[0]
        opt.zero_grad()

        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.manual_backward(loss)
        opt.step()

        if self._schedules:
            for sch in self._schedules:
                sch.step()

        self.log("train_loss", loss.item())
        for metric in self._metrics:
            value = metric(out, y)
            self.log(f"train_{type(metric).__name__}", value)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.log("val_loss", loss.item())
        for metric in self._metrics:
            value = metric(out, y)
            self.log(f"val_{type(metric).__name__}", value)

    def configure_optimizers(self):
        if self._external_scheduler is not None:
            return [self._external_optimizer], [self._external_scheduler]
        return self._external_optimizer
