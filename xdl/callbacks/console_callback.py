"""
控制台日志回调

负责将训练和验证过程中的指标输出到控制台
"""

import logging
import sys
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Callback
    from xdl.trainer.trainer import Trainer
    from xdl.trainer.coreModel import CoreModel
else:
    from .base import Callback


class ConsoleCallback(Callback):
    """
    控制台日志回调

    功能：
    - 将训练和验证指标输出到控制台
    - 支持自定义日志频率和格式
    - 内置独立的步数记录器
    - 区分训练和验证的日志记录策略
    """

    def __init__(
        self,
        log_frequency: int = 50,
        log_train: bool = False,
        log_validation: bool = False,
        log_validation_frequency: str = "epoch",  # "epoch" 或 "step"
        custom_format: Optional[str] = None,
        metric_keys: Optional[List[str]] = None,
        show_epoch_info: bool = True
    ):
        """
        初始化控制台日志回调

        Args:
            log_frequency: 训练日志记录频率(每N个batch记录一次)
            log_train: 是否记录训练日志
            log_validation: 是否记录验证日志
            log_validation_frequency: 验证日志记录频率, "epoch"或"step"
            custom_format: 自定义日志格式字符串
            metric_keys: 要记录的指标键名列表, 如果为None则记录所有指标
            show_epoch_info: 是否显示epoch信息
        """
        super().__init__(priority=200)  # 中等优先级
        self.log_frequency = log_frequency
        self.log_train = log_train
        self.log_validation = log_validation
        self.log_validation_frequency = log_validation_frequency
        self.custom_format = custom_format
        self.metric_keys = metric_keys
        self.show_epoch_info = show_epoch_info

        # 内置步数记录器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0

        # 日志器
        self.logger: Optional[logging.Logger] = None

    def setup(self, trainer: 'Trainer', core_module: 'CoreModel', stage: str) -> None:
        """初始化控制台日志器"""
        self.logger = logging.getLogger(f"console_callback.{id(self)}")
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if self.custom_format:
                formatter = logging.Formatter(self.custom_format)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 重置计数器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0

    def on_train_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """训练开始时的日志"""
        if not self.log_train:
            return

        if self.logger is None:
            return

        self.logger.info("="*50)
        self.logger.info("训练开始")
        if hasattr(trainer, 'max_epochs'):
            self.logger.info(f"最大训练轮数: {trainer.max_epochs}")
        self.logger.info("="*50)

    def on_train_epoch_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """训练epoch开始时的日志"""
        if not self.log_train or self.logger is None:
            return

        self.train_batch_count = 0
        if self.show_epoch_info:
            self.logger.info(f"开始训练 Epoch {core_module.current_epoch}")

    def on_train_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """训练批次结束时的日志"""
        if not self.log_train or self.logger is None:
            return

        self.train_batch_count += 1
        self.train_step += 1

        # 按频率记录日志
        if self.train_batch_count % self.log_frequency == 0:
            # 获取指标数据
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                # 格式化日志信息
                log_msg = f"Step {self.train_step} - "
                if self.show_epoch_info:
                    log_msg += f"Epoch {core_module.current_epoch} - "

                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                log_msg += metric_str

                self.logger.info(log_msg)

    def on_train_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """训练epoch结束时的日志"""
        if not self.log_train or self.logger is None:
            return

        # 获取epoch平均指标
        if hasattr(core_module, '_step_metrics'):
            epoch_metrics = core_module._step_metrics.get_all_epoch_avg()
            if epoch_metrics:
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                self.logger.info(f"Epoch {core_module.current_epoch} 完成 - {metric_str}")

    def on_validation_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """验证开始时的日志"""
        if not self.log_validation or self.logger is None:
            return

        self.val_batch_count = 0
        if self.log_validation_frequency == "epoch":
            self.logger.info(f"开始验证 Epoch {core_module.current_epoch}")

    def on_validation_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """验证批次结束时的日志"""
        if not self.log_validation or self.logger is None:
            return

        self.val_batch_count += 1
        self.val_step += 1

        # 如果设置为按步骤记录验证日志
        if self.log_validation_frequency == "step" and self.val_batch_count % self.log_frequency == 0:
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                log_msg = f"Val Step {self.val_step} - "
                if self.show_epoch_info:
                    log_msg += f"Epoch {core_module.current_epoch} - "

                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                log_msg += metric_str

                self.logger.info(log_msg)

    def on_validation_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """验证epoch结束时的日志"""
        if not self.log_validation or self.logger is None:
            return

        if self.log_validation_frequency == "epoch":
            # 获取验证epoch平均指标
            val_metrics = getattr(core_module, 'last_epoch_avg', {})
            if val_metrics:
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                self.logger.info(f"验证 Epoch {core_module.current_epoch} 完成 - {metric_str}")

    def on_train_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """训练结束时的日志"""
        if not self.log_train or self.logger is None:
            return

        self.logger.info("="*50)
        self.logger.info("训练完成")
        self.logger.info(f"总共训练步骤: {self.train_step}")
        self.logger.info("="*50)

    def _extract_metrics(self, core_module: 'CoreModel', outputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        从core_module和outputs中提取指标

        Args:
            core_module: 核心模块实例
            outputs: 当前步骤的输出

        Returns:
            提取的指标字典
        """
        metrics = {}

        # 优先从core模块的get_current_metrics方法获取
        if hasattr(core_module, '_step_metrics'):
            try:
                metrics = core_module._step_metrics.get_all_current()
            except (AttributeError, TypeError):
                pass

        # 如果没有获取到指标, 尝试从outputs获取
        if not metrics and outputs and isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif hasattr(value, 'item'):  # Tensor
                    metrics[key] = float(value.item())

        # 如果指定了metric_keys, 只返回指定的指标
        if self.metric_keys and metrics:
            metrics = {k: v for k, v in metrics.items()
                      if any(target_key.lower() in k.lower() for target_key in self.metric_keys)}

        return metrics

    def get_train_step(self) -> int:
        """获取当前训练步数"""
        return self.train_step

    def get_validation_step(self) -> int:
        """获取当前验证步数"""
        return self.val_step

    def reset_counters(self):
        """重置步数计数器"""
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0