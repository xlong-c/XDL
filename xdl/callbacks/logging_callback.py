"""
日志记录回调系统

提供自动化的日志记录功能, 基于标准Python logging系统。
"""

import time
import os
import sys
from typing import Optional, Any
from loguru import logger

from .base import Callback


class LoggingCallback(Callback):
    """
    使用 loguru 记录训练指标的回调系统

    功能：
    - 自动记录训练和验证指标
    - 支持控制台彩色输出和文件记录
    - 自动日志滚动 (Rotation)、过期清理 (Retention) 和压缩 (Compression)
    - 异步非阻塞日志写入
    """

    def __init__(
        self,
        log_frequency: int = 50,
        log_val_metrics: bool = True,
        log_train_metrics: bool = True,
        log_learning_rate: bool = True,
        log_dir: Optional[str] = None,
        log_filename: Optional[str] = None,
        rotation: str = "500 MB",
        retention: str = "10 days",
        compression: str = "zip",
        enable_console: bool = False,
    ):
        """
        初始化 loguru 日志回调

        Args:
            log_frequency: 日志记录频率(每N个step记录一次)
            log_val_metrics: 是否记录验证指标
            log_train_metrics: 是否记录训练指标
            log_learning_rate: 是否记录学习率
            log_dir: 日志文件存储目录
            log_filename: 日志文件名
            rotation: 日志滚动条件 (e.g. "500 MB", "12:00", "1 week")
            retention: 日志保留时间 (e.g. "10 days")
            compression: 日志压缩格式 (e.g. "zip", "tar.gz")
            enable_console: 是否在控制台显示日志
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.log_val_metrics = log_val_metrics
        self.log_train_metrics = log_train_metrics
        self.log_learning_rate = log_learning_rate
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.rotation = rotation
        self.retention = retention
        self.compression = compression
        self.enable_console = enable_console

        # 内部状态
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.epoch_start_time = 0
        self._handler_id = None

    def setup(self, trainer: Any, core_module: Any, stage: str):
        """配置 loguru 处理器"""
        if self._handler_id is not None:
            return

        # 移除默认的处理器 (通常 ID 为 0)，避免冗余的模块名和行号显示
        try:
            logger.remove(0)
        except ValueError:
            pass

        # 1. 添加简洁的控制台处理器
        if self.enable_console:
            console_format = "<green>{time:HH:mm:ss}</green> | <level>{message}</level>"
            logger.add(sys.stdout, format=console_format, colorize=True, level="INFO")

        # 2. 获取有效的日志目录
        effective_log_dir = self.log_dir
        if not effective_log_dir and hasattr(trainer, 'log_dir'):
            effective_log_dir = trainer.log_dir
        
        if not effective_log_dir:
            effective_log_dir = os.path.join("others", "logs")

        os.makedirs(effective_log_dir, exist_ok=True)
        
        # 确定文件名
        effective_filename = self.log_filename or f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(effective_log_dir, effective_filename)

        # 3. 添加文件处理器
        self._handler_id = logger.add(
            log_path,
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            level="INFO",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
            encoding="utf-8"
        )
        
        logger.info(f"Loguru 日志系统已就绪 (文件: {log_path})")

    def teardown(self, trainer: Any, core_module: Any, stage: str):
        """移除 loguru 处理器"""
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            self._handler_id = None

    def on_train_start(self, trainer, core_module):
        """训练开始时的钩子"""
        logger.info("🚀 训练开始")

    def on_train_epoch_start(self, trainer, core_module):
        """训练epoch开始时的钩子"""
        self.epoch_start_time = time.time()
        self.train_batch_count = 0
        self.val_batch_count = 0
        logger.info(f"📅 开始训练 Epoch {core_module.current_epoch}")

    def on_train_batch_start(self, trainer, core_module, batch, batch_idx, dataloader_idx=0):
        """训练批次开始时的钩子"""
        self.train_batch_count += 1

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """训练批次结束时记录指标"""
        if not self.log_train_metrics:
            return

        if self.train_batch_count % self.log_frequency == 0:
            metrics = {}
            if hasattr(core_module, 'current_metrics'):
                current_metrics = core_module.current_metrics
                for key, value in current_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics[f"train/{key}"] = float(value)

            # 记录学习率
            if self.log_learning_rate and hasattr(core_module, '_optimizers'):
                optimizers = core_module._optimizers
                for i, optimizer in enumerate(optimizers):
                    for param_group in optimizer.param_groups:
                        lr = param_group.get('lr', 0)
                        if len(optimizers) == 1:
                            metrics["train/lr"] = lr
                        else:
                            metrics[f"train/opt_{i}_lr"] = lr

            if metrics:
                metric_str = " | ".join([f"{k}: <cyan>{v:.4f}</cyan>" for k, v in metrics.items()])
                step = core_module.global_step if hasattr(core_module, 'global_step') else self.train_batch_count
                logger.opt(colors=True).info(f"Step {step:05d} - {metric_str}")

    def on_validation_epoch_start(self, trainer, core_module):
        """验证epoch开始时的钩子"""
        if not self.log_val_metrics:
            return
        logger.info(f"🔍 开始验证 Epoch {core_module.current_epoch}")

    def on_validation_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """验证批次结束时记录指标"""
        if not self.log_val_metrics:
            return

        self.val_batch_count += 1

        if self.val_batch_count % self.log_frequency == 0:
            metrics = {}
            if hasattr(core_module, 'current_metrics'):
                current_metrics = core_module.current_metrics
                for key, value in current_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics[f"val/{key}"] = float(value)

            if metrics:
                metric_str = " | ".join([f"{k}: <yellow>{v:.4f}</yellow>" for k, v in metrics.items()])
                logger.opt(colors=True).info(f"Val Batch {self.val_batch_count} - {metric_str}")

    def on_train_epoch_end(self, trainer, core_module):
        """训练epoch结束时的钩子"""
        if self.log_train_metrics:
            epoch_metrics = getattr(core_module, 'last_epoch_avg', {})
            if epoch_metrics:
                metric_str = " | ".join([f"{k}: <cyan>{v:.4f}</cyan>" for k, v in epoch_metrics.items()])
                duration = time.time() - self.epoch_start_time
                logger.opt(colors=True).success(f"✅ Epoch {core_module.current_epoch} 训练完成 | {metric_str} | Time: {duration:.2f}s")

    def on_validation_epoch_end(self, trainer, core_module):
        """验证epoch结束时的钩子"""
        if not self.log_val_metrics:
            return

        epoch_metrics = getattr(core_module, 'last_epoch_avg', {})
        if not epoch_metrics and hasattr(core_module, '_latest_val_metrics'):
            epoch_metrics = core_module._latest_val_metrics

        if epoch_metrics:
            metric_str = " | ".join([f"{k}: <yellow>{v:.4f}</yellow>" for k, v in epoch_metrics.items()])
            logger.opt(colors=True).success(f"📊 Epoch {core_module.current_epoch} 验证完成 | {metric_str}")

    def on_train_end(self, trainer, core_module):
        """训练结束时的钩子"""
        logger.info("🏁 训练任务结束")


class SystemStatsCallback(Callback):
    """
    系统性能监控回调 (Loguru 版)
    """

    def __init__(self, log_frequency: int = 100):
        super().__init__()
        self.log_frequency = log_frequency
        self.last_time = time.time()

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """记录系统状态"""
        step = getattr(trainer, 'global_step', batch_idx)
        if step % self.log_frequency != 0:
            return

        current_time = time.time()
        time_elapsed = current_time - self.last_time
        self.last_time = current_time

        stats = {}
        if time_elapsed > 0:
            stats["it/s"] = self.log_frequency / time_elapsed

        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1024**3
                    stats[f"gpu{i}_mem_gb"] = mem
        except Exception:
            pass

        if stats:
            stats_str = " | ".join([f"{k}: {v:.2f}" for k, v in stats.items()])
            logger.info(f"💻 System - {stats_str}")


# 导出所有回调类
__all__ = [
    'LoggingCallback',
    'SystemStatsCallback'
]