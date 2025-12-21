"""
LearningRate Monitor Callback
监控和记录学习率变化

参考 PyTorch Lightning 的 LearningRateMonitor 实现。
"""

import logging
from typing import Dict, Any, List, Optional
from .base import Callback


class LearningRateMonitor(Callback):
    """
    学习率监控回调

    参考 Lightning 的 LearningRateMonitor, 提供：
    - 自动监控所有优化器的学习率
    - 支持学习率调度器变化
    - 灵活的日志记录间隔

    Args:
        logging_interval: 日志记录间隔 ('epoch' 或 'step')
        log_momentum: 是否记录动量参数
        log_weight_decay: 是否记录权重衰减
        verbose: 是否输出详细信息
    """

    def __init__(
        self,
        logging_interval: str = "epoch",
        log_momentum: bool = False,
        log_weight_decay: bool = False,
        verbose: bool = False
    ):
        super().__init__()

        self.logging_interval = logging_interval.lower()
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay
        self.verbose = verbose

        # 验证参数
        if self.logging_interval not in ["epoch", "step"]:
            raise ValueError(f"logging_interval must be 'epoch' or 'step', got {logging_interval}")

        # 状态管理
        self._state.update({
            'last_lr_values': {},
            'lr_history': []
        })

        self._logger = logging.getLogger(__name__)

    def on_train_epoch_start(self, trainer, core_module):
        """训练epoch开始时记录学习率(如果间隔为epoch)"""
        if self.logging_interval == "epoch":
            self._log_learning_rates(trainer, core_module, "epoch")

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """训练批次结束时记录学习率(如果间隔为step)"""
        if self.logging_interval == "step":
            # 按频率记录, 避免过于频繁
            if batch_idx % 50 == 0:  # 每50个batch记录一次
                self._log_learning_rates(trainer, core_module, "step")

    def _log_learning_rates(self, trainer, core_module, step_type: str):
        """
        记录学习率到日志

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            step_type: 步骤类型 ('epoch' 或 'step')
        """
        try:
            lr_info = self._get_lr_info(trainer, core_module)

            if not lr_info:
                if self.verbose:
                    self._logger.debug("No learning rate information found")
                return

            # 记录到状态
            self._state['lr_history'].append({
                'step_type': step_type,
                'epoch': getattr(core_module, 'current_epoch', 0),
                'step': getattr(trainer, 'global_step', 0),
                'lr_info': lr_info.copy(),
                'timestamp': __import__('time').time()
            })

            # 输出日志
            for name, info in lr_info.items():
                if isinstance(info, dict):
                    lr = info.get('lr', 0)
                    if self.verbose:
                        self._logger.info(f"LearningRateMonitor - {name}: lr={lr:.6e}")
                else:
                    if self.verbose:
                        self._logger.info(f"LearningRateMonitor - {name}: lr={info:.6e}")

        except Exception as e:
            self._logger.error(f"Error logging learning rates: {e}")

    def _get_lr_info(self, trainer, core_module) -> Dict[str, Any]:
        """
        获取学习率信息

        Args:
            trainer: 训练器实例
            core_module: 核心模块

        Returns:
            Dict[str, Any]: 学习率信息
        """
        lr_info = {}

        # 尝试从不同位置获取优化器信息
        optimizers = self._get_optimizers(trainer, core_module)

        for name, optimizer in optimizers.items():
            if optimizer is None:
                continue

            try:
                # 获取学习率
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                lr_info[f"{name}_lr"] = lrs[0] if len(lrs) == 1 else lrs

                # 记录动量(如果启用)
                if self.log_momentum:
                    momentums = [param_group.get('momentum', 0) for param_group in optimizer.param_groups]
                    if any(m > 0 for m in momentums):
                        lr_info[f"{name}_momentum"] = momentums[0] if len(momentums) == 1 else momentums

                # 记录权重衰减(如果启用)
                if self.log_weight_decay:
                    weight_decays = [param_group.get('weight_decay', 0) for param_group in optimizer.param_groups]
                    if any(wd > 0 for wd in weight_decays):
                        lr_info[f"{name}_weight_decay"] = weight_decays[0] if len(weight_decays) == 1 else weight_decays

            except Exception as e:
                self._logger.warning(f"Error extracting learning rate info from {name}: {e}")

        return lr_info

    def _get_optimizers(self, trainer, core_module) -> Dict[str, Any]:
        """
        从不同位置获取优化器

        Args:
            trainer: 训练器实例
            core_module: 核心模块

        Returns:
            Dict[str, Any]: 优化器字典
        """
        optimizers = {}

        # 尝试从core_module获取
        if hasattr(core_module, '_optimizers') and core_module._optimizers:
            opts = core_module._optimizers
            if isinstance(opts, dict):
                # 字典格式的多个优化器
                optimizers.update(opts)
            elif isinstance(opts, list):
                # 列表格式的多个优化器
                for i, opt in enumerate(opts):
                    optimizers[f'optimizer_{i}'] = opt
            else:
                # 单个优化器
                optimizers['optimizer'] = opts

        # 尝试从trainer获取
        if hasattr(trainer, '_optimizers') and trainer._optimizers:
            opts = trainer._optimizers
            if isinstance(opts, dict):
                optimizers.update(opts)
            elif isinstance(opts, list):
                for i, opt in enumerate(opts):
                    if f'optimizer_{i}' not in optimizers:
                        optimizers[f'trainer_optimizer_{i}'] = opt
            else:
                if 'optimizer' not in optimizers:
                    optimizers['trainer_optimizer'] = opts

        # 尝试从optimizer属性获取
        if hasattr(core_module, 'optimizer') and core_module.optimizer:
            if 'optimizer' not in optimizers:
                optimizers['optimizer'] = core_module.optimizer

        return optimizers

    def get_lr_history(self) -> List[Dict[str, Any]]:
        """获取学习率历史记录"""
        return self._state.get('lr_history', [])

    def get_current_lrs(self) -> Dict[str, float]:
        """获取当前学习率"""
        if self._state['lr_history']:
            return self._state['lr_history'][-1]['lr_info']
        return {}

    def print_lr_summary(self):
        """打印学习率变化摘要"""
        history = self.get_lr_history()
        if not history:
            print("No learning rate history available.")
            return

        print("\n" + "="*60)
        print("LEARNING RATE SUMMARY")
        print("="*60)

        current_lrs = self.get_current_lrs()
        if current_lrs:
            print("\nCurrent learning rates:")
            for name, lr in current_lrs.items():
                if isinstance(lr, (int, float)):
                    print(f"  {name}: {lr:.6e}")
                else:
                    print(f"  {name}: {lr}")

        print(f"\nTotal logged updates: {len(history)}")
        print("="*60)