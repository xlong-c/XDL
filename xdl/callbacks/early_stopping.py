"""
EarlyStopping Callback
早停 Callback
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Callback
    from xdl.trainer.trainer import Trainer
    from xdl.trainer.coreModel import CoreModel
else:
    from .base import Callback


class EarlyStopping(Callback):
    """
    早停 Callback

    Args:
        monitor: 监控的指标
        min_delta: 最小改善阈值
        patience: 等待改善的 epoch 数
        mode: 'min' 或 'max'
    """

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = "min",
    ):
        super().__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode

        self.wait_count = 0
        self.best_score = None
        self.stopped_epoch = 0

        # 验证参数
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_train_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """每个 epoch 结束时检查是否需要早停"""

        # 获取监控指标的值
        monitor_value = self._get_monitor_value(trainer, core_module)
        if monitor_value is None:
            return

        # 判断是否需要早停
        if self._check_early_stop(monitor_value):
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True

    def _get_monitor_value(self, trainer: 'Trainer', core_module: 'CoreModel') -> Optional[float]:
        """获取监控指标的值"""
        if hasattr(core_module, '_latest_val_metrics') and core_module._latest_val_metrics is not None:
            metrics = core_module._latest_val_metrics
            return metrics.get(self.monitor)
        return None

    def _check_early_stop(self, monitor_value: float) -> bool:
        """检查是否应该早停"""

        # 第一次
        if self.best_score is None:
            self.best_score = monitor_value
            self.wait_count = 0
            return False

        # 判断是否有改善
        if self.mode == "min":
            improved = monitor_value < self.best_score - self.min_delta
        else:
            improved = monitor_value > self.best_score + self.min_delta

        if improved:
            self.best_score = monitor_value
            self.wait_count = 0
        else:
            self.wait_count += 1

        # 如果等待的 epoch 数超过 patience, 则早停
        return self.wait_count >= self.patience