"""
训练状态管理模块
统一管理所有训练状态信息
"""

from typing import Dict, Any, Optional
import torch


class TrainerState:
    """训练状态管理类

    统一管理训练过程中的所有状态信息：
    - 全局训练步数
    - 当前 epoch
    - 最大 epoch 数
    - 停止标志
    - 各种指标存储
    - 训练历史
    """

    def __init__(self):
        # 基础状态
        self.global_step = 0
        self.current_epoch = 0
        self.max_epochs = 0
        self.should_stop = False

        # 指标存储
        self.callback_metrics = {}  # 回调指标
        self.logged_metrics = {}  # 记录指标
        self.progress_bar_metrics = {}  # 进度条指标

        # 训练历史
        self.training_history = []  # 存储每个epoch的结果
        self.validation_history = []  # 验证历史

    def epoch_start(self, epoch: int):
        """epoch 开始

        Args:
            epoch: 当前 epoch 编号
        """
        self.current_epoch = epoch
        self.logged_metrics.clear()
        self.progress_bar_metrics.clear()

    def epoch_end(self, epoch: int, epoch_metrics: Dict[str, Any]):
        """epoch 结束

        Args:
            epoch: 当前 epoch 编号
            epoch_metrics: epoch 指标字典
        """
        self.training_history.append({
            'epoch': epoch,
            'step': self.global_step,
            'metrics': epoch_metrics.copy()
        })

    def add_metric(self, name: str, value, where: str = 'logged'):
        """添加指标

        Args:
            name: 指标名称
            value: 指标值
            where: 存储位置 ('callback', 'logged', 'progress_bar')
        """
        if isinstance(value, torch.Tensor):
            value = value.item()

        metric_info = {
            'value': value,
            'step': self.global_step,
            'epoch': self.current_epoch
        }

        if where == 'callback':
            self.callback_metrics[name] = metric_info
        elif where == 'logged':
            self.logged_metrics[name] = metric_info
        elif where == 'progress_bar':
            self.progress_bar_metrics[name] = metric_info

    def get_metric(self, name: str, where: str = 'logged') -> Optional[float]:
        """获取指标

        Args:
            name: 指标名称
            where: 存储位置 ('callback', 'logged', 'progress_bar')

        Returns:
            Optional[float]: 指标值, 不存在返回 None
        """
        metrics = getattr(self, f'{where}_metrics', {})
        return metrics.get(name, {}).get('value')

    def reset(self):
        """重置所有状态"""
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        self.callback_metrics.clear()
        self.logged_metrics.clear()
        self.progress_bar_metrics.clear()
        self.training_history.clear()
        self.validation_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            Dict[str, Any]: 状态字典
        """
        return {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'max_epochs': self.max_epochs,
            'should_stop': self.should_stop,
            'training_history': self.training_history,
            'validation_history': self.validation_history,
        }

    def from_dict(self, state_dict: Dict[str, Any]):
        """从字典加载状态

        Args:
            state_dict: 状态字典
        """
        self.global_step = state_dict.get('global_step', 0)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.max_epochs = state_dict.get('max_epochs', 0)
        self.should_stop = state_dict.get('should_stop', False)
        self.training_history = state_dict.get('training_history', [])
        self.validation_history = state_dict.get('validation_history', [])
