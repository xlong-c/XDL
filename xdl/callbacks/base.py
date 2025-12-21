"""
Callback 基类
所有自定义 Callback 都应该继承此类

参考 PyTorch Lightning 的 Callback 设计模式, 提供完整的生命周期钩子和状态管理支持。
"""

from typing import Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from xdl.trainer.trainer import Trainer
    from xdl.trainer.coreModel import CoreModel


class Callback:
    """
    Callback 基类
    所有自定义 Callback 都应该继承此类

    提供完整的生命周期钩子、状态管理和错误处理支持。
    """

    def __init__(self, priority: int = 999):
        """
        Args:
            priority: 优先级, 数值越小优先级越高, 默认为999
        """
        self.priority = priority
        self._state: Dict[str, Any] = {}     # callback的状态信息

    def on_fit_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """Called when fit begins"""
        pass

    def on_fit_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """Called when fit ends"""
        pass

    def setup(self, trainer: 'Trainer', core_module: 'CoreModel', stage: str) -> None:
        """Called when fit or test begins"""
        pass

    def teardown(self, trainer: 'Trainer', core_module: 'CoreModel', stage: str) -> None:
        """Called when fit or test ends"""
        pass

    def on_train_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_train_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_train_epoch_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_train_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_train_batch_start(self, trainer: 'Trainer', core_module: 'CoreModel', batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        训练批次开始时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_train_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        训练批次结束时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            outputs: 模型输出
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_validation_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_validation_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_validation_epoch_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_validation_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_test_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_test_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_test_epoch_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_test_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_before_backward(self, trainer: 'Trainer', core_module: 'CoreModel', loss: Any) -> None:
        """Called before backward propagation"""
        pass

    def on_after_backward(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """Called after backward propagation"""
        pass

    def on_before_optimizer_step(self, trainer: 'Trainer', core_module: 'CoreModel', optimizer: Any, optimizer_idx: int) -> None:
        """Called before optimizer step"""
        pass

    def on_before_zero_grad(self, trainer: 'Trainer', core_module: 'CoreModel', optimizer: Any) -> None:
        """Called before optimizer's zero_grad"""
        pass

    def on_predict_batch_start(self, trainer: 'Trainer', core_module: 'CoreModel', batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        预测批次开始时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_predict_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        预测批次结束时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            outputs: 模型输出
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_validation_batch_start(self, trainer: 'Trainer', core_module: 'CoreModel', batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        验证批次开始时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_validation_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        验证批次结束时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            outputs: 模型输出
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_test_batch_start(self, trainer: 'Trainer', core_module: 'CoreModel', batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        测试批次开始时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_test_batch_end(self, trainer: 'Trainer', core_module: 'CoreModel', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        测试批次结束时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            outputs: 模型输出
            batch: 当前批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引, 默认为0
        """
        pass

    def on_save_checkpoint(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_load_checkpoint(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    # ========================================
    # 状态管理方法 (参考 Lightning 模式)
    # ========================================

    @property
    def state_key(self) -> str:
        """
        返回callback的唯一状态键, 用于状态保存和恢复

        Returns:
            str: 状态键, 默认使用类名
        """
        return self.__class__.__name__

    def state_dict(self) -> Dict[str, Any]:
        """
        返回callback的状态字典, 用于保存到checkpoint

        Returns:
            Dict[str, Any]: 状态字典
        """
        return self._state.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        从状态字典恢复callback状态

        Args:
            state_dict: 状态字典
        """
        self._state.update(state_dict)

    # ========================================
    # 预测阶段钩子
    # ========================================

    def on_predict_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """Called when predict begins"""
        pass

    def on_predict_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        """Called when predict ends"""
        pass

    def on_predict_epoch_start(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    def on_predict_epoch_end(self, trainer: 'Trainer', core_module: 'CoreModel') -> None:
        pass

    # ========================================
    # 异常处理钩子
    # ========================================

    def on_exception(self, trainer: 'Trainer', core_module: 'CoreModel', exception: Exception) -> None:
        """
        当训练过程中发生异常时调用

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            exception: 发生的异常
        """
        pass
