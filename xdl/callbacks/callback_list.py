"""
回调管理器模块
统一管理所有训练回调

参考 PyTorch Lightning 的 callback 管理机制, 提供错误隔离、执行统计和状态管理功能。
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from xdl.callbacks import Callback
    from xdl.trainer.coreModel import CoreModel
    from xdl.trainer.trainer import Trainer


class CallbackList:
    """回调管理器类

    参考 Lightning 设计, 统一管理所有训练回调, 提供：
    - 错误隔离和恢复机制
    - 回调执行统计和监控
    - 状态管理和持久化
    - 性能诊断和时间统计
    """

    def __init__(self, callbacks: Optional[List["Callback"]] = None, fast_fail: bool = False):
        """
        初始化回调管理器

        Args:
            callbacks: 初始回调列表, 可选
            fast_fail: 是否在回调失败时快速失败, 默认False(错误隔离模式)
        """
        self.callbacks: List[Callback] = []
        self._callback_metadata: Dict[Callback, Dict[str, Any]] = {}  # 存储回调元数据
        self.fast_fail = fast_fail

        # 执行统计和监控
        self._execution_stats: Dict[str, Dict[str, Any]] = {}  # 回调执行统计
        self._logger = logging.getLogger(__name__)

        # 错误处理
        self._error_count = 0
        self._max_errors = 100  # 最大错误记录数
        self._error_history: List[Dict[str, Any]] = []  # 错误历史记录

        # 添加初始回调(如果提供)
        if callbacks:
            for callback in callbacks:
                self.add_callback(callback)

    def add_callback(self, callback: "Callback", priority: int = 999) -> None:
        """添加回调

        Args:
            callback: 回调对象
            priority: 优先级, 数字越小优先级越高
        """
        self._callback_metadata[callback] = {
            "priority": priority,
            "name": callback.__class__.__name__,
        }
        self.callbacks.append(callback)
        self._sort_callbacks()

    def remove_callback(self, callback: "Callback") -> None:
        """移除回调

        Args:
            callback: 要移除的回调对象
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            del self._callback_metadata[callback]

    def _sort_callbacks(self):
        """按优先级排序回调"""
        self.callbacks.sort(key=lambda cb: self._callback_metadata.get(cb, {}).get("priority", 999))

    def invoke_callbacks(
        self, hook_name: str, trainer: "Trainer", core_module: "CoreModel", **kwargs: Any
    ) -> None:
        """统一调用回调, 支持错误隔离和性能监控

        Args:
            hook_name: 钩子名称
            trainer: 训练器实例
            core_module: 核心模块
            **kwargs: 额外参数

        Raises:
            RuntimeError: 当fast_fail=True且回调执行失败时
        """
        for callback in self.callbacks:
            if not hasattr(callback, hook_name):
                continue

            callback_name = callback.__class__.__name__
            start_time = time.time()

            try:
                # 执行回调 - 直接传递参数, 不使用关键字参数
                method = getattr(callback, hook_name)
                method(trainer, core_module, **kwargs)

                # 记录执行统计
                execution_time = time.time() - start_time
                self._record_execution_stats(callback_name, hook_name, execution_time, success=True)

            except Exception as e:
                # 记录执行统计
                execution_time = time.time() - start_time
                self._record_execution_stats(
                    callback_name, hook_name, execution_time, success=False, error=e
                )

                # 处理错误
                self._handle_callback_error(callback, hook_name, e)

                # 快速失败模式
                if self.fast_fail:
                    raise RuntimeError(
                        f"Callback {callback_name}.{hook_name} failed in fast_fail mode"
                    ) from e

    def _handle_callback_error(
        self, callback: "Callback", hook_name: str, error: Exception
    ) -> None:
        """处理回调错误

        Args:
            callback: 回调对象
            hook_name: 钩子名称
            error: 异常对象
        """
        callback_name = callback.__class__.__name__
        error_msg = f"[ERROR] Callback {callback_name}.{hook_name} failed: {error}"

        # 记录错误历史
        self._error_count += 1
        error_info: Dict[str, Any] = {
            "callback": callback_name,
            "hook": hook_name,
            "error": str(error),
            "timestamp": time.time(),
        }

        # 限制错误历史记录数量
        if len(self._error_history) >= self._max_errors:
            self._error_history.pop(0)
        self._error_history.append(error_info)

        # 记录日志
        self._logger.error(error_msg, exc_info=True)

        # 可选：控制台输出(保持向后兼容)
        print(error_msg)

    def save_state(self) -> Dict[str, Any]:
        """保存所有回调状态

        Returns:
            Dict[str, Any]: 回调状态字典
        """
        state = {}
        for callback in self.callbacks:
            if hasattr(callback, "state_dict"):
                state[callback.__class__.__name__] = callback.state_dict()
        return state

    def load_state(self, state: Dict[str, Any]):
        """加载所有回调状态

        Args:
            state: 回调状态字典
        """
        for callback in self.callbacks:
            callback_name = callback.__class__.__name__
            if callback_name in state and hasattr(callback, "load_state_dict"):
                callback.load_state_dict(state[callback_name])

    def __len__(self) -> int:
        """获取回调数量

        Returns:
            int: 回调数量
        """
        return len(self.callbacks)

    def __iter__(self):
        """迭代回调"""
        return iter(self.callbacks)

    def __getitem__(self, index: int) -> "Callback":
        """获取指定索引的回调

        Args:
            index: 回调索引

        Returns:
            Callback: 回调对象
        """
        return self.callbacks[index]

    # ============================================================================
    # 直观的专用回调方法
    # ============================================================================

    # Fit生命周期方法
    def fit_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练开始回调"""
        self.invoke_callbacks("on_fit_start", trainer, core_module)

    def fit_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练结束回调"""
        self.invoke_callbacks("on_fit_end", trainer, core_module)

    # 设置相关方法
    def setup(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        """设置回调"""
        self.invoke_callbacks("setup", trainer, core_module, stage=stage)

    def teardown(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        """清理回调"""
        self.invoke_callbacks("teardown", trainer, core_module, stage=stage)

    # 训练生命周期方法
    def train_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练开始回调"""
        self.invoke_callbacks("on_train_start", trainer, core_module)

    def train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练结束回调"""
        self.invoke_callbacks("on_train_end", trainer, core_module)

    # Epoch相关方法
    def epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """Epoch开始回调"""
        self.invoke_callbacks("on_train_epoch_start", trainer, core_module)

    def epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """Epoch结束回调"""
        self.invoke_callbacks("on_train_epoch_end", trainer, core_module)

    # 验证相关方法
    def validation_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证开始回调"""
        self.invoke_callbacks("on_validation_start", trainer, core_module)

    def validation_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证epoch开始回调"""
        self.invoke_callbacks("on_validation_epoch_start", trainer, core_module)

    def validation_epoch_end(
        self, trainer: "Trainer", core_module: "CoreModel", outputs: Optional[Any] = None
    ) -> None:
        """验证epoch结束回调"""
        kwargs = {"outputs": outputs} if outputs is not None else {}
        self.invoke_callbacks("on_validation_epoch_end", trainer, core_module, **kwargs)

    def validation_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证结束回调"""
        self.invoke_callbacks("on_validation_end", trainer, core_module)

    def train_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """训练批次开始回调"""
        self.invoke_callbacks(
            "on_train_batch_start",
            trainer,
            core_module,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """训练批次结束回调"""
        self.invoke_callbacks(
            "on_train_batch_end",
            trainer,
            core_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def validation_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """验证批次开始回调"""
        self.invoke_callbacks(
            "on_validation_batch_start",
            trainer,
            core_module,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def validation_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """验证批次结束回调"""
        self.invoke_callbacks(
            "on_validation_batch_end",
            trainer,
            core_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    # 测试相关方法
    def test_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """测试开始回调"""
        self.invoke_callbacks("on_test_start", trainer, core_module)

    def test_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """测试epoch开始回调"""
        self.invoke_callbacks("on_test_epoch_start", trainer, core_module)

    def test_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """测试epoch结束回调"""
        self.invoke_callbacks("on_test_epoch_end", trainer, core_module)

    def test_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """测试结束回调"""
        self.invoke_callbacks("on_test_end", trainer, core_module)

    def test_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """测试批次开始回调"""
        self.invoke_callbacks(
            "on_test_batch_start",
            trainer,
            core_module,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def test_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """测试批次结束回调"""
        self.invoke_callbacks(
            "on_test_batch_end",
            trainer,
            core_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    # 预测相关方法
    def predict_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """预测开始回调"""
        self.invoke_callbacks("on_predict_start", trainer, core_module)

    def predict_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """预测epoch开始回调"""
        self.invoke_callbacks("on_predict_epoch_start", trainer, core_module)

    def predict_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """预测epoch结束回调"""
        self.invoke_callbacks("on_predict_epoch_end", trainer, core_module)

    def predict_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """预测结束回调"""
        self.invoke_callbacks("on_predict_end", trainer, core_module)

    def predict_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """预测批次开始回调"""
        self.invoke_callbacks(
            "on_predict_batch_start",
            trainer,
            core_module,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def predict_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """预测批次结束回调"""
        self.invoke_callbacks(
            "on_predict_batch_end",
            trainer,
            core_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    # 检查点相关方法
    def save_checkpoint(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """保存检查点回调"""
        self.invoke_callbacks("on_save_checkpoint", trainer, core_module)

    def load_checkpoint(
        self, trainer: "Trainer", core_module: "CoreModel", checkpoint: Any
    ) -> None:
        """加载检查点回调"""
        self.invoke_callbacks("on_load_checkpoint", trainer, core_module, checkpoint=checkpoint)

    # ========================================
    # 错误处理和性能监控辅助方法
    # ========================================

    def _record_execution_stats(
        self,
        callback_name: str,
        hook_name: str,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ):
        """记录回调执行统计信息

        Args:
            callback_name: 回调名称
            hook_name: 钩子名称
            execution_time: 执行时间
            success: 是否成功
            error: 错误对象(如果失败)
        """
        key = f"{callback_name}.{hook_name}"

        if key not in self._execution_stats:
            self._execution_stats[key] = {
                "total_calls": 0,
                "success_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float("inf"),
                "last_error": None,
            }

        stats = self._execution_stats[key]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)

        if success:
            stats["success_calls"] += 1
        else:
            stats["failed_calls"] += 1
            stats["last_error"] = str(error)

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息

        Returns:
            Dict[str, Any]: 执行统计信息
        """
        return self._execution_stats.copy()

    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史记录

        Returns:
            List[Dict[str, Any]]: 错误历史记录
        """
        return self._error_history.copy()

    def reset_stats(self):
        """重置统计信息"""
        self._execution_stats.clear()
        self._error_history.clear()
        self._error_count = 0

    def print_performance_summary(self):
        """打印性能摘要"""
        if not self._execution_stats:
            print("No callback execution statistics available.")
            return

        print("\n" + "=" * 80)
        print("CALLBACK PERFORMANCE SUMMARY")
        print("=" * 80)

        for key, stats in self._execution_stats.items():
            success_rate = (
                (stats["success_calls"] / stats["total_calls"]) * 100
                if stats["total_calls"] > 0
                else 0
            )

            print(f"\n{key}:")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Avg time: {stats['avg_time']:.4f}s")
            print(f"  Max time: {stats['max_time']:.4f}s")
            print(f"  Min time: {stats['min_time']:.4f}s")

            if stats["failed_calls"] > 0:
                print(f"  Last error: {stats['last_error']}")

        if self._error_count > 0:
            print(f"\nTotal errors: {self._error_count}")
            print("Recent errors:")
            for error in self._error_history[-5:]:  # 显示最近5个错误
                print(f"  - {error['callback']}.{error['hook']}: {error['error']}")

        print("=" * 80)

    # ========================================
    # 异常处理钩子方法
    # ========================================

    def handle_exception(
        self, trainer: "Trainer", core_module: "CoreModel", exception: Exception
    ) -> None:
        """处理训练异常

        Args:
            trainer: 训练器实例
            core_module: 核心模块
            exception: 异常对象
        """
        self.invoke_callbacks("on_exception", trainer, core_module, exception=exception)
