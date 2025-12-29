"""
Lambda Callback
提供简单的lambda函数callback功能

参考 PyTorch Lightning 的 LambdaCallback 实现。
"""

from typing import Callable, Dict

from .base import Callback


class LambdaCallback(Callback):
    """
    Lambda函数回调

    参考 Lightning 的 LambdaCallback, 提供轻量级的自定义回调功能。

    Args:
        on_train_start: 训练开始时的lambda函数
        on_train_end: 训练结束时的lambda函数
        on_train_epoch_start: 训练epoch开始时的lambda函数
        on_train_epoch_end: 训练epoch结束时的lambda函数
        on_train_batch_start: 训练批次开始时的lambda函数
        on_train_batch_end: 训练批次结束时的lambda函数
        on_validation_start: 验证开始时的lambda函数
        on_validation_end: 验证结束时的lambda函数
        on_validation_epoch_start: 验证epoch开始时的lambda函数
        on_validation_epoch_end: 验证epoch结束时的lambda函数
        on_validation_batch_start: 验证批次开始时的lambda函数
        on_validation_batch_end: 验证批次结束时的lambda函数
        on_test_start: 测试开始时的lambda函数
        on_test_end: 测试结束时的lambda函数
        on_test_epoch_start: 测试epoch开始时的lambda函数
        on_test_epoch_end: 测试epoch结束时的lambda函数
        on_test_batch_start: 测试批次开始时的lambda函数
        on_test_batch_end: 测试批次结束时的lambda函数
        on_predict_start: 预测开始时的lambda函数
        on_predict_end: 预测结束时的lambda函数
        on_predict_epoch_start: 预测epoch开始时的lambda函数
        on_predict_epoch_end: 预测epoch结束时的lambda函数
        on_predict_batch_start: 预测批次开始时的lambda函数
        on_predict_batch_end: 预测批次结束时的lambda函数
        on_exception: 异常处理时的lambda函数
        on_keyboard_interrupt: 键盘中断时的lambda函数
        on_save_checkpoint: 保存检查点时的lambda函数
        on_load_checkpoint: 加载检查点时的lambda函数
        **kwargs: 其他自定义钩子
    """

    def __init__(self, **kwargs):
        super().__init__()

        # 存储所有lambda函数
        self._hooks: Dict[str, Callable] = {}

        # 预定义的钩子映射
        predefined_hooks = {
            "setup",
            "teardown",
            "on_train_start",
            "on_train_end",
            "on_train_epoch_start",
            "on_train_epoch_end",
            "on_train_batch_start",
            "on_train_batch_end",
            "on_validation_start",
            "on_validation_end",
            "on_validation_epoch_start",
            "on_validation_epoch_end",
            "on_validation_batch_start",
            "on_validation_batch_end",
            "on_test_start",
            "on_test_end",
            "on_test_epoch_start",
            "on_test_epoch_end",
            "on_test_batch_start",
            "on_test_batch_end",
            "on_predict_start",
            "on_predict_end",
            "on_predict_epoch_start",
            "on_predict_epoch_end",
            "on_predict_batch_start",
            "on_predict_batch_end",
            "on_exception",
            "on_keyboard_interrupt",
            "on_save_checkpoint",
            "on_load_checkpoint",
        }

        # 处理传入的lambda函数
        for hook_name, hook_func in kwargs.items():
            if callable(hook_func):
                self._hooks[hook_name] = hook_func

                # 检查是否是预定义钩子
                if hook_name in predefined_hooks:
                    # 为预定义钩子创建方法
                    self._create_hook_method(hook_name, hook_func)
                else:
                    # 为自定义钩子创建方法
                    self._create_hook_method(hook_name, hook_func)

    def _create_hook_method(self, hook_name: str, hook_func: Callable):
        """动态创建钩子方法"""

        def hook_method(self, trainer, core_module, **kwargs):
            try:
                # 根据钩子类型传递适当的参数
                if hook_name in [
                    "on_train_batch_start",
                    "on_train_batch_end",
                    "on_validation_batch_start",
                    "on_validation_batch_end",
                    "on_test_batch_start",
                    "on_test_batch_end",
                    "on_predict_batch_start",
                    "on_predict_batch_end",
                ]:
                    # 批次级别的钩子, 需要提供 batch, batch_idx, dataloader_idx 等
                    # 如果 hook_func 接受关键字参数，直接传递
                    import inspect

                    sig = inspect.signature(hook_func)
                    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                        hook_func(trainer, core_module, **kwargs)
                    else:
                        # 否则只尝试传递它需要的参数 (简单启发式)
                        args = [trainer, core_module]
                        param_names = list(sig.parameters.keys())
                        for name in param_names[2:]:  # 跳过 trainer, core_module
                            if name in kwargs:
                                args.append(kwargs[name])
                        hook_func(*args)
                elif hook_name == "on_exception":
                    hook_func(trainer, core_module, kwargs.get("exception"))
                elif hook_name == "on_save_checkpoint":
                    hook_func(trainer, core_module)
                elif hook_name == "on_load_checkpoint":
                    hook_func(trainer, core_module, kwargs.get("checkpoint"))
                else:
                    hook_func(trainer, core_module)
            except Exception as e:
                print(f"Error in LambdaCallback.{hook_name}: {e}")

        # 动态绑定方法
        setattr(self.__class__, hook_name, hook_method)

    def add_hook(self, hook_name: str, hook_func: Callable):
        """添加新的钩子函数

        Args:
            hook_name: 钩子名称
            hook_func: lambda函数
        """
        if not callable(hook_func):
            raise ValueError(f"hook_func must be callable, got {type(hook_func)}")

        self._hooks[hook_name] = hook_func
        self._create_hook_method(hook_name, hook_func)

    def remove_hook(self, hook_name: str):
        """移除钩子函数

        Args:
            hook_name: 钩子名称
        """
        if hook_name in self._hooks:
            del self._hooks[hook_name]
            # 移除动态创建的方法
            if hasattr(self, hook_name):
                delattr(self, hook_name)

    def get_hooks(self) -> Dict[str, Callable]:
        """获取所有钩子函数"""
        return self._hooks.copy()

    def has_hook(self, hook_name: str) -> bool:
        """检查是否存在指定的钩子"""
        return hook_name in self._hooks

    # 实现所有预定义钩子的默认方法
    def on_train_start(self, trainer, core_module):
        """训练开始时的钩子"""
        if "on_train_start" in self._hooks:
            self._hooks["on_train_start"](trainer, core_module)

    def on_train_end(self, trainer, core_module):
        """训练结束时的钩子"""
        if "on_train_end" in self._hooks:
            self._hooks["on_train_end"](trainer, core_module)

    def on_train_epoch_start(self, trainer, core_module):
        """训练epoch开始时的钩子"""
        if "on_train_epoch_start" in self._hooks:
            self._hooks["on_train_epoch_start"](trainer, core_module)

    def on_train_epoch_end(self, trainer, core_module):
        """训练epoch结束时的钩子"""
        if "on_train_epoch_end" in self._hooks:
            self._hooks["on_train_epoch_end"](trainer, core_module)

    def on_train_batch_start(self, trainer, core_module, batch, batch_idx, dataloader_idx=0):
        """训练批次开始时的钩子"""
        if "on_train_batch_start" in self._hooks:
            self._hooks["on_train_batch_start"](
                trainer, core_module, batch, batch_idx, dataloader_idx
            )

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """训练批次结束时的钩子"""
        if "on_train_batch_end" in self._hooks:
            self._hooks["on_train_batch_end"](
                trainer, core_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_start(self, trainer, core_module):
        """验证开始时的钩子"""
        if "on_validation_start" in self._hooks:
            self._hooks["on_validation_start"](trainer, core_module)

    def on_validation_end(self, trainer, core_module):
        """验证结束时的钩子"""
        if "on_validation_end" in self._hooks:
            self._hooks["on_validation_end"](trainer, core_module)

    def on_exception(self, trainer, core_module, exception):
        """异常处理钩子"""
        if "on_exception" in self._hooks:
            self._hooks["on_exception"](trainer, core_module, exception)

    def on_keyboard_interrupt(self, trainer, core_module):
        """键盘中断钩子"""
        if "on_keyboard_interrupt" in self._hooks:
            self._hooks["on_keyboard_interrupt"](trainer, core_module)


# 便利函数, 用于快速创建常见的lambda回调
def simple_callback(**hooks) -> LambdaCallback:
    """创建简单的lambda回调

    Args:
        **hooks: 钩子函数字典

    Returns:
        LambdaCallback: 配置好的lambda回调
    """
    return LambdaCallback(**hooks)


def logging_callback(log_func: Callable[[str], None]) -> LambdaCallback:
    """创建日志记录回调

    Args:
        log_func: 日志记录函数

    Returns:
        LambdaCallback: 配置好的日志回调
    """
    return LambdaCallback(
        on_train_start=lambda t, m: log_func("Training started"),
        on_train_end=lambda t, m: log_func("Training completed"),
        on_train_epoch_start=lambda t, m: log_func(
            f"Epoch {getattr(m, 'current_epoch', 0)} started"
        ),
        on_train_epoch_end=lambda t, m: log_func(
            f"Epoch {getattr(m, 'current_epoch', 0)} completed"
        ),
    )


def timing_callback(time_func: Callable[[], float]) -> LambdaCallback:
    """创建计时回调

    Args:
        time_func: 时间获取函数

    Returns:
        LambdaCallback: 配置好的计时回调
    """
    start_time = None

    def on_train_start(t, m):
        nonlocal start_time
        start_time = time_func()
        print(f"Training started at {start_time}")

    def on_train_end(t, m):
        nonlocal start_time
        if start_time is not None:
            end_time = time_func()
            duration = end_time - start_time
            print(f"Training completed at {end_time}")
            print(f"Total duration: {duration:.2f} seconds")

    return LambdaCallback(on_train_start=on_train_start, on_train_end=on_train_end)
