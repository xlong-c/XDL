"""训练期显存管理回调.

工具函数本体在 ``xdl.utils.memory`` (也可显式调用), 这里的回调只是把
梯度检查点与激活卸载接入训练生命周期:

>>> from xdl.callbacks import GradientCheckpointingCallback, ActivationOffloadCallback
>>> trainer = Trainer(
...     callbacks=[
...         GradientCheckpointingCallback(),
...         ActivationOffloadCallback(min_bytes=32 << 20),
...     ],
... )
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from xdl.utils.memory import (
    activation_offload_context,
    enable_gradient_checkpointing,
)
from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


class GradientCheckpointingCallback(Callback):
    """训练开始时开启模型梯度检查点.

    模型需要提供 ``gradient_checkpointing_enable()`` 或
    ``set_gradient_checkpointing(True)``; 两者都没有时仅 warning.
    """

    def __init__(self, priority: int = 50, verbose: bool = True) -> None:
        super().__init__(priority=priority)
        self.verbose = verbose

    def on_fit_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        del trainer
        if enable_gradient_checkpointing(core_module):
            return
        if self.verbose:
            logger.warning(
                "GradientCheckpointingCallback: 模型没有 "
                "gradient_checkpointing_enable()/set_gradient_checkpointing(), 已跳过"
            )


class ActivationOffloadCallback(Callback):
    """用 ``saved_tensors_hooks`` 包住每个训练步, 做选择性激活卸载.

    只在 ``on_train_batch_start`` 到 ``on_train_batch_end`` 之间生效
    (覆盖 ``training_step`` 的前向与反向); 训练步抛异常时通过
    ``on_exception`` 钩子清理上下文.

    Args:
        min_bytes: 超过该字节数的 CUDA 激活才卸载, 默认 32MB.
        pinned: 是否使用 pinned CPU 内存.
    """

    def __init__(
        self,
        min_bytes: int = 32 << 20,
        pinned: bool = True,
        priority: int = 90,
    ) -> None:
        super().__init__(priority=priority)
        self.min_bytes = max(0, int(min_bytes))
        self.pinned = bool(pinned)
        self._active_ctx: Optional[Any] = None

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, core_module, batch, batch_idx, dataloader_idx
        if self._active_ctx is not None:
            return
        context = activation_offload_context(self.min_bytes, self.pinned)
        context.__enter__()
        self._active_ctx = context

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, core_module, outputs, batch, batch_idx, dataloader_idx
        self._close()

    def on_exception(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        exception: Exception,
    ) -> None:
        del trainer, core_module, exception
        self._close()

    def teardown(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        del trainer, core_module, stage
        self._close()

    def _close(self) -> None:
        context, self._active_ctx = self._active_ctx, None
        if context is not None:
            context.__exit__(None, None, None)
