"""Preview callback for task-defined validation or sampling previews."""

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


class PreviewCallback(Callback):
    """Call a task-owned preview method at validation or train-step intervals.

    ``collective=True`` 时所有 rank 都会调用 preview 方法 (适合 FSDP 分片
    参数需要所有 rank 一起前向的采样), 落盘由任务方法内部用
    ``is_main_process()`` 控制; 调用前后自动 ``wait_for_everyone()``.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        method_name: str = "save_preview",
        every_n_epochs: Optional[int] = 1,
        every_n_train_steps: Optional[int] = None,
        first_val_batch_only: bool = True,
        priority: int = 300,
        collective: bool = False,
    ) -> None:
        super().__init__(priority=priority)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.method_name = method_name
        self.every_n_epochs = (
            max(1, int(every_n_epochs)) if every_n_epochs is not None else None
        )
        self.every_n_train_steps = (
            max(1, int(every_n_train_steps))
            if every_n_train_steps is not None
            else None
        )
        self.first_val_batch_only = bool(first_val_batch_only)
        self.collective = bool(collective)
        self._logger = logging.getLogger(__name__)
        self._warned_missing = False

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch_idx, dataloader_idx
        if (
            self.every_n_train_steps
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_train_steps == 0
        ):
            self._call_preview(trainer, core_module, batch)

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, dataloader_idx
        if self.first_val_batch_only and batch_idx != 0:
            return
        if (
            self.every_n_epochs
            and trainer.current_epoch > 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            self._call_preview(trainer, core_module, batch)

    def _is_main_process(self, trainer: "Trainer", core_module: "CoreModel") -> bool:
        if hasattr(core_module, "is_main_process"):
            return bool(core_module.is_main_process())
        if hasattr(trainer, "is_main_process"):
            return bool(trainer.is_main_process())
        return True

    def _call_preview(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
    ) -> None:
        if not self.collective and not self._is_main_process(trainer, core_module):
            return

        method = getattr(core_module, self.method_name, None)
        if not callable(method):
            if not self._warned_missing:
                self._warned_missing = True
                self._logger.warning(
                    "CoreModel 未实现 %s(), PreviewCallback 未生成任何预览",
                    self.method_name,
                )
            return

        if self.collective:
            trainer.wait_for_everyone()
            try:
                if self.output_dir is not None and self._is_main_process(
                    trainer, core_module
                ):
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                self._invoke_method(method, batch)
            finally:
                trainer.wait_for_everyone()
        else:
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            self._invoke_method(method, batch)

    def _invoke_method(self, method: Callable[..., Any], batch: Any) -> None:
        if self.output_dir is None:
            method(batch)
            return

        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if "output_dir" in signature.parameters or has_var_keyword:
            method(batch, output_dir=self.output_dir)
            return

        positional_capacity = sum(
            1
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        has_var_positional = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        if has_var_positional or positional_capacity >= 2:
            method(batch, self.output_dir)
        else:
            method(batch)
