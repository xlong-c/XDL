"""Callback for saving task-specific trainable state."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from xdl.callbacks.base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


class SaveTrainableStateCallback(Callback):
    """Save adapter/LoRA or other task-owned trainable state.

    The callback calls ``core_module.<method_name>(path)`` when the method
    exists. It is intentionally generic so task code can decide the actual
    serialization format.

    ``collective=True`` 时所有 rank 都会调用该方法 (适合 FSDP 下方法内部
    先做 ``accelerator.get_state_dict`` 聚合,再由主 rank 落盘的写法),
    并在调用前后自动 ``wait_for_everyone()``; 默认只在主进程调用.
    """

    fast_fail = True

    def __init__(
        self,
        dirpath: Union[str, Path],
        method_name: str = "save_trainable_state",
        every_n_epochs: Optional[int] = 1,
        every_n_train_steps: Optional[int] = None,
        save_final: bool = True,
        priority: int = 200,
        verbose: bool = False,
        collective: bool = False,
    ) -> None:
        super().__init__(priority=priority)
        self.dirpath = Path(dirpath)
        self.method_name = method_name
        self.every_n_epochs = (
            max(1, int(every_n_epochs)) if every_n_epochs is not None else None
        )
        self.every_n_train_steps = (
            max(1, int(every_n_train_steps))
            if every_n_train_steps is not None
            else None
        )
        self.save_final = bool(save_final)
        self.verbose = verbose
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
        del outputs, batch, batch_idx, dataloader_idx
        if (
            self.every_n_train_steps
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_train_steps == 0
        ):
            self._save(trainer, core_module, self._step_dir(trainer))

    def on_train_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if (
            self.every_n_epochs
            and trainer.current_epoch > 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            self._save(trainer, core_module, self._epoch_dir(trainer))

    def on_train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if self.save_final:
            self._save(trainer, core_module, self.dirpath / "final")

    def _step_dir(self, trainer: "Trainer") -> Path:
        return self.dirpath / f"step_{trainer.global_step:07d}"

    def _epoch_dir(self, trainer: "Trainer") -> Path:
        return (
            self.dirpath
            / f"epoch_{trainer.current_epoch:04d}_step_{trainer.global_step:07d}"
        )

    def _is_main_process(self, trainer: "Trainer", core_module: "CoreModel") -> bool:
        if hasattr(core_module, "is_main_process"):
            return bool(core_module.is_main_process())
        if hasattr(trainer, "is_main_process"):
            return bool(trainer.is_main_process())
        return True

    def _save(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        path: Path,
    ) -> None:
        if not self.collective and not self._is_main_process(trainer, core_module):
            return

        method = getattr(core_module, self.method_name, None)
        if not callable(method):
            if not self._warned_missing:
                self._warned_missing = True
                self._logger.warning(
                    "CoreModel 未实现 %s(), SaveTrainableStateCallback 未保存任何状态 "
                    "(目标路径: %s)",
                    self.method_name,
                    path,
                )
            return

        if self.collective:
            trainer.wait_for_everyone()
            try:
                if self._is_main_process(trainer, core_module):
                    path.mkdir(parents=True, exist_ok=True)
                cast(Callable[[Path], None], method)(path)
            finally:
                trainer.wait_for_everyone()
        else:
            path.mkdir(parents=True, exist_ok=True)
            cast(Callable[[Path], None], method)(path)
        if self.verbose:
            self._logger.info("Trainable state saved to %s", path)
