"""
ModelCheckpoint Callback
自动保存模型的 Callback, 适配 CoreModel 的增强功能。
"""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.coreModel import CoreModel
    from xdl.trainer.trainer import Trainer


class ModelCheckpoint(Callback):
    """
    自动保存模型的 Callback

    Args:
        dirpath: checkpoint 保存目录
        monitor: 监控的指标 (例如: 'val_loss', 'train_loss', 'val_acc')
        save_top_k: 保存最好的 k 个模型. -1 表示保存所有.
        mode: 'min' 或 'max', 判断指标好坏的方式
        every_n_train_steps: 每 N 个训练步长保存一次
        every_n_epochs: 每 N 个 epoch 保存一次
        save_last: 是否总是保存最后一个 checkpoint (命名为 last)
        save_optimizer: 是否保存优化器状态
        save_scheduler: 是否保存调度器状态
        include_components: 指定保存的组件名列表, None表示保存所有
        format: 保存格式, ['pt', 'pth', 'st', 'safetensors', 'accelerator']
        naming_keys: 目录命名键列表, 默认会包含 step 或 epoch 以及 monitor
        verbose: 是否打印详细日志
    """

    def __init__(
        self,
        dirpath: str,
        monitor: Optional[str] = None,
        save_top_k: int = 1,
        mode: str = "min",
        every_n_train_steps: Optional[int] = None,
        every_n_epochs: Optional[int] = 1,
        save_last: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        include_components: Optional[List[str]] = None,
        format: str = "pt",
        naming_keys: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.every_n_train_steps = every_n_train_steps
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.include_components = include_components
        self.format = format
        self.naming_keys = naming_keys or (["step"] if every_n_train_steps else ["epoch"])
        self.verbose = verbose

        # 内部状态
        # List of {"path": str, "score": float}
        self.best_k_models: List[Dict[str, Any]] = []
        self.last_model_path: Optional[str] = None
        self._logger = logging.getLogger(__name__)

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """检查按步长保存"""
        if (
            self.every_n_train_steps
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_train_steps == 0
        ):
            self._save_checkpoint(trainer, core_module, "step")

    def on_train_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """检查按 epoch 保存"""
        if self.every_n_epochs and (trainer.current_epoch) % self.every_n_epochs == 0:
            # 如果有 monitor 且在验证集中, 通常在 on_validation_end 处理
            if self.monitor and "val" in self.monitor:
                return
            self._save_checkpoint(trainer, core_module, "epoch")

    def on_validation_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证结束时检查是否需要保存"""
        if self.monitor:
            self._save_checkpoint(trainer, core_module, "val")

    def on_train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练结束时保存最后状态"""
        if self.save_last:
            self._save_checkpoint(trainer, core_module, "last")

    def _get_monitor_value(self, trainer: "Trainer", core_module: "CoreModel") -> Optional[float]:
        """从 trainer 或 core_module 获取监控指标"""
        if not self.monitor:
            return None

        # 1. 优先尝试从核心组件的最新指标获取 (可能是本步生成的)
        val = core_module.current_metrics.get(self.monitor)

        # 2. 尝试从核心组件的当前 epoch 平均指标获取
        if val is None:
            val = core_module.epoch_avg.get(self.monitor)

        # 3. 尝试从核心组件上一个 epoch 的平均指标获取 (针对步长保存, 可能还没做本轮验证)
        if val is None and hasattr(core_module, "last_epoch_avg"):
            val = core_module.last_epoch_avg.get(self.monitor)

        # 4. 尝试从 trainer 的全局指标字典获取 (兼容逻辑)
        if val is None and hasattr(trainer, "callback_metrics"):
            val = trainer.callback_metrics.get(self.monitor)

        if val is not None:
            return float(val)
        return None

    def _save_checkpoint(
        self, trainer: "Trainer", core_module: "CoreModel", save_type: str
    ) -> None:
        """核心保存逻辑"""
        monitor_val = self._get_monitor_value(trainer, core_module)

        # 如果设置了监控指标但当前不可得, 且不是强制保存类型(如last), 则跳过本次保存
        if self.monitor and monitor_val is None and save_type != "last":
            return

        # 准备命名键和自定义值
        naming_keys = list(self.naming_keys)
        custom_values = {}

        if save_type == "last":
            if "last" not in naming_keys:
                naming_keys.append("last")
            custom_values["last"] = "checkpoint"
        elif self.monitor and self.monitor not in naming_keys:
            naming_keys.append(self.monitor)

        # 调用 CoreModel 的保存方法
        actual_path = core_module.save_checkpoint(
            base_dir=str(self.dirpath),
            format=self.format,
            naming_keys=naming_keys,
            custom_values=custom_values,
            save_optimizer=self.save_optimizer,
            save_scheduler=self.save_scheduler,
            include_components=self.include_components,
            callback_states=trainer.callback_list.save_state()
            if hasattr(trainer, "callback_list")
            else {},
        )

        if not actual_path:
            return

        # 处理 Top K 逻辑: 只有非 last 类型才参与 Top-K 排序管理
        if save_type != "last" and self.monitor and monitor_val is not None:
            self._update_best_models(actual_path, monitor_val)

        # 如果没有 monitor 且 save_top_k == 1, 则保留最新的
        elif self.save_top_k == 1 and not (self.every_n_train_steps or self.every_n_epochs):
            if self.last_model_path and self.last_model_path != actual_path:
                lp = Path(self.last_model_path)
                if lp.exists():
                    if lp.is_dir():
                        shutil.rmtree(lp)
                    else:
                        lp.unlink()

        # 处理 last 引用记录
        if save_type == "last":
            self.last_model_path = actual_path

        if self.verbose:
            msg = f"Checkpoint saved to {actual_path}"
            if monitor_val is not None:
                msg += f" ({self.monitor}={monitor_val:.4f})"
            self._logger.info(msg)

    def _update_best_models(self, filepath: str, score: float):
        """更新并维护 Top K 模型"""
        # 路径查重: 如果该路径已经在列表中, 更新其分数即可
        for item in self.best_k_models:
            if item["path"] == filepath:
                item["score"] = score
                return

        self.best_k_models.append({"path": filepath, "score": score})

        # 排序
        reverse = self.mode == "max"
        self.best_k_models.sort(key=lambda x: x["score"], reverse=reverse)

        # 如果超出 K, 删除最差的
        if self.save_top_k != -1 and len(self.best_k_models) > self.save_top_k:
            worst = self.best_k_models.pop(-1)
            worst_path = Path(worst["path"])

            # 安全检查: 不要删除正在作为 last_model_path 的文件, 也不要重复删除
            if str(worst_path) != str(self.last_model_path) and worst_path.exists():
                if worst_path.is_dir():
                    shutil.rmtree(worst_path)
                else:
                    worst_path.unlink()
                if self.verbose:
                    self._logger.info(f"Removed worst checkpoint: {worst_path}")

    def state_dict(self) -> Dict[str, Any]:
        return {"best_k_models": self.best_k_models, "last_model_path": self.last_model_path}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.best_k_models = state_dict.get("best_k_models", [])
        self.last_model_path = state_dict.get("last_model_path")
