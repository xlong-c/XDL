"""
Tqdm进度条回调

参考 PyTorch Lightning 的进度条设计, 提供内存安全的进度条显示功能。
专门负责训练过程中的进度条显示功能, 与日志系统分离。
"""

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from xdl.trainer.coreModel import CoreModel
    from xdl.trainer.trainer import Trainer

    from .base import Callback
else:
    # 运行时导入, 确保Callback可用
    from .base import Callback

# 尝试导入tqdm
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


# 定义TqdmCallback类, 无论tqdm是否可用
class TqdmCallback(Callback):
    """
    Tqdm进度条回调

    参考 Lightning 的 RichProgressBar 设计, 提供：
    - 内存安全的进度条管理
    - 自动清理和资源释放
    - 线程安全的进度条操作
    - 优雅的多阶段进度条切换

    功能：
    - 统一管理训练、验证、测试阶段的进度条
    - 替代训练器中直接使用tqdm的代码
    - 提供更灵活的配置和自定义选项
    - 支持多阶段进度条同时存在
    """

    def __init__(
        self,
        log_frequency: int = 1,
        show_metrics: bool = True,
        metric_keys: Optional[List[str]] = None,
        leave: bool = False,
        dynamic_ncols: bool = True,
        custom_format: Optional[str] = None,
        position: int = 0,
        bar_format: Optional[str] = None,
        colour: Optional[str] = None,
        disable: bool = False,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: Optional[int] = None,
        ascii: Optional[bool] = None,
        ncols: Optional[int] = None,
        total_progress: bool = False,
    ):
        """
        初始化Tqdm进度条回调

        Args:
            log_frequency: 更新频率(每N个batch更新一次)
            show_metrics: 是否在进度条中显示指标
            metric_keys: 要显示的指标键名列表, 如果为None则自动选择
            leave: 是否在进度条完成后保留显示
            dynamic_ncols: 是否动态调整列宽
            custom_format: 自定义格式字符串
            position: 进度条位置(用于多个进度条)
            bar_format: 自定义进度条格式字符串
            colour: 进度条颜色
            disable: 是否禁用进度条
            mininterval: 最小更新间隔(秒)
            maxinterval: 最大更新间隔(秒)
            miniters: 最小更新次数
            ascii: 是否使用ASCII字符
            ncols: 进度条宽度
            total_progress: 是否使用统一的总进度条模式
        """
        super().__init__(priority=100)  # 高优先级确保进度条及时更新

        self.log_frequency = max(1, log_frequency)
        self.show_metrics = show_metrics
        self.metric_keys = metric_keys or ["loss", "acc", "error", "lr"]
        self.leave = leave
        self.dynamic_ncols = dynamic_ncols
        self.custom_format = custom_format
        self.position = position
        self.bar_format = bar_format
        self.colour = colour
        self.disable = disable
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self.ascii = ascii
        self.ncols = ncols
        self.total_progress = total_progress

        # 内部状态 - 线程安全的进度条管理
        self._progress_bars: Dict[str, Any] = {}
        self._active_bars: Dict[str, Any] = {}  # 当前活跃的进度条引用
        self._batch_counts: Dict[str, int] = {}
        self._last_metrics: Dict[str, Dict[str, Any]] = {}
        self._active_stages: Set[str] = set()
        self._lock = threading.Lock()  # 线程安全锁

        # 性能统计
        self._start_times: Dict[str, float] = {}
        self._batch_times: List[float] = []

    @property
    def state_key(self) -> str:
        """返回callback的唯一状态键"""
        return f"TqdmCallback[priority={self.priority}]"

    def setup(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        """初始化进度条管理器"""
        if not TQDM_AVAILABLE:
            print("Warning: tqdm not available, progress bars disabled")
            return

        # 确保没有遗留的进度条
        self._cleanup_progress_bars(force=True)

    def teardown(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        """清理进度条"""
        self._cleanup_progress_bars()

    def _cleanup_progress_bars(self, force: bool = False) -> None:
        """清理进度条, 防止内存泄漏

        Args:
            force: 是否强制清理所有进度条
        """
        with self._lock:
            stages_to_cleanup = list(self._active_stages) if force else []

            for stage in stages_to_cleanup:
                if stage in self._progress_bars:
                    try:
                        progress_bar = self._progress_bars[stage]
                        if hasattr(progress_bar, "close"):
                            progress_bar.close()
                    except Exception as e:
                        print(f"Warning: Error closing progress bar for {stage}: {e}")
                    finally:
                        del self._progress_bars[stage]
                        self._active_stages.discard(stage)

            if force:
                self._active_bars.clear()

            # 清理其他状态
            if force:
                self._batch_counts.clear()
                self._last_metrics.clear()

    def on_train_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练epoch开始时创建进度条"""
        with self._lock:
            self._batch_counts["train"] = 0
            self._last_metrics["train"] = {}
            self._start_times["train"] = time.time()

        # 优先使用 trainer.state.current_epoch 避免与 model._current_epoch 的自增时机冲突
        current_epoch = getattr(
            trainer.state, "current_epoch", getattr(core_module, "current_epoch", 1)
        )
        max_epochs = getattr(trainer, "max_epochs", 1)

        if self.total_progress:
            # 总进度条模式
            with self._lock:
                progress_bar = self._active_bars.get("train")

            if progress_bar is None:
                # 仅在第一个epoch创建总进度条
                steps_per_epoch = trainer.steps_per_epoch
                total_steps = steps_per_epoch * max_epochs

                if total_steps > 0:
                    bar = self._create_progress_bar(
                        total=total_steps,
                        description=f"Training [{current_epoch}/{max_epochs}]",
                        stage="train",
                        epoch=None,  # 总进度条不绑定具体epoch ID
                        leave=self.leave,
                    )
                    with self._lock:
                        self._active_bars["train"] = bar
                else:
                    print("[DEBUG] TqdmCallback: No train dataloader found for total progress")
            else:
                # 更新现有进度条的描述
                progress_bar.set_description(f"Training [{current_epoch}/{max_epochs}]")
        else:
            # 每个epoch独立进度条模式
            total_steps = trainer.steps_per_epoch
            if total_steps > 0:
                bar = self._create_progress_bar(
                    total=total_steps,
                    description=f"Training Epoch {current_epoch}",
                    stage="train",
                    epoch=current_epoch,
                    leave=self.leave,
                )
                with self._lock:
                    self._active_bars["train"] = bar
            else:
                print("[DEBUG] TqdmCallback: No train dataloader found")

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """训练批次结束时更新进度条"""
        with self._lock:
            self._batch_counts["train"] += 1
            batch_count = self._batch_counts["train"]
            progress_bar = self._active_bars.get("train")

        if not progress_bar:
            return

        # 每步都更新进度条位置
        progress_bar.update(1)

        # 按频率更新指标显示
        if batch_count % self.log_frequency == 0 or batch_idx == 0:
            # 提取显示的指标
            metrics = {}
            
            # 优先从 core_module.current_metrics 获取指标 (xdl 核心推荐方式)
            if hasattr(core_module, "current_metrics"):
                for key, value in core_module.current_metrics.items():
                    if self._should_show_metric(key):
                        metrics[key] = value

            # 兼容性: 同时也从 outputs 获取 (如果 outputs 是字典且包含了不在 metrics 中的键)
            if self.show_metrics and outputs and isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key not in metrics and self._should_show_metric(key):
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        elif hasattr(value, "item"):  # Tensor
                            metrics[key] = float(value.item())

            # 添加学习率信息
            if self.show_metrics:
                lr_metrics = self._extract_learning_rates(core_module)
                metrics.update(lr_metrics)

            # 更新指标显示
            if metrics:
                if self.custom_format:
                    metric_str = self.custom_format.format(**metrics)
                else:
                    metric_str = ", ".join(
                        [
                            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in metrics.items()
                        ]
                    )
                progress_bar.set_postfix_str(metric_str)

            with self._lock:
                self._last_metrics["train"] = metrics

    def on_train_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练epoch结束时关闭进度条"""
        if self.total_progress:
            # 总进度模式下不在此关闭
            return

        with self._lock:
            progress_bar = self._active_bars.get("train")
            if progress_bar:
                progress_bar.close()
                self._active_bars["train"] = None

    def on_train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """训练结束时关闭总进度条"""
        if self.total_progress:
            with self._lock:
                progress_bar = self._active_bars.get("train")
                if progress_bar:
                    progress_bar.close()
                    self._active_bars["train"] = None

    def on_validation_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证epoch开始时创建进度条"""
        if self.total_progress:
            # 总进度模式下隐藏验证进度条
            return

        with self._lock:
            self._batch_counts["val"] = 0
            self._last_metrics["val"] = {}
            self._start_times["val"] = time.time()

        # 获取验证数据加载器长度
        total_steps = self._get_dataloader_length(trainer, "val")
        current_epoch = getattr(
            trainer.state, "current_epoch", getattr(core_module, "current_epoch", 0)
        )

        if total_steps > 0:
            # 创建验证进度条
            bar = self._create_progress_bar(
                total=total_steps,
                description=f"Validating Epoch {current_epoch}",
                stage="val",
                epoch=current_epoch,
                leave=self.leave,
            )
            with self._lock:
                self._active_bars["val"] = bar

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """验证批次结束时更新进度条"""
        if self.total_progress:
            return

        with self._lock:
            self._batch_counts["val"] += 1
            batch_count = self._batch_counts["val"]
            progress_bar = self._active_bars.get("val")

        if not progress_bar:
            return

        # 每步更新
        progress_bar.update(1)

        # 按频率更新指标
        if batch_count % self.log_frequency == 0:
            metrics = {}
            
            # 优先从 core_module.current_metrics 获取指标
            if hasattr(core_module, "current_metrics"):
                for key, value in core_module.current_metrics.items():
                    if self._should_show_metric(key):
                        metrics[key] = value

            # 兼容性处理
            if outputs and isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key not in metrics and self._should_show_metric(key):
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        elif hasattr(value, "item"):  # Tensor
                            metrics[key] = float(value.item())

            if metrics:
                metric_str = ", ".join(
                    [
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items()
                    ]
                )
                progress_bar.set_postfix_str(metric_str)

            with self._lock:
                self._last_metrics["val"] = metrics

    def on_validation_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        """验证epoch结束时关闭进度条"""
        if self.total_progress:
            return

        with self._lock:
            progress_bar = self._active_bars.get("val")
            if progress_bar:
                progress_bar.close()
                self._active_bars["val"] = None

    def on_test_epoch_start(self, trainer, core_module):
        """测试epoch开始时创建进度条"""
        with self._lock:
            self._batch_counts["test"] = 0
            self._last_metrics["test"] = {}
            self._start_times["test"] = time.time()

        # 获取测试数据加载器长度
        total_steps = self._get_dataloader_length(trainer, "test")

        if total_steps > 0:
            # 创建测试进度条
            bar = self._create_progress_bar(
                total=total_steps, description="Testing", stage="test", epoch=None, leave=self.leave
            )
            with self._lock:
                self._active_bars["test"] = bar

    def on_test_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """测试批次结束时更新进度条"""
        with self._lock:
            self._batch_counts["test"] += 1
            batch_count = self._batch_counts["test"]
            progress_bar = self._active_bars.get("test")

        if not progress_bar:
            return

        progress_bar.update(1)

        # 按频率更新
        if batch_count % self.log_frequency == 0:
            metrics = {}
            
            # 优先从 core_module.current_metrics 获取指标
            if hasattr(core_module, "current_metrics"):
                for key, value in core_module.current_metrics.items():
                    if self._should_show_metric(key):
                        metrics[key] = value

            # 兼容性处理
            if outputs and isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key not in metrics and self._should_show_metric(key):
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        elif hasattr(value, "item"):  # Tensor
                            metrics[key] = float(value.item())

            if metrics:
                metric_str = ", ".join(
                    [
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items()
                    ]
                )
                progress_bar.set_postfix_str(metric_str)

            with self._lock:
                self._last_metrics["test"] = metrics

    def on_test_epoch_end(self, trainer, core_module):
        """测试epoch结束时关闭进度条"""
        with self._lock:
            progress_bar = self._active_bars.get("test")
            if progress_bar:
                progress_bar.close()
                self._active_bars["test"] = None

    def _should_show_metric(self, metric_name: str) -> bool:
        """
        判断是否应该显示指定指标

        Args:
            metric_name: 指标名称

        Returns:
            bool: 是否应该显示
        """
        if not self.metric_keys:
            return True

        metric_name_lower = metric_name.lower()
        return any(key.lower() in metric_name_lower for key in self.metric_keys)

    def _get_progress_bar(self, stage: str = "train", epoch: Optional[int] = None):
        """
        获取指定阶段的进度条对象

        Args:
            stage: 阶段
            epoch: 周期数

        Returns:
            tqdm进度条对象或None
        """
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage
        return self._progress_bars.get(bar_id, None)

    def _get_dataloader_length(self, trainer: "Trainer", stage: str) -> int:
        """获取指定阶段的数据加载器长度"""
        dataloader_attr_map = {
            "train": ["_train_dataloader", "train_dataloader", "train_loader"],
            "val": ["_val_dataloader", "val_dataloader", "val_loader"],
            "test": ["_test_dataloader", "test_dataloader", "test_loader"],
        }

        for attr_name in dataloader_attr_map.get(stage, []):
            if hasattr(trainer, attr_name):
                dataloader = getattr(trainer, attr_name)
                if dataloader is not None:
                    # 处理多个数据加载器的情况
                    if isinstance(dataloader, list):
                        return len(dataloader[0]) if dataloader else 0
                    else:
                        return len(dataloader)

        return 0

    def _extract_learning_rates(self, core_module: "CoreModel") -> Dict[str, float]:
        """提取学习率信息"""
        lr_metrics = {}

        # 尝试从不同位置获取优化器
        optimizers = None
        for attr_name in ["_optimizers", "optimizers", "optimizer"]:
            if hasattr(core_module, attr_name):
                optimizers = getattr(core_module, attr_name)
                break

        if optimizers:
            # 如果是字典格式(多个命名优化器)
            if isinstance(optimizers, dict):
                for opt_name, optimizer in optimizers.items():
                    for param_group in optimizer.param_groups:
                        lr = param_group.get("lr", 0)
                        if len(optimizers) == 1:
                            lr_metrics["lr"] = lr
                        else:
                            lr_metrics[f"{opt_name}_lr"] = lr
            # 如果是列表格式(多个优化器)
            elif isinstance(optimizers, list):
                for i, optimizer in enumerate(optimizers):
                    for param_group in optimizer.param_groups:
                        lr = param_group.get("lr", 0)
                        if len(optimizers) == 1:
                            lr_metrics["lr"] = lr
                        else:
                            lr_metrics[f"opt_{i}_lr"] = lr
            # 如果是单个优化器对象
            else:
                for param_group in optimizers.param_groups:
                    lr = param_group.get("lr", 0)
                    lr_metrics["lr"] = lr

        return lr_metrics

    # ============================================================================
    # 进度条管理方法(合并自 ProgressManager)
    # ============================================================================

    def _create_progress_bar(
        self,
        total: int,
        description: str = "Processing",
        stage: str = "train",
        epoch: Optional[int] = None,
        leave: bool = False,
    ):
        """
        创建tqdm进度条

        Args:
            total: 总步数
            description: 描述
            stage: 阶段 ('train', 'val', 'test')
            epoch: 周期数
            leave: 是否保留进度条

        Returns:
            tqdm进度条对象或None
        """
        if tqdm is None or self.disable:
            return None

        # 构造进度条标识 - 使用当前epoch作为唯一标识
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage

        # 构造描述
        desc = f"Epoch {epoch} - {description}" if epoch is not None else description

        with self._lock:
            # 如果已存在进度条, 先关闭旧的
            if bar_id in self._progress_bars:
                with contextlib.suppress(Exception):
                    self._progress_bars[bar_id].close()

            # 创建新的进度条
            try:
                progress_bar = tqdm(
                    total=total,
                    desc=desc,
                    leave=leave,
                    dynamic_ncols=self.dynamic_ncols,
                    unit="batch",
                    position=self.position,
                    bar_format=self.bar_format,
                    colour=self.colour,
                    mininterval=self.mininterval,
                    maxinterval=self.maxinterval,
                    miniters=self.miniters,
                    ascii=self.ascii,
                    ncols=self.ncols,
                )
            except OSError as e:
                # Windows兼容性处理
                print(f"警告: 无法创建进度条, 回退到简单打印: {e}")
                return None

            self._progress_bars[bar_id] = progress_bar
            self._active_stages.add(bar_id)

        return progress_bar

    def _update_progress_bar(
        self,
        stage: str = "train",
        epoch: Optional[int] = None,
        advance: int = 1,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        更新进度条

        Args:
            stage: 阶段
            epoch: 周期数
            advance: 进步步数
            metrics: 要显示的指标字典
        """
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage

        with self._lock:
            if bar_id in self._progress_bars:
                progress_bar = self._progress_bars[bar_id]

                # 确保进度条仍然有效
                if hasattr(progress_bar, "update") and not progress_bar.disable:
                    try:
                        progress_bar.update(advance)

                        # 更新进度条描述中的指标
                        if metrics:
                            if self.custom_format:
                                metric_str = self.custom_format.format(**metrics)
                            else:
                                metric_str = ", ".join(
                                    [
                                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                        for k, v in metrics.items()
                                    ]
                                )
                            progress_bar.set_postfix_str(metric_str)
                    except Exception as e:
                        print(f"警告: 进度条更新失败: {e}")

    def _set_description(self, description: str, stage: str = "train", epoch: Optional[int] = None):
        """
        设置进度条描述

        Args:
            description: 新的描述
            stage: 阶段
            epoch: 周期数
        """
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage

        with self._lock:
            if bar_id in self._progress_bars:
                try:
                    self._progress_bars[bar_id].set_description(description)
                except Exception as e:
                    print(f"警告: 设置进度条描述失败: {e}")

    def _close_progress_bar(self, stage: str = "train", epoch: Optional[int] = None):
        """
        关闭指定进度条

        Args:
            stage: 阶段
            epoch: 周期数
        """
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage

        with self._lock:
            if bar_id in self._progress_bars:
                try:
                    self._progress_bars[bar_id].close()
                except Exception:
                    pass
                finally:
                    del self._progress_bars[bar_id]
                    self._active_stages.discard(bar_id)

    def _close_all_progress_bars(self):
        """关闭所有进度条"""
        with self._lock:
            for _bar_id, progress_bar in self._progress_bars.items():
                with contextlib.suppress(Exception):
                    progress_bar.close()
            self._progress_bars.clear()
            self._active_stages.clear()

    def state_dict(self) -> Dict[str, Any]:
        """返回callback的状态字典"""
        return {
            "total_epochs": getattr(self, "_total_epochs", 0),
            "current_epoch": getattr(self, "_current_epoch", 0),
            "batch_count_epoch": sum(self._batch_counts.values()),
            "last_metrics": self._last_metrics.copy(),
            "active_stages": list(self._active_stages),
            "avg_batch_time": sum(self._batch_times) / len(self._batch_times)
            if self._batch_times
            else 0.0,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典恢复callback状态"""
        self._total_epochs = state_dict.get("total_epochs", 0)
        self._current_epoch = state_dict.get("current_epoch", 0)
        self._last_metrics = state_dict.get("last_metrics", {})
        self._active_stages = set(state_dict.get("active_stages", []))
        # 重置批次计数
        self._batch_counts = dict.fromkeys(["train", "val", "test"], 0)

    def is_available(self) -> bool:
        """检查tqdm是否可用"""
        return TQDM_AVAILABLE

    def get_active_progress_bars(self) -> List[str]:
        """获取当前活跃的进度条列表"""
        return list(self._active_stages)

    def get_batch_count(self, stage: str) -> int:
        """获取指定阶段的批次计数"""
        return self._batch_counts.get(stage, 0)

    def get_last_metrics(self, stage: str) -> Dict[str, Any]:
        """获取指定阶段最后记录的指标"""
        return self._last_metrics.get(stage, {})

    def get_average_batch_time(self) -> float:
        """获取平均批次处理时间"""
        return sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0.0

    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._batch_times.clear()
            self._start_times.clear()
            for stage in self._batch_counts:
                self._batch_counts[stage] = 0

    def force_refresh(self, stage: str = "train", epoch: Optional[int] = None):
        """强制刷新指定进度条的显示"""
        bar_id = f"{stage}_epoch_{epoch}" if epoch is not None else stage
        with self._lock:
            if bar_id in self._progress_bars:
                try:
                    self._progress_bars[bar_id].refresh()
                except Exception as e:
                    print(f"警告: 强制刷新进度条失败: {e}")

    def set_epoch(self, epoch: int):
        """设置当前epoch, 用于显示和统计"""
        with self._lock:
            self._current_epoch = epoch

    def pause(self, stage: str = "train"):
        """暂停指定阶段的进度条更新"""
        bar_id = stage
        with self._lock:
            if bar_id in self._progress_bars:
                with contextlib.suppress(Exception):
                    self._progress_bars[bar_id].disable = True

    def resume(self, stage: str = "train"):
        """恢复指定阶段的进度条更新"""
        bar_id = stage
        with self._lock:
            if bar_id in self._progress_bars:
                with contextlib.suppress(Exception):
                    self._progress_bars[bar_id].disable = False

    def __str__(self) -> str:
        """返回TqdmCallback的字符串表示"""
        info = [
            "TqdmCallback",
            f"  log_frequency: {self.log_frequency}",
            f"  show_metrics: {self.show_metrics}",
            f"  metric_keys: {self.metric_keys}",
            f"  leave: {self.leave}",
            f"  dynamic_ncols: {self.dynamic_ncols}",
            f"  disable: {self.disable}",
            f"  tqdm_available: {TQDM_AVAILABLE}",
        ]

        if self._active_stages:
            info.append(f"  active_stages: {', '.join(self._active_stages)}")

        if self._batch_times:
            info.append(f"  avg_batch_time: {self.get_average_batch_time():.4f}s")

        return "\n".join(info)

    def __repr__(self) -> str:
        """返回TqdmCallback的详细表示"""
        return (
            f"TqdmCallback(log_frequency={self.log_frequency}, "
            f"show_metrics={self.show_metrics}, "
            f"leave={self.leave}, "
            f"disable={self.disable})"
        )
