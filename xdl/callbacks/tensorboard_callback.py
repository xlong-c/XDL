"""
TensorBoard日志回调

负责将训练和验证过程中的指标记录到TensorBoard
"""

from typing import Dict, Any, Optional
from pathlib import Path

# 尝试导入TensorBoard
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from .base import Callback


class TensorBoardCallback(Callback):
    """
    TensorBoard日志回调

    功能：
    - 将训练和验证指标记录到TensorBoard
    - 内置独立的步数记录器
    - 支持训练按步记录, 验证按epoch记录的策略
    - 自动处理指标前缀(train/和val/)
    """

    def __init__(
        self,
        log_dir: str = "./others/logs/tensorboard",
        experiment_name: str = "experiment",
        log_frequency: int = 1,
        log_train: bool = True,
        log_validation: bool = True,
        log_validation_frequency: str = "epoch",  # "epoch" 或 "step"
        flush_frequency: int = 100,
        metric_prefix: bool = True
    ):
        """
        初始化TensorBoard日志回调

        Args:
            log_dir: TensorBoard日志目录
            experiment_name: 实验名称
            log_frequency: 训练日志记录频率(每N个batch记录一次)
            log_train: 是否记录训练日志
            log_validation: 是否记录验证日志
            log_validation_frequency: 验证日志记录频率, "epoch"或"step"
            flush_frequency: 数据刷新到磁盘的频率
            metric_prefix: 是否自动添加指标前缀(train/和val/)
        """
        super().__init__(priority=150)  # 中高优先级
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        self.log_train = log_train
        self.log_validation = log_validation
        self.log_validation_frequency = log_validation_frequency
        self.flush_frequency = flush_frequency
        self.metric_prefix = metric_prefix

        # 内置步数记录器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.last_flush_step = 0
        
        # TensorBoard writer
        self.writer: Optional[Any] = None
        
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            # 创建实验目录
            experiment_dir = self.log_dir / self.experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(experiment_dir))
            print(f"TensorBoard日志目录: {experiment_dir}")
            print(f"查看TensorBoard: tensorboard --logdir={experiment_dir}")
        else:
            print("警告: TensorBoard未安装或不可用, 跳过TensorBoard日志记录")

        # 缓存超参数
        self.hparams_cache: Dict[str, Any] = {}

    def setup(self, trainer, core_module, stage: str):
        """初始化计数器"""
        # 注意: SummaryWriter 已在 __init__ 中初始化

        # 重置计数器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.last_flush_step = 0
    
    def teardown(self, trainer, core_module, stage: str):
        """清理TensorBoard writer"""
        if self.writer:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
    
    def on_train_start(self, trainer, core_module):
        """训练开始时记录超参数"""
        if not self.log_train or not self.writer:
            return

        # 收集超参数
        hparams = self._collect_hyperparams(trainer, core_module)
        if hparams:
            self.writer.add_hparams(hparams, {})
            self.hparams_cache.update(hparams)

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx: int = 0, **kwargs):
        """训练批次结束时的TensorBoard记录"""
        if not self.log_train or not self.writer:
            return

        self.train_batch_count += 1
        self.train_step += 1

        # 按频率记录日志
        if self.train_batch_count % self.log_frequency == 0:
            # 获取指标数据
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                # 记录到TensorBoard
                for name, value in metrics.items():
                    if self.metric_prefix and not name.startswith('train/'):
                        name = f"train/{name}"
                    self.writer.add_scalar(name, value, self.train_step)

                # 定期刷新
                self._maybe_flush()

    def on_train_epoch_end(self, trainer, core_module):
        """训练epoch结束时的TensorBoard记录"""
        if not self.log_train or not self.writer:
            return

        # 记录epoch级别的指标
        epoch_metrics = getattr(core_module, 'last_epoch_avg', {})
        if epoch_metrics:
            for name, value in epoch_metrics.items():
                if self.metric_prefix and not name.startswith('train_epoch/'):
                    name = f"train_epoch/{name}"
                self.writer.add_scalar(name, value, self.train_step)

        # 定期刷新
        self._maybe_flush()

    def on_validation_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx: int = 0, **kwargs):
        """验证批次结束时的TensorBoard记录"""
        if not self.log_validation or not self.writer:
            return

        self.val_batch_count += 1
        self.val_step += 1

        # 如果设置为按步骤记录验证日志
        if self.log_validation_frequency == "step" and self.val_batch_count % self.log_frequency == 0:
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                for name, value in metrics.items():
                    if self.metric_prefix and not name.startswith('val/'):
                        name = f"val/{name}"
                    self.writer.add_scalar(name, value, self.train_step)  # 使用全局训练步数

                # 定期刷新
                self._maybe_flush()

    def on_validation_epoch_end(self, trainer, core_module):
        """验证epoch结束时的TensorBoard记录"""
        if not self.log_validation or not self.writer:
            return

        if self.log_validation_frequency == "epoch":
            # 获取验证epoch平均指标
            val_metrics = getattr(core_module, 'last_epoch_avg', {})
            if val_metrics:
                for name, value in val_metrics.items():
                    if self.metric_prefix and not name.startswith('val/'):
                        name = f"val/{name}"
                    self.writer.add_scalar(name, value, self.train_step)  # 使用全局训练步数

        # 定期刷新
        self._maybe_flush()

    def _collect_hyperparams(self, trainer, core_module) -> Dict[str, Any]:
        """收集超参数信息"""
        hparams = {}

        # 从trainer收集参数
        if hasattr(trainer, 'max_epochs'):
            hparams['trainer/max_epochs'] = trainer.max_epochs
        if hasattr(trainer, 'learning_rate'):
            hparams['trainer/learning_rate'] = trainer.learning_rate
        if hasattr(trainer, 'batch_size'):
            hparams['trainer/batch_size'] = trainer.batch_size

        # 从数据加载器收集参数
        if hasattr(trainer, '_train_dataloader') and trainer._train_dataloader:
            train_loader = trainer._train_dataloader
            if hasattr(train_loader, 'batch_size'):
                hparams['dataset/train_batch_size'] = train_loader.batch_size
            if hasattr(train_loader, '__len__'):
                hparams['dataset/train_steps_per_epoch'] = len(train_loader)

        if hasattr(trainer, '_val_dataloader') and trainer._val_dataloader:
            val_loader = trainer._val_dataloader
            if hasattr(val_loader, 'batch_size'):
                hparams['dataset/val_batch_size'] = val_loader.batch_size

        # 从模型收集参数
        if hasattr(core_module, 'parameters'):
            total_params = sum(p.numel() for p in core_module.parameters())
            trainable_params = sum(p.numel() for p in core_module.parameters() if p.requires_grad)
            hparams['model/total_parameters'] = total_params
            hparams['model/trainable_parameters'] = trainable_params

        return hparams

    def _extract_metrics(self, core_module, outputs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        从core_module和outputs中提取指标

        Args:
            core_module: 核心模块实例
            outputs: 当前步骤的输出

        Returns:
            提取的指标字典
        """
        metrics = {}

        # 优先从core模块的current_metrics属性获取
        if hasattr(core_module, 'current_metrics'):
            metrics = core_module.current_metrics

        # 如果没有获取到指标, 尝试从outputs获取
        if not metrics and outputs and isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif hasattr(value, 'item'):  # Tensor
                    metrics[key] = float(value.item())

        return metrics

    def _maybe_flush(self):
        """根据需要刷新数据到磁盘"""
        if self.writer and (self.train_step - self.last_flush_step) >= self.flush_frequency:
            self.writer.flush()
            self.last_flush_step = self.train_step

    def get_train_step(self) -> int:
        """获取当前训练步数"""
        return self.train_step

    def get_validation_step(self) -> int:
        """获取当前验证步数"""
        return self.val_step

    def reset_counters(self):
        """重置步数计数器"""
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.last_flush_step = 0

    def log_hyperparams(self, hparams: Dict[str, Any]):
        """手动记录超参数"""
        if self.writer:
            self.writer.add_hparams(hparams, {})
            self.hparams_cache.update(hparams)

    def log_text(self, text: str, step: Optional[int] = None):
        """记录文本信息"""
        if self.writer:
            step = step or self.train_step
            self.writer.add_text("logs", text, step)

    def log_graph(self, model, input_to_model=None):
        """记录模型图"""
        if self.writer:
            try:
                self.writer.add_graph(model, input_to_model)
            except Exception as e:
                print(f"警告: 无法记录模型图: {e}")
