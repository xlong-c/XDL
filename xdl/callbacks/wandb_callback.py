"""
WandB日志回调

负责将训练和验证过程中的指标记录到Weights & Biases
"""

from typing import Any, Dict, List, Optional

# 尝试导入WandB
try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from .base import Callback


class WandbCallback(Callback):
    """
    WandB日志回调

    功能：
    - 将训练和验证指标记录到WandB
    - 内置独立的步数记录器
    - 支持训练按步记录, 验证按epoch记录的策略
    - 自动处理项目初始化和配置
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_frequency: int = 1,
        log_train: bool = True,
        log_validation: bool = True,
        log_validation_frequency: str = "epoch",  # "epoch" 或 "step"
        upload_frequency: int = 50,
    ):
        """
        初始化WandB日志回调

        Args:
            project: WandB项目名称
            entity: WandB实体名称(用户名或团队名)
            run_name: 运行名称
            tags: 运行标签列表
            notes: 运行说明
            config: 运行配置字典
            log_frequency: 训练日志记录频率(每N个batch记录一次)
            log_train: 是否记录训练日志
            log_validation: 是否记录验证日志
            log_validation_frequency: 验证日志记录频率, "epoch"或"step"
            upload_frequency: 数据上传频率
        """
        super().__init__(priority=180)  # 中高优先级
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.tags = tags or []
        self.notes = notes
        self.config = config or {}
        self.log_frequency = log_frequency
        self.log_train = log_train
        self.log_validation = log_validation
        self.log_validation_frequency = log_validation_frequency
        self.upload_frequency = upload_frequency

        # 内置步数记录器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0

        # WandB运行实例
        self.run: Optional[Any] = None
        self.last_upload_step = 0
        self.metrics_buffer: List[Dict[str, Any]] = []

    def setup(self, trainer, core_module, stage: str):
        """初始化WandB运行"""

        if wandb is None:
            print("警告: WandB未安装, 跳过WandB日志记录")
            return

        if not self.project:
            print("警告: 未指定wandb项目名称, 跳过WandB日志记录")
            return

        # 初始化WandB运行
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                reinit=True,
            )
            if self.run:
                print(f"WandB运行已初始化: {self.project}/{self.run.name}")
        except Exception as e:
            print(f"警告: WandB初始化失败: {e}")
            self.run = None
            return

        # 重置计数器
        self.train_step = 0
        self.val_step = 0
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.last_upload_step = 0
        self.metrics_buffer.clear()

    def teardown(self, trainer, core_module, stage: str):
        """清理WandB运行"""
        if self.run:
            # 上传剩余的缓冲指标
            self._flush_metrics_buffer()

            # 结束WandB运行
            try:
                self.run.finish()
                print("WandB运行已结束")
            except Exception as e:
                print(f"警告: WandB结束失败: {e}")

            self.run = None

    def on_train_start(self, trainer, core_module):
        """训练开始时记录超参数"""
        if not self.log_train or not self.run:
            return

        # 收集并更新超参数
        hparams = self._collect_hyperparams(trainer, core_module)
        if hparams:
            self.run.config.update(hparams)

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """训练批次结束时的WandB记录"""
        if not self.log_train or not self.run:
            return

        self.train_batch_count += 1
        self.train_step += 1

        # 按频率记录日志
        if self.train_batch_count % self.log_frequency == 0:
            # 获取指标数据
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                # 添加到缓冲区
                self.metrics_buffer.append(
                    {"metrics": metrics, "step": self.train_step, "prefix": "train/"}
                )

                # 按频率上传
                if len(self.metrics_buffer) >= self.upload_frequency:
                    self._flush_metrics_buffer()

    def on_train_epoch_end(self, trainer, core_module):
        """训练epoch结束时的WandB记录"""
        if not self.log_train or not self.run:
            return

        # 记录epoch级别的指标
        epoch_metrics = getattr(core_module, "last_epoch_avg", {})
        if epoch_metrics:
            # 添加epoch前缀
            prefixed_metrics = {f"train_epoch/{k}": v for k, v in epoch_metrics.items()}

            self.metrics_buffer.append(
                {
                    "metrics": prefixed_metrics,
                    "step": self.train_step,
                    "prefix": "",  # 已经有前缀了
                }
            )

        # 定期上传
        self._maybe_upload()

    def on_validation_batch_end(
        self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """验证批次结束时的WandB记录"""
        if not self.log_validation or not self.run:
            return

        self.val_batch_count += 1
        self.val_step += 1

        # 如果设置为按步骤记录验证日志
        if (
            self.log_validation_frequency == "step"
            and self.val_batch_count % self.log_frequency == 0
        ):
            metrics = self._extract_metrics(core_module, outputs)

            if metrics:
                self.metrics_buffer.append(
                    {
                        "metrics": metrics,
                        "step": self.train_step,  # 使用全局训练步数
                        "prefix": "val/",
                    }
                )

                # 按频率上传
                if len(self.metrics_buffer) >= self.upload_frequency:
                    self._flush_metrics_buffer()

    def on_validation_epoch_end(self, trainer, core_module):
        """验证epoch结束时的WandB记录"""
        if not self.log_validation or not self.run:
            return

        if self.log_validation_frequency == "epoch":
            # 获取验证epoch平均指标
            val_metrics = getattr(core_module, "last_epoch_avg", {})
            if val_metrics:
                # 添加val前缀
                prefixed_metrics = {f"val/{k}": v for k, v in val_metrics.items()}

                self.metrics_buffer.append(
                    {
                        "metrics": prefixed_metrics,
                        "step": self.train_step,  # 使用全局训练步数
                        "prefix": "",  # 已经有前缀了
                    }
                )

        # 定期上传
        self._maybe_upload()

    def _collect_hyperparams(self, trainer, core_module) -> Dict[str, Any]:
        """收集超参数信息"""
        hparams = {}

        # 从trainer收集参数
        if hasattr(trainer, "max_epochs"):
            hparams["max_epochs"] = trainer.max_epochs
        if hasattr(trainer, "learning_rate"):
            hparams["learning_rate"] = trainer.learning_rate
        if hasattr(trainer, "batch_size"):
            hparams["batch_size"] = trainer.batch_size

        # 从数据加载器收集参数
        if hasattr(trainer, "_train_dataloader") and trainer._train_dataloader:
            train_loader = trainer._train_dataloader
            if hasattr(train_loader, "batch_size"):
                hparams["train_batch_size"] = train_loader.batch_size
            if hasattr(train_loader, "__len__"):
                hparams["train_steps_per_epoch"] = len(train_loader)

        if hasattr(trainer, "_val_dataloader") and trainer._val_dataloader:
            val_loader = trainer._val_dataloader
            if hasattr(val_loader, "batch_size"):
                hparams["val_batch_size"] = val_loader.batch_size

        # 从模型收集参数
        if hasattr(core_module, "parameters"):
            total_params = sum(p.numel() for p in core_module.parameters())
            trainable_params = sum(p.numel() for p in core_module.parameters() if p.requires_grad)
            hparams["total_parameters"] = total_params
            hparams["trainable_parameters"] = trainable_params

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
        if hasattr(core_module, "current_metrics"):
            metrics = core_module.current_metrics

        # 如果没有获取到指标, 尝试从outputs获取
        if not metrics and outputs and isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif hasattr(value, "item"):  # Tensor
                    metrics[key] = float(value.item())

        return metrics

    def _flush_metrics_buffer(self):
        """上传缓冲区中的所有指标"""
        if not self.run or not self.metrics_buffer:
            return

        try:
            # 批量上传缓冲区中的指标
            for item in self.metrics_buffer:
                metrics = item["metrics"]
                step = item["step"]
                prefix = item["prefix"]

                # 添加前缀(如果需要)
                if prefix:
                    metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

                self.run.log(metrics, step=step)

            # 清空缓冲区
            self.metrics_buffer.clear()
            self.last_upload_step = self.train_step

        except Exception as e:
            print(f"警告: WandB指标上传失败: {e}")

    def _maybe_upload(self):
        """根据需要上传指标"""
        if (self.train_step - self.last_upload_step) >= self.upload_frequency:
            self._flush_metrics_buffer()

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
        self.last_upload_step = 0
        self.metrics_buffer.clear()

    def log_text(self, text: str, step: Optional[int] = None):
        """记录文本信息"""
        if self.run:
            step = step or self.train_step
            self.run.log({"log_text": text}, step=step)

    def log_image(self, key: str, image, step: Optional[int] = None):
        """记录图像"""
        if self.run and wandb is not None:
            step = step or self.train_step
            self.run.log({key: wandb.Image(image)}, step=step)

    def log_table(self, key: str, table, step: Optional[int] = None):
        """记录表格"""
        if self.run:
            step = step or self.train_step
            self.run.log({key: table}, step=step)

    def log_config(self, config: Dict[str, Any]):
        """更新配置"""
        if self.run:
            self.run.config.update(config)
