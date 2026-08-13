"""
训练器 - 简化版(删除加速器相关逻辑)

参考 PyTorch Lightning 设计理念

此文件整合了所有训练器相关的类和功能
"""

import logging
import math
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

# Accelerate 支持
from accelerate import Accelerator
from torch.utils.data import DataLoader

from xdl.callbacks import Callback
from xdl.callbacks.callback_list import CallbackList
from xdl.callbacks.console_callback import ConsoleCallback
from xdl.callbacks.model_checkpoint import ModelCheckpoint
from xdl.callbacks.tensorboard_callback import TensorBoardCallback

# 导入新的独立回调类
from xdl.callbacks.tqdm_callback import TqdmCallback

# 导入拆分后的模块
from xdl.trainer.trainer_state import TrainerState
from xdl.config.accelerate_config import FSDPConfig, build_fsdp_plugin
from xdl.errors import TrainingError
from xdl.utils.memory import global_grad_norm

from .core_model import CoreModel

logger = logging.getLogger(__name__)


def _is_finite(value: Any) -> bool:
    """把值转成 float 并判断是否有限; 无法转换时按有限处理."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return math.isfinite(numeric)


class _GradientClipOptimizer:
    """在 ``optimizer.step()`` 真正执行前按 Trainer 配置裁剪梯度的代理.

    手动优化模式下用户代码会直接调用 ``optimizer.step()``, 无法在训练循环
    里可靠拦截; 通过代理包装让 ``Trainer(grad_clip_max_norm=...)`` 真正生效.
    Accelerate 的梯度累积状态下只在 ``sync_gradients`` 为真时裁剪.
    """

    def __init__(self, optimizer: Any, trainer: "Trainer") -> None:
        object.__setattr__(self, "_clip_optimizer", optimizer)
        object.__setattr__(self, "_clip_trainer", trainer)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        trainer = self._clip_trainer
        if trainer.grad_clip_max_norm is not None:
            accelerator = trainer.accelerator
            sync_gradients = (
                getattr(accelerator, "sync_gradients", True)
                if accelerator is not None
                else True
            )
            if sync_gradients:
                trainer._clip_optimizer_gradients(self._clip_optimizer)
        return self._clip_optimizer.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_clip_optimizer"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_clip_optimizer"), name, value)

    def __repr__(self) -> str:
        return f"_GradientClipOptimizer({self._clip_optimizer!r})"


class Trainer:
    """
    训练器 - 简化版

    Args:
        max_epochs: 最大训练轮数
        device: 使用的设备 ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.) when not using Accelerate, otherwise follows Accelerate config
        precision: 混合精度训练 (None, '16', 'bf16', '32')
        gradient_accumulation_steps: 梯度累积步数
        grad_clip_max_norm: 梯度裁剪最大范数
        grad_clip_norm_type: 梯度裁剪范数类型
        callbacks: 回调函数列表
        accelerate_config: Accelerate 配置
        fsdp: FSDP 版本 (None, 1, 或 2), 也可以传配置 dict / FSDPConfig /
            FullyShardedDataParallelPlugin 实例
        nan_monitor: 训练循环内置 NaN/Inf 指标与梯度监控
        nan_patience: 连续出现 NaN/Inf 的步数达到该值时中止训练
        fail_on_callback_error: 每个 epoch 结束时若存在回调失败则抛 TrainingError
    """

    def __init__(
        self,
        max_epochs: int = 10,
        device: Optional[str] = None,  # auto, cuda, cpu
        precision: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        grad_clip_max_norm: Optional[float] = None,
        grad_clip_norm_type: float = 2.0,
        callbacks: Optional[List[Callback]] = None,
        # Accelerate 配置
        accelerate_config: Optional[Dict[str, Any]] = None,
        fsdp: Optional[Union[int, Dict[str, Any], FSDPConfig]] = None,
        nan_monitor: bool = True,
        nan_patience: int = 3,
        fail_on_callback_error: bool = False,
    ):
        # 训练参数
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps or 1))
        self.grad_clip_max_norm = grad_clip_max_norm
        self.grad_clip_norm_type = grad_clip_norm_type

        # 设备和精度
        self.device_spec = device  # Store the device specification
        self.precision = precision

        # Accelerate 配置
        self.accelerate_config = accelerate_config
        self.fsdp = fsdp

        # 数值稳定性
        self.nan_monitor = bool(nan_monitor)
        self.nan_patience = max(1, int(nan_patience))
        self.fail_on_callback_error = bool(fail_on_callback_error)
        self._nan_steps = 0

        # Type hint for accelerator - will be set during setup
        self._accelerator: Optional[Accelerator] = None

        # 核心组件
        # 训练状态管理
        self.state = TrainerState()
        self.state.max_epochs = max_epochs

        # 模型和数据
        self._model: Optional[CoreModel] = None
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None

        # 回调管理器
        self.callback_list = CallbackList(callbacks)

        # 梯度累积步数计数器
        self._accumulated_batches = 0

        # 设备引用
        self._device = None if self._should_use_accelerate() else self._get_device_from_spec()
        self._is_setup = False

    def _should_use_accelerate(self) -> bool:
        """判断当前配置是否需要走 Accelerate 路径."""
        return (
            self.accelerate_config is not None
            or self.precision is not None
            or self.fsdp is not None
        )

    @classmethod
    def from_setup(cls, setup) -> "Trainer":
        """从 TrainSetup 创建 Trainer 并自动配置日志/检查点.

        自动将 setup 中的 accelerate_config / logging_config / checkpoint_config
        传递给 Trainer 构造函数和 setup_logger().

        Args:
            setup: setup_from_yaml() 返回的 TrainSetup 对象.

        Returns:
            Trainer: 已配置好回调(日志,检查点,进度条)的训练器实例.

        Example:
            >>> setup = setup_from_yaml('config/vgg_cifar100.yaml')
            >>> model = setup.create_model()
            >>> trainer = Trainer.from_setup(setup)
            >>> trainer.fit(model, setup.train_loader, setup.val_loader)
        """
        trainer = cls(
            max_epochs=setup.num_epochs,
            device=setup.device,
            precision=setup.precision
            if setup.precision not in (None, "", "32", "fp32", "float32")
            else None,
            gradient_accumulation_steps=setup.gradient_accumulation_steps,
            grad_clip_max_norm=setup.grad_clip_max_norm,
            grad_clip_norm_type=setup.grad_clip_norm_type,
            callbacks=list(getattr(setup, "callbacks", [])),
            accelerate_config=setup.accelerate_config,
            fsdp=getattr(setup, "fsdp", None),
            nan_monitor=getattr(setup, "nan_monitor", True),
            nan_patience=getattr(setup, "nan_patience", 3),
            fail_on_callback_error=getattr(setup, "fail_on_callback_error", False),
        )

        log_cfg = setup.logging_config
        ckpt_cfg = setup.checkpoint_config
        checkpoint_dir = ckpt_cfg.get("dirpath") or log_cfg.get(
            "output_dir",
            "./others/checkpoints",
        )

        trainer.setup_logger(
            log_dir=log_cfg.get("log_dir", "./others/logs"),
            checkpoint_dir=checkpoint_dir,
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=ckpt_cfg.get("save_top_k", 1),
            every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
            enable_tensorboard=log_cfg.get("enable_tensorboard", True),
            enable_console=log_cfg.get("enable_console", False),
            enable_checkpoint=bool(ckpt_cfg),
        )

        return trainer

    def _get_device_from_spec(self) -> torch.device:
        """根据 device_spec 选择设备 - 直接使用用户提供的设备规格, 让PyTorch处理错误"""
        # 如果用户指定了具体设备, 直接使用, 让PyTorch处理错误
        if self.device_spec is not None:
            return torch.device(self.device_spec)
        # 如果用户没有指定设备或指定为"auto", 使用cuda如果可用, 否则cpu
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def setup_logger(
        self,
        experiment_name: str = "experiment",
        log_dir: str = "./others/logs",
        checkpoint_dir: str = "./others/checkpoints",
        monitor: Optional[str] = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        log_every_n_steps: int = 50,
        every_n_epochs: int = 1,
        enable_tensorboard: bool = True,
        enable_checkpoint: bool = True,
        enable_tqdm: bool = True,
        enable_console: bool = False,
        enable_total_progress: bool = False,
        tqdm_metric_keys: Optional[List[str]] = None,
        console_log_frequency: Optional[int] = None,
        tensorboard_log_frequency: int = 1,
        save_last: bool = True,
    ):
        """
        快速配置常用回调函数(日志,检查点,进度条等)
        """
        # 添加TqdmCallback(进度条管理)
        if enable_tqdm:
            self.callback_list.add_callback(
                TqdmCallback(
                    log_frequency=log_every_n_steps,
                    show_metrics=True,
                    metric_keys=tqdm_metric_keys,
                    leave=True,
                    total_progress=enable_total_progress,
                )
            )

        # 添加ConsoleCallback(控制台日志记录)
        if enable_console:
            self.callback_list.add_callback(
                ConsoleCallback(
                    log_frequency=console_log_frequency if console_log_frequency is not None else log_every_n_steps,
                    log_train=True,
                    log_validation=True,
                    log_validation_frequency="epoch",
                )
            )

        # 添加TensorBoardCallback(TensorBoard日志记录)
        if enable_tensorboard:
            self.callback_list.add_callback(
                TensorBoardCallback(
                    experiment_name=experiment_name,
                    log_dir=log_dir,
                    log_frequency=tensorboard_log_frequency,
                    log_train=True,
                    log_validation=True,
                    log_validation_frequency="epoch",
                )
            )

        # 添加ModelCheckpoint(检查点保存)
        if enable_checkpoint:
            self.callback_list.add_callback(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor=monitor,
                    mode=mode,
                    save_top_k=save_top_k,
                    every_n_epochs=every_n_epochs,
                    save_last=save_last,
                )
            )

    @property
    def steps_per_epoch(self) -> int:
        """获取每轮训练的步数(支持虚拟epoch)"""
        if hasattr(self, "_virtual_steps_per_epoch") and self._virtual_steps_per_epoch is not None:
            return self._virtual_steps_per_epoch
        return len(self._train_dataloader) if self._train_dataloader is not None else 0

    @property
    def device(self) -> torch.device:
        """获取当前设备"""
        if self._accelerator is not None:
            return self._accelerator.device
        if self._device:
            return self._device
        return torch.device("cpu")

    @property
    def accelerator(self):
        """获取 Accelerator 实例"""
        return self._accelerator

    @property
    def global_step(self) -> int:
        """获取全局训练步数"""
        if self._model and hasattr(self._model, "_total_train_steps"):
            return self._model._total_train_steps
        return self.state.global_step if hasattr(self, "state") else 0

    @property
    def accumulation_steps(self) -> int:
        """获取当前梯度累积窗口大小"""
        return self.gradient_accumulation_steps

    @property
    def micro_step(self) -> int:
        """获取全局 micro-batch 训练步数"""
        return self.global_step

    @property
    def micro_step_in_accumulation(self) -> int:
        """获取当前累积窗口内的 1-based micro step"""
        if self.micro_step <= 0:
            return 0
        return ((self.micro_step - 1) % self.accumulation_steps) + 1

    @property
    def optimizer_step(self) -> int:
        """获取按完整累积窗口推导出的优化器更新计数"""
        return self.micro_step // self.accumulation_steps

    @property
    def is_accumulation_start(self) -> bool:
        """当前 micro step 是否为一个累积窗口的起点"""
        return self.micro_step > 0 and self.micro_step_in_accumulation == 1

    @property
    def is_accumulation_boundary(self) -> bool:
        """当前 micro step 是否到达完整累积窗口边界"""
        return self.micro_step > 0 and self.micro_step % self.accumulation_steps == 0

    @property
    def should_optimizer_step(self) -> bool:
        """是否应在当前 micro step 执行 optimizer.step()"""
        return self.is_accumulation_boundary

    @property
    def current_epoch(self) -> int:
        """获取当前epoch"""
        if self._model and hasattr(self._model, "_current_epoch"):
            return self._model._current_epoch
        return self.state.current_epoch if hasattr(self, "state") else 0

    @property
    def should_stop(self) -> bool:
        """获取是否应该停止训练"""
        return self.state.should_stop if hasattr(self, "state") else False

    @should_stop.setter
    def should_stop(self, value: bool):
        """设置是否应该停止训练"""
        if hasattr(self, "state"):
            self.state.should_stop = value

    @property
    def callbacks(self) -> List[Callback]:
        """获取回调列表 - 向后兼容"""
        return self.callback_list.callbacks if hasattr(self, "callback_list") else []

    @property
    def callback_metrics(self) -> Dict[str, float]:
        """获取当前指标字典 - 向后兼容"""
        if self._model:
            return self._model.current_metrics
        return {}

    def fit(
        self,
        model: CoreModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        val_check_interval: Union[int, float] = 1.0,
        check_val_every_n_epoch: int = 1,
        inference_data: Optional[Any] = None,
    ):
        """
        执行训练 - 支持 Accelerate

        Args:
            model: CoreComponent 实例(用户的模型)
            train_dataloader: 训练数据加载器
            val_dataloader: 标准验证数据加载器 (DataLoader)
            val_check_interval: 验证间隔.
            check_val_every_n_epoch: 每隔多少个 epoch 进行一次验证.
            inference_data: 推理采样数据(Prompt列表/字典等).
        """
        # 保存参数
        self._model = model
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._inference_data = inference_data
        self._sync_gradient_accumulation_to_model(model)

        # 阶段1-3:设置 - 提前执行 setup 以确定设备
        if hasattr(model, "setup"):
            model.setup("fit")
        self._setup()

        # 打印使用的设备和已启用的回调
        if self.is_main_process():
            precision_str = self.precision if self.precision else "32 (default)"
            print(f"\n[Trainer] 使用设备: {self.device}")
            print(f"[Trainer] 数据精度: {precision_str}")
            callback_names = [type(cb).__name__ for cb in self.callbacks]
            if callback_names:
                print(f"[Trainer] 已启用的回调: {', '.join(callback_names)}")
            else:
                print("[Trainer] 未启用任何回调")

        original_steps_per_epoch = len(train_dataloader)
        val_step_interval = self._resolve_val_step_interval(
            val_check_interval,
            original_steps_per_epoch,
        )
        self._target_total_train_steps = self.max_epochs * original_steps_per_epoch

        # 如果验证间隔不是原始的 epoch 长度, 则启用虚拟 epoch 模式
        if val_step_interval != original_steps_per_epoch:
            total_steps = self._target_total_train_steps
            self.max_epochs = math.ceil(total_steps / val_step_interval)
            self._virtual_steps_per_epoch = val_step_interval
            self.state.max_epochs = self.max_epochs
        else:
            self._virtual_steps_per_epoch = None

        # 设置模型的 Accelerate 配置
        if self._accelerator:
            model._accelerator = self._accelerator

        # 执行回调的 setup
        self.callback_list.setup(trainer=self, core_module=model, stage="fit")
        self.callback_list.fit_start(trainer=self, core_module=model)

        # 阶段4:训练开始
        self.callback_list.train_start(trainer=self, core_module=model)
        model.on_train_start()

        # 创建持久化训练数据迭代器
        def _infinite_loader(loader):
            while True:
                yield from loader

        self._train_iterator = _infinite_loader(self._train_dataloader)

        # 训练循环 (现在 epoch 指向的是虚拟 epoch)
        for epoch in range(1, self.max_epochs + 1):
            if self.state.should_stop:
                break

            self.state.epoch_start(epoch)

            # 训练一个虚拟 epoch
            self._train_epoch()
            if self.is_main_process():
                self.callback_list.report_epoch_errors()
            if self.fail_on_callback_error:
                self.callback_list.raise_errors()

            # 验证或推理采样
            if (
                self._val_dataloader is not None or self._inference_data is not None
            ) and epoch % check_val_every_n_epoch == 0:
                self._validate_epoch()

            # 早停检查
            if self.state.should_stop:
                break

        # 训练结束
        self.callback_list.train_end(trainer=self, core_module=model)
        if self.is_main_process():
            self.callback_list.report_epoch_errors()
        model.on_train_end()
        if self._accelerator:
            self._accelerator.end_training()
        self.callback_list.fit_end(trainer=self, core_module=model)
        self.callback_list.teardown(
            trainer=self, core_module=model, stage="fit")

    @staticmethod
    def _resolve_val_step_interval(
        val_check_interval: Union[int, float],
        original_steps_per_epoch: int,
    ) -> int:
        if original_steps_per_epoch <= 0:
            raise ValueError("train_dataloader must contain at least one batch")
        if isinstance(val_check_interval, bool):
            raise TypeError("val_check_interval must be an int >= 1 or a float in (0, 1]")
        if isinstance(val_check_interval, int):
            if val_check_interval < 1:
                raise ValueError("integer val_check_interval must be >= 1")
            return val_check_interval
        if isinstance(val_check_interval, float):
            if not 0.0 < val_check_interval <= 1.0:
                raise ValueError("float val_check_interval must be in (0, 1]")
            return max(1, math.ceil(original_steps_per_epoch * val_check_interval))
        raise TypeError("val_check_interval must be an int >= 1 or a float in (0, 1]")

    def _train_epoch(self) -> Dict[str, Any]:
        """训练一个 (虚拟) epoch"""
        model = self._model
        if model is None or self._train_dataloader is None:
            return {}

        # 设置模式和钩子
        model.train()
        model.on_epoch_start()
        self.callback_list.epoch_start(trainer=self, core_module=model)
        model.on_train_epoch_start()

        num_steps = self.steps_per_epoch
        target_total_steps = getattr(self, "_target_total_train_steps", None)
        if target_total_steps is not None:
            remaining_steps = target_total_steps - self.state.global_step
            if remaining_steps <= 0:
                return {"epoch": self.current_epoch, "steps": 0, "avg_metrics": {}}
            num_steps = min(num_steps, remaining_steps)
        for step in range(num_steps):
            batch = next(self._train_iterator)

            # 更新全局步数
            self.state.global_step += 1

            model.on_train_step_start()
            model.on_train_batch_start()
            self.callback_list.train_batch_start(
                trainer=self, core_module=model, batch=batch, batch_idx=step
            )

            batch = self._transfer_to_device(batch)

            # 执行训练 (激活卸载由 ActivationOffloadCallback 包住训练步)
            try:
                if self._accelerator:
                    with self._accelerator.accumulate():
                        model.training_step(batch, step)
                else:
                    model.training_step(batch, step)
            except Exception as exc:
                # 先给回调异常清理机会 (如激活卸载上下文退出), 再传播原异常.
                try:
                    self.callback_list.handle_exception(
                        trainer=self, core_module=model, exception=exc
                    )
                except Exception as callback_exc:
                    logger.error("异常处理回调执行失败: %s", callback_exc)
                raise

            if self.nan_monitor:
                metric_bad = self._monitor_nan_values(model)
                grad_bad = False
                if model.should_optimizer_step:
                    grad_bad = self._monitor_nan_gradients(model)
                if metric_bad or grad_bad:
                    self._nan_steps += 1
                    if self._nan_steps >= self.nan_patience:
                        raise TrainingError(
                            f"训练在 global_step={self.state.global_step} 连续 "
                            f"{self.nan_patience} 步出现 NaN/Inf 指标或梯度"
                        )
                else:
                    self._nan_steps = 0

            model.on_train_batch_end()
            self.callback_list.train_batch_end(
                trainer=self, core_module=model, outputs={}, batch=batch, batch_idx=step
            )

        # 获取平均指标并结束 epoch
        epoch_avg_metrics = (
            model._step_metrics.get_all_epoch_avg() if hasattr(model, "_step_metrics") else {}
        )
        if hasattr(model, "_step_metrics"):
            model._step_metrics.clear_epoch()

        model.on_epoch_end()
        self.callback_list.epoch_end(trainer=self, core_module=model)

        return {"epoch": self.current_epoch, "steps": num_steps, "avg_metrics": epoch_avg_metrics}

    def _validate_epoch(self) -> Dict[str, Any]:
        """验证周期 - 统一处理标准 DataLoader 验证和样本推理采样"""
        model = self._model
        if model is None:
            return {}

        # 只有在至少有一个数据源时才执行验证
        if self._val_dataloader is None and self._inference_data is None:
            return {}

        # 进入验证周期前对齐所有 rank, 避免主进程采样时其他 rank 提前进入
        # 下一个 epoch 的集合通信.
        self.wait_for_everyone()

        # 1. 钩子开始 (沿用标准验证钩子)
        model.eval()
        model.on_validation_epoch_start()
        self.callback_list.validation_epoch_start(
            trainer=self, core_module=model)

        avg_metrics = {}

        # 2. 执行标准验证 (计算 Loss/Accuracy 等)
        if self._val_dataloader is not None:
            with torch.no_grad():
                for step, batch in enumerate(self._val_dataloader):
                    model.on_validation_step_start()
                    model.on_validation_batch_start()

                    batch = self._transfer_to_device(batch)

                    if self._accelerator:
                        model.validation_step(batch, step)
                    else:
                        model.validation_step(batch, step)

                    model.on_validation_batch_end()
                    self.callback_list.validation_batch_end(
                        trainer=self, core_module=model, outputs={}, batch=batch, batch_idx=step
                    )

            # 整理指标
            if hasattr(model, "_step_metrics"):
                avg_metrics = model._step_metrics.get_all_epoch_avg()
                model._step_metrics.clear_epoch()

        # 3. 执行推理采样 (生成式模型生图/采样)
        #    FSDP 下参数分片, 前向必须所有 rank 一起做 all-gather; DDP 下采样
        #    中的集合通信也需要所有 rank 同步. 模型通过
        #    requires_collective_sampling 声明是否集体采样, 落盘仍由模型
        #    内部用 is_main_process() 控制.
        if self._inference_data is not None:
            collective = self._requires_collective_sampling(model)
            if collective or model.is_main_process():
                with torch.no_grad():
                    processed_data = self._transfer_to_device(self._inference_data)
                    # 直接传入完整数据, 由模型自行决定如何处理(如 Batch 推理)
                    model.inference(processed_data)
            self.wait_for_everyone()

        # 4. 钩子结束
        self.wait_for_everyone()
        model.on_validation_epoch_end()
        self.callback_list.validation_epoch_end(
            trainer=self, core_module=model)
        model.on_validation_end()
        self.callback_list.validation_end(trainer=self, core_module=model)

        return {"epoch": self.current_epoch, "avg_metrics": avg_metrics}

    def wait_for_everyone(self) -> None:
        """阻塞当前进程直到所有 rank 都到达该点 (单进程时为空操作)."""
        if self._accelerator is not None and hasattr(self._accelerator, "wait_for_everyone"):
            self._accelerator.wait_for_everyone()

    def _requires_collective_sampling(self, model: CoreModel) -> bool:
        """读取模型的集体采样声明; 支持属性或方法两种写法."""
        value = getattr(model, "requires_collective_sampling", False)
        if callable(value):
            value = value()
        return bool(value)

    def _monitor_nan_values(self, model: CoreModel) -> bool:
        """检查当前步指标是否有 NaN/Inf, 返回是否异常."""
        metrics = getattr(model, "current_metrics", {}) or {}
        if not metrics:
            return False
        bad = {
            name: value
            for name, value in metrics.items()
            if not _is_finite(value)
        }
        if not bad:
            return False
        logger.warning(
            "global_step=%d 出现 NaN/Inf 指标: %s",
            self.state.global_step,
            bad,
        )
        return True

    def _monitor_nan_gradients(self, model: CoreModel) -> bool:
        """在累积窗口边界检查全局梯度范数是否有限, 返回是否异常."""
        norm = global_grad_norm(model, accelerator=self._accelerator)
        if norm is None:
            return False
        try:
            value = float(norm.detach().cpu().item())
        except Exception:
            return False
        if _is_finite(value):
            return False
        logger.warning(
            "global_step=%d 梯度范数为 NaN/Inf (norm=%s)",
            self.state.global_step,
            value,
        )
        return True

    def _transfer_to_device(self, data: Any) -> Any:
        """辅助方法:将 batch/采样数据递归移动到当前设备"""
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            converted = {k: self._transfer_to_device(v) for k, v in data.items()}
            if type(data) is dict:
                return converted
            try:
                return type(data)(converted)
            except TypeError:
                return converted
        if isinstance(data, list):
            return [self._transfer_to_device(v) for v in data]
        if isinstance(data, tuple):
            converted = tuple(self._transfer_to_device(v) for v in data)
            if hasattr(data, "_fields"):
                return type(data)(*converted)
            return converted
        if is_dataclass(data) and not isinstance(data, type):
            converted_fields = {
                field.name: self._transfer_to_device(getattr(data, field.name))
                for field in fields(data)
            }
            return replace(data, **converted_fields)
        return data

    def is_main_process(self) -> bool:
        """判断当前进程是否为主进程"""
        # 如果没有 accelerator, 说明是单机训练, 返回 True
        if self._accelerator is None:
            return True

        # 使用 accelerator 的 is_main_process 方法来判断
        if hasattr(self._accelerator, "is_main_process"):
            value = self._accelerator.is_main_process
            return value() if callable(value) else bool(value)

        # 检查其他可能的属性
        if hasattr(self._accelerator, "is_local_main_process"):
            return self._accelerator.is_local_main_process

        # 默认情况下, 认为是主进程
        return True

    def load_checkpoint(
        self,
        model: CoreModel,
        checkpoint_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        format: str = "pt",
    ) -> Dict[str, Any]:
        """加载模型 checkpoint, 并恢复 callback state."""
        self._model = model
        checkpoint = model.load_checkpoint(
            str(checkpoint_path),
            map_location=str(map_location),
            format=format,
        )
        callback_states = checkpoint.get("callback_states", {}) if checkpoint else {}
        if callback_states:
            self.callback_list.load_state(callback_states)
        self.callback_list.load_checkpoint(
            trainer=self,
            core_module=model,
            checkpoint=checkpoint,
        )
        return checkpoint

    def _configure_optimizers(self):
        """从 model 获取优化器配置"""
        if self._model is None or not hasattr(self._model, "configure_optimizers"):
            return

        optimizers_return = self._model.configure_optimizers()
        # 使用 CoreComponent 的配置处理方法
        if hasattr(self._model, "_configure_optimizers_from_return"):
            self._model._configure_optimizers_from_return(optimizers_return)

        # 如果使用 Accelerate, 通过 Accelerator 准备优化器
        if (
            self._accelerator
            and self._model is not None
            and hasattr(self._model, "_optimizers")
            and self._model._optimizers
        ):
            self._model._optimizers = [
                self._accelerator.prepare_optimizer(_optimizer)
                for _optimizer in self._model.optimizers
            ]

            # 准备调度器
            if hasattr(self._model, "_schedules"):
                self._model._schedules = [
                    self._accelerator.prepare_scheduler(_scheduler)
                    for _scheduler in self._model.schedulers
                ]

        # Trainer 级梯度裁剪: 包装优化器, 让 grad_clip_max_norm 在每次
        # optimizer.step() 前真正生效 (手动优化模式下训练循环无法可靠拦截).
        if self.grad_clip_max_norm is not None and self._model is not None:
            self._model._optimizers = [
                _GradientClipOptimizer(_optimizer, self)
                for _optimizer in self._model._optimizers
            ]

    def _clip_optimizer_gradients(self, optimizer: Any) -> None:
        """对单个优化器的参数组执行梯度裁剪."""
        parameters: List[torch.nn.Parameter] = []
        for group in getattr(optimizer, "param_groups", []) or []:
            parameters.extend(
                param
                for param in group.get("params", [])
                if isinstance(param, torch.nn.Parameter)
            )
        if not parameters:
            return
        if self._accelerator is not None:
            self._accelerator.clip_grad_norm_(
                parameters,
                self.grad_clip_max_norm,
                norm_type=self.grad_clip_norm_type,
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.grad_clip_max_norm,
                norm_type=self.grad_clip_norm_type,
            )

    def _sync_gradient_accumulation_to_model(self, model: CoreModel) -> None:
        """同步 Trainer 的梯度累积配置到 CoreModel helper."""
        if hasattr(model, "_set_gradient_accumulation_steps"):
            model._set_gradient_accumulation_steps(self.gradient_accumulation_steps)

    def _configured_device_objects(self) -> Dict[str, Any]:
        """读取模型声明的额外设备迁移对象."""
        if self._model is None or not hasattr(self._model, "configure_device_objects"):
            return {}

        objects = self._model.configure_device_objects()
        if objects is None:
            return {}
        if not isinstance(objects, Mapping):
            raise TypeError("configure_device_objects() must return a mapping")
        return {str(name): value for name, value in objects.items() if value is not None}

    def _move_device_object(self, obj: Any) -> Any:
        """将非 accelerate.prepare 对象迁移到当前设备."""
        if hasattr(obj, "to"):
            moved = obj.to(self.device)
            return moved if moved is not None else obj
        return obj

    def _assign_model_attribute(self, name: str, value: Any) -> None:
        if self._model is not None:
            setattr(self._model, name, value)

    def _setup(self):
        """初始化设备和其他组件"""
        if getattr(self, "_is_setup", False):
            return

        # 如果指定了精度, Accelerate 配置或 FSDP, 则使用 Accelerate
        if self._should_use_accelerate():
            self._setup_accelerator()
        else:
            # 设置默认设备(不使用 Accelerate)
            self._setup_standard()

        if self._model is not None and hasattr(self._model, "on_after_device_setup"):
            self._model.on_after_device_setup()

        # 配置优化器
        self._configure_optimizers()
        self._is_setup = True

    def _setup_accelerator(self):
        """设置 Accelerate"""

        # 创建 Accelerator
        # 确保 accelerate_config 不为 None 并且是字典类型
        config = self.accelerate_config.copy() if self.accelerate_config is not None else {}

        # 处理 FSDP 配置
        if self.fsdp is not None:
            config["fsdp_plugin"] = build_fsdp_plugin(self.fsdp)
        elif isinstance(config.get("fsdp_plugin"), dict):
            # accelerate_config 里直接给 dict 时同样转成插件对象.
            config["fsdp_plugin"] = build_fsdp_plugin(dict(config["fsdp_plugin"]))

        # 如果 Trainer 初始化时指定了 precision, 覆盖 config 中的 mixed_precision
        if self.precision is not None:
            config["mixed_precision"] = self._normalize_precision()

        self._accelerator = Accelerator(**config)
        self._device = self._accelerator.device

        # 准备模型和数据
        prepare_list = []
        module_names = []
        extra_prepare_names = []
        extra_move_items = []
        unwrapped_names = set(self._unwrapped_module_names())

        # 添加模型(所有 nn.Module 属性)
        for name, value in self._model_modules().items():
            if name in unwrapped_names:
                # 冻结组件 (VAE/text encoder 等) 只做设备迁移,
                # 不进入 accelerator.prepare, 避免被 FSDP 分片.
                extra_move_items.append((name, value))
            else:
                prepare_list.append(value)
                module_names.append(name)

        for name, value in self._configured_device_objects().items():
            if isinstance(value, torch.nn.Module):
                prepare_list.append(value)
                extra_prepare_names.append(name)
            else:
                extra_move_items.append((name, value))

        # 添加数据加载器
        if self._train_dataloader is not None:
            prepare_list.append(self._train_dataloader)
        if self._val_dataloader is not None:
            prepare_list.append(self._val_dataloader)

        # 使用 Accelerator 准备
        if prepare_list:
            prepared_items = self._accelerator.prepare(*prepare_list)

            # 重新分配准备好的对象
            idx = 0
            for name in module_names:
                if idx < len(prepared_items):
                    self._assign_model_attribute(name, prepared_items[idx])
                    idx += 1

            for name in extra_prepare_names:
                if idx < len(prepared_items):
                    self._assign_model_attribute(name, prepared_items[idx])
                    idx += 1

            if self._train_dataloader is not None and idx < len(prepared_items):
                self._train_dataloader = prepared_items[idx]
                idx += 1

            if self._val_dataloader is not None and idx < len(prepared_items):
                self._val_dataloader = prepared_items[idx]

        for name, value in extra_move_items:
            self._assign_model_attribute(name, self._move_device_object(value))

        # 设置模型的 accelerator 引用
        if self._accelerator and self._model:
            self._model._accelerator = self._accelerator

    def _normalize_precision(self) -> str:
        """归一化并校验 precision 参数, 非法值直接报错."""
        raw = str(self.precision).strip().lower()
        precision_map = {
            "16": "fp16",
            "fp16": "fp16",
            "bf16": "bf16",
            "32": "no",
            "fp32": "no",
            "float32": "no",
            "no": "no",
            "fp8": "fp8",
        }
        if raw not in precision_map:
            raise TrainingError(
                f"不支持的 precision: {self.precision!r}; "
                "支持 16/fp16/bf16/32/fp32/no/fp8"
            )
        return precision_map[raw]

    def _unwrapped_module_names(self) -> List[str]:
        """读取模型声明的不需要 prepare 的模块属性名."""
        if self._model is None or not hasattr(self._model, "configure_unwrapped_modules"):
            return []
        try:
            names = self._model.configure_unwrapped_modules()
        except TypeError:
            names = []
        if names is None:
            return []
        if not isinstance(names, (list, tuple, set)):
            raise TypeError(
                "configure_unwrapped_modules() 必须返回属性名列表"
            )
        return [str(name) for name in names]

    def _setup_standard(self):
        """标准设置(不使用 Accelerate)"""
        # 准备 model 和 dataloaders
        prepare_list = []

        # 添加模型(所有 nn.Module 属性)
        for value in self._model_modules().values():
            prepare_list.append(value)
            # 设置 device
            value.to(self._device)

        for name, value in self._configured_device_objects().items():
            moved = self._move_device_object(value)
            self._assign_model_attribute(name, moved)

        # 添加 dataloaders
        if self._train_dataloader is not None:
            prepare_list.append(self._train_dataloader)
        if self._val_dataloader is not None:
            prepare_list.append(self._val_dataloader)

        # 使用简单的设备分配
        for item in prepare_list:
            if hasattr(item, "to"):
                item.to(self._device)

    def _model_modules(self) -> Dict[str, torch.nn.Module]:
        """Return registered and explicitly unregistered module attributes."""

        if self._model is None:
            return {}
        modules = dict(self._model.named_children())
        for name, value in self._model.__dict__.items():
            if isinstance(value, torch.nn.Module) and name not in modules:
                modules[name] = value
        return modules

    def test(
        self,
        model: CoreModel,
        test_dataloader: DataLoader,
    ):
        """
        执行测试

        Args:
            model: CoreComponent 实例
            test_dataloader: 测试数据加载器
        """
        # 设置模型和数据
        self._model = model
        self._test_dataloader = test_dataloader
        self._sync_gradient_accumulation_to_model(model)

        # 确保已 setup
        if not getattr(self, "_is_setup", False):
            self._setup()

        # 调用钩子
        self._model.on_test_start()
        self.callback_list.test_start(trainer=self, core_module=model)

        self.callback_list.test_epoch_start(trainer=self, core_module=model)
        self._model.on_test_epoch_start()

        # 设置为评估模式
        model.eval()

        # 执行测试
        test_loader = self._test_dataloader
        results = []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                batch = self._transfer_to_device(batch)

                model.on_test_step_start()
                model.on_test_batch_start()
                self.callback_list.test_batch_start(
                    trainer=self, core_module=model, batch=batch, batch_idx=step
                )

                # 执行测试步骤
                model.test_step(batch, step)
                
                # 调用批次结束钩子
                model.on_test_batch_end()
                self.callback_list.test_batch_end(
                    trainer=self, core_module=model, outputs={}, batch=batch, batch_idx=step
                )
                
                results.append(None)

        self.callback_list.test_epoch_end(trainer=self, core_module=model)
        self._model.on_test_epoch_end()

        self.callback_list.test_end(trainer=self, core_module=model)
        self._model.on_test_end()

        return results
