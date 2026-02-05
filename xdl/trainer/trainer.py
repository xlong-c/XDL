"""
训练器 - 简化版(删除加速器相关逻辑)

参考 PyTorch Lightning 设计理念

此文件整合了所有训练器相关的类和功能
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

# Conditional imports for type checking
if TYPE_CHECKING:
    from accelerate import Accelerator
# Accelerate 支持
from accelerate import Accelerator, FullyShardedDataParallelPlugin
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

from .coreModel import CoreModel


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
        fsdp: FSDP 版本 (None, 1, 或 2)
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
        fsdp: Optional[int] = None,
    ):
        # 训练参数
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_max_norm = grad_clip_max_norm
        self.grad_clip_norm_type = grad_clip_norm_type

        # 设备和精度
        self.device_spec = device  # Store the device specification
        self.precision = precision

        # Accelerate 配置
        self.accelerate_config = accelerate_config
        self.fsdp = fsdp

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
        self._device = None if accelerate_config is not None else self._get_device_from_spec()
        self._is_setup = False

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
        **kwargs,
    ):
        """
        快速配置常用回调函数(日志、检查点、进度条等)
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
                    log_frequency=kwargs.get(
                        "console_log_frequency", log_every_n_steps),
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
                    log_frequency=kwargs.get("tensorboard_log_frequency", 1),
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
                    save_last=kwargs.get("save_last", True),
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
            val_check_interval: 验证间隔。
            check_val_every_n_epoch: 每隔多少个 epoch 进行一次验证。
            inference_data: 推理采样数据(Prompt列表/字典等)。
        """
        # 保存参数
        self._model = model
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._inference_data = inference_data

        # 阶段1-3：设置 - 提前执行 setup 以确定设备
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

        # 计算验证步数间隔
        original_steps_per_epoch = len(train_dataloader)
        if isinstance(val_check_interval, int):
            val_step_interval = val_check_interval
        elif isinstance(val_check_interval, float):
            val_step_interval = int(
                original_steps_per_epoch * val_check_interval)
        else:
            val_step_interval = original_steps_per_epoch

        # 如果验证间隔不是原始的 epoch 长度, 则启用虚拟 epoch 模式
        if val_step_interval != original_steps_per_epoch:
            total_steps = self.max_epochs * original_steps_per_epoch
            self.max_epochs = total_steps // val_step_interval
            self._virtual_steps_per_epoch = val_step_interval
            self.state.max_epochs = self.max_epochs
        else:
            self._virtual_steps_per_epoch = None

        # 设置模型的 Accelerate 配置
        if self._accelerator:
            model._accelerator = self._accelerator

        # 执行回调的 setup
        self.callback_list.setup(trainer=self, core_module=model, stage="fit")

        # 阶段4：训练开始
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
        model.on_train_end()
        if self._accelerator:
            self._accelerator.end_training()
        self.callback_list.teardown(
            trainer=self, core_module=model, stage="fit")

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
        for step in range(num_steps):
            batch = next(self._train_iterator)

            # 更新全局步数
            self.state.global_step += 1

            model.on_train_step_start()
            model.on_train_batch_start()
            self.callback_list.train_batch_start(
                trainer=self, core_module=model, batch=batch, batch_idx=step
            )

            # 执行训练
            if self._accelerator:
                with self._accelerator.accumulate():
                    model.training_step(batch, step)
            else:
                if hasattr(batch, "__iter__"):
                    batch = [x.to(self.device) if torch.is_tensor(
                        x) else x for x in batch]
                model.training_step(batch, step)

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

                    if self._accelerator:
                        model.validation_step(batch, step)
                    else:
                        if hasattr(batch, "__iter__"):
                            batch = [x.to(self.device) if torch.is_tensor(
                                x) else x for x in batch]
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
        if self._inference_data is not None and model.is_main_process():
            with torch.no_grad():
                processed_data = self._transfer_to_device(self._inference_data)
                # 直接传入完整数据, 由模型自行决定如何处理(如 Batch 推理)
                model.inference(processed_data)

        # 4. 钩子结束
        model.on_validation_epoch_end()
        self.callback_list.validation_epoch_end(
            trainer=self, core_module=model)
        model.on_validation_end()
        self.callback_list.validation_end(trainer=self, core_module=model)

        return {"epoch": self.current_epoch, "avg_metrics": avg_metrics}

    def _transfer_to_device(self, data: Any) -> Any:
        """辅助方法:将采样数据递归移动到当前设备"""
        if torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._transfer_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._transfer_to_device(v) for v in data]
        return data

    def is_main_process(self) -> bool:
        """判断当前进程是否为主进程"""
        # 如果没有 accelerator, 说明是单机训练, 返回 True
        if self._accelerator is None:
            return True

        # 使用 accelerator 的 is_main_process 方法来判断
        if hasattr(self._accelerator, "is_main_process"):
            return self._accelerator.is_main_process

        # 检查其他可能的属性
        if hasattr(self._accelerator, "is_local_main_process"):
            return self._accelerator.is_local_main_process

        # 默认情况下, 认为是主进程
        return True

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
            self.accelerate_config
            and self._accelerator
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

    def _setup(self):
        """初始化设备和其他组件"""
        if getattr(self, "_is_setup", False):
            return

        # 如果指定了精度或提供了加速器配置，则使用 Accelerate
        if self.accelerate_config is not None or self.precision is not None:
            self._setup_accelerator()
        else:
            # 设置默认设备(不使用 Accelerate)
            self._setup_standard()

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
            if self.fsdp == 1:
                if "fsdp_plugin" not in config:
                    config["fsdp_plugin"] = FullyShardedDataParallelPlugin(
                        sharding_strategy="FULL_SHARD",
                        auto_wrap_policy=None,
                        use_orig_params=True,
                    )
            elif self.fsdp == 2:
                if "fsdp_plugin" not in config:
                    config["fsdp_plugin"] = FullyShardedDataParallelPlugin(
                        fsdp_version=2,
                        reshard_after_forward=True,
                    )
                elif isinstance(config["fsdp_plugin"], dict):
                    # 如果是字典配置, 转化为插件对象并强制设置 FSDP2
                    fsdp_dict = config["fsdp_plugin"]
                    fsdp_dict["fsdp_version"] = 2
                    if "reshard_after_forward" not in fsdp_dict:
                        fsdp_dict["reshard_after_forward"] = True
                    config["fsdp_plugin"] = FullyShardedDataParallelPlugin(**fsdp_dict)

        # 如果 Trainer 初始化时指定了 precision, 覆盖 config 中的 mixed_precision
        if self.precision is not None:
            precision_map = {
                "16": "fp16",
                "fp16": "fp16",
                "bf16": "bf16",
                "32": "no",
                "no": "no"
            }
            config["mixed_precision"] = precision_map.get(self.precision.lower(), self.precision)
            
        self._accelerator = Accelerator(**config)
        self._device = self._accelerator.device

        # 准备模型和数据
        prepare_list = []

        # 添加模型(所有 nn.Module 属性)
        if hasattr(self._model, "__dict__"):
            for name, value in self._model.__dict__.items():
                if isinstance(value, torch.nn.Module):
                    prepare_list.append(value)

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
            if hasattr(self._model, "__dict__"):
                for name, value in self._model.__dict__.items():
                    if isinstance(value, torch.nn.Module) and idx < len(prepared_items):
                        setattr(self._model, name, prepared_items[idx])
                        idx += 1

            if self._train_dataloader is not None and idx < len(prepared_items):
                self._train_dataloader = prepared_items[idx]
                idx += 1

            if self._val_dataloader is not None and idx < len(prepared_items):
                self._val_dataloader = prepared_items[idx]

        # 设置模型的 accelerator 引用
        if self.accelerate_config and self._accelerator and self._model:
            self._model._accelerator = self._accelerator

    def _setup_standard(self):
        """标准设置(不使用 Accelerate)"""
        # 准备 model 和 dataloaders
        prepare_list = []

        # 添加模型(所有 nn.Module 属性)
        if hasattr(self._model, "__dict__"):
            for _name, value in self._model.__dict__.items():
                if isinstance(value, torch.nn.Module):
                    prepare_list.append(value)
                    # 设置 device
                    value.to(self._device)

        # 添加 dataloaders
        if self._train_dataloader is not None:
            prepare_list.append(self._train_dataloader)
        if self._val_dataloader is not None:
            prepare_list.append(self._val_dataloader)

        # 使用简单的设备分配
        for item in prepare_list:
            if hasattr(item, "to"):
                item.to(self._device)

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
                # 确保batch在正确的设备上
                if hasattr(batch, "__iter__"):
                    batch = [x.to(self.device) if torch.is_tensor(
                        x) else x for x in batch]

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
