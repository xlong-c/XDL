from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from xdl.utils.checkpoint import (
    detect_and_load_checkpoint,
    generate_checkpoint_dirname,
)
from xdl.utils.checkpoint import (
    save_checkpoint as save_checkpoint_to_dir,
)
from xdl.utils.tools import print_model_parameters


@dataclass
class PipelineInput:
    """
    用于存储流水线输入的通用dataclass
    """

    inputs: Any = None
    targets: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOutput:
    """
    用于存储流水线输出的通用dataclass
    """

    output: Any = None  # 模型输出
    target: Any = None  # 目标值
    input: Any = None  # 输入值
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValueDictData:
    """
    用于存储多个数值信息的通用dataclass, 可以处理损失和指标
    增强版本:支持epoch级别管理和自动计算
    """

    values: Dict[str, List[float]] = field(default_factory=dict)  # 总历史值列表
    current_values: Dict[str, float] = field(default_factory=dict)  # 当前值
    epoch_values: Dict[str, List[float]] = field(
        default_factory=dict)  # 当前epoch的值
    last_epoch_avg: Dict[str, float] = field(
        default_factory=dict)  # 上一个epoch的平均值

    def log(self, name: str, value: float) -> None:
        """记录新指标值"""
        # 记录到总历史
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(value)
        self.current_values[name] = value

        # 记录到当前epoch
        if name not in self.epoch_values:
            self.epoch_values[name] = []
        self.epoch_values[name].append(value)

    def get(self, name: str) -> Optional[List[float]]:
        """获取指定名称的数值列表"""
        return self.values.get(name)

    def append(self, name: str, value: float) -> None:
        """添加新的数值(兼容旧接口)"""
        self.log(name, value)

    def get_current(self, name: str) -> Optional[float]:
        """获取指标的当前值"""
        return self.current_values.get(name)

    def get_epoch_avg(self, name: str) -> float:
        """获取指标在当前epoch的平均值"""
        epoch_vals = self.epoch_values.get(name, [])
        return sum(epoch_vals) / len(epoch_vals) if epoch_vals else 0.0

    def get_epoch_sum(self, name: str) -> float:
        """获取指标在当前epoch的总和"""
        return sum(self.epoch_values.get(name, []))

    def get_all_current(self) -> Dict[str, float]:
        """获取所有指标的当前值"""
        return self.current_values.copy()

    def get_all_epoch_avg(self) -> Dict[str, float]:
        """获取所有指标在当前epoch的平均值"""
        return {name: self.get_epoch_avg(name) for name in self.epoch_values}

    def clear(self) -> None:
        """清空所有数值列表"""
        for value in self.values.values():
            value.clear()
        for value in self.epoch_values.values():
            value.clear()
        self.epoch_values.clear()
        self.current_values.clear()
        self.last_epoch_avg.clear()

    def clear_epoch(self) -> None:
        """清空当前epoch的数据, 并将结果存入 last_epoch_avg"""
        self.last_epoch_avg = self.get_all_epoch_avg()
        for value in self.epoch_values.values():
            value.clear()
        self.epoch_values.clear()
        # 清除当前值, 以便重新开始记录新epoch的指标
        self.current_values.clear()

    def reset(self) -> None:
        """重置所有数据(保留结构)"""
        self.values.clear()
        self.epoch_values.clear()
        self.current_values.clear()
        self.last_epoch_avg.clear()

    @property
    def current(self) -> Dict[str, float]:
        """获取所有数值的当前值"""
        return self.current_values.copy()

    @property
    def avg(self) -> Dict[str, float]:
        """获取所有数值的平均值"""
        return {
            name: sum(values) / len(values) if values else 0.0
            for name, values in self.values.items()
        }

    @property
    def epoch_avg(self) -> Dict[str, float]:
        """获取当前epoch所有数值的平均值"""
        return self.get_all_epoch_avg()

    def __len__(self) -> int:
        """返回记录的指标数量"""
        return len(self.values)

    def items(self):
        """返回所有指标的名称和当前值"""
        return self.current_values.items()


class CoreModel(Module):
    """
    用户继承此类并实现:
    - training_step()
    - validation_step()
    - configure_optimizers()

    在 __init__ 中定义:
    - 模型属性 (如 self.model)
    - 损失函数 (如 self.loss_fn)
    """

    def __init__(self):
        """
        初始化 CoreComponent
        """
        super().__init__()

        # 指标存储系统(简化版本)
        # 会在验证/训练周期开始的时候被清空
        self._step_metrics = ValueDictData()  # 统一管理所有指标

        # 优化器存储(由 configure_optimizers 自动填充)
        self._optimizers: List[Optimizer] = []

        # 调度器存储(由 configure_optimizers 自动填充)
        self._schedules: List[LRScheduler] = []

        # 步骤计数器(双层设计, 私有属性, 通过 @property 暴露)
        # 累计计数器(不重置)
        self._total_train_steps = 0
        self._total_valid_steps = 0
        self._total_test_steps = 0

        # 可重置计数器(按epoch/阶段重置)
        self._train_steps_epoch = 0
        self._valid_steps_epoch = 0
        self._test_steps_epoch = 0

        # 训练状态跟踪
        self._current_epoch = 0  # 当前epoch数

        # 最新验证指标(供 callbacks 使用)
        self._latest_val_metrics: Optional[Dict[str, float]] = None

        # Type hint for conditional accelerator attribute
        self._accelerator: Optional[Accelerator] = None
        self._batch_data = None

    def setup_data(self, batch_data):
        self._batch_data = batch_data

    @property
    def batch_data(self):
        return self._batch_data

    @property
    def current_metrics(self):
        return self._step_metrics.current

    @property
    def mean_metrics(self):
        return self._step_metrics.avg

    @property
    def epoch_avg(self):
        """获取当前 epoch 的平均指标"""
        return self._step_metrics.epoch_avg

    @property
    def last_epoch_avg(self):
        """获取上一个 epoch 的平均指标"""
        return self._step_metrics.last_epoch_avg

    # ========== 核心方法(用户必须重写) ==========

    def training_step(self, batch, batch_idx: int):
        """
        单步训练逻辑(用户必须重写)

        注意:
            此方法为手动优化模式, 需要在方法中自行处理前向传播、反向传播和优化步骤
            所有指标都应通过 self.log() 记录, 此方法不应有返回值

        示例(手动优化模式):
            def training_step(self, batch, batch_idx):
                x, y = batch
                # 手动前向和反向传播
                opt = self.optimizers
                opt.zero_grad()
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                self.manual_backward(loss)
                self.clip_gradients(gradient_clip_val=1.0)
                opt.step()

                # 每步调用调度器
                sch = self.lr_schedulers()
                sch.step()

                self.log('train_loss', loss.item())

        多个调度器示例:
            def training_step(self, batch, batch_idx):
                # ... 计算损失 ...
                sch1, sch2 = self.lr_schedulers()
                sch1.step()
                sch2.step()

        ReduceLROnPlateau 使用示例:
            # 在 on_train_epoch_end 中调用
            def on_train_epoch_end(self):
                sch = self.lr_schedulers()
                # 传递验证指标给调度器
                avg_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
                sch.step(avg_loss)
        """
        raise NotImplementedError("请在子类中实现 training_step() 方法")

    def validation_step(self, batch, batch_idx: int):
        """
        单步验证逻辑(用户必须重写)
        """
        raise NotImplementedError("请在子类中实现 validation_step() 方法")

    def inference(self, data: Any):
        """
        推理/采样方法(可选)
        专门用于生成式模型(如扩散模型采样、LLM生成)

        Args:
            data: 采样配置数据(通常是一个 Prompt 列表或字典)

        示例:
            def inference(self, data):
                # data 可能是 ['prompt1', 'prompt2']
                images = self.generate(data)
                self.log_images("samples", images)
        """
        pass

    def test_step(self, batch, batch_idx: int):
        """
        单步测试逻辑(可选, 默认调用 validation_step)
        """
        return self.validation_step(batch, batch_idx)

    @property
    def device(self):
        """获取模型所在的设备"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def configure_optimizers(self):
        """
        配置优化器(用户必须重写)

        返回格式支持:
        1. 单个优化器: return optimizer
        2. 优化器列表: return [optimizer1, optimizer2]
        3. 优化器和调度器元组: return [optimizer], [scheduler]

        示例:
            # 单个优化器
            def configure_optimizers(self):
                return torch.optim.Adam(self.model.parameters(), lr=1e-3)

            # 多个优化器
            def configure_optimizers(self):
                optimizer1 = torch.optim.Adam(self.model.parameters(), lr=1e-3)
                optimizer2 = torch.optim.SGD(self.model.parameters(), lr=1e-2)
                return [optimizer1, optimizer2]

            # 优化器 + 调度器
            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                return [optimizer], [scheduler]
        """
        raise NotImplementedError("请在子类中实现 configure_optimizers() 方法")

    # ========== 钩子函数方法(可选重写) ==========

    # ===== START 钩子 =====

    def on_train_start(self):
        """训练开始钩子"""
        print_model_parameters(self)

    def on_validation_start(self):
        """验证开始钩子"""
        self._step_metrics.reset()  # 重置指标

    def on_test_start(self):
        """测试开始钩子"""
        self._test_steps_epoch = 0
        self._step_metrics.reset()  # 重置指标

    def on_predict_start(self):
        """预测开始钩子"""
        self._step_metrics.reset()  # 重置指标

    # ===== EPOCH START 钩子 =====

    def on_train_epoch_start(self):
        """训练 epoch 开始钩子"""
        self._current_epoch += 1
        self._train_steps_epoch = 0
        self._step_metrics.reset()  # 重置指标

    def on_validation_epoch_start(self):
        """验证 epoch 开始钩子"""
        self._valid_steps_epoch = 0

    def on_test_epoch_start(self):
        """测试 epoch 开始钩子"""
        pass

    def on_predict_epoch_start(self):
        """预测 epoch 开始钩子"""
        pass

    # ===== STEP START 钩子 =====

    def on_train_step_start(self):
        """训练步长开始钩子(自动递增训练步数计数器)"""
        self._total_train_steps += 1
        self._train_steps_epoch += 1

    def on_validation_step_start(self):
        """
        验证步长开始钩子(用户可重写)
        自动递增验证步数计数器
        """
        self._total_valid_steps += 1
        self._valid_steps_epoch += 1

    def on_test_step_start(self):
        """
        测试步长开始钩子(用户可重写)
        自动递增测试步数计数器
        """
        self._total_test_steps += 1
        self._test_steps_epoch += 1

    def on_predict_step_start(self):
        """
        预测步长开始钩子(用户可重写)
        """
        pass

    # ===== BATCH START 钩子 =====

    def on_train_batch_start(self):
        """训练 batch 开始钩子"""
        pass

    def on_validation_batch_start(self):
        """验证 batch 开始钩子"""
        pass

    def on_test_batch_start(self):
        """测试 batch 开始钩子"""
        pass

    def on_predict_batch_start(self):
        """预测 batch 开始钩子"""
        pass

    # ===== STEP END 钩子 =====

    def on_train_step_end(self):
        """训练步长结束钩子"""
        pass

    def on_validation_step_end(self):
        """
        验证步长结束钩子(用户可重写)
        """
        pass

    def on_test_step_end(self):
        """
        测试步长结束钩子(用户可重写)
        """
        pass

    def on_predict_step_end(self):
        """
        预测步长结束钩子(用户可重写)
        """
        pass

    # ===== EPOCH END 钩子 =====

    def on_train_epoch_end(self):
        """训练 epoch 结束钩子"""
        pass

    def on_validation_epoch_end(self):
        """验证 epoch 结束钩子"""
        pass

    def on_test_epoch_end(self):
        """测试 epoch 结束钩子"""
        pass

    def on_predict_epoch_end(self):
        """预测 epoch 结束钩子"""
        pass

    # ===== BATCH END 钩子 =====

    def on_train_batch_end(self):
        """训练 batch 结束钩子"""
        pass

    def on_validation_batch_end(self):
        """验证 batch 结束钩子"""
        pass

    def on_test_batch_end(self):
        """测试 batch 结束钩子"""
        pass

    def on_predict_batch_end(self):
        """预测 batch 结束钩子"""
        pass

    # ===== END 钩子 =====

    def on_train_end(self):
        """训练结束钩子"""
        pass

    def on_validation_end(self):
        """验证结束钩子"""
        pass

    def on_test_end(self):
        """测试结束钩子"""
        pass

    def on_predict_end(self):
        """预测结束钩子"""
        pass

    # ===== 通用钩子 =====

    def setup(self, stage: str):
        """
        初始化设置钩子(由 Trainer 调用)
        Args:
            stage: 阶段名称 ('fit', 'validate', 'test', 'predict')
        """
        pass

    def on_epoch_start(self):
        """Epoch开始钩子"""
        pass

    def on_epoch_end(self):
        """Epoch结束钩子"""
        pass

    def on_save_checkpoint(self):
        """保存检查点钩子"""
        pass

    def on_load_checkpoint(self):
        """加载检查点钩子"""
        pass

    def on_device_change(self):
        """设备变更钩子"""
        pass

    # ========== 预测相关方法 ==========

    def predict_step(self, batch):
        """
        单步预测逻辑(可选, 默认调用 __call__)
        """
        return self(batch)

    def __call__(self, batch, **kwargs):
        """
        前向传播(默认实现)
        如果模型有 nn.Module 类型的属性, 则调用第一个
        子类可以重写此方法来自定义前向传播逻辑

        Args:
            batch: 输入批次
            **kwargs: 其他参数

        Returns:
            模型输出
        """
        # 查找第一个 nn.Module 属性并调用
        for _name, value in self.__dict__.items():
            if isinstance(value, torch.nn.Module):
                return value(batch, **kwargs)
        raise NotImplementedError("模型中未找到 nn.Module 属性, 请实现 __call__ 方法")

    # ========== 设备相关方法 ==========

    def is_main_process(self) -> bool:
        """
        判断当前进程是否为主程序(主进程)

        在分布式训练环境中, 只有主进程应该执行某些操作如日志记录、保存模型等
        在单机训练环境中, 始终返回 True

        Returns:
            bool: 如果是主进程返回 True, 否则返回 False

        使用示例:
            def training_step(self, batch, batch_idx):
                # ... 计算损失和指标 ...
                self.log('train_loss', loss.item())

                # 只有主进程才执行某些操作
                if self.is_main_process():
                    print(f"Epoch {self.current_epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                    # 保存某些信息到文件等操作
        """
        # 如果没有 accelerator, 说明是单机训练, 返回 True
        if self._accelerator is None:
            return True

        # 使用 accelerator 的 is_main_process 方法来判断
        # 这是 Accelerate 库的标准方法
        if hasattr(self._accelerator, "is_main_process"):
            return self._accelerator.is_main_process

        # 如果 accelerator 没有 is_main_process 方法, 检查其他可能的属性
        # 某些版本的 accelerator 可能使用不同的属性名
        if hasattr(self._accelerator, "is_local_main_process"):
            return self._accelerator.is_local_main_process

        # 默认情况下, 认为是主进程
        return True

    # ========== 日志记录方法 ==========

    def log(self, name: str, value: Union[float, int], prefix: Optional[str] = None):
        """
        在 Component 内记录指标

        这是用户在 training_step 和 validation_step 中使用的日志方法
        增强版本:自动管理指标的当前值、历史值和epoch值

        Args:
            name: 指标名称 (例如: 'loss', 'accuracy', 'val_loss')
            value: 指标值
            step: 步骤数 (可选, 默认使用 self.global_step)
            epoch: 周期数 (可选, 默认使用 self.current_epoch)
            prefix: 前缀 (可选, 例如: 'train', 'val')

        使用示例:
            def training_step(self, batch, batch_idx):
                # ... 计算损失和指标 ...
                self.log('train_loss', loss.item())
                self.log('train_accuracy', accuracy.item())

            def validation_step(self, batch, batch_idx):
                # ... 计算验证指标 ...
                self.log('val_accuracy', val_accuracy.item())
        """
        # 确保值是数值类型
        if isinstance(value, torch.Tensor):
            value = value.item()

        value = float(value)

        # 使用统一的指标存储系统记录指标
        self._step_metrics.log(name, value)

    def log_metrics(self, metrics: Dict[str, Union[float, int]], prefix: Optional[str] = None):
        """
        批量记录指标

        Args:
            metrics: 指标字典 {'metric_name': value, ...}
            step: 步骤数 (可选)
            epoch: 周期数 (可选)
            prefix: 前缀 (可选)

        使用示例:
            def training_step(self, batch, batch_idx):
                # ... 计算 ...
                self.log_metrics({
                    'train/loss': loss.item(),
                    'train/accuracy': accuracy.item(),
                    'learning_rate': current_lr
                })
        """
        for name, value in metrics.items():
            self.log(name, value, prefix)

    def get_lr(self) -> float:
        """
        获取优化器的学习率

        Returns:
            float: 学习率
        """
        if not self._optimizers:
            return 0.0

        # 获取第一个优化器
        optimizer = self._optimizers[0]

        if optimizer and optimizer.param_groups:
            return optimizer.param_groups[0]["lr"]
        return 0.0

    # ========== 梯度相关方法 ==========

    def clip_gradients(
        self,
        model: Module,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        check_sync_gradients: bool = True,
    ):
        if gradient_clip_val is None:
            return

        # 检查是否需要同步梯度(仅在Accelerate环境下)
        if (
            check_sync_gradients
            and self._accelerator
            and hasattr(self._accelerator, "sync_gradients")
        ):
            if not self._accelerator.sync_gradients:
                return

        # 获取模型参数
        model_parameters = model.parameters()
        if not model_parameters:
            return

        # 执行梯度裁剪
        if self._accelerator:
            # 使用 Accelerate 的分布式梯度裁剪
            if gradient_clip_algorithm == "norm":
                self._accelerator.clip_grad_norm_(
                    model_parameters, gradient_clip_val)
            elif gradient_clip_algorithm == "value":
                self._accelerator.clip_grad_value_(
                    model_parameters, gradient_clip_val)
        else:
            # 使用 PyTorch 原生梯度裁剪
            if gradient_clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(
                    model_parameters, gradient_clip_val)
            elif gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    model_parameters, gradient_clip_val)

    def manual_backward(self, loss: torch.Tensor):
        """
        手动反向传播, 兼容 Accelerate 分布式训练

        Args:
            loss: 损失张量

        示例:
            def training_step(self, batch, batch_idx):
                loss = self.compute_loss(batch)
                self.manual_backward(loss)
                self.optimizer_step()
                self.zero_grad()
        """
        if self._accelerator:
            # 使用 Accelerate 的分布式反向传播
            self._accelerator.backward(loss)
        else:
            # 使用 PyTorch 原生反向传播
            loss.backward()

    # ========== Checkpoint 管理方法 ==========

    def _get_checkpoint_values(
        self, custom_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取用于checkpoint命名的值字典"""
        values = {"step": self._total_train_steps,
                  "epoch": self._current_epoch}
        values.update(self._step_metrics.current_values)
        if custom_values:
            values.update(custom_values)
        return values

    def save_checkpoint(
        self,
        base_dir: str,
        format: str = "pt",
        naming_keys: Optional[List[str]] = None,
        custom_values: Optional[Dict[str, Any]] = None,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        include_components: Optional[List[str]] = None,
    ) -> str:
        """保存检查点"""
        if not self.is_main_process():
            return ""

        save_dir = Path(base_dir) / generate_checkpoint_dirname(
            naming_keys or ["step"], self._get_checkpoint_values(custom_values)
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        if format == "accelerator" and self._accelerator:
            self._accelerator.save_state(str(save_dir))
            return str(save_dir)

        ckpt = self._prepare_checkpoint_data(
            save_optimizer, save_scheduler, include_components)
        save_checkpoint_to_dir(save_dir, ckpt, format)
        return str(save_dir)

    def _prepare_checkpoint_data(
        self,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        include_components: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """收集所有组件的状态数据"""
        # 搜集所有 nn.Module 属性 (包括未注册的)
        all_modules = dict(self.named_children())
        all_modules.update(
            {
                n: m
                for n, m in self.__dict__.items()
                if isinstance(m, torch.nn.Module) and n not in all_modules
            }
        )

        return {
            "epoch": self._current_epoch,
            "step": self._total_train_steps,
            "state_dict": {
                n: m.state_dict()
                for n, m in all_modules.items()
                if not include_components or n in include_components
            },
            "optimizer_states": {f"opt_{i}": o.state_dict() for i, o in enumerate(self._optimizers)}
            if save_optimizer
            else None,
            "scheduler_states": {
                f"sch_{i}": s.state_dict()
                for i, s in enumerate(self._schedules)
                if hasattr(s, "state_dict")
            }
            if save_scheduler
            else None,
        }

    def load_checkpoint(self, checkpoint_path: str, map_location: str = "cpu", format: str = "pt"):
        """加载检查点"""
        ckpt_path = Path(checkpoint_path)
        if format == "accelerator" and self._accelerator:
            self._accelerator.load_state(str(ckpt_path))
        else:
            self._restore_from_checkpoint(
                detect_and_load_checkpoint(ckpt_path, map_location))

    def _restore_from_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        """执行状态恢复"""
        self._current_epoch = ckpt.get("epoch", 0)
        self._total_train_steps = ckpt.get("step", 0)

        # 恢复模型权重
        for name, state in ckpt.get("state_dict", {}).items():
            module = getattr(self, name, None)
            if isinstance(module, torch.nn.Module):
                module.load_state_dict(state)

        # 恢复优化器与调度器
        for i, opt in enumerate(self._optimizers):
            if f"opt_{i}" in ckpt.get("optimizer_states", {} or {}):
                opt.load_state_dict(ckpt["optimizer_states"][f"opt_{i}"])

        for i, sch in enumerate(self._schedules):
            state = ckpt.get("scheduler_states", {} or {}).get(f"sch_{i}")
            if state and hasattr(sch, "load_state_dict"):
                sch.load_state_dict(state)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, map_location: str = "cpu", format: str = "pt", **kwargs
    ):
        """从文件直接实例化模型"""
        model = cls(**kwargs)
        model.load_checkpoint(checkpoint_path, map_location, format)
        return model

    # ========== 配置处理方法 ==========

    def _configure_optimizers_from_return(self, optimizers_return):
        """
        处理 configure_optimizers 的返回值, 配置优化器和调度器
        支持3种标准格式:
        1. 单个优化器: return optimizer
        2. 优化器列表: return [optimizer1, optimizer2]
        3. 优化器和调度器元组: return [optimizers], [schedulers]

        Args:
            optimizers_return: configure_optimizers 方法的返回值
        """
        if optimizers_return is None:
            self._optimizers = []
            self._schedules = []
            return
        elif isinstance(optimizers_return, (Tuple, List)):
            # 检查是否为空列表/元组
            if len(optimizers_return) == 0:
                self._optimizers = []
                self._schedules = []
                return

            # 检查是否是 [optimizers, schedulers] 格式
            # 这里的逻辑是：如果是两个元素的列表/元组，且第一个元素本身也是列表/元组，或者是包含调度器的格式
            if len(optimizers_return) == 2 and isinstance(optimizers_return[0], (list, tuple)):
                self._optimizers = list(optimizers_return[0])
                self._schedules = list(optimizers_return[1])
            else:
                # 否则视为优化器列表
                # 过滤掉非优化器对象（以防用户混入调度器但没按格式传）
                self._optimizers = [
                    opt for opt in optimizers_return if isinstance(opt, Optimizer)]
                # 如果列表里还有调度器，则提取出来
                self._schedules = [
                    sch for sch in optimizers_return if not isinstance(sch, Optimizer)
                ]
        else:
            # 单个优化器, 需要包装到列表中
            self._optimizers = [optimizers_return]
            self._schedules = []

    @property
    def schedulers(self):
        return self._schedules

    def _ensure_optimizers_initialized(self) -> None:
        """
        确保优化器已配置
        """
        if len(self._optimizers) > 0:
            return
        optimizers_return = self.configure_optimizers()
        self._configure_optimizers_from_return(optimizers_return)

    @property
    def optimizers(self) -> List[Optimizer]:
        """
        获取所有优化器列表

        Returns:
            List[Optimizer]: 优化器列表
        """
        if len(self._optimizers) == 0:
            self._ensure_optimizers_initialized()
        return self._optimizers

    # ========== 步骤计数器访问方法 ==========

    @property
    def total_train_steps(self) -> int:
        return self._total_train_steps

    @property
    def total_valid_steps(self) -> int:
        return self._total_valid_steps

    @property
    def total_test_steps(self) -> int:
        return self._total_test_steps

    @property
    def train_steps_epoch(self) -> int:
        return self._train_steps_epoch

    @property
    def valid_steps_epoch(self) -> int:
        return self._valid_steps_epoch

    @property
    def test_steps_epoch(self) -> int:
        return self._test_steps_epoch

    # ========== 兼容性别名(指向我们的计数器) ==========

    @property
    def current_epoch(self) -> int:
        return self._current_epoch
