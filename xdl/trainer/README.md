# xdl.trainer

`xdl/trainer/` 是 XDL 的训练编排层，负责把任务逻辑、训练循环和配置构建结果接起来。

## 当前文件

- `trainer.py`：`Trainer` 主类
- `coreModel.py`：`CoreModel` 基类
- `trainSetupModel.py`：把 `TrainSetup` 外部组件包装成 `CoreModel`
- `trainer_state.py`：训练状态管理

## 三个关键对象

新代码推荐从子包入口导入：

```python
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
```

历史文件级路径仍保持兼容，但文档和新示例优先使用上面的稳定入口。

### `CoreModel`

承载任务逻辑。通常需要实现：

- `training_step()`
- `validation_step()`
- `configure_optimizers()`

当前采用手动优化模式，训练步里需要自行处理：

- `optimizer.zero_grad()`
- `self.manual_backward(loss)`
- `self.clip_gradients(...)`
- `optimizer.step()`

常规单优化器场景可以用 `self.manual_optimization_step(loss, ...)` 减少重复模板. 它会按 `accumulation_steps` 缩放 loss, 在累积窗口起点清梯度, 在 `should_optimizer_step` 为真时裁剪和 step, 并返回本步是否执行了 optimizer step.

手动梯度累积优先使用公开 helper，而不是直接取模私有字段：

- `accumulation_steps`
- `micro_step`
- `micro_step_in_accumulation`
- `optimizer_step`
- `is_accumulation_start`
- `is_accumulation_boundary`
- `should_optimizer_step`

指标记录支持兼容命名：

- `self.log("train_loss", value)` 保持旧行为。
- `self.log("loss", value, prefix="train")` 记录为 `train_loss`。
- `self.log_metrics({"loss": value}, prefix="val")` 记录为 `val_loss`。
- `value` 支持 Python 数值或单元素 `torch.Tensor`，内部记录为 `float`。
- 已有 `train_loss` / `train/loss` 这类键不会重复加前缀。

### `Trainer`

承载训练循环编排。负责：

- `model.setup("fit")`
- 设备与 accelerate 初始化
- epoch / step 循环
- `Tensor / dict / list / tuple / dataclass` batch 递归迁移
- callback 调度
- 验证与推理采样周期

最小用法：

```python
from xdl.trainer import CoreModel, Trainer

trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

Accelerate 路径会使用 `accelerator.accumulate()` 包裹 `training_step()`; XDL 仍保持手动优化语义, 因此任务代码需要继续用 `self.accumulation_steps` / `manual_optimization_step()` 控制实际 step 时机. 不要再叠加一套与 Trainer 不一致的私有取模逻辑.

传入 `precision`, `accelerate_config` 或 `fsdp` 时都会启用 Accelerate 路径. FSDP 通过 Accelerate 的 `FullyShardedDataParallelPlugin` 接入, `fsdp=1` 使用 FSDP1, `fsdp=2` 使用 FSDP2. 最小用法:

```python
trainer = Trainer(max_epochs=10, fsdp=2)
```

需要混合精度或更多 Accelerate 选项时可以继续传入 `accelerate_config`:

```python
trainer = Trainer(
    max_epochs=10,
    fsdp=2,
    accelerate_config={"mixed_precision": "bf16"},
)
```

外部大模型 pipeline 可以在 `CoreModel.configure_device_objects()` 返回 `{属性名: 对象}`. 标准路径会调用对象的 `.to(device)`, Accelerate 路径会 prepare 其中的 `nn.Module`, 其他对象同样走 `.to(device)`. 设备设置完成后会调用 `CoreModel.on_after_device_setup()`.

### `TrainSetupModel`

桥接 YAML 配置流和 `Trainer.fit()`。

用法：

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

`Trainer.from_setup()` 会消费配置中的 `max_epochs`、`precision`、`gradient_accumulation_steps` 和梯度裁剪字段；默认 `precision: "32"` 保持普通 32-bit 路径。

## 当前边界

- `Trainer` 负责编排，不负责具体任务前向和损失细节
- `CoreModel` 负责任务逻辑，不负责日志、检查点等横切能力
- 日志、检查点、早停、监控优先通过 `xdl/callbacks/` 接入
- 大模型、外部 pipeline、LoRA 等重组件适合放到 `model.setup("fit")` 中惰性初始化
- `Trainer.load_checkpoint()` 会恢复模型 checkpoint 中的 callback state

## 阅读顺序

建议按下面顺序看：

1. [coreModel.py](/root/workspace/xdl/xdl/trainer/coreModel.py)
2. [trainer.py](/root/workspace/xdl/xdl/trainer/trainer.py)
3. [trainSetupModel.py](/root/workspace/xdl/xdl/trainer/trainSetupModel.py)
4. [../../docs/XDL.md](/root/workspace/xdl/docs/XDL.md)
5. [../../docs/CONFIG.md](/root/workspace/xdl/docs/CONFIG.md)
6. [../../docs/API.md](/root/workspace/xdl/docs/API.md)
