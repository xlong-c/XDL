# xdl.trainer

`xdl/trainer/` 是 XDL 的训练编排层,负责把任务逻辑,训练循环和配置构建结果接起来.

## 当前文件

- `trainer.py`:`Trainer` 主类
- `core_model.py`:`CoreModel` 基类
- `train_setup_model.py`:把 `TrainSetup` 外部组件包装成 `CoreModel`
- `trainer_state.py`:训练状态管理

## 三个关键对象

新代码推荐从子包入口导入:

```python
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
```

历史文件级路径仍保持兼容,但文档和新示例优先使用上面的稳定入口.

### `CoreModel`

承载任务逻辑.通常需要实现:

- `training_step()`
- `validation_step()`
- `configure_optimizers()`

当前采用手动优化模式,训练步里需要自行处理:

- `optimizer.zero_grad()`
- `self.manual_backward(loss)`
- `self.clip_gradients(...)`
- `optimizer.step()`

常规单优化器场景可以用 `self.manual_optimization_step(loss, ...)` 减少重复模板. 它会按 `accumulation_steps` 缩放 loss, 在累积窗口起点清梯度, 在 `should_optimizer_step` 为真时裁剪和 step, 并返回本步是否执行了 optimizer step.

手动梯度累积优先使用公开 helper,而不是直接取模私有字段:

- `accumulation_steps`
- `micro_step`
- `micro_step_in_accumulation`
- `optimizer_step`
- `is_accumulation_start`
- `is_accumulation_boundary`
- `should_optimizer_step`

指标记录支持兼容命名:

- `self.log("train_loss", value)` 保持旧行为.
- `self.log("loss", value, prefix="train")` 记录为 `train_loss`.
- `self.log_metrics({"loss": value}, prefix="val")` 记录为 `val_loss`.
- `value` 支持 Python 数值或单元素 `torch.Tensor`,内部记录为 `float`.
- 已有 `train_loss` / `train/loss` 这类键不会重复加前缀.

### `Trainer`

承载训练循环编排.负责:

- `model.setup("fit")`
- 设备与 accelerate 初始化
- epoch / step 循环
- `Tensor / dict / list / tuple / dataclass` batch 递归迁移
- callback 调度
- 验证与推理采样周期

最小用法:

```python
from xdl.trainer import CoreModel, Trainer

trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

### `TBSMCoreModel`

`TBSMCoreModel` puts TBSM's one-step scattering objective into the normal XDL
`CoreModel + Trainer` lifecycle. It owns the online generator, one or more
frozen-representor scattering fields, an optional tracker optimizer, and an EMA
generator for sampling.

The generator should expose `forward(x, labels, t)` and preferably
`to_image(x)`. `TBSMGenerator` adapts an existing conditional backbone and
uses `x * 0.5 + 0.5` as the default image conversion. A representor must expose
`feat_dim` and either `extract_featmap(x)`, `extract(x)`, or `forward(x)`.

```python
from xdl.model.generate import (
    IdentityRepresentor,
    RepresentationScatteringField,
    TBSMGenerator,
)
from xdl.trainer import TBSMCoreModel, Trainer

field = RepresentationScatteringField(
    representor=IdentityRepresentor(),
    lambda_weight=0.0,
    rho=0.0,
    num_classes=10,
)
model = TBSMCoreModel(
    generator=TBSMGenerator(backbone),
    representation_fields=[field],
    gen_lr=1e-5,
    t_sampling=[],  # uniform time; use [1.0] for pure-noise input
)
Trainer(max_epochs=100, device="cuda").fit(model, train_loader, val_loader)
samples = model.sample(noise, labels)
```

Training batches may be `(images, labels)` or mappings with
`images`/`image`/`x` and `labels`/`label`/`y`. Gradient accumulation remains
owned by `Trainer`; TBSMCoreModel uses the standard `CoreModel` accumulation
helpers and performs generator/tracker steps at the same boundary.

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

### 大模型多卡训练要点

#### FSDP 配置

`fsdp` 参数除了 `1` / `2` 快捷分支, 还接受配置 dict,`FSDPConfig` 或
`FullyShardedDataParallelPlugin` 实例:

```python
from xdl.trainer import Trainer

trainer = Trainer(
    max_epochs=10,
    fsdp={
        "fsdp_version": 1,
        "sharding_strategy": "FULL_SHARD",
        "auto_wrap_policy": "transformer_based_wrap",
        "transformer_cls_names_to_wrap": [
            "MyTransformerBlock",
        ],
        "limit_all_gathers": True,
        "use_orig_params": True,
    },
)
```

`transformer_cls_names_to_wrap` 支持按类名自动生成 transformer block 的
wrap policy, 不再需要手写 `auto_wrap_policy` 闭包. `cpu_offload` 直接传
bool 即可, 但注意参数 CPU offload 在多卡训练中通常极慢, 优先用下面的激活卸载.

冻结组件 (VAE / text encoder 等) 不希望被 FSDP 分片时, 在 `CoreModel` 里
声明跳过 prepare:

```python
class MyModel(CoreModel):
    def configure_unwrapped_modules(self) -> List[str]:
        return ["vae", "text_encoder"]  # 只做 .to(device), 不进入 accelerator.prepare
```

#### FSDP 下 checkpoint 的正确姿势

- 默认 `pt` 格式在 FSDP 下会自动对所有 rank 做 `FULL_STATE_DICT` 聚合,
  只由主 rank 落盘, 不再出现"非主 rank 提前 return 导致挂起/分片权重"的问题.
- FSDP 下 `save_optimizer=True` 会**明确报错**而不是静默存坏: 优化器/调度器
  状态在 FSDP 里是分片的, 需要完整恢复时用
  `ModelCheckpoint(..., format="accelerator")` (内部走 `accelerator.save_state`);
  只想存权重型 checkpoint 就显式传 `save_optimizer=False, save_scheduler=False`.
- 加载时 pt 格式会在 `FULL_STATE_DICT` 上下文内恢复, `format="accelerator"`
  走 `accelerator.load_state`, 两种路径都按 FSDP 语义处理.

#### 多卡采样 / 预览

FSDP 下参数分片, 主进程单独前向会挂起; DDP 下采样里的集合通信需要所有
rank 同步. 模型声明需要集体采样后, `Trainer` 会让所有 rank 一起执行
`inference()` / preview, 只在主进程落盘:

```python
class MyModel(CoreModel):
    def requires_collective_sampling(self) -> bool:
        return True  # 也可以直接返回 bool 属性
```

采样回调同样支持 `collective=True`:

```python
from xdl.callbacks import PreviewCallback, SamplingAnimationCallback, SaveTrainableStateCallback

preview = PreviewCallback(output_dir="runs/previews", collective=True)
animation = SamplingAnimationCallback(collective=True)
save_adapter = SaveTrainableStateCallback("runs/adapters", collective=True)
```

`collective=True` 时回调会在调用前后自动 `wait_for_everyone()`, 任务方法内部
用 `self.is_main_process()` 控制落盘; FSDP 下保存 LoRA 等可训练状态时,
任务方法应先用 `accelerator.get_state_dict(...)` 聚合再过滤.

#### 梯度裁剪与 NaN 监控

- `Trainer(grad_clip_max_norm=...)` 现在真正生效: 框架会包装优化器,
  在每次 `optimizer.step()` 前按 `grad_clip_norm_type` 裁剪, 并且只在
  Accelerate 梯度累积窗口边界 (`sync_gradients=True`) 时执行.
- 手动优化代码里 `self.clip_gradients(...)` 和
  `self.manual_optimization_step(loss, max_grad_norm=...)` 仍然可用,
  二者与 Trainer 级裁剪互不冲突.
- `nan_monitor=True` (默认) 会在每步检查指标,在每个累积窗口边界检查全局
  梯度范数; 连续 `nan_patience` (默认 3) 步出现 NaN/Inf 时抛
  `TrainingError` 中止训练. `EarlyStopping` 对 NaN/Inf 指标也按"无改善"
  计数, 不再一路 NaN 训到底.

#### 显存管理

梯度检查点与激活卸载是横切能力, 以 callback 方式接入:

```python
from xdl.callbacks import ActivationOffloadCallback, GradientCheckpointingCallback
from xdl.trainer import Trainer

trainer = Trainer(
    max_epochs=10,
    callbacks=[
        GradientCheckpointingCallback(),               # 需要模型实现 gradient_checkpointing_enable()
        ActivationOffloadCallback(min_bytes=32 << 20), # 反向用的大激活同步卸载到 pinned CPU
    ],
)
```

`ActivationOffloadCallback` 用 `saved_tensors_hooks` 包住每个训练步, 只搬运
超过阈值的 CUDA 张量, 小张量留在 GPU 避免 stride/对齐语义被破坏; 卸载使用
同步拷贝, 避免非阻塞 D2H/H2D 的竞态. 训练步异常时回调通过 `on_exception`
自动清理上下文.

工具函数本体在 `xdl.utils` (`activation_offload_context`,
`enable_gradient_checkpointing`, `global_grad_norm`), 需要显式调用 (比如
任务代码想自己包某个前向) 时可以直接从 `xdl.utils` 导入, callback 只是
薄封装.

#### 条件预编码缓存

"text encoder 编码一次 -> 缓存到磁盘 -> 释放 encoder -> 训练时复用向量"的
流程属于任务侧逻辑, 在 `CoreModel.setup()` / `on_after_device_setup()` 里
自行实现"自己存,自己加载"即可, 框架不提供设施也不介入; 多 rank 同步时
用 `self.wait_for_everyone()` 对齐.

#### 回调错误处理

- 保存类回调 (`ModelCheckpoint`, `SaveTrainableStateCallback`) 默认
  `fast_fail=True`, 保存失败立刻抛错, 不再"训练照跑, checkpoint 悄悄没了".
- 其余回调默认仍隔离执行; 每个 epoch 结束后 `CallbackList` 汇总打印本周期
  失败的回调, 传 `Trainer(fail_on_callback_error=True)` 会把汇总升级为
  `TrainingError`.
- 日志类回调 (tqdm / tensorboard / console / wandb / loguru) 现在只会在主
  rank 初始化与输出, DDP 4 卡不再出现 4 个 writer / 4 条进度条.

### `TrainSetupModel`

桥接 YAML 配置流和 `Trainer.fit()`.

用法:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

`Trainer.from_setup()` 会消费配置中的 `max_epochs`,`precision`,`gradient_accumulation_steps` 和梯度裁剪字段;默认 `precision: "32"` 保持普通 32-bit 路径.

## 当前边界

- `Trainer` 负责编排,不负责具体任务前向和损失细节
- `CoreModel` 负责任务逻辑,不负责日志,检查点等横切能力
- 日志,检查点,早停,监控优先通过 `xdl/callbacks/` 接入
- 大模型,外部 pipeline,LoRA 等重组件适合放到 `model.setup("fit")` 中惰性初始化
- `Trainer.load_checkpoint()` 会恢复模型 checkpoint 中的 callback state

## 阅读顺序

建议按下面顺序看:

1. [core_model.py](/root/workspace/xdl/xdl/trainer/core_model.py)
2. [trainer.py](/root/workspace/xdl/xdl/trainer/trainer.py)
3. [train_setup_model.py](/root/workspace/xdl/xdl/config/train_setup_model.py)
4. [XDL 项目结构与使用说明](/root/workspace/xdl/docs/md/README.md#xdl-项目结构与使用说明)
5. [XDL Config 系统说明](/root/workspace/xdl/docs/md/README.md#xdl-config-系统说明)
6. [XDL API 稳定边界](/root/workspace/xdl/docs/md/README.md#xdl-api-稳定边界)
