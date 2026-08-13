# xdl-jax 设计文档

状态: `0.1.0 alpha / production candidate`. 第一阶段一级项目化和第二阶段
CPU/GPU 单设备训练闭环已经落地. CPU data parallel 只保留 correctness
验证子集. 多平台实测,多 GPU 正式验收,多主机和跨拓扑 sharded checkpoint
仍未交付.

日期: 2026-08-13.

本文定义 `xdl-jax` 的架构和首版 API,并记录与当前实现的契约.

## 1. 设计目标

### 1.1 目标

- 提供 JAX-native 的标准训练闭环.
- 复用 XDL 的配置, registry, callback 和实验组织思想.
- 支持 Flax NNX 和 functional model.
- 支持 Optax optimizer.
- 支持 Orbax checkpoint 和可恢复训练.
- 支持 CPU 单设备, GPU/TPU 单设备和单主机多设备 data parallel.
- 让 compiled data plane 和 host control plane 边界清晰.
- 为后续 JAX checkpoint 到 XQT 的模型产物交接保留明确 metadata.

### 1.2 非目标

- 不兼容 `torch.nn.Module`.
- 不兼容 `torch.optim.Optimizer`.
- 不实现 PyTorch autograd bridge.
- 不实现 LLM serving.
- 不实现 QAT, quantization, pruning 或 deployment.
- 不实现完整 Tensor Parallel, Pipeline Parallel 或 Expert Parallel.
- 不实现多优化器和复杂 RL 训练.
- 不复制 MaxText, Tunix 或 PaxML.

## 2. 运行模型

`xdl-jax` 分为三个层面:

```text
configuration plane:
  OmegaConf, registry, setup, validation

host control plane:
  JaxTrainer, callback, logger, checkpoint manager, stop control

compiled data plane:
  model apply, loss, grad, accumulation, optimizer update, metrics
```

训练主循环:

```text
setup task
  -> create model adapter
  -> initialize JaxTrainState
  -> create input pipeline
  -> create compiled train/eval step
  -> for epoch:
       callback epoch start
       for batch:
         host batch validation
         place batch
         compiled step
         transfer scalar metrics at configured frequency
         callback batch end
       optional validation
       optional checkpoint
  -> teardown
```

compiled step 不应直接看到:

- callback 实例.
- logger.
- Python iterator.
- 文件路径.
- 训练器对象.
- 任意 host service.

## 3. 包和目录

一级项目目录:

```text
xdl-jax/
├── pyproject.toml
├── README.md
├── constraints/
├── docs/
├── examples/
├── tests/
└── xdl_jax/
    ├── __init__.py
    ├── errors.py
    ├── registry.py
    ├── config/
    ├── trainer/
    ├── model/
    ├── optimizer/
    ├── loss/
    ├── metric/
    ├── data/
    ├── callbacks/
    ├── checkpoint/
    ├── distributed/
    └── utils/
```

当前项目可以从 `xdl-jax/` 独立构建 wheel,不依赖根目录的 PyTorch
training stack.

## 4. 核心数据结构

### 4.1 ModelAdapter

```python
from typing import Any, Protocol

import jax


class ModelAdapter(Protocol):
    def initialize(
        self,
        rng: jax.Array,
        sample_batch: Any,
    ) -> Any:
        ...

    def apply(
        self,
        model_state: Any,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[Any, Any]:
        ...

    def state_spec(self, model_state: Any) -> Any:
        ...
```

`apply()` 的第二个返回值用于 mutable state, 例如 BatchNorm statistics. 没有 mutable state 时返回 `None` 或空 PyTree, 但必须统一语义.

### 4.2 JaxTask

```python
from collections.abc import Mapping
from typing import Any, Protocol

import jax
import optax


class JaxTask(Protocol):
    def build_model(self) -> ModelAdapter:
        ...

    def loss_and_metrics(
        self,
        model_state: Any,
        batch: Any,
        rng: jax.Array,
        *,
        training: bool,
    ) -> tuple[jax.Array, Mapping[str, jax.Array], Any]:
        ...

    def configure_optimizer(
        self,
        *,
        total_steps: int,
    ) -> optax.GradientTransformation:
        ...
```

任务只描述模型, loss, metric 和 optimizer. 训练 loop 由 `JaxTrainer` 负责.

### 4.3 JaxTrainState

建议使用 immutable dataclass 或 Flax struct:

```python
from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class JaxTrainState:
    model_state: Any
    optimizer_state: Any
    rng_key: jax.Array
    micro_step: jax.Array
    optimizer_step: jax.Array
    epoch: jax.Array
    accumulation_state: Any
```

实际实现需要确认 Orbax 对 dataclass/PyTree registration 的约束. 不能为了方便把运行时对象直接放入 state.

### 4.4 StepOutput

```python
@dataclass(frozen=True)
class StepOutput:
    loss: Any
    metrics: Any
    did_optimizer_step: Any
    gradient_norm: Any
```

`StepOutput` 只允许保存 compiled step 的结果. host callback 接收经过 `device_get` 和标量归约后的 snapshot.

## 5. Train step 设计

### 5.1 单步更新

逻辑等价于:

```text
step(state, batch):
  step_key, next_key = split(state.rng_key)
  (loss, metrics, mutable), grads = value_and_grad(loss_fn, has_aux=True)(...)
  grads = clip(grads)
  updates, new_opt_state = optimizer.update(grads, state.optimizer_state, params)
  new_model_state = apply_updates(model_state, updates, mutable)
  return new_state, output
```

实现上应区分:

- model parameters.
- mutable model variables.
- optimizer target.
- non-trainable state.

不能默认对整个 model PyTree 做 optimizer update.

### 5.2 梯度累积

MVP 推荐由 runtime 显式累积:

```text
micro batch:
  grads_i = grad(loss_i)
  accumulated_grads += grads_i / accumulation_steps

boundary:
  clip accumulated_grads
  optimizer.update(accumulated_grads)
  accumulated_grads = zero tree
  optimizer_step += 1
```

需要明确最后窗口不足时的策略:

- 默认 `drop_incomplete_accumulation=true`.
- 用户显式设置为 false 时, 对实际窗口大小重新归一化.
- 两种模式都必须写入 checkpoint metadata.

### 5.3 评估

评估不改变 optimizer state 和 model trainable parameters. 如果模型存在 mutable statistics, 必须通过 task/adapter 显式声明是否更新.

默认:

```text
training=True  -> dropout on, mutable update allowed
training=False -> dropout off, mutable update rejected or isolated
```

### 5.4 NaN 和 Inf

compiled step 返回:

- loss finite flag.
- gradient finite flag.
- optional global gradient norm.

host trainer 根据 `nan_patience` 决定是否停止. 不在设备侧抛 Python 异常.

## 6. Model adapter 设计

### 6.1 NNX adapter

职责:

- 使用 RNG 和 sample batch 初始化 NNX module.
- 将 NNX state split 为 trainable/non-trainable 部分.
- 在 compiled step 中 merge/apply.
- 将 mutable variables 合并回新 state.
- 输出结构化 state spec.

NNX API 版本变化必须集中在 adapter 中, 不得散落在 Trainer 和 callback.

### 6.2 Functional adapter

Functional adapter 的最低输入:

```text
init_fn(rng, sample_batch) -> variables
apply_fn(variables, batch, rng, training) -> outputs/variables
```

它是 runtime 的最小参考实现, 也用于 CPU 单元测试.

### 6.3 Linen adapter

不进入第一实现阶段, 但 state schema 应支持:

```text
params
batch_stats
other mutable collections
```

后续新增 Linen adapter 时, 不得改变 `JaxTrainState` 的公共字段语义.

## 7. Optimizer 设计

### 7.1 Factory

```python
def build_optimizer(
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> optax.GradientTransformation:
    ...
```

支持:

- `adam`.
- `adamw`.
- `sgd`.
- `adafactor`.
- gradient clipping chain.
- learning-rate schedule.
- parameter labels/masks.

不支持:

- Torch optimizer instance.
- mutable optimizer class.
- 自动从 model 属性读取参数.

### 7.2 参数 mask

参数 mask 应基于显式 label function 或 model state path:

```text
path -> "decay" | "no_decay" | "adapter" | "frozen"
```

最终 mask 必须能打印和测试. 不通过字符串模糊匹配静默选择参数.

## 8. Data API 设计

### 8.1 Protocol

```python
from collections.abc import Iterable, Iterator
from typing import Any, Protocol


class JaxDataSource(Protocol):
    def __iter__(self) -> Iterator[Any]:
        ...

    def __len__(self) -> int:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...
```

`state_dict` 为可选增强能力, 但如果没有它, checkpoint report 必须标记 data iterator 不可 exact restore.

### 8.2 Batch validation

每个 loader 应在第一个 batch 检查:

- PyTree structure.
- leaf dtype.
- leaf shape.
- batch dimension.
- device placement.
- 是否包含不允许的 Python object.

shape 变化时给出包含路径的错误, 例如 `batch["input_ids"]`.

### 8.3 XDL adapter

后续可以提供:

```python
from xdl_jax.data import from_xdl_dataset
```

该 adapter 只复用 XDL dataset 的样本读取和 transform 结果, 不复用 Torch DataLoader 设备迁移和 collate 假设.

## 9. Distributed strategy

### 9.1 Strategy protocol

```python
class JaxStrategy(Protocol):
    def setup(self) -> None:
        ...

    def place_batch(self, batch: Any) -> Any:
        ...

    def initialize_state(self, state: JaxTrainState) -> JaxTrainState:
        ...

    def compile_train_step(self, step_fn: Any) -> Any:
        ...

    def compile_eval_step(self, step_fn: Any) -> Any:
        ...

    def report(self) -> Mapping[str, Any]:
        ...
```

### 9.2 Single device

默认 strategy:

```text
single_device
```

不创建 Mesh, 不做 collective, 适合 CPU/GPU/TPU 单设备训练和调试.

平台选择由 `TrainerConfig.platform` 或 `SingleDeviceStrategy(platform=...)`
控制. `cpu`, `gpu` 和 `tpu` 会调用对应的 JAX backend; `auto` 使用 JAX
默认 backend. 指定 accelerator 但插件或驱动不可用时必须显式失败,不得静默
降级到 CPU.

### 9.3 Single-host data parallel

```text
Mesh(devices, ("data",))
batch sharding: PartitionSpec("data", ...)
model state: replicated
optimizer state: replicated or derived from model state
```

当前实现:

- device count validation.
- batch divisibility validation.
- input placement.
- gradient reduction.
- state sharding report.

实现方式为一维 `Mesh` + `jax.shard_map`, batch 使用
`PartitionSpec("data", ...)`, model/optimizer state 使用 replicated
placement, compiled train/eval step 内使用 `jax.lax.pmean` 做梯度和指标归约.
`pmap` 不作为公共 API.

GPU 单设备验证覆盖 CUDA 13 plugin, FP32, bfloat16, GPU placement,
compiled train/eval, loss decrease 和 Orbax exact checkpoint round trip.
JAX 默认显存预分配属于进程级运行时设置,共享 GPU 的多进程场景由启动命令
显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE` 或
`XLA_PYTHON_CLIENT_MEM_FRACTION`.

### 9.4 Multi-host

多主机不进入 MVP. 设计必须预留:

- process index/count.
- global device topology.
- initialization barrier.
- process-aware data partition.
- global checkpoint coordination.

没有真实多主机测试时, 不把字段标成 supported.

## 10. Callback 设计

### 10.1 Base class

```python
class Callback:
    priority: int = 999

    def on_fit_start(self, trainer, state) -> None:
        ...

    def on_train_batch_end(self, trainer, snapshot) -> None:
        ...

    def on_validation_epoch_end(self, trainer, snapshot) -> None:
        ...

    def on_checkpoint_end(self, trainer, report) -> None:
        ...
```

callback 签名使用 JAX 侧 trainer/state/snapshot, 不引用当前 Torch `CoreModel` 类型.

### 10.2 Snapshot

```python
@dataclass(frozen=True)
class MetricSnapshot:
    epoch: int
    micro_step: int
    optimizer_step: int
    metrics: dict[str, float]
    compile_time_s: float | None
    step_time_s: float | None
    is_main_process: bool
```

callback 读取 snapshot, 而不是直接读取设备侧 JAX Array.

### 10.3 Callback error policy

- checkpoint callback 默认 fast-fail.
- logging/progress callback 默认隔离.
- `fail_on_callback_error=true` 时在 epoch boundary 汇总并抛错.
- callback 的 state 必须可选地纳入 checkpoint.

## 11. Checkpoint 设计

### 11.1 Public API

```python
class JaxCheckpointManager:
    def save(
        self,
        step: int,
        state: JaxTrainState,
        *,
        data_state: Any = None,
        callback_state: Any = None,
        metadata: Mapping[str, Any],
    ) -> CheckpointReport:
        ...

    def restore(
        self,
        step: int | None = None,
        *,
        mode: str = "exact",
        target: Any = None,
    ) -> RestoredCheckpoint:
        ...
```

### 11.2 Restore modes

```text
exact:
  model + optimizer + rng + loop + data + callback state

weights_only:
  model state only, optimizer and loop state reset

model_and_optimizer:
  model + optimizer, RNG/data/loop state explicit reset or supplied
```

默认只允许 `exact`. 其他模式必须显式指定.

### 11.3 Metadata

```yaml
format_version: 1
xdl_jax_version: ...
jax_version: ...
flax_version: ...
optax_version: ...
orbax_version: ...
config_hash: ...
precision: ...
mesh:
  axis_names: ...
  shape: ...
sharding:
  model_state: ...
  batch: ...
loop:
  epoch: ...
  micro_step: ...
  optimizer_step: ...
restore_capabilities:
  exact_data_resume: true
  exact_rng_resume: true
```

### 11.4 XQT handoff

checkpoint metadata 只描述模型状态和训练上下文. 如果需要交给 XQT, 另提供 export/materialize helper:

```text
JAX checkpoint
  -> model state extraction
  -> host array materialization
  -> model artifact + metadata
  -> XQT session
```

不把 XQT quantization pass 放进 checkpoint manager.

## 12. Config 和 registry

### 12.1 独立 registry

`xdl_jax.registry` 独立维护:

```text
MODEL
DATASET
OPTIMIZER
SCHEDULER
LOSS
METRIC
TRANSFORM
COLLATE
CALLBACK
```

当前 XDL 的 registry 默认承载 Torch 组件. 两者不共用全局 namespace, 但可以沿用注册命名规则和 `target + params` 形式.

### 12.2 Schema

顶层必须包含:

```yaml
config_version: 1
backend: jax
runtime: ...
trainer: ...
model: ...
train_data: ...
optimization: ...
checkpoint: ...
```

schema 直接拒绝:

- `backend: torch`.
- Torch optimizer target.
- 缺少 batch size.
- 动态 shape 与 fixed-shape strategy 冲突.
- 不完整的 Mesh 配置.

### 12.3 Setup

```python
@dataclass
class JaxTrainSetup:
    task: JaxTask
    train_data: JaxDataSource
    val_data: JaxDataSource | None
    optimizer_config: Mapping[str, Any]
    trainer_config: Mapping[str, Any]
    callbacks: list[Any]
    full_config: Mapping[str, Any]
```

与当前 `TrainSetup` 不同, 不暴露 Torch model, loss_fn 和 optimizer instance.

## 13. Logging 和 benchmark

训练报告至少拆分:

- initialization time.
- first compile time.
- first executable step.
- steady-state step p50/p90.
- input pipeline time.
- metric transfer time.
- checkpoint time.
- peak host/device memory, 如果环境支持.

每条性能结果同时记录:

- device.
- platform.
- process/device count.
- dtype.
- batch shape.
- accumulation steps.
- mesh/sharding.
- warmup steps.
- measured steps.
- compilation cache setting.

不得把 CPU smoke 性能外推成 GPU/TPU 性能.

## 14. API 稳定级别

### Stable, 目标

- `xdl_jax.JaxTrainer`.
- `xdl_jax.JaxTask`.
- `xdl_jax.JaxTrainState`.
- `xdl_jax.config.setup_from_yaml`.
- `xdl_jax.callbacks.Callback`.
- `xdl_jax.checkpoint.JaxCheckpointManager`.

### Provisional

- NNX adapter.
- Grain adapter.
- distributed strategy.
- sharding report schema.
- XDL dataset adapter.

### Internal

- compiled step closure.
- Orbax handler selection.
- NNX split/merge helpers.
- host/device transfer helpers.
- metric reduction implementation.

## 15. 设计决策记录

### ADR-001: 独立项目而不是修改现有 Trainer

结论: `xdl-jax` 独立于当前 `xdl/trainer`.

原因:

- 当前 `CoreModel` 绑定 Torch module.
- 当前 `Trainer` 绑定 Torch device transfer 和 Accelerate.
- 当前 `TrainSetup` 绑定 Torch DataLoader 和 optimizer.
- JAX 的 state, compile 和 sharding 需要不同生命周期.

### ADR-002: 采用显式 immutable state

结论: runtime 以 PyTree state 为核心, NNX 只作为 adapter.

原因:

- 适应 JAX transform.
- 适应 Orbax.
- 适应 sharding.
- 降低对 Flax 某个 API 风格的锁定.

### ADR-003: 先 data parallel

结论: 第一分布式版本只做单主机 data parallel.

原因:

- 能覆盖真实多设备训练收益.
- 复杂度低于 Tensor/Pipeline/Expert Parallel.
- 可以先验证 Mesh, batch placement, reduction 和 checkpoint.

### ADR-004: Optax transformation 而不是 optimizer object

结论: 用户配置构建 `GradientTransformation`, 不构建可变 optimizer instance.

原因:

- 与 JAX 函数式训练一致.
- 便于 JIT.
- optimizer state 可以明确进入 checkpoint.

## 16. 未决问题

必须通过 Spike 解决:

1. NNX state split/merge 的目标版本稳定写法.
2. Orbax 对 dataclass state 和 sharded array 的推荐 handler.
3. `shard_map` 与梯度累积的实现方式.
4. 单设备 checkpoint 到不同 mesh 的 restore 策略.
5. Grain iterator state 的多进程恢复.
6. BatchNorm/dropout mutable state 的统一模型 contract.
