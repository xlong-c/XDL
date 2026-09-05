# xdl-jax 抽象抽离与接口对齐优化方案

本文定义 `xdl-jax` 与 PyTorch `xdl` 之间的底层抽象关系,代码复用策略以及对称接口对照规范.
目标是在不引入跨框架依赖(避免 JAX 环境强依赖 PyTorch)的前提下, 最小化 `xdl-jax` 的新增代码量, 并提供清晰对称的心智模型.

---

## 1. 现状问题与优化动机

### 1.1 现状诊断
当前 `xdl-jax` 作为面向 JAX 生态的独立训练项目, 实现了从单设备到单主机多设备的训练能力(代码规模约 3300 行). 但在代码实现层面存在以下痛点:
1. **控制层逻辑重合**: `Registry` 机制, `CallbackList` 调度器, `TimerCallback`, `ConsoleLoggerCallback`, `EarlyStoppingCallback` 等基础调度逻辑在两端心智相同.
2. **物理隔离与依赖冲突的矛盾**: 现阶段为了防止 `xdl-jax` 在纯 CPU/JAX 环境下引入庞大的 PyTorch 生态依赖, 必须保证两端在构建和运行时互不强行引用.
3. **高层心智模型缺乏显式对照**: 开发者从 PyTorch `xdl` 切换到 `xdl-jax` 时, 缺少一张权威清晰的概念与接口对照图景, 增加了学习与迁移成本.

### 1.2 优化原则
1. **计算面原生隔离, 控制面协议对齐**: 张量计算,自动微分(Autograd vs JIT),优化器状态更新等数据平面保持原生; 注册机制,配置解析,回调生命周期调度,标量指标跟踪等控制平面保持接口协议对称.
2. **避免小代码量过度设计(遵循 Python 之禅)**: 通用控制层代码总量不足 300 行时, 不单独建立额外的物理包(避免引发复杂的 monorepo 多包构建,CI 依赖链和 IDE 路径解析成本).
3. **接口严格对照**: 在保持 JAX 原生不可变特性的前提下, 顶层 API 命名,配置字段与事件钩子尽量与 `xdl` 形成一一对应.

---

## 2. 架构决策与演进路径

```text
当前阶段 (方案 A: 契约对齐, 零多包摩擦)
┌────────────────────────────────────────────────────────┐
│                   统一的接口与生命周期契约               │
├────────────────────────────┬───────────────────────────┤
│       xdl (PyTorch)        │        xdl-jax (JAX)      │
│  - 自包含轻量控制层 (Registry,│  - 自包含轻量控制层 (Registry,│
│    CallbackList, Config)   │    CallbackList, Config)  │
│  - 原生动态图与 Autograd 训练 │  - 原生 JIT 编译与不可变状态   │
└────────────────────────────┴───────────────────────────┘

后续规模化演进 (方案 B: 触发条件见 2.2)
┌────────────────────────────────────────────────────────┐
│           xdl-base (厚重的跨框架 AI 共享基座)            │
│  - 实验看板 (WandB/TensorBoard)  - 纯数学 LR 调度曲线    │
│  - 通用度量引擎 (Metrics Engine) - XQT 模型产物交付契约 │
└────────────────────────────┬───────────────────────────┘
```

### 2.1 当前阶段策略: 方案 A (协议契约对齐, 物理保持自包含)
- **原因**: 注册表,基础回调和配置加载仅有约 200 行代码. 为此单开独立顶层包(`xdl-base`), 会带来额外的 `pyproject.toml`,wheel 打包,CI 构建步骤与 IDE Pylance 诊断路径负担, 属于典型的过度设计.
- **做法**: 在 `xdl-jax` 内部保留自包含,极简的原生实现, 严格保证事件名,方法签名和行为与 `xdl` 对齐. 不需要跨目录引用任何中间包, 也不引入任何第三方框架污染.

### 2.2 未来演进触发条件: 方案 B (独立共享基座)
当且仅当出现以下跨框架通用重资产且累积代码量达到 1000+ 行时, 再正式设立独立的 `xdl-base` 共享包:
1. **实验监控集成**: 统一打通 WandB, TensorBoard, MLflow 的通用标量/图表上报;
2. **纯数学学习率调度**: CosineAnnealing, LinearWarmup 等纯 step->lr 衰减数学公式;
3. **度量统计引擎**: 滑动平均(EMA), Confusion Matrix, PR 曲线等纯 Python/NumPy 评估计算器;
4. **下游 XQT 产物契约**: 跨框架的 Safetensors / NPZ 权重序列化与结构元数据标准导出器.

---

## 3. 接口与心智对照表 (Interface Correspondence)

PyTorch `xdl` 与 JAX `xdl-jax` 形成清晰的镜像对照关系:

### 3.1 核心组件对照

| 抽象职责 | PyTorch 侧 (`xdl`) | JAX 侧 (`xdl-jax`) | 架构设计考量 |
| :--- | :--- | :--- | :--- |
| **训练编排器** | `Trainer` | `JaxTrainer` | 均通过 `.fit(data)` 和 `.validate(data)` 交互, 均接收 `TrainerConfig`. |
| **任务/模型定义** | `CoreModel` (继承 `nn.Module`) | `JaxTask` (实现 Protocol) | Torch 侧模型即任务; JAX 侧任务编排模型 Adapter 与纯函数 Loss. |
| **模型接入** | 原生 `nn.Module` | `ModelAdapter` (`Functional` / `NNX`) | JAX 侧通过 Adapter 解耦纯函数与 Flax NNX 对象, 屏蔽 mutable state. |
| **状态载体** | 隐式内部属性 (in-place) | `JaxTrainState` (不可变 PyTree) | JAX 每一步函数式输入输出新 state (params, opt_state, rng, step). |
| **优化器与调度** | `torch.optim` + `LRScheduler` | `optax.GradientTransformation` | 配置层均支持 `target + params` 声明式解析. |
| **随机数管理** | 全局隐式 `torch.manual_seed` | 显式 `jax.random.key` 树流转 | JAX 侧严禁全局随机状态, 确保完全确定性可复现. |
| **检查点管理器** | `ModelCheckpoint` (`.pt`) | `JaxCheckpointManager` (Orbax) | 统一提供 `exact`(断点全状态续训)与 `weights-only`(仅权重). |
| **生命周期回调** | `xdl.callbacks.Callback` | `xdl_jax.callbacks.Callback` | 共享一致的事件名: `on_fit_start`, `on_train_batch_end`, `on_train_epoch_end` 等. |
| **下游交付 (XQT)** | `model.state_dict()` 导出 | `export_model_artifact()` | 均输出无框架绑定的 NumPy 权重字典与 JSON metadata 给 XQT. |

### 3.2 训练任务编写代码对照

#### PyTorch `xdl`:
```python
from xdl.trainer import CoreModel, Trainer
import torch.nn as nn
import torch.nn.functional as F

class LinearTask(CoreModel):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def training_step(self, batch: dict, batch_idx: int):
        preds = self.linear(batch["x"])
        loss = F.mse_loss(preds, batch["y"])
        self.log("loss", loss, prefix="train")
        return loss

trainer = Trainer(max_epochs=5)
trainer.fit(LinearTask(4, 1), train_loader)
```

#### JAX `xdl-jax`:
```python
from xdl_jax import JaxTrainer, JaxTask, NNXModelAdapter, TrainerConfig
from flax import nnx
import optax, jax.numpy as jnp

class LinearTask(JaxTask):
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim, self.out_dim = in_dim, out_dim

    def build_model(self):
        return NNXModelAdapter(nnx.Linear(self.in_dim, self.out_dim, rngs=nnx.Rngs(0)))

    def configure_optimizer(self, *, total_steps: int):
        return optax.adamw(learning_rate=1e-3)

    def loss_and_metrics(self, model, model_state, batch, rng, *, training=True):
        preds, new_state = model.apply(model_state, batch["x"], rng=rng)
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, {"loss": loss}, new_state

trainer = JaxTrainer(LinearTask(4, 1), config=TrainerConfig(max_epochs=5))
trainer.fit(train_data)
```
