# xdl-jax 长期设计与实现指导

本文是 `xdl-jax` 的长期指导文件. 它约束项目定位, 模块边界, API 取舍和实现方式. 具体进度以 [TODO.md](TODO.md) 为准, 具体接口以 [DESIGN.md](DESIGN.md) 为准.

## 1. 项目定位

`xdl-jax` 是 XDL 的 JAX 训练后端, 目标是把 XDL 已有的训练组织能力映射到 JAX 的函数式计算, 编译和分片模型.

它提供:

- JAX-native 训练循环.
- Flax NNX 和 functional model adapter.
- Optax optimizer 和 schedule.
- Orbax checkpoint.
- Python/NumPy data source,并为 Grain 等 JAX-compatible pipeline 预留 adapter.
- host-side callback, logging 和实验状态.
- Mesh, PartitionSpec 和单主机 data parallel.
- OmegaConf 配置构建.

它不提供:

- 新的张量计算框架.
- PyTorch API 兼容层.
- LLM serving engine.
- continuous batching, scheduler 或 KV cache manager.
- 量化, 剪枝, QAT, 蒸馏和部署导出.
- MaxText 或 Tunix 的替代实现.

## 2. 总体原则

### 2.1 训练职责单一

训练循环, loss, gradient update, optimizer state, evaluation 和训练 checkpoint 属于 `xdl-jax`.

模型压缩, 图变换, 低比特 kernel, 导出和部署 benchmark 属于 XQT. 训练后的 JAX checkpoint 可以作为 XQT 输入, 但 XQT 不接管训练状态或训练循环.

### 2.2 JAX 语义优先

不要用 PyTorch 术语遮蔽 JAX 语义. 特别是:

- 不公开 `backward()`.
- 不公开 `zero_grad()`.
- 不公开 `optimizer.step()`.
- 不把可变 optimizer 对象放进 compiled train step.
- 不用 Python 属性变更表达设备侧状态更新.

JAX 的训练步应表达为:

```text
state + batch + rng
  -> loss_and_grad
  -> gradient transform
  -> new_state
```

### 2.3 显式优于隐式

以下内容必须显式:

- RNG key 的拆分和消费.
- batch shape 和 `drop_remainder`.
- dtype 和 precision policy.
- model state 与 optimizer state.
- Mesh axis 和 PartitionSpec.
- checkpoint restore 的 target structure.
- callback 是否在主进程执行.
- host/device 数据边界.

不要对旧配置字段做隐式迁移, 不要静默修正 shape, dtype 或 sharding. 不符合 schema 时直接报错.

### 2.4 编译边界清楚

训练步分为两个区域:

```text
host control plane:
  epoch, callback, logging, checkpoint, stop request

compiled data plane:
  forward, loss, grad, accumulation, clipping, optimizer update, device metrics
```

callback 属于 host control plane, 不得在 JIT 函数内部创建文件, 修改 Python list 或调用任意 Python service.

### 2.5 可恢复优先

checkpoint 不是只保存参数. 最小完整恢复对象应包含:

- model state.
- optimizer state.
- RNG state.
- micro step.
- optimizer step.
- epoch.
- gradient accumulation state.
- data iterator state, 如果数据源支持.
- config hash.
- dtype, mesh 和 sharding metadata.
- 依赖版本.

如果某一项无法恢复, 必须在 checkpoint report 中明确标记, 不能声称 exact resume.

## 3. 模型抽象指导

### 3.1 用户 API 和 runtime API 分离

用户可以使用 Flax NNX, Linen 或纯函数式模型. runtime 不应把其中任意一个模型库的 Python 对象直接当作唯一训练状态.

推荐分层:

```text
user model:
  Flax NNX / Linen / functional callable

model adapter:
  init, apply, extract_state, merge_state, state_spec

runtime:
  immutable ModelState PyTree
```

### 3.2 NNX 是首个用户层 adapter

NNX 适合承载对象式模块, 变量和状态管理, 因此作为第一种 adapter. 但 NNX 对象不得成为 `JaxTrainState` 的唯一公共表示.

首版至少定义:

- `NnxModelAdapter`.
- `FunctionalModelAdapter`.
- adapter 的初始化, 前向, 状态提取和状态合并.

### 3.3 不把 task 和 model 混在一起

模型负责网络和模型状态. task 负责:

- 如何创建模型.
- 如何计算 loss.
- 如何计算 metrics.
- 如何创建 optimizer.
- 如何创建 eval/inference 输入.

这样可以支持同一个模型用于分类, 回归, 生成或蒸馏实验, 而不把任务逻辑塞进模型类.

## 4. 优化器和 step 指导

### 4.1 Optax 是基础, 不再造 optimizer hierarchy

`xdl-jax` 的 optimizer factory 只负责:

- 从配置构建 `optax.GradientTransformation`.
- 处理 schedule.
- 处理参数 mask 或 label.
- 对 optimizer state 做初始化.

不重新实现 Adam, AdamW, SGD 等 optimizer.

### 4.2 梯度累积统一由 runtime 编排

梯度累积必须定义清楚:

- `micro_step`: 每个 batch 的计数.
- `optimizer_step`: 实际执行一次参数更新的计数.
- `accumulation_steps`: 窗口大小.
- 窗口末尾不足时的处理.

默认建议:

- `drop_remainder=true` 的固定 shape 训练.
- 只在完整累积窗口执行参数更新.
- 最后不完整窗口默认丢弃, 除非配置显式允许.

### 4.3 多优化器延后

MVP 只支持一个 Optax transformation 和一套 model state. GAN, actor-critic, alternating optimization 和多个 optimizer state 不进入第一阶段.

多优化器需要独立设计:

- 多个 loss/grad graph.
- step schedule.
- 多份 optimizer state.
- checkpoint schema.
- callback metric 语义.

不能通过把多个 transformation 放进 list 伪装支持.

## 5. RNG 指导

JAX 的随机数 key 是训练状态的一部分. 必须定义固定的 key stream:

```text
root_key
  -> init_key
  -> train_key
  -> eval_key
  -> callback/inference key
```

每个 step 内:

1. 从 state 中取当前 key.
2. 用 `jax.random.split` 生成 next key 和 step key.
3. 只使用 step key 计算当前 batch.
4. 将 next key 写回新 state.

不得在 task 内使用全局随机状态或基于 Python `random` 的隐式随机.

## 6. Sharding 指导

### 6.1 先 data parallel, 再 model parallel

第一版只实现单主机多设备 data parallel:

```text
mesh = Mesh(devices, axis_names=("data",))
batch -> PartitionSpec("data", ...)
model_state -> replicated
gradient -> data-axis reduction
```

Tensor parallel, pipeline parallel, expert parallel 和自动策略搜索必须在 data parallel 通过后单独立项.

### 6.2 pmap 不是公共核心抽象

`pmap` 可以作为兼容或 Spike 路径, 但公共设计围绕:

- `jax.Array`.
- `Mesh`.
- `NamedSharding`.
- `PartitionSpec`.
- `shard_map`.

理由是这些对象能直接表达全局数组和分片布局, 更适合后续扩展.

### 6.3 所有分片都要可报告

训练启动时应能输出:

- device topology.
- mesh shape.
- axis names.
- model state sharding.
- batch sharding.
- optimizer state sharding.
- 是否发生 reshard.

出现隐式 reshard 时至少记录 warning 和计数, 不能让性能问题不可见.

## 7. 数据指导

### 7.1 JAX batch contract

一个可交给 compiled step 的 batch 必须是:

- JAX-compatible PyTree.
- 数组 dtype 明确.
- shape 在一个 compiled region 内稳定.
- 不包含文件句柄, Python generator 或任意不可转换对象.

字符串, metadata 和路径可以在 host-side data pipeline 保留, 但不能直接进入默认 numeric train step.

### 7.2 Grain 作为可选增强

MVP 可以使用 Python iterable 或 NumPy source. Grain 接入后, 应负责:

- 确定性 shuffle.
- process-aware sharding.
- prefetch.
- iterator checkpoint.

不要为了接入 Grain 重写所有已有 XDL dataset. 使用 thin adapter.

## 8. Checkpoint 指导

### 8.1 Orbax 封装而非替代

只在 `xdl_jax.checkpoint` 中封装 Orbax. 上层不直接散落 `CheckpointManager` 的细节.

公共对象建议:

- `JaxCheckpointManager`.
- `CheckpointMetadata`.
- `RestoreOptions`.
- `CheckpointReport`.

### 8.2 restore 必须显式

restore API 应要求用户选择:

- exact restore.
- restore weights only.
- restore weights and optimizer.
- restore with new mesh.

默认使用 exact restore. weights-only 是显式降级, 不应通过缺字段自动触发.

## 9. Callback 指导

callback 复用 XDL 的生命周期命名和优先级规则, 但使用独立基类和类型定义.

建议优先实现:

- `ConsoleCallback`.
- `TimerCallback`.
- `MetricLoggerCallback`.
- `OrbaxCheckpointCallback`.
- `EarlyStoppingCallback`.
- `CompileReportCallback`.

保存, checkpoint 和数据一致性相关 callback 默认 fast-fail. 普通可视化和日志 callback 可以隔离错误.

## 10. 配置指导

配置沿用 XDL 的 OmegaConf 和 `target + params` 组织方式, 但使用独立 schema 和 registry.

必须显式声明:

```yaml
backend: jax
```

不得把 JAX target 发送到当前 Torch builder, 也不得把 Torch target 发送到 `xdl_jax` builder.

配置校验不负责:

- 路径重写.
- `"null"` 字符串兼容.
- 旧字段迁移.
- 自动改变 batch shape.
- 自动改变 group size.
- 自动转换不兼容 dtype.

## 11. 性能指导

每个性能结论都要拆分:

```text
compile time
steady-state step time
host input pipeline time
device compute time
metric transfer time
checkpoint time
```

不能只报告端到端 wall time, 也不能把首次编译时间和稳态吞吐混在一起.

必须避免:

- 每步 `jax.device_get` 全量参数.
- 每步 Python callback.
- 动态 shape 导致重复编译.
- 训练循环中频繁创建 Mesh.
- 未报告的隐式 reshard.

## 12. 与 XQT 的交接

`xdl-jax` 输出给 XQT 的内容应是模型侧产物:

- JAX/Flax checkpoint.
- 物化后的 NumPy 或 safetensors 权重.
- 模型结构和参数 metadata.
- dtype 和 sharding metadata.
- 训练配置和实验指标.

XQT 消费这些产物后负责:

- 量化.
- 剪枝.
- 图变换.
- kernel/runtime 适配.
- 导出和 benchmark.

`xdl-jax` 不在训练过程中调用 XQT quantizer, 也不把 XQT runtime 放进训练循环.
