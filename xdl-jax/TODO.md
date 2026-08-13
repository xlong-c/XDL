# xdl-jax TODO

本文是 `xdl-jax` 的可执行任务清单. 长期原则见 [GUIDE.md](GUIDE.md), 调研事实见 [research/xdl-jax/RESEARCH.md](../research/xdl-jax/RESEARCH.md), 设计接口见 [DESIGN.md](DESIGN.md).

当前阶段: `0.1.0 alpha / production candidate - 第一,二阶段完成`.

## 完成规则

- `[ ]` 未开始.
- `[>]` 当前执行中.
- `[x]` 已完成, 且有测试, smoke, artifact, report 或文档证据.
- `[-]` 明确不做或移出当前范围.

设计草案, 空模块, import 成功和 metadata-only 不能单独作为完成证据.

## P0 - 设计验证和 Spike

### P001. 建立技术验证矩阵

- [x] 固定 Python, JAX, Flax, Optax, Orbax 的首个 tested combination.
- [x] 记录 CPU-only 安装命令和最小 import.
- [x] 记录 GPU/TPU 安装不纳入基础依赖的原因.
- [x] 明确哪些测试不需要 accelerator.

验收:

- 有可复跑的环境说明.
- CPU 环境可以安装并运行最小 smoke.
- 依赖版本写入 Spike 输出.

### P002. 验证 functional train step

- [x] 用纯 JAX 参数 PyTree 实现 linear regression.
- [x] 使用 `jax.value_and_grad`.
- [x] 使用 `optax.adamw`.
- [x] 使用 `jax.jit`.
- [x] 记录首次编译和稳态 step.
- [x] 验证 loss 下降.

验收:

- CPU 上可复跑.
- 没有 compiled region 内的 Python side effect.
- state 可以作为单一 PyTree 传递.

### P003. 验证 Flax NNX adapter

- [x] 初始化 NNX MLP.
- [x] 验证 trainable state 和 mutable state 的拆分.
- [x] 验证 dropout RNG.
- [x] 验证 BatchNorm 或等价 mutable statistics.
- [x] 验证 state merge 后下一步输出连续.
- [x] 记录目标 Flax 版本和 adapter 限制.

验收:

- adapter 不依赖 Trainer 私有字段.
- mutable state 的更新和恢复语义明确.
- 失败时有明确的 API 版本说明.

### P004. 验证 RNG 和 exact resume

- [x] 固定 root key.
- [x] 定义 init/train/eval key stream.
- [x] 训练中断后恢复 RNG.
- [x] 对比连续运行和恢复运行的参数, loss 和 metrics.
- [x] 验证不同 eval 频率不会意外消耗 train RNG.

验收:

- 同配置和同数据顺序下, 中断恢复结果一致.
- RNG state 进入 checkpoint candidate.

### P005. 验证 Orbax state checkpoint

- [x] 保存 model state.
- [x] 保存 optimizer state.
- [x] 保存 RNG 和 loop counters.
- [x] 保存 metadata.
- [x] 验证 weights-only restore.
- [x] 验证 exact restore.
- [x] 验证异步保存完成状态.

验收:

- 不能静默丢失 optimizer 或 RNG.
- restore mode 有显式差异.
- metadata 可读并包含版本和 config hash.

### P006. 验证 sharding 方向

- [x] 验证 single device strategy.
- [x] 检查本机可见 JAX device 数量.
- [x] 若有多设备, 验证一维 data mesh.
- [x] 验证 batch `PartitionSpec`.
- [x] 验证 replicated model state.
- [x] 验证 gradient reduction.
- [x] 对比 single device 和 data parallel 数值结果.

验收:

- 明确 `shard_map` 或其他内部实现选择.
- 不把 `pmap` 作为长期公共 API.
- 无目标设备时将多设备任务保留为未完成, 不伪造证据.

## P1 - 包骨架和单设备 runtime

### P010. 创建独立项目骨架

- [x] 创建 `xdl-jax/pyproject.toml`.
- [x] 包名使用 `xdl-jax`.
- [x] import 包使用 `xdl_jax`.
- [x] 增加 `py.typed`.
- [x] 顶层 import 保持轻量.
- [x] 增加 CPU-only test job.
- [x] 增加 constraints 文件.
- [x] 增加 CUDA 12/13 GPU optional dependency 和 constraints.

验收:

- `import xdl_jax` 不加载 Torch training stack.
- `import xdl` 不加载 `xdl_jax`.
- 独立包可构建 wheel.

### P011. 实现核心类型

- [x] `JaxTask`.
- [x] `ModelAdapter`.
- [x] `JaxTrainState`.
- [x] `StepOutput`.
- [x] `MetricSnapshot`.
- [x] `JaxTrainerState`.
- [x] `xdl_jax.errors`.

验收:

- 所有函数有类型注解.
- PyTree 注册行为有测试.
- public 和 internal 类型边界写入 API 文档.

### P012. 实现 single-device trainer

- [x] `JaxTrainer.fit`.
- [x] `JaxTrainer.validate`.
- [x] epoch/step counters.
- [x] fixed-shape batch validation.
- [x] precision validation.
- [x] gradient clipping.
- [x] NaN/Inf monitoring.
- [x] stop request.
- [x] GPU platform selection.
- [x] GPU FP32 smoke.
- [x] GPU bfloat16 smoke.

验收:

- synthetic MLP 可以训练.
- validation 指标可以汇总.
- 首次编译时间单独记录.
- 无每步参数 host transfer.
- CUDA-enabled JAX 上 GPU device placement 和 loss decrease 有真实证据.

### P013. 实现梯度累积

- [x] 完整窗口更新.
- [x] `micro_step`.
- [x] `optimizer_step`.
- [x] incomplete window 默认丢弃.
- [x] explicit incomplete window normalization.
- [x] 累积状态保存和恢复.

验收:

- accumulation 结果与等效大 batch 在容差内一致.
- boundary 计数和 checkpoint 恢复正确.

### P014. 实现最小 model adapters

- [x] Functional adapter.
- [x] NNX adapter.
- [x] adapter init/apply contract.
- [x] mutable state contract.
- [x] dropout RNG contract.

验收:

- 两类 adapter 都能运行同一 MLP task.
- adapter 与 trainer 解耦.

## P2 - Callback, logging 和 config

### P020. 实现 JAX callback 基类

- [x] 生命周期 hook.
- [x] priority 排序.
- [x] error isolation.
- [x] fast-fail.
- [x] callback state.
- [x] compile hooks.

验收:

- callback 顺序与 priority 测试通过.
- checkpoint callback 失败不会被吞掉.

### P021. 实现基础 callback

- [x] Console.
- [x] Timer.
- [x] Metric logger.
- [ ] Progress.
- [x] Early stopping.
- [x] Compile report.
- [x] Orbax checkpoint callback.

验收:

- CPU smoke 可以通过 callback 完成日志和保存.
- 多进程输出策略有测试或明确未支持标记.

### P022. 建立独立 registry

- [x] MODEL.
- [x] DATASET.
- [x] OPTIMIZER.
- [x] SCHEDULER.
- [x] LOSS.
- [x] METRIC.
- [x] TRANSFORM.
- [x] COLLATE.
- [x] CALLBACK.

验收:

- 不与当前 Torch registry 共用 global namespace.
- duplicate, missing target 和 suggestion 行为有测试.

### P023. 实现 OmegaConf 配置链

- [x] `schema.py`.
- [x] `resolver.py`.
- [x] `builder.py`.
- [x] `JaxTrainSetup`.
- [x] `setup_from_yaml`.
- [x] `backend: jax` 校验.
- [x] `target + params`.
- [x] MLP YAML 示例.

验收:

- YAML 可以完成 CPU 训练.
- Torch target 进入 JAX builder 时显式失败.
- 未知字段显式失败.
- 不做隐式路径和字段迁移.

## P3 - Checkpoint 和数据恢复

### P030. Orbax checkpoint facade

- [x] `JaxCheckpointManager`.
- [x] `CheckpointMetadata`.
- [x] `CheckpointReport`.
- [x] exact restore.
- [x] weights-only restore.
- [x] model+optimizer restore.
- [x] async save.
- [ ] save_last.
- [x] retention policy.

验收:

- continuous run 和 restored run 结果一致.
- restore mode 不可混淆.

### P031. 数据源 protocol

- [x] Python iterable source.
- [x] NumPy source.
- [x] `drop_remainder`.
- [x] deterministic shuffle.
- [x] batch shape validation.
- [x] optional `state_dict`.
- [x] data resume report.

验收:

- 数据顺序可复现.
- 不支持 iterator restore 时 report 明确标记.

### P032. Grain adapter

- [ ] Grain source adapter.
- [ ] deterministic index policy.
- [ ] process-aware sharding.
- [ ] prefetch.
- [ ] iterator checkpoint.

验收:

- 单进程恢复一致.
- 多进程恢复有真实测试或明确阻塞记录.

## P4 - 单主机多设备

### P040. Mesh strategy

- [x] `MeshConfig`.
- [x] `JaxStrategy`.
- [x] device topology report.
- [x] batch placement.
- [x] replicated state placement.
- [x] gradient reduction.
- [x] sharding validation.

验收:

- 2-device data parallel correctness.
- sharding report 可读.
- 不支持的 shape/spec 快速失败.

### P041. 多设备 checkpoint

- [ ] 保存 sharded model state.
- [ ] 保存 sharded optimizer state.
- [ ] restore 同拓扑.
- [ ] restore 到不同 mesh 的策略.
- [ ] reshard 成本记录.

验收:

- 同拓扑 exact restore.
- 不同拓扑要么有显式转换, 要么清晰拒绝.

### P042. 性能基线

- [x] compile time.
- [x] steady-state step p50/p90.
- [ ] input pipeline time.
- [ ] metric transfer time.
- [ ] checkpoint time.
- [x] single vs multi-device comparison.

验收:

- benchmark report 包含设备和 warmup, dtype/batch shape/mesh 由调用方补充到外层报告.
- 不把首次编译混入 steady-state.

## P5 - XDL 和 XQT 交接

### P050. XDL 交接

- [x] 写清 PyTorch XDL 与 `xdl-jax` 的边界.
- [ ] 评估是否提供 `xdl` 侧可选 adapter.
- [x] 不让 Torch Trainer 依赖 JAX runtime.
- [x] 不让 JAX runtime 依赖 Torch DataLoader.

### P051. XQT 模型产物交接

- [x] JAX checkpoint state extraction.
- [x] host array materialization.
- [x] model metadata export.
- [x] dtype metadata.
- [x] sharding metadata.
- [ ] XQT input smoke.

验收:

- 交接只通过模型产物或 checkpoint.
- `xdl-jax` 不调用 XQT quantizer.
- XQT 不读取 JAX optimizer/RNG/data state 作为压缩输入.

## P6 - 文档和发布

### P060. 文档

- [x] 独立项目 README.
- [x] 安装和 CPU 验证.
- [x] model adapter guide.
- [x] config guide.
- [x] checkpoint guide.
- [x] distributed guide.
- [x] performance guide.
- [x] API boundary.

### P061. 发布门槛

- [x] CPU wheel smoke.
- [x] synthetic MLP example.
- [x] YAML example.
- [x] checkpoint resume test.
- [x] tested dependency matrix.
- [x] changelog.
- [x] known limitations.

## 明确不做

- [-] 把 JAX 类型加入当前 Torch `CoreModel`.
- [-] 复制 `manual_backward` 和 `manual_optimization_step`.
- [-] 在 `xdl-jax` 实现量化和部署.
- [-] 在 XQT 实现 JAX training loop.
- [-] 复刻 MaxText/Tunix/PaxML.
- [-] 第一阶段支持多优化器.
- [-] 第一阶段支持 Tensor/Pipeline/Expert Parallel.
- [-] 用 planned 或 metadata-only 结果作为 native 训练能力.

## 阶段门槛

### P0 -> P1

- functional step, NNX adapter, RNG 和 Orbax restore 的 Spike 有可复跑证据.
- 核心 state schema 已冻结.
- single-device 和 data parallel 的边界已确定.

### P1 -> P2

- CPU synthetic MLP 可训练和验证.
- gradient accumulation 有 correctness 测试.
- 首次 compile 和稳态 step 有报告.

### P2 -> P3

- callback 和 YAML 闭环可运行.
- registry 与 Torch registry 隔离.

### P3 -> P4

- exact checkpoint resume 通过.
- data iterator restore 能力有明确 report.

### P4 -> P5

- 2-device data parallel 通过 correctness.
- sharding 和 checkpoint report 完整.

## 完成记录

| 日期 | 阶段 | 事项 | 证据 |
| --- | --- | --- | --- |
| 2026-08-13 | P0 | 创建调研目录和初版规划文档 | `../research/xdl-jax/RESEARCH.md`, `GUIDE.md`, `DESIGN.md`, `TODO.md` |
| 2026-08-13 | P0-P5 | CPU 单设备核心实现,可恢复训练,CPU data parallel correctness,GPU single-device 和模型侧 artifact | `xdl_jax/`, `tests/`, `examples/`, CPU `20 passed + 4 skipped`, GPU full `28 passed`, 2-device CPU `4 passed + 1 skipped` |
| 2026-08-13 | P1-P2 | 一级项目正式化,NNX mutable contract,YAML checkpoint callback,版本元数据,API 文档,CPU CI 和 GPU 手动 workflow | `pyproject.toml`, `docs/`, `.github/workflows/`, `pyright 0 errors`, `ruff passed` |
