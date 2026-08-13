# xdl-jax API 边界

## Stable candidate

以下对象是 `0.1.0` 的公开训练入口:

- `JaxTask`: 创建 model adapter,计算 loss/metrics,构建 Optax optimizer.
- `ModelState`: 参数和 mutable state 的 immutable 容器.
- `JaxTrainState`: model, optimizer, RNG, accumulation 和 loop state.
- `JaxTrainer`: 单设备训练和验证循环.
- `TrainerConfig`: 训练步, precision, platform 和错误策略.
- `FunctionalModelAdapter`: 纯函数式模型接入.
- `NNXModelAdapter`: Flax NNX 模型接入.
- `JaxCheckpointManager`: Orbax state 保存和显式 restore mode.
- `CheckpointCallback`: epoch boundary checkpoint.
- `export_model_artifact`: 只导出模型侧数组和 metadata.

## State contract

`ModelState.params` 是唯一交给 Optax 更新的 PyTree.
`ModelState.mutable` 保存 BatchNorm statistics, Dropout RNG state 或其他
前向更新变量. `JaxTrainState` 另外保存 optimizer state, RNG, accumulation
grads 和 loop counters.

NNX adapter 在每次 apply 前复制 mutable variable wrappers,避免 Flax NNX
原地更新泄漏到上一个 immutable state. `train()`/`eval()` 改变的 GraphDef
静态属性是合法状态转换.

## Checkpoint contract

`exact` restore 需要目标 `JaxTrainState` 结构. `weights_only` 和
`model_and_optimizer` 必须显式指定. 当前默认数据源只报告
`exact_iterator_resume: false`; 不得把它描述为完整数据迭代器恢复.

## Boundary

`xdl-jax` 不承载:

- PyTorch `Module`, `DataLoader` 或 optimizer.
- 量化,剪枝,QAT,部署 runtime 和 serving.
- 多主机训练.
- 未经真实设备证据验收的 TPU 或多 GPU 能力.

