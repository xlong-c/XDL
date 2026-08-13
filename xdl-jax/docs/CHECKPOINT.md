# Checkpoint

`JaxCheckpointManager` 是 Orbax 的薄封装,默认保存:

- `JaxTrainState.model_state`.
- optimizer state.
- RNG key.
- micro/optimizer step, epoch 和 accumulation state.
- data source state.
- callback state.
- JSON metadata.

恢复必须提供同一任务初始化得到的 target state:

```python
with JaxCheckpointManager("outputs/checkpoints") as manager:
    restored = manager.restore(target=target_state, mode="exact")
```

支持的 mode:

- `exact`: 恢复完整训练状态.
- `weights_only`: 只替换模型状态,保留 target 的 optimizer/RNG/loop state.
- `model_and_optimizer`: 替换模型和 optimizer,保留 target 的 RNG/loop state.

当前 NumPy data source 不承诺恢复 Python iterator 的精确位置, metadata 会
明确写出 `exact_iterator_resume: false`.

