# P2-09: batch_size fallback 失败时静默回退到 1

## 涉及文件

`xdl/config/setup.py:241-261`

## 问题代码

```python
selected_batch_size = trainer_batch_size
if selected_batch_size is None:
    selected_batch_size = dataloader_defaults.get("batch_size")
if selected_batch_size is None and "train" in dataloader_config:
    selected_batch_size = dataloader_config["train"].get("params", {}).get("batch_size")
if selected_batch_size is None and built_dataloaders.get("train") is not None:
    selected_batch_size = built_dataloaders["train"].batch_size

return TrainSetup(
    ...
    batch_size=int(selected_batch_size or 128),  # 最终 fallback: 128
)
```

## 问题说明

batch_size 有四级 fallback：

1. `trainer.batch_size`
2. `dataloader_defaults.batch_size`
3. `train_dataloader.params.batch_size`
4. 已构建 DataLoader 的 `batch_size`

如果全部为 None，最终 `int(selected_batch_size or 128)` 回退到 128。

**但这不是真正的问题。真正的问题是**：`_merge_dataloader_params`（`setup.py:142-165`）中，传给 PyTorch `DataLoader()` 的 params 可能也不包含 `batch_size`。DataLoader 的默认 batch_size=1，这意味着：

- `TrainSetup.batch_size` 显示 128
- 实际 DataLoader 的 batch_size 是 1
- 两个值不一致

如果用户没有在任何地方配置 batch_size，会出现"我以为 batch_size=128，实际每次取 1 条数据"的混淆。

## 修复建议

在 `setup_from_yaml` 返回前加校验：

```python
if selected_batch_size is None:
    raise ConfigValidationError(
        "batch_size must be specified in at least one of: "
        "trainer.batch_size, dataloader_defaults.batch_size, "
        "or train_dataloader.params.batch_size"
    )
```
