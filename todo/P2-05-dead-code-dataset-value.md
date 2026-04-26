# P2-05: _resolve_dataset_value 中存在死代码

## 涉及文件

`xdl/config/setup.py:69-82`

## 问题代码

```python
def _resolve_dataset_value(
    dataset_name: str,
    dataset_value: Any,
    built_datasets: Dict[str, Any],
    built_transforms: Dict[str, Any],
) -> Any:
    if dataset_name in built_datasets:
        return built_datasets[dataset_name]
    if isinstance(dataset_value, str) and dataset_value in built_datasets:  # ← 死代码
        return built_datasets[dataset_value]
    if isinstance(dataset_value, dict):
        transform_value = _resolve_dataset_transform_value(dataset_value, built_transforms)
        return build_dataset(dataset_value, transform=transform_value)
    raise ConfigValidationError(f"Unable to resolve dataset for dataloader '{dataset_name}'")
```

## 为什么是死代码

在 `_resolve_dataset_value` 被调用时（`setup.py:214`），`dataset_value` 是 `dataloader_cfg.get("dataset")`，而 dataloader 配置已经经过 OmegaConf 的 `${...}` 插值解析。

在 YAML 中，`dataset: ${train_dataset}` 是一个引用。经过 `to_plain_dict(merged_config, resolve=True)` 后，`${train_dataset}` 已被 resolve 为实际的 dict（即 train_dataset 的完整配置 dict），不再是字符串。

因此：
- `isinstance(dataset_value, str)` 永远为 `False`
- 即使假设插值未解析（保留了 `"${train_dataset}"` 字符串），`"${train_dataset}"` 也不会在 `built_datasets` 中（built_datasets 的 key 是 `"train"`, `"val"`, `"test"`）

## 影响

- 不影响功能（dict 分支正常工作）
- 代码审查时的困惑
- 如果未来有人看到这个分支并试图依赖它，会产生 bug

## 修复建议

移除该分支，简化为：

```python
def _resolve_dataset_value(dataset_name, dataset_value, built_datasets, built_transforms):
    if dataset_name in built_datasets:
        return built_datasets[dataset_name]
    if isinstance(dataset_value, dict):
        transform_value = _resolve_dataset_transform_value(dataset_value, built_transforms)
        return build_dataset(dataset_value, transform=transform_value)
    raise ConfigValidationError(f"Unable to resolve dataset for dataloader '{dataset_name}'")
```

注意：`built_datasets` 的 key 与 `dataset_name` 相同（都是 `"train"`, `"val"`, `"test"`），所以第一个 `if dataset_name in built_datasets` 分支也需要审视——调用方在构建 dataloader 之前已经构建了 datasets（`setup.py:201-204`），所以给 train_dataloader 解析 dataset 时 `"train" in built_datasets` 为 `True`，不会继续往下走到 dict 构建分支。这说明第二个 `isinstance(dataset_value, dict)` 分支在正常路径下也是走不到的——dataset 已经在 step 5 构建好了，step 7 只是通过名称引用。

这是一个更深层的问题：dataloader 配置中的 `dataset: ${train_dataset}` 经过 resolve 后变成 dict，但实际构建时用的是 step 5 已经构建好的 dataset 对象。如果 `${train_dataset}` resolve 后的 dict 与 step 5 构建的不是同一个对象，可能导致不一致。建议重新梳理这里的引用解析逻辑。
