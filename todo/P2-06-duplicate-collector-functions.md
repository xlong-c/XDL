# P2-06: 三层收集函数完全重复

## 涉及文件

`xdl/config/setup.py:85-139`

## 问题代码

三个函数结构完全一致，唯一的区别是 `alias_mapping` 的键名：

```python
def _collect_transform_configs(root_config):
    alias_mapping = (
        ("train_transforms", "train"),
        ("val_transforms", "val"),
        ("test_transforms", "test"),
    )
    collected = {}
    for alias_key, transform_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[transform_name] = alias_value
    return collected

def _collect_dataset_configs(root_config):
    alias_mapping = (
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "test"),
    )
    collected = {}
    for alias_key, dataset_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[dataset_name] = alias_value
    return collected

def _collect_dataloader_configs(root_config):
    alias_mapping = (
        ("train_dataloader", "train"),
        ("val_dataloader", "val"),
        ("test_dataloader", "test"),
    )
    collected = {}
    for alias_key, dataloader_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[dataloader_name] = alias_value
    return collected
```

总共 55 行，核心逻辑完全一致。

## 修复建议

参数化合并为一个函数：

```python
def _collect_configs(root_config, alias_mapping):
    collected = {}
    for alias_key, short_name in alias_mapping:
        alias_value = root_config.get(alias_key)
        if alias_value is not None:
            collected[short_name] = alias_value
    return collected

# 调用处：
transform_config = _collect_configs(resolved_config, (
    ("train_transforms", "train"),
    ("val_transforms", "val"),
    ("test_transforms", "test"),
))
dataset_config = _collect_configs(resolved_config, (
    ("train_dataset", "train"),
    ("val_dataset", "val"),
    ("test_dataset", "test"),
))
dataloader_config = _collect_configs(resolved_config, (
    ("train_dataloader", "train"),
    ("val_dataloader", "val"),
    ("test_dataloader", "test"),
))
```
