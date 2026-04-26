# P2-07: setup_from_yaml 内 resolve 流程绕路

## 涉及文件

| 文件 | 位置 |
|------|------|
| `xdl/config/setup.py:182-183` | `setup_from_yaml` 调用处 |
| `xdl/config/resolver.py:117-126` | `load_config_with_schema` 实现 |

## 问题代码

```python
# setup.py:182-183
merged_config = load_config_with_schema(raw_config)  # 内部已做 resolve_config
resolved_config = to_plain_dict(merged_config, resolve=True)  # 又 resolve 一次
```

`load_config_with_schema` 的实现（`resolver.py:117-126`）：

```python
def load_config_with_schema(config, *, schema=None, resolve=False):
    merged_cfg = merge_with_schema(config, schema=schema)
    return resolve_config(merged_cfg) if resolve else merged_cfg
```

默认 `resolve=False`，所以 `load_config_with_schema(raw_config)` 只做了 merge 没做 resolve。

但是 `load_config_with_schema` 返回的 `DictConfig`（merge 后）仍然包含结构化 schema 类型信息（因为 `OmegaConf.merge(base_cfg, raw_cfg)` 保留了 `base_cfg` 的类型）。然后 `to_plain_dict(merged_config, resolve=True)` 调用 `OmegaConf.to_container(config, resolve=True)`，此时发生了 resolve。

## 实际流程

1. `load_config_with_schema(raw_config)` → merge with schema → 返回 `DictConfig`（未 resolve，`${...}` 还在）
2. `to_plain_dict(merged_config, resolve=True)` → to_container + resolve → 返回 `Dict[str, Any]`

这个流程本身是**正常**的——只 resolve 了一次。但在 `resolver.py` 中存在另一个函数 `resolve_config` 从未被 `setup_from_yaml` 调用，而 `load_config_with_schema` 的 `resolve` 参数默认是 `False`。

## 问题

- `load_config_with_schema` 有 `resolve` 参数但默认 `False`，`setup_from_yaml` 没传 `resolve=True`，转而在 `to_plain_dict` 中做 resolve。两条路径都可以 work，但含义不清晰。
- `resolve_config` 函数（`resolver.py:90`）与 `to_plain_dict(config, resolve=True)` 做了相同的事，但一个保留 `DictConfig`，一个转成 plain dict。命名上 `resolve_config` 更明确。
- 如果 `load_config_with_schema` 的 `resolve` 参数永不被使用，它属于死参数。

## 修复建议

统一 resolve 路径，明确意图：

**方案 A**：让 `load_config_with_schema` 一步到位

```python
merged_config = load_config_with_schema(raw_config, resolve=True)
resolved_config = to_plain_dict(merged_config, resolve=False)  # 已 resolve，直接转 dict
```

**方案 B**：去掉 `load_config_with_schema` 的 `resolve` 参数，只让它做 merge。resolve 由调用方自行决定用 `resolve_config` 还是 `to_plain_dict`：
```python
merged = load_config_with_schema(raw_config)     # 只管 merge
resolved = resolve_config(merged)                 # 显式 resolve（保留 DictConfig）
plain = to_plain_dict(resolved, resolve=False)    # 转 dict
```

推荐方案 A，改动最小。
