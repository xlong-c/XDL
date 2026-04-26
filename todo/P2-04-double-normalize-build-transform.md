# P2-04: build_transform 重复调用 _normalize_transform_shorthand

## 涉及文件

`xdl/config/builder.py:263-277`

## 问题代码

```python
def build_transform(config: Any) -> Any:
    if not config:
        return None
    if isinstance(config, list):
        config_dict = _normalize_transform_shorthand(config)   # 第一次 normalize
    elif isinstance(config, str):
        config_dict = _normalize_transform_shorthand(config)   # 第一次 normalize
    elif isinstance(config, Mapping):
        config_dict = dict(config)
    else:
        raise ConfigValidationError("...")

    return _build_component(
        _normalize_transform_shorthand(config_dict),           # 第二次 normalize
        kind="transform",
    )
```

## 问题说明

- 当 `config` 是 `list` 时：`_normalize_transform_shorthand(config)` 返回 `{"target": "torchvision.transforms:Compose", "params": {...}}` 这种 dict。然后这个 dict 又被传入 `_normalize_transform_shorthand`，进入 Mapping 分支，再次遍历 inline params。
- 当 `config` 是 `string` 时：第一次返回 `{"target": "...", "params": {}}`。第二次 normalize 是幂等的（走 Mapping 分支但无 inline params），浪费了函数调用和分支判断。
- 当 `config` 是 Mapping 且已经包含 `target` 键时：只 normalize 一次，路径正常。

## 影响

- 功能上无影响（第二次 normalize 是幂等的）
- 性能微损无关紧要
- 代码阅读时产生困惑："为什么 normalize 了两次？"

## 修复建议

将第一次 normalize 的结果直接传入 `_build_component`：

```python
def build_transform(config: Any) -> Any:
    if not config:
        return None
    if isinstance(config, (list, str)):
        normalized = _normalize_transform_shorthand(config)
    elif isinstance(config, Mapping):
        normalized = _normalize_transform_shorthand(dict(config))
    else:
        raise ConfigValidationError("...")
    return _build_component(normalized, kind="transform")
```
