# P1-03: config_version 声明但不校验

## 概述

Schema 中定义了 `config_version` 字段，YAML 文件中也写了 `config_version: 1`，但代码中没有版本校验逻辑。用户写 `config_version: 2` 也能正常通过，不会报错。

## 涉及文件

| 文件 | 位置 | 内容 |
|------|------|------|
| `xdl/config/schema.py:15` | `CONFIG_SCHEMA_VERSION = 1` | 当前版本常量 |
| `xdl/config/schema.py:118` | `config_version: int = CONFIG_SCHEMA_VERSION` | Schema 字段，默认值为 1 |
| `xdl/config/resolver.py:71` | `merge_with_schema()` | 合并不做版本校验 |
| `xdl/config/setup.py:182` | `load_config_with_schema(raw_config)` | 入口未校验 |
| `config/vgg_cifar100.yaml:3` | `config_version: 1` | YAML 中声明 |
| `config/unified_logger_example.yaml:3` | `config_version: 1` | YAML 中声明 |

## 问题分析

`merge_with_schema` 使用 OmegaConf 的 `OmegaConf.merge(base, raw)` 将用户 YAML 合并到 `ConfigSchemaV1` dataclass。如果用户写：

```yaml
config_version: 2
```

OmegaConf 会把这个值合并到 `config_version` 字段上（因为字段在 dataclass 中存在），不会报错。之后也没有任何代码检查这个值是否等于 `CONFIG_SCHEMA_VERSION`。

## 为什么重要

- 如果未来 schema 有 breaking change（例如 `optimization` 拆分为 `optimizer` + `scheduler` 两个顶层键），需要根据 `config_version` 做兼容或抛错
- 没有版本校验意味着未来新增 schema v2 时，v1 和 v2 YAML 都会被静默接受，导致难以排查的配置问题
- 语义上 `config_version` 是一个协议版本号，应该作为第一道校验关卡

## 修复建议

在 `setup_from_yaml` 或 `merge_with_schema` 中加入校验：

```python
# 在 setup_from_yaml 中，获取 loaded config 后立即校验
raw_config = _load_raw_yaml(path)
declared_version = raw_config.get("config_version")
if declared_version is not None and declared_version != CONFIG_SCHEMA_VERSION:
    raise ConfigValidationError(
        f"Unsupported config version {declared_version}. "
        f"Expected {CONFIG_SCHEMA_VERSION}."
    )
```

或者更优雅地在 `load_config_with_schema` / `merge_with_schema` 内部，merge 完成后读取 `merged_cfg.config_version` 并与 `CONFIG_SCHEMA_VERSION` 比较。
