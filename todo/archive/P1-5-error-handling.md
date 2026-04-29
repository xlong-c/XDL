# TODO-P1-5: 错误处理增强 (统一异常层次)

> **优先级**: P1 | **预估工时**: 2h | **状态**: ✅ DONE

---

## 问题

- 仅 `xdl/config/errors.py` 有结构化异常 (`ConfigError` 及其子类)
- 其余模块 (registry, dataset, model, trainer) 使用裸 `ValueError`/`KeyError`/`FileNotFoundError`
- 回调层 log-and-swallow，库代码 raise-bare，两套模式互不衔接

## 修复

### 1. 新建 `xdl/errors.py` — 统一根异常层次

```
XDLError(Exception)
├── RegistryError   — 注册表查找/注册失败
├── DataError       — 数据加载/预处理失败
├── ModelError      — 模型构建/前向失败
└── TrainingError   — 训练过程异常
```

### 2. `xdl/config/errors.py`

`ConfigError` 改为继承 `XDLError`（原继承 `Exception`）

### 3. `xdl/utils/registry.py`

`Registry.get()` 中 `KeyError` → `RegistryError`

### 4. `xdl/config/builder.py`

`except KeyError` → `except (RegistryError, KeyError)` 保持向后兼容

## 验证

```python
python -c "
from xdl.errors import XDLError, RegistryError
from xdl.config.errors import ConfigError, ConfigValidationError
assert issubclass(ConfigError, XDLError)
assert issubclass(ConfigValidationError, ConfigError)
from xdl.utils.registry import Registry
r = Registry('t'); r.register('x')(lambda: 1)
try: r.get('y')
except RegistryError: print('OK')
"
```

## 完成标准

- [x] 统一根异常 `XDLError`
- [x] `ConfigError` 继承 `XDLError`
- [x] `Registry.get()` 抛出 `RegistryError`
- [x] `builder.py` 兼容新旧异常类型
- [x] 所有注册表正常运行
