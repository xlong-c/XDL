# TODO-P0-1: 未注册组件补全

> **优先级**: P0 (功能缺口) | **预估工时**: 1.5h | **风险**: 极低

---

## 问题

以下组件已完整实现但未接入注册系统，无法通过 YAML 配置使用。

| 组件 | 文件 | 行数 | 状态 |
|------|------|------|------|
| `SOAP` 优化器 | `xdl/optimizer/soap.py:10` | 431 行 | 已实现，未注册 |
| `MeanAbsoluteError` | `xdl/metric/metrics.py` | ~30 行 | 已实现，未导入 |
| `MeanSquaredError` | `xdl/metric/metrics.py` | ~30 行 | 已实现，未导入 |
| `RootMeanSquaredError` | `xdl/metric/metrics.py` | ~30 行 | 已实现，未导入 |
| `FATT` 分割模型 | `xdl/model/segment/fatt.py:169` | 267 行 | 已实现，未导入未注册 |

---

## 修复步骤

### Step 1: 注册 SOAP 优化器

**文件**: `xdl/optimizer/__init__.py`

```python
from .soap import SOAP                          # 新增导入

def _register_optimizers():
    ...
    register_optimizer("SOAP")(SOAP)            # 新增注册
```

### Step 2: 注册 MAE/MSE/RMSE 指标

**文件**: `xdl/metric/__init__.py`

```python
from .metrics import (
    ...,
    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,  # 新增
)

def _register_metrics():
    ...
    register_metric("MeanAbsoluteError")(MeanAbsoluteError)
    register_metric("MeanSquaredError")(MeanSquaredError)
    register_metric("RootMeanSquaredError")(RootMeanSquaredError)
```

### Step 3: 注册 FATT 分割模型

**文件**: `xdl/model/__init__.py`

```python
from .segment.fatt import FATT                  # 新增导入

def _register_models():
    ...
    register_model("FATT")(FATT)                # 新增注册

__all__ = [..., "FATT"]                         # 新增导出
```

---

## 验证

```bash
# SOAP
python -c "from xdl.optimizer import OPTIMIZER_REGISTRY; assert 'SOAP' in OPTIMIZER_REGISTRY.list_available()"

# 指标
python -c "from xdl.metric import METRIC_REGISTRY; names = METRIC_REGISTRY.list_available(); assert all(n in names for n in ['MeanAbsoluteError','MeanSquaredError','RootMeanSquaredError'])"

# FATT
python -c "from xdl.model import MODEL_REGISTRY; assert 'FATT' in MODEL_REGISTRY.list_available()"
```

---

## 影响范围

| 文件 | 改动 |
|------|------|
| `xdl/optimizer/__init__.py` | +1 import +1 register |
| `xdl/metric/__init__.py` | +3 import +3 register |
| `xdl/model/__init__.py` | +1 import +1 register +1 `__all__` |

---

## 完成标准

- [ ] `SOAP` 可通过 `OPTIMIZER_REGISTRY.get("SOAP")` 获取
- [ ] `MeanAbsoluteError/MeanSquaredError/RootMeanSquaredError` 可通过 `METRIC_REGISTRY.get(...)` 获取
- [ ] `FATT` 可通过 `MODEL_REGISTRY.get("FATT")` 获取
