# TODO-P0-2: 回调 `__init__.py` 导出修复

> **优先级**: P0 (功能缺口) | **预估工时**: 10min | **风险**: 极低

---

## 问题

以下 5 个回调已完整实现但在 `xdl/callbacks/__init__.py` 的 `__all__` 中未导出：

| 回调 | 文件 | 功能 |
|------|------|------|
| `DeviceStatsMonitor` | `device_stats_monitor.py` | CPU/GPU/内存监控 |
| `LambdaCallback` | `lambda_callback.py` | 动态 Lambda 钩子 |
| `LearningRateMonitor` | `learning_rate_monitor.py` | 学习率追踪 |
| `ModelSummary` | `model_summary.py` | 模型结构分析 (priority=1) |
| `Timer` | `timer.py` | 训练计时 + ETA |

---

## 修复

**文件**: `xdl/callbacks/__init__.py`

```python
# 添加 5 行导入
from .device_stats_monitor import DeviceStatsMonitor
from .lambda_callback import LambdaCallback
from .learning_rate_monitor import LearningRateMonitor
from .model_summary import ModelSummary
from .timer import Timer

# __all__ 中添加 5 个名称
__all__ = [
    ...,
    "DeviceStatsMonitor",
    "LambdaCallback",
    "LearningRateMonitor",
    "ModelSummary",
    "Timer",
]
```

---

## 验证

```bash
python -c "
from xdl.callbacks import (DeviceStatsMonitor, LambdaCallback,
    LearningRateMonitor, ModelSummary, Timer)
print('All 5 callbacks imported OK')
"
```

---

## 完成标准

- [x] 所有 5 个回调可通过 `from xdl.callbacks import Xxx` 导入
