# TODO-P1-3: IoU/Dice 多类别接口统一

> **优先级**: P1 | **预估工时**: 1h | **状态**: ✅ DONE

---

## 问题

IoU/Dice 用单独的 `__call_multi_class__` 处理多类（调用者必须手动选择调用），而 Accuracy 在 `__call__` 中通过 `pred.shape[1] > 1` 自动检测。接口风格不一致。

## 修复

**文件**: `xdl/metric/metrics.py`

1. IoU / Dice `__init__` 增加 `num_classes`、`average` 参数
2. 多分类逻辑合并入 `__call__`，通过 `pred.dim() == 4 and pred.shape[1] > 1` 自动检测
3. `__call_multi_class__` 保留为 deprecated 别名，委托到 `__call__`

## 验证

```python
python -c "
import torch
from xdl.metric.metrics import IoU, DiceCoefficient

iou = IoU()
# Binary 3D/4D-C1
iou(torch.rand(4, 32, 32), (torch.rand(4, 32, 32)>0.5).float())
iou(torch.rand(4, 1, 32, 32), (torch.rand(4, 1, 32, 32)>0.5).float())
# Multi-class — auto-detected
iou(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
# Backward compat
iou.__call_multi_class__(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)), 5)

dice = DiceCoefficient()
dice(torch.rand(4, 32, 32), (torch.rand(4, 32, 32)>0.5).float())
dice(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
print('OK')
"
```

---

## 完成标准

- [x] IoU.__call__ 自动检测多分类
- [x] DiceCoefficient.__call__ 自动检测多分类
- [x] __call_multi_class__ 保留为 deprecated 别名
- [x] num_classes / average 参数可用
