# TODO-P1-4: Precision/Recall/F1 averaging 模式扩展

> **优先级**: P1 | **预估工时**: 1h | **状态**: ✅ DONE

---

## 问题

Precision/Recall/F1Score 的 `average` 参数（`"macro"` / `"micro"`）已定义但从未使用，三种指标始终只做 macro 平均。`self.average` 为死代码。

## 修复

**文件**: `xdl/metric/metrics.py`

Precision / Recall / F1Score 的 `__call__` 中增加分支：
- `average="micro"`: 全局聚合 TP/FP/FN 后统一计算，micro Precision = micro Recall = micro F1
- `average="macro"` (默认): 逐类计算后取平均，保持向后兼容

## 验证

```python
python -c "
import torch
from xdl.metric.metrics import Precision, Recall, F1Score

pred = torch.randn(8, 5)
target = torch.randint(0, 5, (8,))

p = Precision(average='micro')(pred, target)
r = Recall(average='micro')(pred, target)
f1 = F1Score(average='micro')(pred, target)
assert abs(p - r) < 1e-6, 'micro P should equal micro R'

p_m = Precision(average='macro')(pred, target)
f1_m = F1Score()(pred, target)  # 默认 macro
print('OK')
"
```

## 完成标准

- [x] `average="macro"` — 逐类平均（向后兼容默认行为）
- [x] `average="micro"` — 全局聚合计算
- [x] micro 模式下 Precision == Recall == F1
- [x] 默认 macro 向后兼容
