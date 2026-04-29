# TODO-P0-3: TwinFlow 生成模型文档

> **优先级**: P0 | **预估工时**: 2h | **风险**: 极低

---

## 问题

`xdl/model/generate/twinflow.py` (513 行) 是框架唯一的生成模型，完全无文档。

涉及复杂功能：
- `training_step` — RCGM + 一致性正则化 + 分布匹配 + 增强目标
- `sampling_loop` — 一阶/二阶 ODE 求解器 + SDE 校正 + 3 种采样模式 (few/mul/any)
- 8 个超参数，4 个 forward 返回值

---

## 修复

### 1. 模块级 docstring

```
TwinFlow: Continuous-Time Generative Model.
Combines RCGM, consistency regularization, and distribution matching.
Refs: https://github.com/LINs-lab/RCGM, https://arxiv.org/abs/2505.07447
```

### 2. 类 docstring

包含所有 8 个参数说明 + 使用示例：

```python
model:
  target: "registry:TwinFlow"
  params:
    consistc_ratio: 1.0
    ema_decay_rate: 0.99
    estimate_order: 2
```

### 3. 关键方法 docstring

`forward` / `training_step` / `sampling_loop` 至少各一行参数和返回值说明。

---

## 验证

```bash
python -c "
from xdl.model.generate.twinflow import TwinFlow
assert TwinFlow.__doc__, 'Missing class docstring'
assert TwinFlow.training_step.__doc__, 'Missing training_step docstring'
assert TwinFlow.sampling_loop.__doc__, 'Missing sampling_loop docstring'
print('TwinFlow docs OK')
"
```

---

## 完成标准

- [x] 模块级 docstring 存在
- [x] 类 docstring 包含 8 个参数说明
- [x] `forward` / `training_step` / `sampling_loop` / `prepare_inputs` / `enhance_target` / `get_rcgm_target` / `dist_match` 有 docstring
