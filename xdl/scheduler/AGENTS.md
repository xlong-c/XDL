# xdl/scheduler — 学习率调度器子模块

## 目录职责

- 封装和注册学习率调度器与工厂函数
- 为训练流程提供统一 LR 调度能力

## 当前内容

- `step_lr.py`
- `multi_step_lr.py`
- `exponential_lr.py`
- `cosine_annealing_lr.py`
- `cosine_annealing_warm_restarts.py`

## 修改约束

- 新增 scheduler 后必须在 `__init__.py` 中集中注册
- 区分按 step 调度还是按 epoch 调度的语义
- 保持与 PyTorch scheduler 行为和参数命名尽量一致
- 工厂函数与类注册名保持稳定

## 验证建议

- 至少验证若干步后的学习率曲线是否符合预期
- 涉及配置构建时同步检查 builder 兼容
