# TODO: XDL 优化任务总览

> 最后更新: 2026-04-29 | autopilot 模块分析生成

---

## 已归档

YAML 配置启动体系审查 — 10 项全部完成。详见 [COMPLETED.md](COMPLETED.md)。

---

## 当前活跃: XDL 模块全面优化

### P0 — 立即修复（功能缺口）

| 编号 | 任务 | 预估 | 状态 |
|------|------|------|------|
| [P0-1](P0-1-unregistered-components.md) | 未注册组件补全 (SOAP/MAE/MSE/RMSE/FATT) | 1.5h | ⬜ TODO |
| [P0-2](P0-2-callbacks-export.md) | 回调 `__init__.py` 导出修复 (5 个回调) | 10min | ⬜ TODO |
| [P0-3](P0-3-twinflow-docs.md) | TwinFlow 生成模型文档 | 2h | ⬜ TODO |

### P1 — 短期改进（质量提升）

| 编号 | 任务 | 预估 | 状态 |
|------|------|------|------|
| [P1-1](P1-1-agents-md-update.md) | model/AGENTS.md 更新到 28 条目 | 1h | ⬜ TODO |
| P1-2 | 数据集增强代码去重 (hairdata 3 文件) | 3h | ⬜ TODO |
| P1-3 | IoU/Dice 多类别接口统一 | 1h | ⬜ TODO |
| P1-4 | Precision/Recall/F1 averaging 模式扩展 | 1h | ⬜ TODO |
| P1-5 | 错误处理增强 (统一异常层次) | 2h | ⬜ TODO |

### P2 — 中期增强（功能扩展）

| 编号 | 任务 | 预估 | 状态 |
|------|------|------|------|
| P2-1 | 原生 DeepSpeed 集成 | 1w | ⬜ TODO |
| P2-2 | 新增损失函数 (Huber/InfoNCE 等) | 1-2w | ⬜ TODO |
| P2-3 | 通用数据集支持 (CIFAR/MNIST 等) | 3-5d | ⬜ TODO |
| P2-4 | collate_fn 支持 | 2h | ⬜ TODO |
| P2-5 | 完整测试覆盖 | 2-3w | ⬜ TODO |

---

## 推荐执行顺序

```
本周  → P0-1, P0-2 (补全注册缺口，零风险)
下周  → P0-3, P1-1, P1-3 (文档 + 接口修复)
本月  → P1-2, P1-4, P1-5 (代码质量)
下月+ → P2 系列 (新功能)
```

---

## 相关分析文档

- [XDL 模块功能边界文档](../docs/xdl-functional-boundary.md)
- [XDL 优化方向分析](../docs/xdl-optimization-plan.md)
- [Autopilot Spec](../.omc/autopilot/spec.md)
