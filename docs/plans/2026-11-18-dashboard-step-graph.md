# Dashboard DAG 节点和训练步子图 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 明确装配图的节点契约, 接入必要的中间节点, 并在 Dashboard 编辑实际驱动 XDL 训练的 batch 计算图.

**Architecture:** 快照增加可选 `step_graph` (独立 ID 和端口命名空间), 只有 `dag-runtime` 可以执行它. 它引用外层组件, 但每条边只在所属图内连接; 一个 `CoreModel` 适配器调用已编译的计算图并独占梯度和优化生命周期. 旧快照及 schema v1 不变.

**Tech Stack:** Python/PyTorch, XDL CoreModel 与 Trainer, OmegaConf, Go, Preact/TypeScript.

---

### 批次 1: 一个 batch 的计算

- 新增 `xdl/workflow/step_graph.py` 和 `tests/workflow/test_step_graph.py`. 先写测试: `BatchInput -> Select -> Forward -> LossCall -> StepOutput` 能对 tuple/dict batch 产生可求导标量; 添加带明确左右权重的 `WeightedSum`, 和只输出日志指标的 `MetricCall`. 非法端口, 环, 重复来源和死节点在预检失败, 缺字段或 loss 非标量在运行时报节点 ID.
- 在 `xdl/workflow/ir.py` 和 `dashboard/backend/internal/workflow/model.go` 保留可选图, 使用既有节点和边的规模限制, 不允许任意源码或 target. `dashboard/worker/workflow_validator.py` 只允许显式引用已装配的 loss/metric, 不更改旧快照语义.
- 在 `xdl/workflow/executor.py` 挂接子图 CoreModel: `training_step` 经现有 `manual_optimization_step` 做反传、累积、裁剪和更新; 验证复用计算图但不更新状态. 不改 `setup_from_yaml()` 或 Trainer 主循环.
- 运行 `python -m pytest tests/workflow dashboard/worker -q` 和 `go test ./...` (在 `dashboard/backend`), 加一条真实 CPU 训练断言参数变化、训练步数和日志.

### 批次 2: 装配图端口和中间节点

- 在 `xdl/workflow/node_catalog.json` 定义节点参数的约束和端口载荷说明; `DataLoader` 允许显式 `batch_size` 并兼容 Trainer 的旧默认来源, 接可选 `Collate`.
- 在 `xdl/workflow/executor.py` 实现带 seed 的 `DatasetSplit`, 复用现有 builder 接入 `Collate`. 无法 lowering 到 schema v1 的图必须明确拒绝; 不用任意 target 导入.
- `dashboard/worker/workflow_validator.py` 校验组件绑定、多态透传和组合、已知的样本契约不匹配. 不把未知形状当作已推导结果. 旧版 metric/checkpoint 端口策略不得破坏历史快照.

### 批次 3: 编辑和跨层验收

- 在 `dashboard/frontend/src/workflow_editor.tsx` 与现有工作流序列化逻辑接入双层编辑, 编辑端口、参数及组件引用, 并隔离各层选择和连线. `step_graph` 随草稿与服务端快照一起保存.
- 扩展 `dashboard/backend/internal/api/workflow_e2e_test.go` 从真实 API 创建并执行显式训练步子图. 保持旧 profile 的 CPU 验收, 再验证前端测试和构建通过.
- 同步 `docs/md/architecture/dashboard-workflow-ir.md` 和 `dashboard/README.md`, 区分已实现能力与计划. 浏览器检查编辑、保存、重新加载及错误定位.
