# XQT Session / Stage 改造 TODO

本文记录 `XQTOptimizationSession` 这条改造线的完成状态. 它只服务本轮 `session / stage / payload / runtime artifact / provider` 目标, 不替代 [xqt.md](/root/workspace/xdl/docs/md/architecture/xqt.md) 的长期架构正文.

当前状态: 本轮 TODO 已完成. 后续如果继续扩展 runtime stage 或新的 transform provider, 应以 [stage.py](/root/workspace/xdl/xqt/workflows/stage.py), [stage_provider.py](/root/workspace/xdl/xqt/workflows/stage_provider.py) 和 [optimization.py](/root/workspace/xdl/xqt/workflows/optimization.py) 为内部事实源继续演进.

## 负责什么

- 记录本轮目标范围内已经完成的结构性修改.
- 记录已完成项的验证锚点.
- 给后续 agent 或开发者提供最小继续上下文.

## 不负责什么

- 不重新定义 `XQT` 的整体边界.
- 不替代 `xqt/FRAMEWORK.md` 或 `docs/md/architecture/xqt.md`.
- 不把尚未实现的规划写成既成事实.
- 不展开无关的 quant backend, export backend 或 kernel 细节.

## 本轮目标边界

本轮只覆盖 `XQTOptimizationSession` 的 stage 化演进, 以及围绕它形成的内部 payload / provider 协议:

```text
baseline model
  -> SessionStage
  -> StagePayload
  -> typed runtime artifacts
  -> stage provider output
```

它解决的是:

- session 需要正式管理阶段图, 而不是只靠零散 snapshot.
- stage 需要能表达 "当前产物是什么", 不能默认所有阶段产物都是 `nn.Module`.
- quant / operator / export 这类阶段需要有稳定的 payload 协议.
- workflow 主流程不能继续堆积 `if stage.kind == ...` 的分支细节.
- session 内比较需要有 payload capability-aware 和 artifact-aware 的结构化结果.

## 已完成修改

### 1. 引入正式的 SessionStage / StagePayload 协议

已新增 [stage.py](/root/workspace/xdl/xqt/workflows/stage.py), 当前包含:

- `SessionStage`
- `StagePayload`
- `StagePersistence`
- `TransformLineage`
- `StageComparison`
- payload capability helper
- typed payload 类型

当前 session 不再只把阶段看成一组 `OptimizationStageResult`, 而是同时维护可追踪的 `SessionStage` 列表.

### 2. baseline 已正式成为 session 的第一个 stage

`XQTOptimizationSession` 初始化时现在会显式注册:

- stage name: `baseline`
- stage kind: `baseline`
- created_by.kind: `session_init`

这意味着后续 `quant / operator / export` 都是在一个显式 stage 图上继续演进, 而不是隐式从 "初始模型" 开始.

### 3. session 现在正式维护 stage 图和快照

[optimization.py](/root/workspace/xdl/xqt/workflows/optimization.py) 当前已经维护:

- `stages_by_name`
- `stage_order`
- `stage_counter`
- `model_snapshots`
- `baseline_stage`
- `best_stage`

当前语义已经分清:

- `SessionStage.payload`: 阶段产物语义
- `model_snapshots`: session 可恢复的模型态

### 4. use / revert_to 已恢复为模型语义

`session.use(name)` / `session.revert_to(name)` 现在优先通过 `model_snapshots` 恢复模型.

这保证了:

- operator stage 即使 payload 是 `runtime_plan`, session 仍能恢复出模型态
- export stage 即使 payload 是 `export_bundle`, session 仍不会把导出产物误当成当前模型
- quant stage 即使 payload 是 `quantized_model`, session 仍优先使用 snapshot 恢复模型态

### 5. stage lineage 已结构化

当前 `created_by` 不再是松散 dict, 而是 `TransformLineage`.

已稳定记录的字段包括:

- `kind`
- `transform`
- `transform_family`
- `transform_name`
- `params`
- `from_stage`
- `compare_to`

### 6. stage parent 语义已修正

当前 parent 计算已经修正为基于真正的来源 stage, 而不是在更新 `best_stage` 之后再回推.

当前已验证:

- 默认 quant stage 会挂到 `baseline`
- 显式 `from_stage="fp4_quant"` 的 operator stage 会挂到 `fp4_quant`

### 7. stage payload capability 已集中定义

当前 payload kind 和能力映射已经下沉到 [stage.py](/root/workspace/xdl/xqt/workflows/stage.py).

已覆盖:

- `torch_module`
- `quantized_model`
- `runtime_plan`
- `export_bundle`
- `runtime_handle`

当前 payload capability 至少显式表达:

- `can_evaluate`
- `can_export`
- `can_quantize`
- `can_optimize_ops`
- `can_restore_model`

### 8. quant stage 已使用类型化 QuantizedModelPayload

当前 quant stage 的 `SessionStage.payload.value` 不再只是默认模型态对象, 而是 `QuantizedModelPayload`.

当前字段包括:

- `artifact_kind`
- `stage_name`
- `source_model_stage`
- `model`
- `backend`
- `method`
- `strategy`
- `quantized_module_count`
- `quantized_modules`
- `calibration_samples`
- `calibration_summary`
- `components`
- `artifacts`
- `capability`

同时:

- `payload.payload_kind == "quantized_model"`
- `payload.metadata["quantized_model"]` 保留 plain mapping 视图
- `model_snapshots` 继续保存可恢复模型态, `use()` / `revert_to()` 不依赖 payload 直接恢复

### 9. operator stage 已使用类型化 RuntimePlanPayload

当前 operator stage 的 `SessionStage.payload.value` 不再是松散 dict, 而是 `RuntimePlanPayload`.

当前字段包括:

- `artifact_kind`
- `stage_name`
- `source_model_stage`
- `engine`
- `target_count`
- `targets`
- `artifacts`

同时:

- `payload.payload_kind == "runtime_plan"`
- `payload.metadata["runtime_plan"]` 仍保留 plain mapping 视图, 便于 report 和 JSON 输出

### 10. export stage 已使用类型化 ExportBundlePayload

当前 export / deploy stage 的 `SessionStage.payload.value` 不再拿 `context.model` 伪装成 `export_bundle`, 而是 `ExportBundlePayload`.

当前字段包括:

- `artifact_kind`
- `stage_name`
- `source_model_stage`
- `format`
- `target_count`
- `targets`
- `artifacts`

同时:

- `payload.payload_kind == "export_bundle"`
- `payload.metadata["export_bundle"]` 保留 plain mapping 视图

### 11. runtime_handle 已补齐 typed payload 协议

当前 `runtime_handle` 不再只存在 payload kind 和 capability 映射中, 已新增 `RuntimeHandlePayload`.

当前字段包括:

- `artifact_kind`
- `stage_name`
- `source_model_stage`
- `runtime`
- `handle_kind`
- `target_count`
- `targets`
- `handle`
- `artifacts`
- `metadata`

本轮没有新增 runtime_handle stage producer. 这是有意为之: 当前 XQT workflow 还没有稳定的 executable runtime session 生产阶段, 因此只补协议类型, 不伪造执行句柄.

### 12. 已形成统一的 RuntimeArtifactPayload 家族

当前 `RuntimePlanPayload`, `ExportBundlePayload` 和 `RuntimeHandlePayload` 都收敛到 `RuntimeArtifactPayload` 这层共享协议.

当前共享字段:

- `artifact_kind`
- `stage_name`
- `source_model_stage`
- `artifacts`

这让 "非模型态 payload" 走统一表达.

### 13. workflow 输出已包含 session_stages

当前 `OptimizedModelResult` 已新增:

- `session_stages`

当前 `workflow_result.json` 已输出:

- 旧 `stages`
- 新 `session_stages`

并且 stage payload 的序列化已经补齐 JSON-safe 规则, 避免 `nn.Module`, `Path` 等对象直接进入 JSON.

### 14. stage provider 已独立为内部模块

[stage_provider.py](/root/workspace/xdl/xqt/workflows/stage_provider.py) 现在是 transform-side stage 描述的内部落点.

当前内部结构包括:

- `StagePayloadBuildContext`
- `StageProviderOutput`
- `StageProvider`
- `DefaultStageProvider`
- `ModelQuantizerProvider`
- `OperatorOptimizerProvider`
- `ExportProvider`
- `resolve_stage_provider`

当前 provider 一次性产出:

- `session_stage_kind`
- `lineage`
- `payload_value`
- `payload_metadata`

当前 provider 分工:

- `DefaultStageProvider`: benchmark / prune / analyze 等模型态默认 stage
- `ModelQuantizerProvider`: quant stage 和 `QuantizedModelPayload`
- `OperatorOptimizerProvider`: operator stage 和 `RuntimePlanPayload`
- `ExportProvider`: export / deploy stage 和 `ExportBundlePayload`

`optimization.py` 只负责运行 pass, acceptance, snapshot 和注册 stage, 不再内联 provider / payload builder 细节.

### 15. session stage compare helper 已落地

当前 `XQTOptimizationSession` 已新增:

- `compare_stages(source_stage, target_stage)`
- `compare_to_baseline(stage_name)`

底层结构化结果为 `StageComparison`, 覆盖:

- stage kind / payload kind 变化
- payload capability delta
- metrics added / removed / changed
- artifacts added / removed / changed
- source / target 是否可恢复模型
- plain mapping 序列化

### 16. 本轮 todo 已无未完成实现项

原待办中的 5 项处理结论:

- transform-side provider: 已通过 `stage_provider.py` 落地.
- quant typed payload: 已通过 `QuantizedModelPayload` 落地.
- runtime_handle typed payload: 已通过 `RuntimeHandlePayload` 落地, 暂不新增 producer.
- stage provider 独立模块: 已落地.
- stage compare / diff helper: 已通过 `StageComparison` 和 session helper 落地.

## 当前测试锚点

本轮改动当前主要由 [test_optimization_session_stages.py](/root/workspace/xdl/tests/xqt/test_optimization_session_stages.py) 覆盖, 并已联动验证:

- [test_fp4_tilelang_workflow.py](/root/workspace/xdl/tests/xqt/test_fp4_tilelang_workflow.py)
- [test_framework_contract.py](/root/workspace/xdl/tests/xqt/test_framework_contract.py)

当前已稳定覆盖的关键行为:

- baseline stage 创建
- quant / operator lineage
- quant typed payload
- operator runtime plan payload
- export bundle payload
- runtime handle typed payload
- `workflow_result` 输出 `session_stages`
- provider 驱动的 `stage_kind / payload` 行为
- stage compare helper

## 后续扩展约束

本轮 TODO 已完成. 后续扩展仍需遵守:

- 新 provider 先放入 `stage_provider.py`, 不新增 public registry.
- 新 payload kind 必须先在 `stage.py` 定义 typed payload 和 capability.
- 新 runtime_handle producer 必须有真实 executable runtime session 或 materialized handle, 不能只写 metadata.
- compare helper 只做 session 内 stage 摘要和结构化 diff, 不引入 task-level validation.

## 明确不在本次目标内

以下内容本轮明确没有纳入:

- 在 `XQT` 中新增训练循环
- 在 `XQT` 中新增 QAT / finetune / recovery 流程
- 在 `XQT` 中引入 dataset / dataloader / evaluation provider
- 接管 vLLM 等外部 inference service 编排
- 改造公开 YAML schema
- 改造 `xqt` 顶层 public API
- 把所有 quant / export / backend 细节统一成完整 registry 系统

## 相关代码锚点

- [stage.py](/root/workspace/xdl/xqt/workflows/stage.py)
- [stage_provider.py](/root/workspace/xdl/xqt/workflows/stage_provider.py)
- [optimization.py](/root/workspace/xdl/xqt/workflows/optimization.py)
- [test_optimization_session_stages.py](/root/workspace/xdl/tests/xqt/test_optimization_session_stages.py)
- [test_fp4_tilelang_workflow.py](/root/workspace/xdl/tests/xqt/test_fp4_tilelang_workflow.py)
- [test_framework_contract.py](/root/workspace/xdl/tests/xqt/test_framework_contract.py)

## 使用方式

继续沿这条线推进时, 优先按下面顺序阅读:

1. 当前文件
2. [xqt.md](/root/workspace/xdl/docs/md/architecture/xqt.md)
3. [FRAMEWORK.md](/root/workspace/xdl/xqt/FRAMEWORK.md)
4. [stage.py](/root/workspace/xdl/xqt/workflows/stage.py)
5. [stage_provider.py](/root/workspace/xdl/xqt/workflows/stage_provider.py)
6. [optimization.py](/root/workspace/xdl/xqt/workflows/optimization.py)
