# XQT 配置单轨化与架构重构 TODO

本文是 XQT v0.x 重构的执行清单. 它记录计划和验收标准, 不把未完成事项写成已实现事实.

## 目标

XQT 只保留一套配置标准: `OptimizationConfig` + 按 `stage.kind` 分型的 `StageSpec`. YAML workflow 和 `XQTOptimizationSession` 必须构造同一棵 dataclass 树. 旧 `XQTConfig` / `load_xqt_config()` / `context.config.compression.*` 已从代码路径删除, 不再作为兼容垫片保留.

长期目标形态:

```text
YAML workflow / XQTOptimizationSession
  -> OptimizationConfig
  -> OptimizationStageConfig.spec: QuantStageSpec | PruneStageSpec | ...
  -> transform 子系统直接消费 StageSpec
  -> XQTContext 只承载 model, artifacts, metrics, manifest, runtime inputs
```

## 原则

- 配置对象只有 `OptimizationConfig`; 不新增第三种 recipe schema.
- `stage.params` 是 YAML 表达面, loader 后必须解析成 `stage.spec`.
- Stage 执行函数读取 `StageSpec`, 不再从全局 `context.config` 里找当前 stage 的设置.
- `backend` 表示外部 quant/export/runtime 选择, 例如 `torchao`, `onnxruntime_qdq`, `tensorrt`.
- `engine` 表示 XQT 内部 kernel/lowering 实现, 例如 `torch_compile`, `triton`, `tilelang`, `cutile`.
- v0.x 可以破坏旧内部 API; 不做旧顶层字段自动迁移.

## Phase 0: 止血与可导航性

- [x] 拆 `xqt/operator_opt/materialize.py`.
  - [x] `plan.py`: plan 构建.
  - [x] `reference_wrappers.py`: CuTile / CuTe DSL reference-guarded linear wrapper, compatibility probe 和 candidate materialization.
  - [x] `triton_wrappers.py`: Triton RMSNorm wrapper, compatibility probe, candidate materialization 和 execution metadata.
  - [x] `tilelang_wrappers.py`: TileLang attention, conv, linear, norm 和 dequant wrapper family.
  - [x] `materialize.py`: 仅保留跨 engine candidate dispatch, component replacement 和 contract 校验; 运行辅助位于 `execution_support.py`, engine metadata / preflight 位于 `metadata.py`, TileLang candidate 构造位于 `tilelang_wrappers.py`.
  - [x] `reporting.py`: report summary helper.
  - [x] `execute.py`: 调度, benchmark, report.
  - [x] `runtime.py`: CUDA Graph capture/replay 与固定 shape runtime helper.
  - [x] 删除旧 `executor.py`; public API 从职责模块或 `xqt.operator_opt` 聚合入口导入.
- [x] 拆 `xqt/prune/structured.py`.
  - [x] `graph.py`: 依赖图.
  - [x] candidate helpers: conv, residual, MBConv, attention, ViT width, MLP, MoE, container.
    - [x] `candidates.py`: candidate dataclass 和 adapter dispatch.
    - [x] `candidates_attention.py`: attention head collector.
    - [x] `candidates_conv.py`: chain-conv 和 concat-branch collector.
    - [x] `candidates_residual.py`: residual stage collector.
    - [x] `candidates_mbconv.py`: MBConv collector.
    - [x] `candidates_mlp.py`: plain and gated MLP collector.
    - [x] `candidates_moe.py`: MoE expert collector.
    - [x] `candidates_container.py`: CNN stage and composite block collector.
    - [x] `candidates_vit.py`: ViT hidden/embedding width collector.
  - [x] `plan.py`, `rewrite.py`.
    - [x] `plan.py`: plan 构建 helper 和 `plan_structured_pruning(...)` 主 planner.
    - [x] `rewrite.py`: layer rewrite helper, attention rewrite, topology materialization 和 `apply_structured_pruning_plan(...)` action apply.
  - [x] `report.py`.
  - [x] `sparsity.py`: in-place N:M 和 block-sparse 权重稀疏实现及其 report 组装.
  - [x] 删除 `structured.py`: public operation 位于 `api.py`, discovery wiring 位于 `discovery.py`, toy model 位于 `toy_models.py`.
- [x] 给 quant/backend/operator engine 写入统一 maturity.
  - [x] `executable`.
  - [x] `reference_guarded`.
  - [x] `metadata_only`.
  - [x] `planned`.
- [x] 治理 placeholder.
  - [x] 空 `awq.py`, `gptq.py`, `rtn.py`, `smoothquant.py` 改为明确 re-export 或删除.
  - [x] 空 recipe 目录补 smoke 或移出 recipe 树.
- [x] 静默 fallback 显式化.
  - [x] engine fallback 写入 stage metrics: `fallback_reason`, `fallback_policy`.
  - [x] 支持 `strict` / `prefer_fallback` 配置.

## Phase A: StageSpec 落地, 对外行为不变

- [x] 新增 `xqt/workflows/stage_specs.py`.
- [x] 定义 `QuantStageSpec`, `PruneStageSpec`, `OperatorStageSpec`, `ExportStageSpec`, `DeployStageSpec`, `AnalyzeStageSpec`, `BenchmarkStageSpec`.
- [x] `load_optimization_config()` 为每个 stage 解析并挂载 `stage.spec`.
- [x] `XQTOptimizationSession` 动态创建的 stage 也必须挂载同一类 `stage.spec`.
- [x] workflow 内部读取 `StageSpec`; 早期通过集中兼容垫片过渡, 当前主链已由 stage helper 直接消费 runtime config.
- [x] `StageSpec` 字段继续收紧为完全 structured schema.
  - [x] `OperatorStageSpec.benchmark` 改为 typed benchmark override.
  - [x] `ExportStageSpec.validate` 接入 `OutputDiffConfig`.
  - [x] `DeployStageSpec` 与 `ExportStageSpec` 的 runtime handle 字段分清.
  - [x] `DeployRuntimeHandleSpec` 将 ONNX Runtime providers 和 TensorRT device/runtime plugin libraries 分别收敛为 typed 子配置, 不再保留 `runtime_handle.params` 或 engine-target plugin fallback.
  - [x] `ExportTargetConfig.onnx` 收敛 ONNX 的 input/output names, dynamo, validate, runtime diff, pre-export fusion/lowering 和 graph optimization; 旧 target `params` ONNX 键会明确拒绝.
  - [x] `ExportTargetConfig.tensorrt` 收敛 TensorRT 的 ONNX source, builder, plugin, performance threshold 和 runtime benchmark 配置; `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口经同一 StageSpec 解析. 旧 target `params` TensorRT 键会明确拒绝, 且 engine-build plugin 不会隐式成为 runtime handle plugin.
  - [x] `ExportTargetConfig.openvino` 收敛 OpenVINO 的 ONNX source, input shape, dry-run, runtime diff 和 device 配置; `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口经同一 StageSpec 解析. 旧 target `params` OpenVINO 键会明确拒绝.
  - [x] `ExportTargetConfig.torch_export` 与 `.torchscript` 收敛 PyTorch-native export 配置; 单 target Session 入口与 YAML 同样解析为 StageSpec. 旧 target `params` 同名键会明确拒绝, TorchScript `method` 限制为 `trace` 或 `script`.
  - [x] `ExportTargetConfig.executorch`, `.ncnn` 与 `.mnn` 收敛移动 / 嵌入式 export 配置; 单 target Session 入口与 YAML 同样解析为 StageSpec. ncnn 的 `converter` 显式区分 `onnx2ncnn` 和 `pnnx`, ExportPass 与 preflight 选择同一工具, dry-run 不要求 optional package 或 converter executable 已安装. 旧 target `params` 同名键会明确拒绝.
- [x] loader 拒绝旧顶层键的错误信息补齐迁移对照.

## Phase B: Pass 直接消费 StageSpec

- [x] 改造 `QuantPass` 为 `run_quant_stage(context, spec)`.
- [x] 改造 `PrunePass` 为 `run_prune_stage(context, spec)`.
- [x] 改造 `OperatorOptimizationPass` 为 `run_operator_stage(context, spec)`.
- [x] 改造 `ExportPass` 为 `run_export_stage(context, spec)`.
- [x] 改造 `AnalyzePass` 和 `BenchmarkPass`.
- [x] 删除 workflow 中对 `context.config.compression.quant`, `context.config.compression.prune`, `context.config.operator_optimization`, `context.config.export`, `context.config.analysis`, `context.config.benchmark` 的写入.
- [x] `XQTContext.config` 已删除; workflow context 不再持有旧 recipe view.

当前真实状态:

- workflow 主链已不再写入 `context.config.compression.quant`, `context.config.compression.prune`, `context.config.operator_optimization`, `context.config.export`, `context.config.analysis`, `context.config.benchmark`; `XQTContext` 已无 `config` 字段.
- `run_quant_stage(context, spec)` / `run_prune_stage(context, spec)` / `run_operator_stage(context, spec)` / `run_export_stage(context, spec)` / `run_analyze_stage(context, spec)` / `run_benchmark_stage(context, spec)` 已落地, workflow 和 `XQTOptimizationSession` 的 stage 主链已通过这些 typed stage 入口调度.
- `QuantPass` / `PrunePass` 包装层已删除; `run_quant_stage(...)` / `run_prune_stage(...)` 已成为唯一公开主路径. `OperatorOptimizationPass.run(...)`, `ExportPass.run(...)`, `AnalyzePass.run(...)`, `BenchmarkPass.run(...)` 仍保留直接接收 typed runtime config 的能力, 但 `StageSpec -> runtime config` 的投影逻辑已从 workflow 调度层下沉到 `xqt.pipeline` stage helper.
- `xqt.quant.quantizers.fake_qdq.build_fake_qdq_surrogate(...)` 已支持显式 `QuantConfig` / `QuantStageSpec`, 并读取 `context.quant_config`, 不再从旧 `context.config.compression.quant` 获取当前量化设置.
- `XQTContext` 已新增独立 runtime 字段 `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `export_targets`. public `create_context(...)` 只从 workflow schema 初始化这些字段, workflow `_stage_context(...)` 也会在每个 stage 前直接从 `OptimizationConfig` 刷新它们; 旧 `XQTContext(config=...)` 构造方式已删除.
- `XQTContext` 现也承载当前 stage 的 runtime 配置视图: `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`. workflow stage 调度会刷新这些字段, `LoadModelPass`, `run_quant_stage(...)`, `run_prune_stage(...)`, `AnalyzePass`, `BenchmarkPass`, `OperatorOptimizationPass`, `ExportPass`, `fake_qdq` 和量化 layer analysis helper 已直接读取这些 runtime 视图, 不再用旧 `context.config` 兜底当前 stage runtime 配置.
- `xqt/run_workflow.py` 的 CLI 输出已从 `result.context.project_name` 读取项目名, 不再通过旧 `result.context.config.project.name` 展示 workflow project.
- `export_pass`, `passes.py` 的报告落盘路径, `operator_opt/execute.py`, `quant/backends/onnx_qdq.py` 等路径已开始优先读取 `context.device` / `context.artifact_dir`, 不再继续扩散 `context.config.model.device` / `context.config.project.artifact_dir` 的运行时直连.
- 旧 `run_xqt_recipe(...)`, `build_pipeline_from_config(...)`, pass order helper 和 `preflight_xqt_config(...)` 已删除. `create_context(...)` 已收窄为 workflow-only, 传入 workflow 时会构造 runtime-only context; 旧 `XQTConfig` dataclass, 旧 `load_xqt_config()` 和旧 recipe mapping 入口已删除. `QuantPass` / `PrunePass` 包装层也已删除, 不再保留并行 runtime 入口.
- Phase 0 已完成 structured prune 的职责拆分: dependency graph 位于 `graph.py`, concrete candidate collector 位于 `candidates*.py`, plan/report 位于 `plan.py` / `report.py`, rewrite 位于 `rewrite.py`, N:M / block-sparse 位于 `sparsity.py`, public operation 位于 `api.py`, discovery binding 位于 `discovery.py`, toy model 位于 `toy_models.py`. `structured.py` 已删除, 现有公开 prune 行为仍由 `tests/xqt/test_prune_structured.py` 覆盖.

## Phase C: 删除 XQTConfig 主路径

- [x] 删除 `load_xqt_config()`.
- [x] 删除 `XQTConfig` recipe schema.
- [x] 删除 `XQTContext.config` legacy view.
- [x] `xqt.core` 聚合入口不再重导出 `load_xqt_config` / `XQTConfig`.
- [x] `xqt.core.config.__all__` / `xqt.core.schema.__all__` 不再 star-export 旧 loader/schema.
- [x] 删除无调用者的旧 `xqt_config_to_dict()` helper.
- [x] 删除 `pipeline/runner.py` 的旧 recipe runner: `run_xqt_recipe()`, `build_pipeline_from_config()` 和 pass order helper.
- [x] `preflight_xqt_config()` 改为 `preflight_optimization_config()`.
- [x] public `create_context()` 不再接受旧 recipe mapping; 旧 `XQTConfig` dataclass 已不存在.
- [x] 删除 `pipeline/runner.py` 的旧 `create_manifest(XQTConfig)` helper 及 `xqt.pipeline` 重导出.
- [x] `xdl_adapter` 只接受 `OptimizationConfig` 或 workflow 输入, 不接旧 recipe schema.
- [x] `xqt/FRAMEWORK.md` 删除 "XQTConfig 作为内部主链路" 的描述.

## Phase D: Contract 层与产品面闭合

- [x] 新增 `xqt/contracts/`.
  - [x] `PrecisionPolicy`.
  - [x] `ModuleContract`.
  - [x] `FusionIntent`.
  - [x] `RuntimePlan`.
  - [x] `RuntimeHandle`.
  - [x] `OptimizationCapability` maturity 字段.
- [x] `xqt.convert()` 的 TileLang lowerings 与 Triton FeedForward lowering 和 operator materialize 共用 `materialize_module(contract, target)`; 没有对应 executor 的 runtime facade 不强行走 materializer.
- [x] `QuantizedModelPayload` 收敛到 `xqt/contracts/`; workflow stage 仅保留重导出.
- [x] `QuantizedModel` 成为模型侧量化算法的通用语义 contract. FP4, MXFP, AWQ/GPTQ, INT8 MMA, W4 storage INT8 MMA, SVD 和 TorchAO 的结果均继承它; `QuantizedModelPayload` 只追加 workflow provenance. analysis-only FakeQDQ surrogate 与 ONNX QDQ graph artifact 不伪装为量化模型.
- [x] `PrunedModelPayload` 收敛到 `xqt/contracts/`; prune stage 不再用裸 `torch_module` 表达模型侧结果, 而是记录 method,目标/实际 sparsity,execution state,完整 prune report,lineage,artifacts 和 capability.
- [x] `PrecisionPolicy` 收敛 module conversion, `xqt.nn` facade runtime intent 与 GEMM precision. 原 `MatmulPrecisionSpec` 在 `gemm_precision` 和 `conversion` 中仅保留为 `PrecisionPolicy` identity alias; role/mapping 解析统一归 contract, 删除 conversion 的双向转换与平行字段 schema. facade 的 runtime-only `auto` 不构成第二套静态 schema.
- [x] `xqt.nn.Linear`, `Conv2d`, `LayerNorm` 从 torch alias 变成真正 XQT semantic facade.
- [x] `FeedForward` 失败回退写入 runtime/report metadata, 不再静默吞掉 engine 错误.
- [x] deploy stage 的 runtime-handle producer.
  - [x] ONNX Runtime: deploy stage 同时导出 ONNX 后可 materialize `InferenceSession` 并产出 `RuntimeHandlePayload`.
  - [x] TensorRT: 同一 deploy stage 的非 dry-run engine 可反序列化并创建 execution context, 产出 `RuntimeHandlePayload`. 这只证明 runtime session 可创建; 数值和性能验收仍需目标硬件与显式执行.

## Phase E: 能力面收敛

- [x] 主推黄金路径: `weight_only_tilelang_onnx_tensorrt_golden.yaml` 声明 weight-only low-bit -> TileLang -> benchmark/analyze -> ONNX -> TensorRT deploy. ONNX target 在 `onnx` typed config 中使用 `dynamo: false`, `dynamic_shapes.input.0: batch` 和显式 `pre_export_lowering: fp4_weight_only_to_dense_linear`; lowering 只 materialize export copy 的 dense dequantized 权重, 不把 TensorRT artifact 伪装为 packed-FP4 runtime.
- [x] PyTorch quant 主推 `torchao` 加 XQT weight-only FP4/INT8 路径, 并由 capability 和量化测试覆盖其不同成熟度.
- [x] ONNX QDQ -> TensorRT 检测部署链保留为 stage workflow recipe; 真实 TensorRT build / runtime 验证受目标环境约束.
- [x] TileLang 已覆盖 attention / linear / dequant / norm wrapper family 和仅在 runtime-compatible TileLang 环境运行的 CUDA correctness tests. 已知 packed-tensor ABI 不兼容时, CUDA kernel entry 会显式报错, wrapper 可按 `fallback_policy` 记录 eager fallback; 无硬件结果不提升为生产声明.
- [x] OpenVINO, ExecuTorch, ncnn, MNN 维持 optional adapter, maturity 为 `reference_guarded`.
- [x] CuTile, CuTe DSL, CUTLASS 分别维持 `reference_guarded` 或 `metadata_only`, 直到有硬件验收.
- [x] XQT 不引入 QAT, recovery, mAP 闭环, dataset/dataloader 构建或 task provider.

## 验收标准

- [x] `grep XQTConfig / load_xqt_config` 在 workflow 生产路径为零; 当前只允许出现在文档和删除断言测试中.
- [x] 任意 recipe 只通过 `load_optimization_config()` 加载.
- [x] 每个 stage 执行函数签名接受对应 `StageSpec`.
- [x] `Session` 与 YAML workflow 生成同一类 `StageSpec` dataclass.
- [x] `test_framework_contract` 断言公开 API 只有 `OptimizationConfig` 族.
- [x] readiness 和 capability 明确区分 `executable`, `reference_guarded`, `metadata_only`, `planned`.
- [x] 每个 stage report 至少记录 backend/engine, device, shape, warmup, iterations, fallback 状态和 artifact kind.

当前核对:

- `Session` 与 YAML workflow 已生成同一类 `StageSpec` dataclass, 并由 `tests/xqt/test_framework_contract.py` 覆盖 `stage.spec` rebuild, typed benchmark override, export validate 和 deploy runtime handle schema.
- workflow 主链已不再通过 `quant/prune` compat 投影驱动 stage 执行, `preflight_optimization_config()` 已成为 workflow preflight 入口, `run_xqt_recipe()` / `build_pipeline_from_config()` / pass order helper / `preflight_xqt_config()` 也已删除. `xqt.workflows.optimization` 已不再导入或构造 `XQTConfig`, `_stage_context(...)` 也不再写 `context.config`.
- `create_context(...)`, `xdl_setup_to_xqt_context(...)` / `xdl_checkpoint_to_xqt_context(...)` 已接受 `OptimizationConfig`, workflow 映射和带 `stages` 的 workflow 路径, 并直接构造 runtime-only context. 旧 recipe schema mapping 会直接报错; 旧 `XQTConfig` dataclass 已不存在.
- `XQTContext` 目前已处于 "workflow 去配置化" 状态: `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`, `export_targets` 已独立成 runtime 字段; `create_context(...)` 和 workflow stage 调度会维护这些字段, 相关 pass 与默认输出路径已优先读取它们. `XQTContext.config` 已删除.
- `xqt.core.__all__` 已不再重导出 `load_xqt_config` 或 `XQTConfig`; `xqt.core.config.__all__` / `xqt.core.schema.__all__` 也不再通过 star-export 带出旧 loader/schema. 无调用者的旧 `xqt_config_to_dict()` helper 已删除; 旧 loader/schema 文件级入口也已删除.
- `xdl_setup_to_xqt_context(...)` / `xdl_checkpoint_to_xqt_context(...)` 现已收窄为只接受 `OptimizationConfig` 或 workflow 输入; 传入旧 recipe schema mapping 会直接报错.
- `tests/xqt/quant/test_quant_fp4_weight_only.py`, `test_quant_mxfp_weight_only.py`, `test_quant_planned_backends.py`, `test_quant_onnx_qdq_report.py`, `test_quant_method_schema.py` 已脱离 `load_xqt_config()` 和旧 `XQTConfig`, 改为直接构造 `QuantConfig`, `QuantStageSpec` 或 runtime-only `XQTContext`.
- `xqt.core.reporting.OptimizationCapability` 已新增共享 `maturity` 字段, 并由 quant / prune / operator / export capability producer 与 `assess_xqt_readiness()` capability matrix 统一输出 `executable`, `reference_guarded`, `metadata_only`, `planned` 四档成熟度. 相关断言已由 `tests/xqt/test_reporting_schema.py`, `tests/xqt/test_readiness.py`, `tests/xqt/quant/test_quant_method_schema.py`, `tests/xqt/test_cutile_backend.py`, `tests/xqt/test_cute_dsl_backend.py`, `tests/xqt/test_tilelang_capability_preflight.py` 覆盖.
- operator target config / plan / report 已新增 `fallback_policy`, 当前支持 `strict` / `prefer_fallback`. `execute_operator_optimization_plan(...)`, `summarize_operator_optimization_reports(...)`, manifest metric 与 stage summary 已显式记录 `fallback_reason` / `fallback_policy`; 相关断言已由 `tests/xqt/test_operator_tilelang_skeleton.py` 和 `tests/xqt/test_framework_contract.py` 覆盖.
- `tests/xqt/test_operator_tilelang_skeleton.py`, `test_operator_tilelang_linear.py`, `test_operator_tilelang_conv.py`, `test_operator_tilelang_norm.py`, `test_operator_tilelang_conv3d.py`, `test_operator_tilelang_cuda.py`, `test_operator_tilelang_dequant_gemm_cuda.py`, `test_cutile_backend.py`, `test_cute_dsl_backend.py`, `test_operator_triton_rmsnorm.py` 已脱离 `load_xqt_config()` 和旧 `XQTConfig`, 改为直接构造 `OperatorOptimizationConfig`, `BenchmarkConfig` 与 runtime-only `XQTContext`.
- `tests/xqt/test_onnx_optimizer.py` 已脱离旧 `XQTConfig`, export 测试直接传递 runtime `export_targets` 和 `output_diff_config`.
- `preflight_xqt_config(...)` 已删除, 现有 preflight 测试已改为直接调用 `preflight_optimization_config(...)`.
- `run_xqt_recipe(...)` / `build_pipeline_from_config(...)` / pass order helper / `create_manifest(XQTConfig)` 已删除, `xqt.pipeline.runner` 目前只保留 workflow-only `create_context(...)`.
- `materialize.py` 现只保留跨 engine dispatch / component replacement / contract 校验. `execution_support.py`, `metadata.py` 和 `tilelang_wrappers.py` 分别承载运行辅助, metadata/preflight 和 TileLang candidate construction. `tests/xqt/test_operator_tilelang_*.py`, `test_operator_triton_rmsnorm.py` 与 `test_convert_api.py` 覆盖重构后主链.
- `structured.py` 已删除. `api.py`, `discovery.py`, `toy_models.py` 与既有 planner/rewrite/report/sparsity 模块共同承载 structured prune, `tests/xqt/test_prune_structured.py` 维持公开行为覆盖.
- deploy stage 已能创建 ONNX Runtime `InferenceSession` 或 TensorRT runtime session. TensorRT producer 拒绝 dry-run engine, 并记录 engine deserialization / execution-context creation 的验证 metadata; `tests/xqt/test_onnx_optimizer.py` 覆盖两类 producer. 设定 `XQT_RUN_TENSORRT_HARDWARE_TESTS=1` 后, `tests/xqt/test_tensorrt_runtime_hardware.py` 会在真实 CUDA + ONNX + TensorRT 环境覆盖 ONNX export, non-dry-run engine build, runtime handle, session execution, `1e-3` 数值差异和小型 benchmark. 该验证已在 NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`) 上通过.
- 每个 `StageReport` 已带固定 `execution` envelope: `backend`, `engine`, `device`, `shape`, `warmup`, `iterations`, `fallback`, `artifact_kinds`. `tests/xqt/test_reporting_schema.py` 覆盖 schema 和 workflow 记录.
- `xqt/recipes/smoke/weight_only_tilelang_onnx_tensorrt_golden.yaml` 声明主推 stage 顺序, 使用 `dynamo: false` 的 legacy dynamic axes export 路径, 并要求调用方通过 `optimize_model(..., example_inputs=...)` 提供运行时样本. `tests/xqt/test_golden_path_recipe.py` 覆盖 typed spec, 在设定 `XQT_RUN_GOLDEN_HARDWARE_TESTS=1` 且具备 CUDA + ONNX + TileLang 时运行完整的 CUDA dry-run workflow, 并在具备 TensorRT 时构建 non-dry-run engine, materialize runtime handle, 执行 `1e-3` 数值对比和 `warmup=2` / `iterations=5` 的 `[8,64]` runtime benchmark. 测试隔离产物目录, 验证 ONNX artifact / dynamic batch profile 和 TensorRT deploy, 并断言 TileLang fallback 保持 `applied=false`. 该 non-dry-run deploy 使用明确记录的 dense dequantized export lowering, 不代表 packed-FP4 TensorRT runtime 或 TileLang kernel 性能验收. 两项 golden hardware test 已在 NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`) 上通过. TileLang CUDA tests 还会拒绝已知不兼容的 packed-tensor runtime ABI, 相关守卫由 `tests/xqt/test_tilelang_capability_preflight.py` 覆盖.

## 后续验证

1. 本机已升级 TileLang `0.1.12` (仍保留 `0.1.11` 为 known-incompatible). RTX 4070 Ti SUPER (`sm_89`) 上 linear / conv / norm / dequant / attention CUDA correctness tests, Triton RMSNorm / FeedForward, 以及 golden dry-run + TensorRT materialize 已通过. 更大 shape 与正式性能基线仍待扩展.
2. TensorRT golden 路径已通过 FP4 export-copy dense lowering 的 non-dry-run engine, materialized runtime handle, `1e-3` 数值验收和 `[8,64]` 最小 benchmark; 这不证明 packed-FP4 TensorRT runtime.
3. Stage report 序列化已修复: `json_safe_value` 对含 live TensorRT module 的 dataclass 不再 `asdict` 崩溃; `TensorRTRuntimeSession.to_dict()` 只保留 engine path / device / handle_materialized / inspector metadata.
4. Contract 合流进度:
   - payload contracts 提供 `from_stage_metrics(...)` 与可选 `module_contract`; `stage_provider` 从 model 自动提取.
   - `FeedForward.fusion_intent()` 产出 `FusionIntent`.
   - `xqt.convert` 全路径挂载 `_xqt_module_contract`; operator report 可透传 `module_contract`.
   - `xqt.nn.Attention` 已 materialize 到 `_TileLangXqtAttentionWrapper`; `TransformerBlock` tilelang 会 materialize 内部 `attn` 子模块, 完整 block-level 单 kernel fusion 仍为后续.
   - quant report / operator report 的 `to_dict` 可透传 `module_contract`; payload 层已可携带 ModuleContract.
   - 多 shape Attention / TransformerBlock convert 正确性覆盖见 `tests/xqt/test_nn_attention.py`.
5. 新增 engine 或 backend 时, 先更新 capability maturity, recipe hardware constraints 和对应的 StageReport assertions.
