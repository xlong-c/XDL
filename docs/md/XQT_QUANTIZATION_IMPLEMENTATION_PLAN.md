# XQT 量化实现拆解文档

## 负责内容

- 把 [XQT_QUANTIZATION_REQUIREMENTS.md](XQT_QUANTIZATION_REQUIREMENTS.md) 的需求翻译成具体实现任务.
- 明确 `schema`, `quant/`, `pipeline`, `data`, `export`, `recipe`, `tests` 的改动面.
- 给后续量化开发提供可执行顺序,而不是只停留在模型族分类和策略建议.

## 不负责内容

- 不把本文中的拆解项视为当前已完成事实.
- 不替代 [XQT.md](XQT.md) 的整体路线图.
- 不给每个任务写完整代码实现; 具体实现应回到源码和测试中收口.
- 不把还未接入的第三方 backend 写成既成事实.

## 当前现状

当前 `xqt.quant` 已有能力:

- 量化 backend:
  - `torchao` PyTorch runtime 路径
  - ONNX Runtime static QDQ INT8 路径
- 量化策略:
  - `QuantizationPolicy`
  - 基于模块类型,名字模式和参数量的筛选
- 分析:
  - `calibrate_activation_statistics()`
  - `analyze_activation_drift()`
  - `analyze_layer_sensitivity()`
  - `analyze_layer_errors()`
  - `recommend_high_precision_modules()`
- runner 接线:
  - 内置 `quant` pass
  - ONNX QDQ 自动导出和前置融合 metadata 透传
  - manifest metric 和 artifact 记录
  - `QuantPass` 已拆为 plan + executor 编排
- 预检查:
  - `torchao` 依赖
  - FP8 CUDA 要求
  - ONNX QDQ 对 calibration / validation 的要求
  - 组件级 backend capability,数据来源和 runtime mix 检查
- 组件化:
  - `QuantConfig.component_policies`
  - `QuantizationExecutionPlan`
  - `QuantizationReport`
  - `multi_component_quant_smoke.yaml`

见:

- [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:66)
- [xqt/core/config.py](/root/workspace/xdl/xqt/core/config.py:107)
- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:588)
- [xqt/pipeline/preflight.py](/root/workspace/xdl/xqt/pipeline/preflight.py:152)
- [xqt/quant/policy.py](/root/workspace/xdl/xqt/quant/policy.py:25)
- [xqt/quant/types.py](/root/workspace/xdl/xqt/quant/types.py)
- [xqt/quant/plan.py](/root/workspace/xdl/xqt/quant/plan.py)
- [xqt/quant/executor.py](/root/workspace/xdl/xqt/quant/executor.py)
- [xqt/quant/capability.py](/root/workspace/xdl/xqt/quant/capability.py)
- [xqt/quant/torchao_backend.py](/root/workspace/xdl/xqt/quant/torchao_backend.py:80)
- [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py:131)
- [xqt/quant/calibration.py](/root/workspace/xdl/xqt/quant/calibration.py:177)
- [xqt/quant/sensitivity.py](/root/workspace/xdl/xqt/quant/sensitivity.py:81)

当前已落地的首批修正:

1. `QuantConfig` 已可表达组件级量化,高精度保留,skip list,force list 和 analysis-only 组件.
2. `QuantPass` 已下沉为 `plan.py` + `executor.py`,pass 本身只负责上下文回填和 manifest 记录.
3. batch 输入拆分已收敛到共享 helper,ONNX QDQ 无标签多输入 tuple 已有测试覆盖.
4. `calibration` / `sensitivity` 默认只分析 `quantize=True` 的模块.
5. preflight 已覆盖组件级 backend,data source,runtime mix 和 backend capability metadata.
6. `context.metrics["quant"]` 已统一为 `mode/components/summary/artifacts`,并保留 legacy 快捷字段.
7. 已新增 toy 异构 recipe 验证 `vision_encoder` QDQ artifact,`projector` analysis-only,`decoder` torchao 组件量化表达.

仍保留的后续缺口:

1. `torchao_backend.py` 还没有完整返回 backend 侧 skip reason 和 high precision reason,当前主要由 executor 按 config 记录.
2. ONNX QDQ 组件 artifact 当前不会回写替换 PyTorch 子模块,只产出组件级部署 artifact.
3. GPTQ,AWQ,bitsandbytes 当前只是配置和 preflight 层的 planned backend,runner 执行会明确拒绝; ModelOpt,OpenVINO PTQ,PT2E,QAT 和 KV cache quantization 仍是后续方向.
4. 视觉大模型/LLM/扩散模型的真实大样本精度和性能验收仍需要独立 recipe 和硬件环境验证.

## 总体实现原则

### 1. 先修正确性,后扩 backend

不要在当前输入/分析/报告边界还有已知缺陷时,直接叠加 GPTQ,AWQ,ModelOpt 或 QAT.

第一批任务应先修:

- 多输入 calibration 输入拆分
- 默认分析模块选择
- quant report 统一结构

### 2. 先把 `quant pass` 变薄,再加复杂策略

`xqt/pipeline/passes.py` 里的 `QuantPass` 应该只负责:

- 读取 config
- 组装执行 plan
- 调度 backend executor
- 回填 artifacts / metrics / manifest

backend 细节应尽量下沉到 `xqt/quant/`.

### 3. 组件优先,不要继续默认“整个模型只走一个 backend”

异构模型是常态:

- vision encoder
- projector
- LLM decoder
- text encoder
- UNet / DiT
- VAE

XQT 后续量化设计应允许:

- 同一模型的不同子模块走不同 backend
- 同一组件内部 mixed precision
- 某些组件只分析不实际量化

### 4. 数据,导出,分析和量化必须共用同一套输入语义

`export`, `quant`, `analysis` 都会碰到:

- Tensor
- tuple/list
- mapping
- 有 label 的 batch
- 无 label 的多输入 batch

这些逻辑不能继续在多处各自维护.

### 5. 报告和 manifest 不是附属物

最低要求不是“量化成功”,而是要能回答:

- 哪些模块被量化了
- 哪些模块被跳过了
- 使用了什么 backend 和策略
- 用了哪个 calibration split
- 输出差异和任务指标怎么变化

## 建议新增的数据结构

建议先把量化配置和执行计划从 `dict` 风格收紧成更清晰的 dataclass.

### 1. `QuantComponentPolicyConfig`

建议放在 [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:66).

作用:

- 描述一个组件级量化策略

建议字段:

- `name: str`
- `target: Optional[str] = None`
- `enabled: bool = True`
- `backend: Optional[str] = None`
- `strategy: Optional[str] = None`
- `policy: Dict[str, Any] = field(default_factory=dict)`
- `calibration_split: Optional[str] = None`
- `validation_split: Optional[str] = None`
- `keep_high_precision: List[str] = field(default_factory=list)`
- `skip_quantize: List[str] = field(default_factory=list)`
- `force_quantize: List[str] = field(default_factory=list)`
- `analysis_only: bool = False`

### 2. `QuantConfig` 扩展字段

当前只有:

- `enabled`
- `backend`
- `policy`

建议增加:

- `strategy: Optional[str] = None`
- `calibration_split: Optional[str] = None`
- `validation_split: Optional[str] = None`
- `keep_high_precision: List[str] = field(default_factory=list)`
- `skip_quantize: List[str] = field(default_factory=list)`
- `force_quantize: List[str] = field(default_factory=list)`
- `analysis_only_modules: List[str] = field(default_factory=list)`
- `component_policies: List[QuantComponentPolicyConfig] = field(default_factory=list)`

兼容策略:

- 保留 `backend + policy` 作为单组件 legacy 路径.
- 当 `component_policies` 非空时,将顶层 `backend/policy` 视为默认值或 fallback.

### 3. `QuantizationExecutionPlan`

建议新增文件:

- `xqt/quant/types.py`

作用:

- 表达一次 `quant` pass 实际要执行的组件列表和 backend 参数

建议字段:

- `components: list[QuantizationComponentPlan]`
- `analysis_enabled: bool`
- `artifact_prefix: str`
- `metadata: dict[str, Any]`

### 4. `QuantizationComponentPlan`

作用:

- 表达某个组件如何被量化

建议字段:

- `name`
- `target_path`
- `backend`
- `strategy`
- `policy`
- `calibration_split`
- `validation_split`
- `keep_high_precision`
- `skip_quantize`
- `force_quantize`
- `analysis_only`

### 5. `QuantizationReport`

建议放在:

- `xqt/quant/types.py`

作用:

- 统一 `torchao` 和 `onnxruntime_qdq` 的结果结构

建议字段:

- `backend`
- `strategy`
- `runtime`
- `component_name`
- `quantized_modules`
- `skipped_modules`
- `high_precision_modules`
- `artifacts`
- `calibration_summary`
- `analysis_summary`
- `metadata`

## Schema 改动建议

### P0-1 扩展量化配置字段

涉及:

- [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:66)
- [xqt/core/config.py](/root/workspace/xdl/xqt/core/config.py:107)
- [tests/xqt/test_config.py](/root/workspace/xdl/tests/xqt/test_config.py:1)

要改什么:

- 在 `QuantConfig` 中增加组件级和 mixed precision 相关字段.
- 新增 `QuantComponentPolicyConfig`.

怎么改:

- 先用 dataclass 明确定义字段,避免所有新能力都塞进 `policy: dict`.
- 保持旧 YAML 兼容: 顶层 `backend/policy` 仍能直接工作.

验收:

- 顶层单组件配置仍可加载.
- `component_policies` 配置可加载.
- `keep_high_precision` / `skip_quantize` / `force_quantize` 类型被校验.

### P0-2 增加 config 校验

涉及:

- [xqt/core/config.py](/root/workspace/xdl/xqt/core/config.py:107)
- [tests/xqt/test_config.py](/root/workspace/xdl/tests/xqt/test_config.py:1)

要改什么:

- backend 枚举校验
- `component_policies[*].name` 唯一性校验
- `target` 非空字符串校验
- `keep_high_precision` / `skip_quantize` / `force_quantize` 必须是字符串列表
- 冲突校验:
  - 同一个规则同时出现在 `force_quantize` 和 `skip_quantize`
  - `analysis_only=true` 但又要求落地 artifact 的非法组合

怎么改:

- 在 `_validate_config()` 中新增量化专用校验 helper,不要继续把所有逻辑塞进一个函数体里.

验收:

- 非法 backend,重复组件名,冲突规则都能报清晰错误.

## `xqt/data/` 与输入语义改动建议

### P0-3 新增共享 batch 输入工具

涉及:

- 建议新增 `xqt/data/input_utils.py`
- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:61)
- [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py:27)
- [xqt/export/input_utils.py](/root/workspace/xdl/xqt/export/input_utils.py:15)
- [tests/xqt/test_quant.py](/root/workspace/xdl/tests/xqt/test_quant.py:1)
- [tests/xqt/test_runner.py](/root/workspace/xdl/tests/xqt/test_runner.py:905)

要改什么:

- 把“从 batch 中提取模型输入”的逻辑抽成共享 helper.

建议首批函数:

- `split_inputs_from_batch(batch, *, has_target: bool | None = None) -> Any`
- `infer_batch_signature(batch) -> dict[str, Any]`
- `strip_target_fields(mapping_batch) -> dict[str, Any]`

怎么改:

- `pipeline._split_inputs_from_batch()` 删除本地实现,改用共享 helper.
- `onnx_qdq._as_numpy_inputs()` 不再自己猜测二元 tuple 就一定是 `(inputs, target)`.
- 对 tuple/list 的处理改成:
  - 明确 label key 或显式 target 才剥离 target
  - 否则按“多输入模型输入”处理

验收:

- `(x1, x2)` 这种无标签多输入 batch 能进入 ONNX auto-export 和 QDQ.
- `(x1, x2, y)` 这种带标签 batch 仍能正确剥离最后一个 target.
- mapping batch 的 label 过滤语义保持一致.

### P0-4 补 calibration summary 能力

涉及:

- [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py:65)
- [xqt/data/calibration.py](/root/workspace/xdl/xqt/data/calibration.py)
- [tests/xqt/test_quant.py](/root/workspace/xdl/tests/xqt/test_quant.py:284)
- [tests/xqt/test_runner.py](/root/workspace/xdl/tests/xqt/test_runner.py:834)

要改什么:

- 当前 `calibration_summary` 只有 `input_names`, `batch_count`, `shapes`, `dtypes`.
- 后续需要补:
  - `source_split`
  - `sample_limit`
  - `input_structure`
  - `has_targets`

怎么改:

- `IterableCalibrationDataReader` 构造时保存输入签名和 split 来源.
- 由 `QuantPass` 在调用时显式传入 split 名称.

验收:

- manifest 和 metrics 里能看出量化到底用的是 `calibration` 还是 `validation`.

## `xqt/quant/` 目录实现拆分建议

### 1. 保持 `policy.py` 只负责规则表达和候选筛选

当前 [xqt/quant/policy.py](/root/workspace/xdl/xqt/quant/policy.py:25) 已经基本符合这个定位.

但需要补:

- `selected_only` 语义 helper
- richer skip reason

建议增加:

- `selected_quantizable_modules()`
- `summarize_quantization_policy()`

### 2. 修正默认分析模块集合

涉及:

- [xqt/quant/calibration.py](/root/workspace/xdl/xqt/quant/calibration.py:187)
- [xqt/quant/sensitivity.py](/root/workspace/xdl/xqt/quant/sensitivity.py:72)
- [tests/xqt/test_quant.py](/root/workspace/xdl/tests/xqt/test_quant.py:1)

要改什么:

- 默认分析不应把 `quantize=False` 的模块纳入结果.

怎么改:

- `calibration.py` 和 `sensitivity.py` 只对 `candidate.quantize` 为真的模块构造默认 `names`.
- 保留 `module_names` 显式覆盖能力.

验收:

- 默认分析不再包含 `head`,`classifier`,`norm` 和容器模块.
- 指定 `module_names` 时仍允许显式分析这些模块.

### 3. 新增 `plan.py` 或 `executor.py`

建议新增文件:

- `xqt/quant/plan.py`
- `xqt/quant/executor.py`
- 或合并为 `xqt/quant/runtime.py`

职责:

- 解析 `QuantConfig` -> `QuantizationExecutionPlan`
- 调度 backend executor
- 统一返回 `QuantizationReport`

建议首批函数:

- `build_quantization_plan(context: XQTContext) -> QuantizationExecutionPlan`
- `run_quantization_plan(context: XQTContext, plan: QuantizationExecutionPlan) -> list[QuantizationReport]`
- `resolve_component_module(model, target_path: str) -> nn.Module`

### 4. `torchao_backend.py` 演进建议

涉及:

- [xqt/quant/torchao_backend.py](/root/workspace/xdl/xqt/quant/torchao_backend.py:80)
- [tests/xqt/test_quant.py](/root/workspace/xdl/tests/xqt/test_quant.py:240)

要改什么:

- 除 `quantized_modules` 外,还应返回:
  - `skipped_modules`
  - `high_precision_modules`
  - `policy_snapshot`
  - 组件名

怎么改:

- `filter_fn` 除记录命中的模块外,也记录被规则排除的原因.
- 将返回结果升级成更统一的 `QuantizationReport` 风格结构.

验收:

- runner 输出里能同时看到“量化了哪些层”和“明确跳过了哪些层”.

### 5. `onnx_qdq.py` 演进建议

涉及:

- [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py:131)
- [tests/xqt/test_quant.py](/root/workspace/xdl/tests/xqt/test_quant.py:284)
- [tests/xqt/test_runner.py](/root/workspace/xdl/tests/xqt/test_runner.py:834)

要改什么:

- 统一输入拆分
- 更完整 calibration summary
- 支持组件级 artifact 命名

怎么改:

- `_as_numpy_inputs()` 只负责把“已确认是模型输入”的对象转成 numpy.
- 是否剥离 target 由共享 batch helper 决定.
- `ONNXQDQQuantizationResult.metadata` 补充:
  - `component_name`
  - `source_split`
  - `input_structure`
  - `pre_export_fusion`

验收:

- 多输入无标签 tuple 可正常通过.
- metadata 足够驱动 manifest 和后续报告.

## `pipeline quant pass` 的演进建议

当前 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:588) 的 `QuantPass` 逻辑是:

- 读顶层 `quant.backend`
- 直接执行 `torchao` 或 `onnxruntime_qdq`
- 在 pass 内部处理 ONNX auto-export 和 metric 拼装

建议演进为:

1. 读取并标准化 `QuantConfig`
2. 构建 `QuantizationExecutionPlan`
3. 调度 backend executor
4. 统一回填:
   - `context.model`
   - `context.artifacts`
   - `context.metrics["quant"]`
   - manifest metrics / artifacts

### P0-5 拆分 `QuantPass`

涉及:

- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:588)
- 建议新增 `xqt/quant/plan.py`
- 建议新增 `xqt/quant/executor.py`
- [tests/xqt/test_runner.py](/root/workspace/xdl/tests/xqt/test_runner.py:834)

怎么改:

- 保留 `QuantPass` 类,但内部只保留薄 orchestration.
- ONNX auto-export helper 可拆成 `_prepare_onnx_source_for_quant()`.
- torchao / qdq 两条路径不要再在同一方法里直接拼 metrics dict.

验收:

- 单组件 legacy recipe 行为不变.
- 新测试可验证 `QuantPass` 支持多个组件 plan 的串行执行.

### P0-6 统一 `context.metrics["quant"]` 结构

涉及:

- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:652)
- [tests/xqt/test_runner.py](/root/workspace/xdl/tests/xqt/test_runner.py:834)

建议统一为:

- `mode`: `single_component` / `multi_component`
- `components`: `list[QuantizationReport.to_dict()]`
- `summary`
- `artifacts`

兼容策略:

- 保留顶层 `backend` / `strategy` / `quantized_modules` 快捷字段一段时间,但它们应从 `components[0]` 派生,而不是与底层字段分叉维护.

验收:

- `torchao` 和 `onnxruntime_qdq` 的结果结构对齐.
- 后续异构 recipe 不需要另写一套 metric schema.

## `preflight` 演进建议

### P0-7 扩展量化 preflight 检查

涉及:

- [xqt/pipeline/preflight.py](/root/workspace/xdl/xqt/pipeline/preflight.py:152)
- [tests/xqt/test_preflight.py](/root/workspace/xdl/tests/xqt/test_preflight.py:1)

要改什么:

- 当前只知道 `torchao` / `onnxruntime_qdq` 单组件检查.
- 后续需要覆盖:
  - 组件级 backend 依赖
  - component target 是否可解析
  - calibration split 是否存在
  - `keep_high_precision` / `skip_quantize` 是否命中已知模块
  - hetero recipe 的多 runtime 风险提示

怎么改:

- 新增 `_check_quant_component_policy()`
- 顶层 `quant` 和 `component_policies` 分别生成检查项
- 对 `validation` fallback 维持 warning,但要明确是哪个组件在 fallback

验收:

- preflight 能输出“vision_encoder 走 qdq,decoder 走 torchao”的组件级检查结果.

## Recipe 改动建议

### P0-8 更新现有 recipe

涉及:

- [xqt/recipes/smoke_cpu.yaml](/root/workspace/xdl/xqt/recipes/smoke_cpu.yaml)
- [xqt/recipes/image_resnet_onnx_qdq_int8.yaml](/root/workspace/xdl/xqt/recipes/image_resnet_onnx_qdq_int8.yaml)
- [xqt/recipes/image_resnet_cifar100_qdq_cpu.yaml](/root/workspace/xdl/xqt/recipes/image_resnet_cifar100_qdq_cpu.yaml)
- [xqt/recipes/image_vit_torchao_fp8.yaml](/root/workspace/xdl/xqt/recipes/image_vit_torchao_fp8.yaml)

要改什么:

- 显式加入:
  - `keep_high_precision`
  - `skip_quantize`
  - `calibration_split`
  - `strategy`

怎么改:

- `torchao` recipe 显式写出 `skip_quantize: [head, norm]`
- `onnxruntime_qdq` recipe 显式写出 `calibration_split: calibration`

验收:

- recipe 本身能表达策略,而不是靠代码里的默认经验.

### P1-1 新增组件级异构 recipe

建议新文件:

- `xqt/recipes/vlm_hetero_quant_smoke.yaml`
- 或 `xqt/recipes/multi_component_quant_smoke.yaml`

建议模型:

- toy vision encoder + projector + decoder 组合模块

能力:

- encoder 走 `onnxruntime_qdq`
- projector 保 `bf16`
- decoder 走 `torchao`

验收:

- runner,preflight,metrics,manifest 都能表达组件级差异.

## P0 实现任务 - 正确性与基础编排

### P0-1 扩展 `QuantConfig` 和校验

涉及:

- `xqt/core/schema.py`
- `xqt/core/config.py`
- `tests/xqt/test_config.py`

最小目标:

- 为 component-level 和 mixed precision 打地基,同时兼容旧配置.

### P0-2 抽共享 batch 输入工具,修复无标签多输入 tuple

涉及:

- `xqt/data/input_utils.py` 或 `xqt/data/calibration.py`
- `xqt/pipeline/passes.py`
- `xqt/quant/onnx_qdq.py`
- `tests/xqt/test_quant.py`
- `tests/xqt/test_runner.py`

最小目标:

- `PairModel(left, right)` 这种模型能走 QDQ auto-export.

### P0-3 修正默认分析模块选择

涉及:

- `xqt/quant/policy.py`
- `xqt/quant/calibration.py`
- `xqt/quant/sensitivity.py`
- `tests/xqt/test_quant.py`

最小目标:

- 默认分析只覆盖真正被策略选中的模块.

### P0-4 新增 quant plan / executor,薄化 `QuantPass`

涉及:

- `xqt/quant/plan.py`
- `xqt/quant/executor.py`
- `xqt/pipeline/passes.py`
- `tests/xqt/test_runner.py`

最小目标:

- backend 分派与 metrics 拼装从 pass 中下沉.

### P0-5 统一 quant report 与 manifest 表达

涉及:

- `xqt/quant/types.py`
- `xqt/pipeline/passes.py`
- `tests/xqt/test_runner.py`

最小目标:

- `torchao` 和 `onnxruntime_qdq` 共享一套结果形状.

### P0-6 扩展 preflight

涉及:

- `xqt/pipeline/preflight.py`
- `tests/xqt/test_preflight.py`

最小目标:

- 输出更清晰的量化数据来源和组件级依赖检查.

### P0-7 更新现有 recipe 与 README

涉及:

- `xqt/recipes/*.yaml`
- `xqt/README.md`

最小目标:

- 现有 recipe 明确量化策略,而不是隐式依赖默认值.

## P1 实现任务 - 组件级量化与异构模型

### P1-1 组件级 submodule 解析与替换

涉及:

- `xqt/quant/plan.py`
- `xqt/quant/executor.py`
- `xqt/pipeline/passes.py`
- `tests/xqt/test_runner.py`

能力:

- 用 `target` 路径解析子模块
- 对 PyTorch runtime 组件替换为量化后的子模块
- 对 ONNX QDQ 组件生成独立 artifact

验收:

- 组件名和 artifact 名稳定可追踪

### P1-2 异构模型 metrics / artifact 组织

涉及:

- `xqt/pipeline/passes.py`
- `xqt/core/artifact.py`
- `tests/xqt/test_runner.py`

能力:

- 组件级 artifact 命名
- 组件级 metric 聚合
- manifest 中明确组件和 runtime

### P1-3 新增 backend capability matrix

建议新增:

- `xqt/quant/capability.py`

涉及:

- `xqt/pipeline/preflight.py`
- `tests/xqt/test_preflight.py`

能力:

- backend + 模块族 + 硬件的规则化能力描述
- 当前已新增 `xqt/quant/capability.py`,覆盖 `torchao` 和 `onnxruntime_qdq` 已接线后端,并为 `gptq`,`awq`,`bitsandbytes` 保留 planned 描述.
- preflight 已把 capability metadata 写入 `compression.quant.capability` 和 `compression.quant.component_policies.<name>.capability`.

例子:

- `torchao/fp8` 需要 CUDA
- `onnxruntime_qdq` 需要 exportable ONNX graph
- future `gptq/awq` 只支持 Linear-heavy 权重路径

### P1-4 大模型 weight-only backend 接口预留

建议新增:

- `xqt/quant/hf_backend.py`
- 或 `xqt/quant/transformers_backend.py`

涉及:

- `xqt/core/schema.py`
- `xqt/core/config.py`
- `xqt/pipeline/preflight.py`
- `tests/xqt/test_config.py`
- `tests/xqt/test_preflight.py`

说明:

- P1 先做接口和 preflight,再决定优先接 GPTQ,AWQ,bitsandbytes 还是 torchao HF integration.
- 当前已完成最小接口预留:
  - `compression.quant.backend` 和 `component_policies[*].backend` 可加载 `gptq`,`awq`,`bitsandbytes`.
  - `xqt.quant.capability` 把这些 backend 标为 `planned`.
  - preflight 对 planned backend 输出 warning.
  - runner 执行 planned backend 时明确报错,避免把未接 adapter 的路径伪装成可运行能力.

### P1-5 新增异构 smoke recipe

涉及:

- `xqt/recipes/multi_component_quant_smoke.yaml`
- `tests/xqt/test_runner.py`

说明:

- 这是验证 component policy 和 metrics schema 的关键闭环.

## P2 实现任务 - 新 backend 与服务场景

### P2-1 ModelOpt / SmoothQuant / OpenVINO PTQ

建议新增:

- `xqt/quant/modelopt_backend.py`
- 可选 `xqt/quant/openvino_ptq.py`

涉及:

- `xqt/core/schema.py`
- `xqt/pipeline/preflight.py`
- `tests/xqt/test_preflight.py`
- `tests/xqt/test_runner.py`

### P2-2 KV cache quantization

涉及:

- `xqt/core/schema.py`
- `xqt/benchmark/latency.py`
- `xqt/pipeline/passes.py`
- `tests/xqt/test_runner.py`

能力:

- 区分 prefill / decode benchmark
- 单独记录 KV cache policy

### P2-3 PT2E / QAT

建议新增:

- `xqt/quant/pt2e_backend.py`

涉及:

- `xqt/core/schema.py`
- `xqt/pipeline/passes.py`
- `tests/xqt/test_config.py`

说明:

- 这类路径与训练环节耦合更深,应明显晚于 P0/P1.

## 测试拆解建议

### 单元测试

建议继续放在:

- `tests/xqt/test_quant.py`

新增测试组:

- 共享 batch 输入拆分:
  - `(tensor,)`
  - `(x, y)`
  - `(x1, x2)`
  - `(x1, x2, y)`
  - mapping + label keys
- policy:
  - `skip_quantize`
  - `force_quantize`
  - `keep_high_precision`
- calibration / sensitivity 默认只选 `quantize=True`
- `torchao` report 包含 skip / high precision metadata
- `onnx_qdq` calibration summary 包含 split 和结构信息

### runner 测试

建议继续放在:

- `tests/xqt/test_runner.py`

新增测试组:

- `QuantPass` legacy 单组件路径保持兼容
- 多组件 quant plan
- 异构 recipe metrics 聚合
- ONNX QDQ 无标签多输入 tuple 真正跑通

### preflight 测试

建议继续放在:

- `tests/xqt/test_preflight.py`

新增测试组:

- 组件级 backend 依赖
- 组件级 calibration split fallback
- `force_quantize` / `skip_quantize` 冲突
- hetero runtime warning

### config 测试

建议继续放在:

- `tests/xqt/test_config.py`

新增测试组:

- `component_policies` 合法配置
- 非法 backend
- 重复组件名
- 冲突规则校验

## 推荐的开发顺序

建议按下面顺序推进:

1. `QuantConfig` 扩展
2. config 校验
3. 共享 batch 输入工具
4. 修正默认分析模块选择
5. `QuantizationReport` 统一结构
6. `QuantPass` 拆分为 plan + executor
7. preflight 扩展
8. 更新现有 recipe
9. 多组件 smoke recipe
10. 再接新 backend

这个顺序的原因:

- 先修正确性,避免在错误输入语义上叠加复杂功能.
- 先统一 schema 和 report,后面加新 backend 成本更低.
- 异构模型支持的前提不是更多算法,而是更稳的编排和更清晰的 artifact/metric 结构.

## 暂不建议现在就做的事

- 还没统一 `QuantPass` 前就直接接 GPTQ/AWQ/bitsandbytes.
- 还没修多输入 batch 语义前就做异构多模态 recipe.
- 一开始就做 KV cache quantization.
- 一开始就做 QAT/PT2E.
- 一开始就把 `xqt.quant` 提升为 Provisional API.

## 对外文档同步建议

如果开始实现本文内容,至少同步更新:

- [XQT_QUANTIZATION_REQUIREMENTS.md](XQT_QUANTIZATION_REQUIREMENTS.md)
- [XQT_DATA.md](XQT_DATA.md)
- [XQT_ANALYSIS.md](XQT_ANALYSIS.md)
- [XQT.md](XQT.md)
- [xqt/README.md](/root/workspace/xdl/xqt/README.md)
