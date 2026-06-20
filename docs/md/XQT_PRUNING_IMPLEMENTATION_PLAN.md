# XQT 结构化剪枝实现拆解文档

## 负责内容

- 把 [XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md) 的需求翻译成具体实现任务.
- 明确 `schema`, `prune/`, `pipeline`, `recipe`, `tests` 的分阶段改动面.
- 给后续结构化剪枝开发提供可执行的实现顺序,而不是只给概念分类.

## 不负责内容

- 不把本文中的拆解项视为当前已完成事实.
- 不替代 [XQT.md](XQT.md) 的整体路线图.
- 不给每个任务写详细代码实现;具体实现应回到源码和测试中收口.

## 当前现状

当前 `xqt.prune` 已有能力:

- 非结构化剪枝: `global_l1_unstructured`
- 规则化稀疏: `nm_structured`
- 剪枝 schedule: `PruningSchedule`, `run_prune_kd_loop()`
- 局部 rewrite helper:
  - `prune_linear_out_features()`
  - `prune_linear_in_features()`
  - `prune_linear_in_out_features()`
  - `prune_conv2d_out_channels()`
  - `prune_conv2d_in_channels()`
  - `prune_batchnorm_channels()`
- importance ranking: `collect_module_importance()`, `rank_prune_candidates()`
- 结构化 target / plan / report:
  - `PruningTarget`
  - `StructuredPruningPlan`
  - `StructuredPruningReport`
  - `NMStructuredPruningReport`
- 已支持的结构化粒度:
  - CNN `channel/filter`
  - ViT/Transformer `mlp_neuron`
  - ViT `head`
  - `block`
- 已支持的 runner recipe:
  - `xqt/recipes/cnn_structured_prune.yaml`
  - `xqt/recipes/vit_structured_prune.yaml`

当前缺失:

- block sparsity
- backend-aware 稀疏 runtime 真加速闭环
- 更复杂拓扑的 CNN residual/stage 级依赖传播
- 更通用的 attention/Transformer 实现适配
- MoE expert pruning

## 总体实现原则

### 1. 先 plan,后 rewrite

不要直接在 pass 里边算边改模型.

先产出:

- 哪些模块可剪
- 每个模块保留哪些 index
- 哪些后继模块需要联动改写

再执行 rewrite.

### 2. 先支持最清晰的结构,再扩难结构

建议顺序:

1. CNN channel/filter pruning
2. ViT/Transformer 的 MLP neuron pruning
3. Attention head pruning
4. Block pruning
5. N:M / block sparsity

### 3. 把“候选评分”和“结构改写”分模块

`importance.py` 负责“给分”.

`structured.py` 负责:

- plan 生成
- 剪枝粒度逻辑
- rewrite 调度

### 4. 把“shape 变了”作为结构化剪枝的最低验收线

结构化剪枝不能只看 sparsity.

最低验收必须包括:

- 模块 shape 变了
- 模型 forward 能跑
- 参数量/FLOPs 变了

## 建议新增的数据结构

建议放在 `xqt/prune/structured.py` 或拆到 `xqt/prune/types.py`.

### 1. `StructuredPruningSpec`

作用:

- 表达一次结构化剪枝请求

建议字段:

- `granularity: str`
- `scope: str`
- `target_sparsity: float | None`
- `module_name_patterns: list[str]`
- `exclude_module_names: list[str]`
- `importance_type: str`
- `selection_mode: str`
- `keep_indices: dict[str, list[int]] | None`
- `score_threshold: float | None`

### 2. `PruningTarget`

作用:

- 表达一个可剪结构对象

建议字段:

- `module_name`
- `module_type`
- `granularity`
- `group_size`
- `dependency_group`
- `metadata`

例子:

- Conv 输出通道 target
- Attention head target
- ViT block target

### 3. `StructuredPruningAction`

作用:

- 表达一个模块级改写动作

建议字段:

- `module_name`
- `action_type`
- `keep_indices`
- `depends_on`
- `metadata`

例子:

- `conv_out_channels`
- `conv_in_channels`
- `linear_out_features`
- `linear_in_features`
- `batchnorm_channels`
- `attention_heads`
- `drop_blocks`

### 4. `StructuredPruningPlan`

作用:

- 表达一次结构化剪枝的完整 plan

建议字段:

- `granularity`
- `targets`
- `actions`
- `summary`

### 5. `StructuredPruningReport`

作用:

- 结构化剪枝执行后的统一报告

建议字段:

- `granularity`
- `action_count`
- `modules_before_after`
- `params_before`
- `params_after`
- `flops_before`
- `flops_after`
- `output_diff`
- `metric_delta`

## Schema 改动建议

当前 [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:57) 的 `PruneConfig` 只有:

- `enabled`
- `method`
- `target_sparsity`
- `schedule`
- `params`

这对结构化剪枝不足.

### 第一阶段最小扩展

建议在 `PruneConfig` 增加:

- `granularity: Optional[str] = None`
- `scope: str = "global"`
- `importance: Dict[str, Any] = field(default_factory=dict)`
- `selection: Dict[str, Any] = field(default_factory=dict)`
- `rewrite: Dict[str, Any] = field(default_factory=dict)`

这样能保持兼容现有 `params`,又能先把结构化配置显式化.

### `core/config.py` 校验建议

需要增加:

- `granularity` 枚举校验
- `scope` 枚举校验
- `selection.keep_indices` 格式校验
- `importance.type` 枚举校验
- `method=structured` 时的必填字段校验

## `xqt/prune/` 目录实现拆分建议

### 1. 保持 `rewrite.py` 只做单模块 helper

继续保持:

- `prune_linear_out_features()`
- `prune_linear_in_features()`
- `prune_conv2d_out_channels()`
- `prune_conv2d_in_channels()`
- `prune_batchnorm_channels()`

不要把模型级依赖传播塞进 `rewrite.py`.

### 2. 新增 `structured.py`

建议职责:

- 枚举可剪结构
- 生成结构化 plan
- 执行模型级 rewrite
- 汇总结构化报告

建议首批函数:

- `find_structured_pruning_targets()`
- `score_structured_targets()`
- `build_structured_pruning_plan()`
- `apply_structured_pruning_plan()`
- `summarize_structured_pruning()`

### 3. 可选新增 `cost.py`

建议职责:

- 估算 params / FLOPs before-after

建议首批函数:

- `count_model_parameters()`
- `estimate_conv_linear_flops()`
- `compare_model_cost()`

### 4. 可选新增 `report.py`

建议职责:

- 把结构化剪枝结果写成统一可序列化对象

## P0 实现任务 - CNN channel/filter pruning

这是首个真正应该落地的结构化剪枝闭环.

### P0-1 新增 schema 字段

涉及:

- `xqt/core/schema.py`
- `xqt/core/config.py`
- `tests/xqt/test_config.py`

验收:

- `method=structured`
- `granularity=channel/filter`
- `importance.type=bn_gamma/l1/l2`

### P0-2 枚举 Conv-BN 结构 target

涉及:

- `xqt/prune/structured.py`
- `tests/xqt/test_prune.py`

能力:

- 找到 Conv 输出通道 target
- 找到依赖的 BN
- 找到下游 Conv/Linear 输入依赖

验收:

- 能输出 target 列表
- target 中含依赖信息

### P0-3 生成 channel pruning plan

涉及:

- `xqt/prune/structured.py`
- `xqt/prune/importance.py`
- `tests/xqt/test_prune.py`

能力:

- 根据 importance 选择 keep indices
- 生成 `StructuredPruningPlan`

验收:

- plan 可序列化
- plan 包含每层 keep indices

### P0-4 执行 Conv-BN-Linear rewrite

涉及:

- `xqt/prune/structured.py`
- `xqt/prune/rewrite.py`
- `tests/xqt/test_prune.py`

能力:

- 批量应用 plan
- 更新父模块引用
- 重写下游依赖层

验收:

- 模型 forward 正常
- shape 真实变化

### P0-5 增加结构化 prune pass 分支

涉及:

- `xqt/pipeline/passes.py`
- `tests/xqt/test_runner.py`

能力:

- `compression.prune.method=structured`
- 调用 plan + rewrite + report

验收:

- runner 能跑结构化 recipe
- `context.metrics["prune"]` 返回结构化报告

### P0-6 增加 `cnn_structured_prune` recipe

建议新文件:

- `xqt/recipes/cnn_structured_prune.yaml`

建议模型:

- 小型 ResNet / Conv-heavy toy model

验收:

- 参数量下降
- ONNX 图输入输出 channel 真实变小

## P1 实现任务 - ViT / Transformer 结构化剪枝

### P1-1 MLP neuron pruning

涉及:

- `xqt/prune/structured.py`
- `xqt/prune/rewrite.py`
- `tests/xqt/test_prune.py`

能力:

- 识别 FFN 对:
  - `fc1.out_features`
  - `fc2.in_features`
- 联动裁剪

验收:

- block 内部宽度变小
- block 输入输出 hidden size 不变

### P1-2 Attention head pruning

涉及:

- `xqt/prune/structured.py`
- 可能新增 `rewrite_attention_heads()`
- `tests/xqt/test_prune.py`

能力:

- 识别 attention head 结构
- 生成 `keep_head_indices`
- 重写 qkv / proj

验收:

- `num_heads` 更新
- reshape 逻辑正确
- forward 正常

### P1-3 Block pruning

涉及:

- `xqt/prune/structured.py`
- `tests/xqt/test_prune.py`

能力:

- 对 `ModuleList` 中 block 打分
- 保留 block 子集
- 重建 block 列表

验收:

- block 数减少
- benchmark latency 下降

### P1-4 增加 `vit_structured_prune` recipe

建议新文件:

- `xqt/recipes/vit_structured_prune.yaml`

建议先支持:

- `mlp_neuron`
- `block`

head pruning 可后续补.

验收:

- 至少一种 ViT 结构真实改写
- 报告包含 pre/post block 或 width 变化

## P2 实现任务 - 规则化结构稀疏

### P2-1 N:M structured sparsity

涉及:

- `xqt/prune/structured.py`
- 可选 `xqt/prune/masks.py`
- `tests/xqt/test_prune.py`

能力:

- 生成满足 N:M 约束的 mask
- 报告约束满足率

验收:

- 报告里明确这是 `nm` 而不是 `channel/head/block`

### P2-2 backend-aware benchmarking

涉及:

- `xqt/pipeline/preflight.py`
- `xqt/benchmark/`
- `tests/xqt/test_runner.py`

能力:

- 明确某后端是否支持该格式
- benchmark 报告标记 capability

## `pipeline prune pass` 的演进建议

当前 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:415) 的 `PrunePass` 逻辑是:

- 非结构化 one-shot
- 非结构化 schedule + 可选 KD 微调

建议演进为:

1. `method=global_l1_unstructured`
   - 维持现状
2. `method=structured`
   - build plan
   - apply plan
   - summarize structured report
3. `method=nm_structured`
   - 规则化 mask 路径

注意:

- 不要让结构化剪枝硬复用非结构化 `target_sparsity` 语义.
- 对 `block pruning` 更适合 `keep_indices` 或 `drop_count`.

## 测试拆解建议

### 单元测试

建议放在:

- `tests/xqt/test_prune.py`

新增测试组:

- target 枚举
- plan 生成
- Conv-BN-Linear rewrite
- MLP pair rewrite
- attention head rewrite
- block list rebuild
- 结构化报告序列化

### runner 测试

建议放在:

- `tests/xqt/test_runner.py`

新增测试组:

- `cnn_structured_prune` recipe
- `vit_structured_prune` recipe
- 剪枝后 benchmark/report/manifest

### config 测试

建议放在:

- `tests/xqt/test_config.py`

新增测试组:

- `method=structured` 合法配置
- 非法 granularity / scope / importance 校验

## 推荐的开发顺序

建议按下面顺序推进:

1. schema 扩展
2. `StructuredPruningPlan` 数据结构
3. CNN target 枚举
4. CNN rewrite 执行器
5. `structured` prune pass
6. `cnn_structured_prune` recipe
7. MLP neuron pruning
8. block pruning
9. attention head pruning
10. ViT recipe

这个顺序的原因:

- CNN 的依赖传播最清晰.
- ViT 的 head/block rewrite 更容易因为模块实现细节出错.
- 先把报告,plan,pass 框架打稳,后面加新粒度会更便宜.

## 暂不建议现在就做的事

- 一开始就做全局 hidden size 重写
- 一开始就做 MoE expert pruning
- 一开始就把所有论文名都映射成一个 `method` 字段
- 先做复杂后端适配,再补基础 rewrite

## 对外文档同步建议

如果开始实现本文内容,至少同步更新:

- [docs/md/XQT.md](XQT.md)
- [docs/md/XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md)
- `xqt/README.md`
