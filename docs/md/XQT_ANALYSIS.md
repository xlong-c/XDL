# XQT 分析与优化能力规划

## 负责内容

- 定义 `xqt/` 内部的分析,诊断和优化建议能力.
- 说明压缩,导出和部署路径里哪些情况需要分析.
- 约束第一批分析 API,报告结构和 recipe 接入方式.

## 不负责内容

- 不替代 `docs/md/XQT.md`,这里只细化分析能力线.
- 不承诺本文中的规划已经全部实现.
- 不覆盖 `xdl/` 训练期 callback 的一般监控职责.

## 定位

`xqt` 不应只负责"做变换",还要负责"解释变换为什么可行,哪里出问题,下一步怎么调".

对 `xqt` 来说,分析能力不是可选装饰,而是压缩和部署链路的一等能力:

- 量化需要 sensitivity 和 calibration,否则很难决定哪些层保高精度.
- 剪枝需要 importance 和误差交叉分析,否则很难选结构化裁剪目标.
- 蒸馏需要 teacher/student 对齐分析,否则 loss 只是一团总数.
- 导出和后端对齐需要逐层 diff,否则很难定位 backend 漂移.
- benchmark 需要和误差一起看,否则无法做真实的精度-性能权衡.

因此,近期实现应优先落在 `xqt/` 内,服务压缩,导出,报告和 recipe runner,而不是先抽到更通用的仓库基础层.

## 当前基础

`xqt/` 已经有几块相关能力,但还没有形成统一的分析与优化闭环:

| 现有位置 | 当前能力 | 现状问题 |
| --- | --- | --- |
| `xqt.eval.compare` | tensor 级基础 diff | 还不够覆盖结构化误差,分布漂移和离散决策误差 |
| `xqt.quant.sensitivity` | 逐层输出 diff,高误差层排序 | 目前更偏量化,缺少统一报告和权重/激活联合视角 |
| `xqt.quant.calibration` | 激活范围统计 | 还缺少 clipping,saturation,outlier 等更直接的优化信号 |
| `xqt.prune` | sparsity report,剪枝流程 | 缺少 importance x sensitivity 的联合分析 |
| `xqt.eval.report` | JSON/CSV/Markdown 写出 | 还没有统一的 analysis record schema |

现阶段最合理的路线不是立刻重做一层全新系统,而是先把这些已有点位收束成同一条能力线.

## XQT 里哪些情况必须做分析

### 1. 量化

重点回答:

- 哪些层最敏感.
- 误差主要来自激活还是权重.
- 哪些层应该跳过量化或升高精度.
- 校准是否不足,量化范围是否过窄.

### 2. 剪枝

重点回答:

- 哪些模块低重要度但低敏感度,适合先裁.
- 哪些结构一旦裁掉就会放大量化或任务误差.
- 稀疏率提升后,真实 latency 是否跟着下降.

### 3. 蒸馏

重点回答:

- teacher/student 是 logits 差得多,还是 feature 对不齐.
- 误差主要集中在哪几层,哪几个 token/head/channel.
- student 的结构缩小后,是宽度瓶颈还是深度瓶颈.

### 4. 导出与后端对齐

重点回答:

- PyTorch 和后端输出从哪一层开始漂.
- 是数值小偏差,还是 top-k/argmax 这种决策已经变了.
- 哪些 op 或 shape 组合最容易触发偏差.

### 5. 步数压缩和生成路径

重点回答:

- 每步 latent / hidden state 漂移趋势如何.
- 误差是首步就放大,还是尾步累积.
- 加速收益和采样质量的 Pareto 前沿在哪里.

## 分析对象和优化动作应该配套

XQT 里的分析不应该只停留在"把误差打印出来". 每类分析都要能接到下一步优化动作.

| 场景 | 主要分析对象 | 核心误差 | 直接优化动作 |
| --- | --- | --- | --- |
| 量化 | activation,weight,layer output | `max_abs`,`mean_abs`,`cosine`,`saturation ratio` | skip list,混合精度,增加校准,改 granularity |
| 剪枝 | importance,sensitivity,module output | `importance score`,`layer diff`,`sparsity delta` | 调整 pruning target,改结构化粒度,局部回退 |
| 蒸馏 | logits,feature,attention | `KL`,`MSE`,`cosine`,`top-k overlap` | 调 loss 权重,改层对齐,改 student 宽深 |
| 导出 | backend output,layer output | `allclose`,`max_abs`,`argmax mismatch` | fallback op,禁用 fusion,约束 shape |
| 生成少步 | step-wise latent,attention map | `step diff`,`trajectory drift` | 调 timestep schedule,teacher cache,student step 数 |
| benchmark 取舍 | latency,memory,error,task metric | `p50/p90`,`metric delta`,`error delta` | 选择 Pareto 最优配置 |

## 模块边界建议

首期不急着新建 `xqt.analysis/`. 先沿着现有模块边界收束:

- `xqt.eval.compare`: 放基础 tensor diff 和 tensor summary.
- `xqt.eval.report`: 放 analysis record 的序列化和导出.
- `xqt.quant.sensitivity`: 放逐层输出误差,权重/激活联合分析,混合精度建议.
- `xqt.quant.calibration`: 放激活统计,分布漂移,clipping/saturation 分析.
- `xqt.prune`: 放 importance ranking 和 prune candidate 建议.
- `xqt.distill.hooks`: 放 teacher/student 中间层抓取和对齐.
- `xqt.pipeline.passes`: 放 recipe 级 `analyze` pass 和 artifact 落盘.

中期判断标准:

- 如果 `eval/`, `quant/`, `prune/`, `distill/` 之间开始重复维护同样的 tensor summary,diff record 和 layer table 逻辑,再把这部分抽成 `xqt.analysis/`.
- 在那之前,优先复用现有目录,避免在规划期先拆出第三套边界.

## 第一批建议 API

下面按"已有"和"建议补齐"区分,避免把规划写成已实现事实.

### 已有基础

| 状态 | API | 位置 | 作用 |
| --- | --- | --- | --- |
| 已有 | `compare_tensors()` | `xqt.eval.compare` | tensor 级数值对比 |
| 已有 | `analyze_layer_sensitivity()` | `xqt.quant.sensitivity` | 逐层输出误差排序 |
| 已有 | `calibrate_activation_statistics()` | `xqt.quant.calibration` | 激活范围统计 |
| 已有 | `write_json_report()` / `write_csv_report()` / `write_markdown_report()` | `xqt.eval.report` | 报告落盘 |

### 建议首批补齐

| 优先级 | API | 作用 |
| --- | --- | --- |
| P0 | `summarize_tensor()` | 输出 `shape`,`dtype`,`numel`,`mean`,`std`,`min`,`max`,`quantiles`,`zero_ratio`,`nan_count`,`inf_count` |
| P0 | `compare_tensors(..., structured=False)` | 在现有 `TensorDiff` 基础上补相对误差,可选 per-channel/per-token 结构化 diff |
| P0 | `analyze_layer_errors()` | 联合输出 layer output diff,activation summary,weight diff,排序键和 tag |
| P0 | `records_to_dataframe()` | 把逐层记录稳定转成表格,服务 CSV 和可视化 |
| P1 | `analyze_activation_drift()` | 比较 baseline/optimized 两侧激活统计,给出 clipping/outlier/saturation 信号 |
| P1 | `recommend_high_precision_modules()` | 用 sensitivity + policy 生成 skip list / mixed precision 建议 |
| P1 | `rank_prune_candidates()` | 用 importance x sensitivity 做剪枝候选排序 |
| P1 | `build_pareto_points()` | 组合 latency,memory,error,task metric,用于配置筛选 |

## 建议数据结构

目前已有 `TensorDiff`,`LayerSensitivityRecord`,`ActivationStatistic`. 首期建议在这个基础上补到可以直接做分析报告的层级.

### 1. `TensorSummary`

建议字段:

- `shape`
- `dtype`
- `numel`
- `mean`
- `std`
- `min`
- `max`
- `quantiles`
- `zero_ratio`
- `nan_count`
- `inf_count`

### 2. 扩展后的 `TensorDiff`

建议新增字段:

- `relative_error`
- `correlation`
- `valid`
- `message`
- `structured` 或 `details`

### 3. `LayerAnalysisRecord`

建议字段:

- `name`
- `module_type`
- `scope`
- `reference_summary`
- `candidate_summary`
- `diff`
- `weight_diff`
- `parameter_count`
- `tags`
- `recommendation`

### 4. `AnalysisReport`

建议字段:

- `scenario`
- `baseline`
- `candidate`
- `records`
- `aggregates`
- `metadata`

## 建议 recipe 配置块

XQT 现有 runner 已经走 YAML/OmegaConf 路线. 分析能力也应通过 recipe 显式打开,不要引入命令行参数解析.

建议增加一个可选的 `analysis` 配置块:

```yaml
analysis:
  enabled: true
  compare_to: baseline
  module_names: null
  top_k: 20
  metrics:
    - max_abs
    - mean_abs
    - cosine_similarity
  structured:
    per_channel: false
    per_token: false
  recommendations:
    mixed_precision: true
    prune_candidates: true
  export:
    json: true
    csv: true
    markdown: true
```

这个块的目标不是描述具体图表,而是控制采样范围,输出指标和是否生成优化建议.

## 建议新增 `analyze` pass

XQT 的 pass 现在覆盖 baseline,quant,prune,distill,export,benchmark 和 manifest. 分析应成为显式 pass,而不是散落在各个脚本里.

建议 pass 语义:

| Pass | 输入 | 输出 | 最小验证 |
| --- | --- | --- | --- |
| `analyze` | baseline/candidate model + sample batch + analysis config | `analysis.json`,`analysis.csv`,`analysis.md` | 报告完整,排序字段存在,可选阈值检查通过 |

建议插入位置:

- `baseline_eval` 之后,用于建立 baseline 统计.
- `quant` / `prune` / `distill` 之后,用于分析压缩后误差.
- `export` 之后,用于分析 backend 对齐误差.
- `benchmark` 之后,用于合并 Pareto 数据点.

## 优化建议不应只靠硬编码规则

XQT 需要的是"基于数据的建议",不是单纯打印几条经验.

首期可以接受规则式启发,但输出必须回到分析数据:

### 量化建议

- 某层 `mean_abs` 高且 `cosine` 低: 提示保留高精度.
- 某层 saturation/clipping 高: 提示增加校准样本或更换 granularity.
- 某些 norm/embedding/head 层长期高敏感: 提示加入 denylist.

### 剪枝建议

- importance 低且 sensitivity 低: 优先进入候选集.
- importance 低但 sensitivity 高: 标记为"风险候选",默认不先裁.
- 稀疏率提升但 latency 不降: 提示当前 pattern 对目标后端无收益.

### 蒸馏建议

- logits 接近但中间 feature 偏差大: 提示增加 feature KD.
- 早层误差低,末层误差高: 提示 student 宽度或 head 数不足.
- token/head 误差集中: 提示改 layer map 或注意力蒸馏权重.

### 导出建议

- backend 只在特定层后开始漂: 提示检查该层对应 op/fusion.
- 数值误差小但 argmax/top-k mismatch 高: 提示业务上已不可接受.
- 特定 shape 下 diff 激增: 提示限制 dynamic shape 或分 profile 导出.

## 可视化真正需要的数据

分析函数最终要服务图表和报告. 对 XQT 来说,第一批最有价值的可视化输入不是花哨界面,而是稳定的数据表.

首期应稳定输出:

- 逐层排序表: `layer,module_type,max_abs,mean_abs,cosine,recommendation`
- 激活统计表: `layer,min,max,mean,std,zero_ratio,saturation_ratio`
- 剪枝候选表: `layer,importance,sensitivity,recommended_action`
- 导出对齐表: `output_name,max_abs,allclose,argmax_mismatch`
- Pareto 点表: `config_id,latency_ms,memory_mb,error,metric_delta`

这些表已经足够支撑:

- 柱状图
- 热图
- 直方图
- token/head 排名图
- 精度-延迟散点图

## 分阶段实施建议

### M1: 先把已有能力接成统一报告

- 扩展 `compare_tensors()`
- 增加 `summarize_tensor()`
- 给 `analyze_layer_sensitivity()` 增加表格化导出
- 统一 `records_to_dataframe()`

### M2: 面向量化给出第一批建议

- 联合 `sensitivity + calibration + policy`
- 输出 mixed precision skip list 建议
- 在 recipe 中支持 `analysis.enabled`

### M3: 把剪枝和蒸馏接进来

- 增加 `prune.importance`
- 增加 `rank_prune_candidates()`
- 增加 teacher/student feature 对齐分析

### M4: 把分析 pass 接入 pipeline

- 在 `xqt.pipeline.passes` 中注册 `analyze`
- 报告纳入 manifest
- 把 benchmark 与 error 合并为 Pareto 报告

## 暂定结论

- 这条能力线应首先落在 `xqt/`,因为它直接服务压缩,导出和部署优化决策.
- 近期重点不是造一个抽象很重的新包,而是把 `eval`,`quant`,`prune`,`distill`,`pipeline` 现有点位收束成统一分析闭环.
- 第一批实现必须同时满足两件事: 能输出可视化所需数据,也能给出下一步优化建议.
