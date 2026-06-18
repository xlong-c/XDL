# 通用误差分析与可视化数据需求

## 负责内容

- 说明哪些工程场景需要做误差分析.
- 归纳常见误差类型,粒度和对比对象.
- 归纳为可视化和诊断提供数据支持时,需要哪些通用函数和数据结构.

## 不负责内容

- 不定义某个具体压缩算法,导出后端或训练任务的业务逻辑.
- 不承诺某个模块路径已经实现.
- 不替代任务指标文档,例如分类指标,生成质量指标或检索指标的领域细节.

## 定位

这里讨论的"误差分析"是中性的基础能力,不应天然耦合到 `xqt/`.

原因很直接:

- 模型压缩只是误差分析的一个使用场景.
- 导出验证,后端对齐,训练诊断,重构回归,可视化排障同样需要误差分析.
- 这类函数更像通用的数据观察和比较能力,应该先定义清楚输入,输出和粒度,再决定最终落点.

当前仓库里已经有一些分散实现可作为参考:

- `xqt/eval/compare.py`: tensor 数值对比.
- `xqt/quant/sensitivity.py`: 层输出对比.
- `xqt/quant/calibration.py`: 激活统计.
- `xdl/callbacks/layer_monitor.py`: 权重和梯度统计.

它们说明需求真实存在,但不意味着最终归属必须在 `xqt/`.

## 哪些情况需要分析误差

### 1. 模型压缩前后

典型场景:

- 量化前后.
- 剪枝前后.
- 蒸馏 teacher / student.
- 混合精度豁免层筛选.

关注点:

- 哪些层最敏感.
- 误差主要来自权重,激活,还是离散决策变化.
- 精度损失是否集中在少数层或少数 batch.

### 2. 导出与部署后端对齐

典型场景:

- PyTorch vs ONNX Runtime.
- PyTorch vs TensorRT.
- PyTorch vs OpenVINO.
- eager vs torch.compile / torch.export.

关注点:

- 算子替换后输出是否一致.
- dtype,layout,broadcast,插值,归一化等语义是否漂移.
- 某些层是否出现 NaN,Inf,裁剪或范围压缩.

### 3. 训练过程异常诊断

典型场景:

- loss 突增或不下降.
- 梯度消失,梯度爆炸.
- 某些层长期不更新.
- AMP 或自定义 kernel 引入数值不稳定.

关注点:

- 梯度范数和更新量是否异常.
- 激活是否塌缩到常数,全零或极窄范围.
- 参数和梯度分布是否突然漂移.

### 4. 数据或预处理链路变更

典型场景:

- 归一化参数变更.
- tokenizer / resize / crop / pad 策略变更.
- 新增数据增强.
- dataloader / collate 重构.

关注点:

- 变化是输入层面的,还是经过模型后被放大.
- 哪些中间表示最先发生漂移.
- 误差来自值偏移,形状变化,还是离散 token 对齐问题.

### 5. 模型重构或替换子模块

典型场景:

- 用 fused kernel 替换原始算子.
- 用新 attention / norm / activation 实现替换旧实现.
- 做性能优化或代码清理后验证无回归.

关注点:

- 功能等价实现是否真正数值等价.
- 误差是局部可接受偏差,还是已经影响最终任务输出.
- 某一层的小误差是否被后续层放大.

### 6. 生成模型和序列模型质量漂移

典型场景:

- 采样器或 scheduler 改动.
- cache,KV layout,位置编码实现变更.
- 多步推理变单步或少步推理.

关注点:

- 每步 latent / hidden state 的漂移趋势.
- attention map 或 token logit 的结构变化.
- 误差是早期放大还是末端累积.

### 7. 在线回归和 A/B 排障

典型场景:

- 同一个 checkpoint 在两套环境结果不一致.
- 线上与离线评测偏差较大.
- 某次依赖升级后指标回退.

关注点:

- 是环境问题,数据问题,还是模型执行路径问题.
- 误差是否只出现在特定 batch,特定 shape 或特定设备.

## 误差应该按什么对象来分析

误差分析先决定"比较谁",再决定"怎么算".

常见对比对象:

- 输入 vs 参考输入.
- 中间层输出 A vs 中间层输出 B.
- 权重 A vs 权重 B.
- 梯度 A vs 梯度 B.
- 更新量 A vs 更新量 B.
- 最终 logits / score / mask / image / latent / hidden states.
- 任务指标 A vs 任务指标 B.

常见粒度:

- 标量级,例如单个 loss.
- tensor 整体级,例如一个 activation tensor 的摘要.
- 维度级,例如 per-channel,per-head,per-token.
- 层级,例如每层排序.
- 模块组级,例如所有 attention 层,所有 MLP 层.
- batch 级.
- 数据集级.
- 时间级,例如训练 step 轨迹或生成 step 轨迹.

## 常见误差类型

不是每种场景都需要所有误差. 重点是根据目标选最能解释问题的那几类.

### 1. 有效性异常

这是最先要查的一层.

- shape mismatch
- dtype mismatch
- device mismatch
- NaN count
- Inf count
- empty tensor
- requires_grad 状态异常

这类问题通常不需要复杂可视化,但需要稳定的前置检查函数.

### 2. 绝对误差类

适合回答"差了多少".

- max absolute error
- mean absolute error
- median absolute error
- percentile absolute error,例如 p95,p99

适合:

- 后端对齐.
- 层输出排序.
- 找局部尖峰异常.

### 3. 相对误差类

适合回答"相对原值偏了多少".

- mean relative error
- max relative error
- symmetric relative error
- norm ratio

适合:

- 不同量纲之间比较.
- 小值和大值混在一起时看偏差比例.

注意:

- 参考值接近 0 时需要稳定分母或单独屏蔽.

### 4. 二次误差类

适合放大大偏差样本.

- MSE
- RMSE
- normalized MSE

适合:

- 数值回归场景.
- 关注大误差是否集中爆发.

### 5. 相似性类

适合回答"趋势是否还一致".

- cosine similarity
- Pearson correlation
- Spearman rank correlation

适合:

- 激活方向是否一致.
- 排序结构是否保留.
- feature drift 但幅值不完全可比的场景.

### 6. 分布误差类

适合回答"整体分布有没有变".

- mean,std,min,max
- quantile,p01,p50,p99
- histogram
- zero ratio
- saturation ratio
- clipping ratio
- outlier ratio
- KL divergence
- JS divergence
- Wasserstein distance

适合:

- 量化校准.
- 激活塌缩诊断.
- AMP / 低比特 / 剪枝后的统计漂移.

### 7. 离散决策误差类

适合看输出值之外的决策变化.

- argmax mismatch rate
- top-k overlap
- sign mismatch rate
- threshold flip rate
- token mismatch rate

适合:

- 分类.
- 检索排序.
- 二值化或阈值触发模块.
- token 预测.

### 8. 结构化张量误差类

适合图像,序列和注意力这类有内部结构的表示.

- per-channel error
- per-head error
- per-token error
- per-time-step error
- spatial heatmap diff
- attention map diff

适合:

- 定位误差究竟来自哪一部分结构.
- 为热图,箱线图,直方图,token 排名图提供输入.

### 9. 梯度与优化误差类

适合训练期诊断.

- gradient mean,std,min,max
- gradient L1 / L2 norm
- update norm
- grad / weight ratio
- zero-gradient ratio
- exploding-gradient flag
- vanishing-gradient flag

适合:

- 训练不收敛.
- 某些层学不到.
- 新优化器或 AMP 路径验证.

### 10. 任务级误差类

适合回答"最终业务上损失了多少".

- accuracy delta
- F1 delta
- perplexity delta
- mAP delta
- PSNR / SSIM delta
- BLEU / ROUGE delta

注意:

- 任务级指标不能替代中间误差,只能说明后果.
- 中间误差和任务级误差都要有,才能解释"为什么退化".

### 11. 稳定性误差类

适合回答"结果是否可重复".

- seed-to-seed variance
- batch-to-batch variance
- run-to-run variance
- time-drift curve

适合:

- 压缩后模型偶发失败.
- 某条后端路径只在特定输入上出问题.

## 不同场景推荐看哪些误差

| 场景 | 优先误差 | 常见补充 |
| --- | --- | --- |
| 量化 | 激活范围,绝对误差,相似性,分布漂移 | saturation ratio,clipping ratio,top-k overlap |
| 剪枝 | 层输出误差,任务指标变化 | 零值比例,通道级误差 |
| 蒸馏 | feature 相似性,logit 误差 | token / channel 级热图 |
| 导出 / 后端对齐 | max abs,mean abs,allclose,NaN / Inf | dtype / shape 检查,per-layer diff |
| 训练异常 | 梯度范数,激活统计,更新量 | zero-gradient ratio,分布突变 |
| 数据链路变更 | 输入统计,早期层输出误差 | token mismatch,分布偏移 |
| 生成模型 | step-wise latent diff,attention diff | 最终图像质量指标 |
| 重构回归 | 层级 diff + 最终指标 diff | 模块组聚合排序 |

## 可视化误差场景需要哪些通用函数

这里先只定义函数族,不绑定最终模块路径.

### 1. 基础摘要函数

目标: 把单个 tensor 变成稳定可序列化的统计摘要.

建议能力:

- `summarize_tensor()`
- `histogram_summary()`
- `quantile_summary()`
- `finite_check()`
- `sparsity_summary()`

建议输出:

- shape
- dtype
- device
- numel
- mean,std,min,max
- quantiles
- zero ratio
- NaN / Inf count

### 2. 基础对比函数

目标: 比较两个 tensor 或两个标量容器.

建议能力:

- `compare_tensors()`
- `compare_scalars()`
- `compare_histograms()`
- `compare_sequences()`
- `compare_named_tensors()`

建议输出:

- 有效性检查结果
- abs / relative / squared error
- cosine / correlation
- allclose
- 可选的结构化误差,例如 per-channel 或 per-token

### 3. 捕获与对齐函数

目标: 让比较发生在"对应的位置".

建议能力:

- `collect_module_outputs()`
- `collect_parameter_snapshots()`
- `collect_gradient_snapshots()`
- `match_module_names()`
- `align_tensor_shapes()`
- `flatten_for_compare()`

要解决的问题:

- 同名模块才能比.
- tuple,list,dict 输出需要统一展开策略.
- 形状不同时,是报错,裁剪,还是跳过,要显式配置.

### 4. 聚合与排序函数

目标: 把大量逐层记录整理成表格和排行榜.

建议能力:

- `rank_layer_errors()`
- `group_error_records()`
- `aggregate_by_module_type()`
- `aggregate_by_stage()`
- `filter_error_records()`

常见排序键:

- max abs
- mean abs
- cosine similarity 升序
- relative error
- task impact score

### 5. 轨迹与时序函数

目标: 支持训练过程和多步生成过程的误差曲线.

建议能力:

- `track_metric_series()`
- `compare_stepwise_tensors()`
- `rolling_summary()`
- `detect_distribution_shift()`

适合:

- loss / grad 曲线.
- diffusion latent 逐步漂移.
- online drift 监控.

### 6. 导出与可视化适配函数

目标: 给 CSV,DataFrame,前端面板或 notebook 直接喂数据.

建议能力:

- `records_to_dicts()`
- `records_to_dataframe()`
- `export_error_report()`
- `build_heatmap_matrix()`
- `build_histogram_bins()`

原则:

- 计算函数与渲染函数分开.
- 先定义稳定数据 schema,再接 Matplotlib,TensorBoard,W&B 或 HTML.

## 建议的中性数据结构

下面是比具体实现更重要的一层. 先把记录长什么样定清楚,可视化才能稳定复用.

### 1. TensorSummary

建议字段:

- `shape`
- `dtype`
- `device`
- `numel`
- `mean`
- `std`
- `min`
- `max`
- `quantiles`
- `zero_ratio`
- `nan_count`
- `inf_count`

### 2. TensorDiff

建议字段:

- `max_abs`
- `mean_abs`
- `mean_squared`
- `relative_error`
- `cosine_similarity`
- `correlation`
- `allclose`
- `atol`
- `rtol`
- `valid`
- `message`

### 3. LayerErrorRecord

建议字段:

- `name`
- `module_type`
- `stage`
- `reference_summary`
- `candidate_summary`
- `diff`
- `parameter_count`
- `tags`

### 4. ErrorReport

建议字段:

- `scenario`
- `source_a`
- `source_b`
- `records`
- `aggregates`
- `metadata`

## 分析顺序建议

误差分析不要一上来就画很多图. 更有效的顺序通常是:

1. 先做有效性检查,排除 shape,dtype,NaN,Inf 等硬错误.
2. 再做整体摘要,确认是幅值问题,分布问题,还是离散决策问题.
3. 再做逐层排序,定位最敏感的层或步骤.
4. 最后再做结构化可视化,例如 heatmap,per-token 图或时序曲线.

## 当前仓库的启发,但不是归属承诺

当前代码已经说明了几种真实需求:

- `xqt/eval/compare.py` 适合沉淀基础 tensor diff.
- `xqt/quant/sensitivity.py` 说明逐层输出对比是刚需.
- `xqt/quant/calibration.py` 说明激活分布统计是刚需.
- `xdl/callbacks/layer_monitor.py` 说明训练期权重 / 梯度统计是刚需.

但从职责上看,这些需求已经超出压缩实验本身. 后续如果要沉淀公共函数,应优先保持中性边界:

- 纯函数和轻量数据结构,可以放到通用基础层.
- 面向训练生命周期的在线统计,可以放到 callback 或 observer 层.
- 面向压缩策略的特化分析,再由 `xqt/` 在其上复用.

## 第一批最值得做的通用函数

如果后续开始实现,建议先做最小闭环:

1. `finite_check()`
2. `summarize_tensor()`
3. `compare_tensors()`
4. `collect_module_outputs()`
5. `compare_named_tensors()`
6. `rank_layer_errors()`
7. `records_to_dataframe()`

这 7 个函数已经足够支撑:

- 逐层误差表.
- 激活分布表.
- 热图和直方图输入.
- 后端一致性验证.
- 压缩前后敏感层排序.

## 暂定结论

- 误差分析不是 `xqt/` 私有需求,而是横跨训练,压缩,导出,部署和可视化的通用能力.
- 真正需要先统一的是"比较对象,误差类型,记录结构和聚合方式",不是某个具体前端图表.
- 一旦这些基础数据函数稳定下来,上层无论是 `xqt/`,训练 callback,notebook 还是 HTML 阅读页,都可以复用同一套结果.
