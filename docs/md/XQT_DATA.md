# XQT 数据需求与边界

## 负责内容

- 定义 `xqt/` 中各类数据分片的角色,边界和消费方式.
- 约束蒸馏数据集,校准数据集,验证数据和 prompt 数据在 recipe 中的职责分工.
- 说明当前已实现的数据消费点,以及后续扩展时必须补齐的能力.

## 不负责内容

- 不替代 [XQT.md](XQT.md) 的整体架构草案和任务排期.
- 不重复 [DATASET.md](DATASET.md) 中 `xdl.dataset` 的通用模板选择和注册约束.
- 不把本文中的需求边界直接升级为已稳定 API 承诺.
- 不定义具体第三方数据集的下载脚本,版权策略或标注规范.

## 背景

`xqt` 的主线不是训练通用框架,而是压缩与部署工具链:

```text
PyTorch checkpoint
    -> baseline eval
    -> optional distill
    -> optional prune
    -> optional quant
    -> optional diffusion_distill
    -> export
    -> validate
    -> benchmark
```

这条链路会消费多种用途不同的数据:

- 蒸馏数据集: 给 teacher/student 训练或离线缓存用.
- 校准数据集: 给 PTQ/QDQ observer 或 calibration reader 用.
- 验证数据集: 给 baseline/optimized/exported artifact 做精度和 diff 验证用.
- prompt 数据: 给扩散少步蒸馏,生成预览和固定 seed 报告用.

如果这些数据只用一个 `data` 概念笼统表示,后续很容易把责任混在一起:

- 用验证集冒充校准集,导致量化策略和真实部署分布不匹配.
- 用训练集和蒸馏缓存混写,导致 recipe 无法复现.
- 把 prompt 列表当成普通 dataloader,导致扩散链路接口混乱.

因此,`xqt` 需要单独把“数据角色”这件事讲清楚.

## 当前真实现状

当前 schema 已定义四类数据字段,位置见 [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:45):

- `data.calibration`
- `data.train`
- `data.validation`
- `data.prompts`

当前内置 pass 已经能构建四类分片:

- `load_data` 会构建 `calibration`,`train`,`validation`,`prompts`,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:113).
- `distill` pass 消费 `train`,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:188).
- `baseline_eval`,`analyze`,`export`,`benchmark` 消费 `validation`,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:233),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:263),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:560),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:951).
- `quant` pass 中 `onnxruntime_qdq` 路径优先消费 `calibration`,没有时回退 `validation`; `torchao` 路径当前不消费校准集,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:440).
- `load_data` 当前会把 prompt 摘要写入 `context.metrics["prompts"]` 和 manifest metric `prompts.count`,作为最小消费闭环.

当前数据侧的真实边界是:

| 数据角色 | schema 已有 | 内置构建已通 | 内置 pass 已消费 | 当前说明 |
| --- | --- | --- | --- | --- |
| calibration | 是 | 是 | 是 | 主要服务 ONNX Runtime static QDQ |
| train | 是 | 是 | 是 | 主要服务 distill 和 prune finetune |
| validation | 是 | 是 | 是 | baseline/export/benchmark 主入口 |
| prompts | 是 | 是 | 是 | 当前已接入 prompt list / prompt file 构建,并有最小 metrics/manifest 消费 |

## 为什么要分角色

### 1. 蒸馏数据集不是校准数据集

蒸馏数据集需要支持:

- teacher/student 同步前向
- label 或 teacher logits/features
- 训练 shuffle
- 多 step 迭代
- 可选缓存命中

校准数据集只需要支持:

- 覆盖部署期输入分布
- 稳定抽样若干 batch
- 提供模型输入张量
- 不要求 label 一定存在

这两者混用会让量化和蒸馏都失去边界.

### 2. 校准数据集不是验证数据集

校准目标是“统计范围和分布”,验证目标是“衡量误差和任务指标”.

常见现实情况:

- 校准集可以小,但要分布代表性强.
- 验证集通常更大,用于统计 top1/F1/mAP/LPIPS 等指标.
- 某些部署场景下,校准集来自线上采样,验证集来自标准 benchmark.

因此 `xqt` 应允许两者独立配置,只在无法提供校准集时显式回退到验证集.

### 3. prompt 数据不是普通监督数据集

扩散少步蒸馏和生成报告的 prompt 数据可能包含:

- prompt 文本
- negative prompt
- seed
- guidance scale
- width/height
- scheduler 或 step 配置
- 条件图像/参考图像/latent cache key

这类对象不应被勉强塞进 image classification dataloader 语义.

## 四类核心数据角色

### 1. 蒸馏数据集

#### 负责内容

- 提供 teacher/student 训练或离线 distill 所需样本.
- 支持 logit KD,feature KD,relation KD 或 task loss 组合.
- 在需要时支持 teacher logits/features cache 的读写键空间.

#### 不负责内容

- 不承担 PTQ/QDQ 激活范围统计.
- 不默认作为最终精度验收集.
- 不负责导出时的 example input 签名推导,除非 recipe 明确复用.

#### 最小需求

- 可以构造成稳定 dataloader.
- batch 中必须能抽出 student 可消费的模型输入.
- 如需 task loss,应提供 label/target.
- 如需 cache,应提供稳定 sample identity 或可重复索引.

#### 典型形态

- image classification: `(image, label)`
- text classification: `{"input_ids","attention_mask","labels"}`
- teacher cache: `{"inputs": ..., "labels": ..., "cache_key": ...}`
- feature distill: 输入样本本身不变,但 recipe 需要额外层对齐配置

#### 当前现状

- 当前 `distill` pass 只要求 `context.data["train"]` 存在,并把它当普通 dataloader 使用,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:203).
- 当前还没有“蒸馏专用数据 schema”或 cache key 契约.

### 2. 校准数据集

#### 负责内容

- 为 PTQ/QDQ/static observer 提供代表性输入分布.
- 支持 sample limit 和稳定顺序迭代.
- 允许无标签输入.

#### 不负责内容

- 不负责最终任务指标验收.
- 不默认参与训练反向传播.
- 不要求覆盖全量数据集规模.

#### 最小需求

- 能提取模型输入张量,支持 Tensor / tuple/list / mapping 三类批次结构.
- 能限制 batch 数量而非必须遍历全量.
- 分布应尽量接近部署期真实输入.

#### 当前现状

- `onnxruntime_qdq` 路径会把 dataloader batch 转成 numpy 输入,见 [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py:27).
- `QuantPass` 中 `onnxruntime_qdq` 优先使用 `calibration`,缺失时回退 `validation`,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:451).
- `torchao` 当前不依赖校准集,所以配置里即使写了 `data.calibration`,对 torchao 路径也不会生效.

#### 需求约束

- 后续若增加 `torchao` 之外的 PTQ/QAT/PT2E observer 路径,校准数据必须继续独立于验证集存在.
- 回退到 `validation` 必须是显式兼容行为,不能把“未提供 calibration”当作最佳实践.

### 3. 验证数据集

#### 负责内容

- 提供 baseline 和 optimized model 的任务指标评估.
- 提供 export/runtime diff 的 example batch.
- 提供 benchmark 的真实输入形态.

#### 不负责内容

- 不负责 teacher/student 的训练驱动.
- 不负责专门的 calibration 统计,除非 recipe 明确降级复用.

#### 最小需求

- 至少能提供一个 batch 供 `analyze`,`export`,`benchmark` 取样.
- 如需任务指标,应提供 labels/targets 或任务特定 ground truth.
- 输入形态必须与导出/部署的模型签名一致.

#### 当前现状

- `baseline_eval`,`analyze`,`export`,`benchmark` 都依赖 `validation`,见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:245),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:282),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:585),[xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:973).
- 当前内置 exporter 和 ONNX QDQ auto-export 已支持 Tensor / tuple / mapping 三类 example input 提取,但更复杂的多模态条件 schema 仍未统一.

### 4. Prompt 数据

#### 负责内容

- 为扩散少步蒸馏,采样对比,固定 seed 报告和生成预览提供输入.
- 支持 prompt 文本及附加生成条件.
- 支持固定顺序和可复现采样配置.

#### 不负责内容

- 不替代普通监督训练 dataloader.
- 不强行套用 classification dataset 语义.
- 不要求每条 prompt 都绑定标准 label.

#### 典型形态

- `{"prompt": str, "negative_prompt": str, "seed": int}`
- `{"prompt": str, "guidance_scale": float, "steps": int}`
- `{"prompt": str, "image": Tensor/PIL, "mask": Tensor/PIL}`

#### 当前现状

- schema 已有 `data.prompts`,当前内置 `load_data` 已支持 `prompt_list` / `prompt_file` 构建,并会写入 prompt 摘要 metric,见 [xqt/core/schema.py](/root/workspace/xdl/xqt/core/schema.py:46) 和 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:149).
- 当前 `PromptBatch` / `PromptRecord` 已支持 text-only 和最小 image-conditioned schema: `condition_image`,`condition_mask`,`reference_image`,`latent_cache_key`,同时兼容 `image` / `mask` 作为输入别名,见 [xqt/data/prompts.py](/root/workspace/xdl/xqt/data/prompts.py:14) 和 [xqt/diffusion_distill/spec.py](/root/workspace/xdl/xqt/diffusion_distill/spec.py:42).
- 当前这些条件字段已经能进入 prompt summary 和 diffusion cache,但还没有扩展到真实的 diffusion distill sampler / trainer 消费.

## Recipe 层的边界要求

Recipe 中各类数据分片应按角色分开写,不要因为样本来源相同就合并成一个字段.

推荐结构:

```yaml
data:
  train:
    ...
  calibration:
    ...
  validation:
    ...
  prompts:
    ...
```

约束:

- `train` 只表达训练/蒸馏/微调数据.
- `calibration` 只表达量化校准输入.
- `validation` 只表达评估/导出 diff/benchmark 输入.
- `prompts` 只表达生成式输入集合.

允许复用同一底层数据源,但不能省略角色声明. 例如:

- 同一 CIFAR-100 val split 可同时配置为 `calibration` 和 `validation`,但两个分片仍应独立写出 `sample_limit`,`batch_size` 和用途.
- 同一 prompt 列表可同时用于 `diffusion_distill` 训练预览和最终报告,但其消费点应在 recipe 中明确.

## 当前内置数据构建能力的现实边界

当前 `LoadDataPass` 内置已支持五类 target:

- `synthetic_classification`
- `torchvision_image_classification`
- `hf_text_classification`
- `xdl_dataset`
- `prompt_list`
- `prompt_file`

见 [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:124) 和 [xqt/data/builders.py](/root/workspace/xdl/xqt/data/builders.py:1).

这意味着:

- image classification smoke recipe 已经有最小闭环.
- HF 文本分类和扩散 prompt 已接入统一内置 data pass.
- `xdl_dataset` bridge 已允许 `Record*` / `Image*` 等 `xdl.dataset` 通用模板通过 `xqt.data` 角色层接入内置 data pass.
- 多模态条件输入 schema 仍未统一.
- 后续新增数据角色时,优先扩展 `xqt.data` 的专用构建器,不要把角色逻辑散落进 `pipeline/passes.py`.

## 要实现的任务列表

下面的任务列表是把上面的需求边界转成可执行工作项. 它们仍然是规划任务,不是已完成事实.

### P0 - 先打通数据角色闭环

#### 任务 1: 接入 `data.prompts` 的构建和消费链路

状态: 已完成

目标:

- 让 `data.prompts` 不再只是 schema 占位字段.
- 为扩散少步蒸馏,采样预览和固定 seed 报告提供统一输入入口.

涉及位置:

- `xqt/core/schema.py`
- `xqt/pipeline/passes.py`
- `xqt/data/`
- `tests/xqt/`

最小实现:

- `load_data` 支持构建 `prompts` 分片.
- `xqt.data` 提供最小 prompt list / prompt file 构建器.
- pipeline 中至少有一个 pass 能显式消费 `context.data["prompts"]`.

验收标准:

- recipe 中配置 `data.prompts` 后,`context.data["prompts"]` 可用.
- prompt 样本顺序可复现.
- 缺失 `prompts` 时,非扩散 recipe 不受影响.

#### 任务 2: 为量化路径增加“缺少 calibration”的 preflight 提示

状态: 已完成

目标:

- 区分“兼容回退到 validation”和“推荐提供 calibration”的边界.
- 避免用户误以为只配 `validation` 就等价于完成量化数据准备.

涉及位置:

- `xqt/pipeline/preflight.py`
- `tests/xqt/test_preflight.py`
- 相关 recipe 文档

最小实现:

- 当 `compression.quant.enabled = true` 且量化后端需要校准数据时:
  - 若存在 `data.calibration`,报告正常.
  - 若仅存在 `data.validation`,给出通过但带 warning/提示的信息.
  - 若两者都不存在,继续报错.

验收标准:

- preflight 输出能区分三种状态: 已提供 calibration,回退 validation,完全缺失.
- 不改变当前 `onnxruntime_qdq` 真实运行时的兼容行为.

#### 任务 3: 为蒸馏数据增加最小 sample identity / cache key 契约

状态: 已完成

目标:

- 让 teacher logits/features cache 不再只依赖隐含顺序.
- 为后续缓存一致性校验打基础.

涉及位置:

- `xqt/distill/cache.py`
- `xqt/data/`
- `xqt/core/types` 或相关 dataclass
- `tests/xqt/`

最小实现:

- 约定蒸馏样本可选提供 `cache_key` 或等价稳定 identity.
- cache 写入和读取时优先使用 identity,而不是只按 batch 顺序拼接.

验收标准:

- 同一数据源重复运行时能命中同一 cache key.
- 样本顺序变化时,缓存不会静默错配到错误样本.

### P1 - 补角色化构建器和基础报告

#### 任务 4: 在 `xqt.data` 中增加角色化构建器

状态: 已完成第一阶段

目标:

- 把数据角色相关逻辑从 `pipeline/passes.py` 中拆回 `xqt.data`.
- 为后续多任务 recipe 提供统一数据入口.

建议首批覆盖:

- calibration iterable
- text classification train/validation
- prompt list / prompt file

涉及位置:

- `xqt/data/`
- `xqt/pipeline/passes.py`
- `xqt/README.md`
- `tests/xqt/`

验收标准:

- `LoadDataPass` 不再只硬编码两类 target.
- 新增数据 target 有最小测试覆盖和 README 说明.

#### 任务 5: 增加校准集采样报告

状态: 已完成

目标:

- 让量化前的数据准备有可见反馈,而不是只在量化时报错.
- 给 manifest 和调试报告提供校准样本概况.

涉及位置:

- `xqt/quant/onnx_qdq.py`
- `xqt/pipeline/passes.py`
- `xqt/core/artifact.py`
- `tests/xqt/`

最小实现:

- 记录 calibration batch 数,样本数和输入 shape 摘要.
- 将报告写入 metrics 或 manifest metadata.

验收标准:

- 成功量化后可以看到 `calibration_samples` 之外的更完整摘要.
- 摘要不依赖任务特定字段,至少适用于 Tensor / tuple / mapping 输入.

#### 任务 6: 为蒸馏缓存增加数据版本和 identity 校验

状态: 已完成

目标:

- 避免更换数据源后继续误用旧 cache.
- 让 cache 真正可复现而不是“碰巧能跑”.

涉及位置:

- `xqt/distill/cache.py`
- `xqt/core/artifact.py`
- `tests/xqt/`

最小实现:

- cache metadata 记录数据签名或样本 identity 摘要.
- 读取 cache 时校验当前数据配置和缓存元数据是否匹配.

验收标准:

- 数据源变化或 identity 不一致时,能给出明确错误或拒绝复用.
- 兼容当前不带完整数据版本信息的简单 smoke 路径.

### P2 - 扩展到更通用的任务输入

#### 任务 7: 支持多输入模型和更通用的 example input 提取

状态: 已完成

目标:

- 解除当前 export/quant 路径对“单 Tensor 输入”的偏强假设.
- 为文本,多模态和扩散条件输入铺路.

涉及位置:

- `xqt/pipeline/passes.py`
- `xqt/quant/onnx_qdq.py`
- `xqt/export/`
- `tests/xqt/`

验收标准:

- 至少支持 mapping 和 tuple 多输入的 example input 提取.
- ONNX QDQ 和 export 路径对多输入 batch 的错误信息清晰可诊断.

#### 任务 8: 支持从 `xdl.dataset` 通用模板桥接到 `xqt.data`

状态: 已完成第一版

目标:

- 减少 `xqt` 和 `xdl.dataset` 之间重复维护数据构建逻辑.
- 让 `record/manifest` 类通用数据能更自然进入 `xqt`.

涉及位置:

- `xqt/data/`
- `xqt/core/imports.py` 或配置构建链
- `docs/md/DATASET.md`
- `tests/xqt/`

验收标准:

- 至少有一类 `xdl.dataset` 通用模板可被 `xqt` recipe 复用.
- 角色边界仍保留在 `xqt.data` 这一层,不把 `train/calibration/validation/prompts` 语义泄漏回 `xdl.dataset`.

#### 任务 9: 继续扩展扩散和多模态 prompt/condition schema

状态: 已完成最小实现

目标:

- 在当前最小 text-only / image-conditioned schema 基础上,继续扩展到更完整的扩散和多模态条件表达.

涉及位置:

- `xqt/core/schema.py`
- `xqt/data/`
- `xqt/diffusion_distill/`
- `tests/xqt/`

验收标准:

- prompt schema 能覆盖 text-only 和 image-conditioned 两类最小场景,并为后续 sampler / trainer 消费保留稳定字段.
- 不把这套 schema 强行推广到 classification 或 distill train dataloader.

## 推荐执行顺序

建议按下面顺序实现:

1. 任务 2: preflight 提示
2. 任务 1: `data.prompts` 构建和消费
3. 任务 3: sample identity / cache key
4. 任务 4: 角色化构建器
5. 任务 5: 校准集采样报告
6. 任务 6: 蒸馏缓存校验
7. 任务 7-9: 多输入,`xdl.dataset` 桥接和扩散条件 schema

这样排的原因是:

- 先补提示和最小入口,能尽快减少误用.
- 再补 identity 和构建器,把后续功能建立在更稳的数据契约上.
- 最后再扩到多输入和多模态,避免在当前单输入主链路未收敛时过早扩面.

## 与其他文档的关系

- 整体压缩与部署工具链规划见 [XQT.md](XQT.md).
- 分析,诊断和优化建议能力见 [XQT_ANALYSIS.md](XQT_ANALYSIS.md).
- 导出前前置融合和 QDQ 前处理边界见 [XQT_PRE_EXPORT_FUSION.md](XQT_PRE_EXPORT_FUSION.md).
- 通用 dataset 模板和 `xdl.dataset` 注册规则见 [DATASET.md](DATASET.md).

## 暂定结论

- `xqt` 后续必须把“数据角色”而不是“数据来源”作为第一层边界.
- 蒸馏数据集,校准数据集,验证数据和 prompt 数据要长期并列存在,不能继续只靠一个模糊的 `data` 概念承载.
- 当前实现已经有 `train/calibration/validation/prompts` 的最小消费点.
- 后续扩展应优先在 `xqt.data` 建立角色化构建器和最小契约,再把新能力接入 `pipeline`.
