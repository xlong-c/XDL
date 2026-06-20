# XQT 结构化剪枝需求文档

## 负责内容

- 定义 `xqt.prune` 后续扩展时,结构化剪枝的分类,工程目标和模块边界.
- 把“结构化剪枝有哪些分类”翻译成 XQT 需要提供的具体功能.
- 为后续实现 `structured.py`, `rewrite.py`, `pipeline prune pass`, recipe 和报告格式提供需求基线.

## 不负责内容

- 不把本文中的需求直接视为“当前已实现事实”.
- 不替代 [XQT.md](XQT.md) 的整体架构草案.
- 不重复解释非结构化剪枝,量化,蒸馏和扩散少步蒸馏的完整规划.
- 不承诺首期一次支持所有模型族,所有硬件稀疏格式和所有论文路线.

## 背景

当前 `xqt.prune` 已经具备三类基础能力:

- 非结构化 baseline: `global_l1_unstructured`, sparsity report, prune schedule, prune + KD smoke 流程.
- 结构改写 helper: `Conv2d`, `Linear`, `BatchNorm` 的基础维度裁剪与重写.
- 分析侧入口: importance 统计和 `rank_prune_candidates()`.

截至当前代码状态, 已额外具备:

- 结构化 plan / report / target 枚举
- CNN channel/filter pruning 闭环
- ResNet stage 输出宽度的 residual add-aware channel pruning 闭环,覆盖 BasicBlock / Bottleneck 的主分支,shortcut downsample 和下一 stage 输入联动
- 一般 grouped `Conv2d` 的 group-aligned channel 选择,输入通道传播和 keep-index 校验
- shape-compatible CNN stage pruning,支持输入/输出通道一致且不改变空间尺寸的 `ModuleList` / `Sequential` stage 容器按 `keep_indices` 删除
- ViT/Transformer 的 `mlp_neuron`,gated `gate_proj/up_proj/down_proj`,`head`,`block` 结构化剪枝闭环
- XDL `VisionTransformer` 的 `hidden_width` / `embedding_width` 全局 residual width 缩减最小闭环
- toy MoE `expert` pruning,支持 `router + experts` 结构的 usage-aware expert 删除和 `topology_changes` 报告
- `nm_structured` 规则化稀疏报告与 runner 分支
- `block_sparse` 固定二维块稀疏 mask 生成,报告与 runner 分支
- `cnn_structured_prune` / `vit_structured_prune` recipe
- preflight / benchmark 对 N:M backend capability 的显式标记

见:

- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:409)
- [xqt/prune/rewrite.py](/root/workspace/xdl/xqt/prune/rewrite.py:25)
- [xqt/prune/importance.py](/root/workspace/xdl/xqt/prune/importance.py:146)
- [docs/md/XQT.md](XQT.md)

当前剩余缺口不再是“有没有结构化剪枝入口”,而主要是更复杂结构和更强后端闭环. 否则仍然容易出现三个问题:

- 把“zero 变多”误当成“模型真的变小或变快”.
- 把 CNN 的 channel 剪枝和 ViT/Transformer 的 head/block 剪枝混为一种实现.
- 只有局部 helper,没有 recipe/report/export 验证闭环,最终无法验收真实部署收益.

## 结构化剪枝的分类

这里按“被裁掉的结构对象”分类,这是最适合工程落地的划分方式.

### 1. 宽度剪枝

宽度剪枝裁掉的是每层内部的并行维度,典型对象包括:

- CNN: output channels, input channels, filters.
- Transformer/ViT: attention heads, head_dim, MLP hidden neurons, embedding width 的一部分.

对应功能:

- 真实减少参数量.
- 真实减少 FLOPs.
- 在导出图中真实减少张量维度,不是只把权重置零.
- 对部分后端带来真实 latency 收益.

典型子类:

#### 1.1 Channel pruning / Filter pruning

主要面向 `Conv2d`-heavy 模型.

压缩对象:

- `Conv2d.out_channels`
- `Conv2d.in_channels`
- 后继 `BatchNorm.num_features`
- 下游卷积或线性层的输入维度

对应工程功能:

- 通道重要性评分
- 按层或全局决定保留哪些 channels
- 联动重写后继 `Conv2d`, `BatchNorm`, `Linear`
- 输出每层裁剪前后 shape

适用场景:

- ResNet, UNet encoder/decoder, Conv-heavy detection backbones

#### 1.2 MLP neuron pruning

主要面向 Transformer/ViT 的 FFN/MLP.

压缩对象:

- `Linear.out_features`
- 下游 `Linear.in_features`

对应工程功能:

- 评估 FFN hidden neurons 重要性
- 裁掉 MLP 中间扩展维度
- 保持 residual 主干维度不变,只改 block 内部宽度

适用场景:

- ViT, BERT, GPT 类 block 内 FFN

#### 1.3 Attention head pruning

主要面向 Multi-Head Attention.

压缩对象:

- head 个数
- 可选的 q/k/v projection 中对应 head 分块

对应工程功能:

- 评估每个 attention head 的重要性
- 保留 head 子集
- 重写 qkv/proj 权重和 `num_heads`
- 维护 `embed_dim % num_heads == 0` 等结构约束

适用场景:

- ViT, BERT, GPT, 多头跨注意力模块

注意:

- 单纯把某个 head 权重置零不算结构化完成.
- 真正的结构化 head pruning 需要修改模块元数据和权重 shape.

#### 1.4 Embedding width / hidden width pruning

主要面向更激进的 Transformer/LLM/ViT 压缩.

压缩对象:

- token embedding width
- hidden size
- block 主干宽度

对应工程功能:

- 联动所有依赖 hidden size 的线性层,层归一化和残差路径
- 提供更全局的 rewrite 计划
- 当前 XDL 最小闭环支持 XDL `VisionTransformer`: 同步缩减 patch embedding 输出,`cls_token`,`pos_embed`,所有 block 的 LayerNorm,attention `qkv/proj`,MLP 输入/输出和分类 head 输入.

适用场景:

- 更像“窄 student 重写”而不是局部 helper
- 当前已验证 XDL ViT forward 和一次优化器训练步,尚不等于任意 HF Transformer / LLM 的自动 student rewrite.

实现难度:

- 明显高于 channel/head/MLP neuron 剪枝

### 2. 深度剪枝

深度剪枝裁掉的是重复堆叠的层或 block.

典型对象:

- CNN stage/block
- Transformer encoder block
- Decoder block
- Diffusion UNet/DiT 中可裁减的重复块

对应功能:

- 真实缩短前向路径
- latency 收益通常比稀疏更直接
- 需要处理残差,skip connection,stage 边界和位置编码兼容

典型子类:

#### 2.1 Layer pruning / Block pruning

压缩对象:

- `ModuleList` / `Sequential` 中的整层或整块

对应工程功能:

- 给 block 打分
- 选择保留层索引
- 重建新的 block 列表
- 保持 forward 路径和 state_dict 可解释

适用场景:

- ViT blocks
- Transformer encoder layers
- CNN residual blocks

#### 2.2 Stage pruning

压缩对象:

- 更粗粒度的 stage 或 branch

对应工程功能:

- 重建更大范围的 topology
- 更新下游 head / neck 的输入通道和分辨率假设

适用场景:

- FPN backbone
- 多 stage CNN

实现难度:

- 高于 block pruning

### 3. 模式化结构剪枝

这类路线仍然是结构化,但约束来自硬件或 kernel 格式.

#### 3.1 N:M structured sparsity

压缩对象:

- 固定局部块内保留 `N` 个权重,其余置零,例如 `2:4`

对应功能:

- 生成满足 N:M 约束的 mask
- 报告格式兼容性
- 明确后端支持矩阵

适用场景:

- NVIDIA Ampere+ 稀疏 Tensor Core 路径

注意:

- 这类不是“shape 缩小”,而是“规则化稀疏”.
- 需要和普通非结构化剪枝分开建模.

#### 3.2 Block sparsity

压缩对象:

- 固定 block 大小的权重块

对应功能:

- block 级 mask
- block 稀疏率报告
- 后端可执行性检查

适用场景:

- 大矩阵,部分加速器

### 4. 路由与专家剪枝

主要面向 MoE 或条件计算模型.

压缩对象:

- experts
- routing branches
- selective modules

对应功能:

- expert usage 统计
- 删除低使用专家
- 更新 router 输出维度与映射

适用场景:

- MoE Transformer

当前优先级:

- 不建议作为 XQT 首期结构化剪枝主线

## 哪些结构可以剪枝,对应什么剪枝,应该怎么做

这一节按“具体结构对象”来写. 目标是让后续拆任务时可以直接回答三个问题:

- 这个结构能不能剪?
- 对应哪一种结构化剪枝?
- 在工程上具体该怎么做?

### 1. Conv2d 输出通道

对应剪枝:

- channel pruning
- filter pruning

为什么能剪:

- `Conv2d.out_channels` 决定了该层输出特征图宽度.
- 裁掉部分输出通道后,只要同步更新消费这些通道的后继模块,前向图仍然成立.

应该怎么做:

1. 计算每个输出通道的重要性.
2. 生成 `keep_indices`.
3. 重写当前 `Conv2d.weight` 和 `bias`.
4. 同步重写后继 `BatchNorm.num_features`.
5. 同步重写下游 `Conv2d.in_channels` 或 `Linear.in_features`.
6. 记录 pre/post shape, 参数量和 FLOPs 变化.

至少要验证:

- 剪枝后模型不依赖 mask.
- 前向输出 shape 合法.
- 下游模块输入维度匹配.

### 2. Conv2d 输入通道

对应剪枝:

- input-channel pruning
- 通常作为上游输出通道剪枝的联动结果

为什么能剪:

- 当前卷积的输入通道由上游特征决定.
- 当上游输出通道减少时,这里必须同步减少.

应该怎么做:

1. 不单独把它当成第一裁剪目标,优先把它视为依赖传播结果.
2. 根据上游 `keep_indices` 重写当前 `Conv2d.weight[:, keep_indices, ...]`.
3. 校验 groups/depthwise 约束是否仍成立.

至少要验证:

- `groups=1` 情况必须稳定支持.
- depthwise/group conv 需要单独约束和错误提示.

### 3. BatchNorm 通道

对应剪枝:

- BN-assisted channel pruning
- 依赖式结构改写

为什么能剪:

- `BatchNorm` 的 `weight`, `bias`, `running_mean`, `running_var` 都和 channel 一一对应.

应该怎么做:

1. 若使用 `bn_gamma` 评分,先基于 `gamma` 排序得到保留通道.
2. 按相同索引裁剪 `weight`, `bias`, `running_mean`, `running_var`.
3. 更新 `num_features`.

至少要验证:

- 运行统计量和 affine 参数都同步裁剪.
- 与前后 Conv2d 的 channel 数一致.

### 4. Linear 输出维度

对应剪枝:

- MLP neuron pruning
- classifier width pruning
- projection width pruning

为什么能剪:

- `Linear.out_features` 是一组并行输出单元.
- 对 MLP 中间层来说,删掉部分输出单元后只需同步更新下游输入维度.

应该怎么做:

1. 对输出维度打分.
2. 选择保留的 neuron 索引.
3. 重写当前 `Linear.weight[keep_indices, :]` 和 `bias[keep_indices]`.
4. 同步重写后继 `Linear.in_features`.

至少要验证:

- MLP 中间层可以稳定裁剪.
- 最终任务 head 默认应谨慎处理,不要误裁掉类别维度.

### 5. Linear 输入维度

对应剪枝:

- 联动式 width pruning

为什么能剪:

- 它不是独立压缩目标,更常见是跟随上游输出维度同步收缩.

应该怎么做:

1. 把它当成 rewrite 的依赖项而非首要评分对象.
2. 依据上游 `keep_indices` 重写 `weight[:, keep_indices]`.
3. 校验下游输出语义未被破坏.

至少要验证:

- 对分类 head, detection head, language head 要区分“可裁剪输入”和“不可裁剪输出语义”.

### 6. Attention heads

对应剪枝:

- attention head pruning

为什么能剪:

- 多头注意力天然由多个并行 head 组成.
- 某些 head 可以整体删除,前提是同步修改 qkv/proj 结构和 `num_heads`.

应该怎么做:

1. 先定义 head 粒度的 score.
2. 为每层 attention 生成 `keep_head_indices`.
3. 把 q/k/v projection 按 head 分块重排后裁剪.
4. 重写 output projection 的输入维度.
5. 更新 `num_heads`, 必要时更新 `embed_dim/head_dim` 元数据.
6. 校验 reshape / view / transpose 逻辑仍然成立.

至少要验证:

- `embed_dim % num_heads == 0`
- forward 中所有和 head 相关的 reshape 都能通过
- 剪枝后 attention 输出与残差维度兼容

注意:

- 如果实现里 `embed_dim` 固定,而只减少 head 个数,需要明确是“减少 heads”还是“减少每层总 attention width”.
- 这部分通常不能只靠通用 `Linear` helper 自动完成,需要 attention-aware rewrite.

### 7. MLP hidden neurons

对应剪枝:

- FFN / MLP neuron pruning

为什么能剪:

- Transformer block 的 FFN 一般是 `fc1 -> act -> fc2`.
- 中间扩展维度是天然可裁剪的并行单元集合.

应该怎么做:

1. 对 `fc1.out_features` 或中间激活通道打分.
2. 选择保留的 neuron.
3. 裁剪 `fc1.out_features`.
4. 同步裁剪 `fc2.in_features`.
5. 保持 block 主残差维度不变.

至少要验证:

- MLP 内部宽度变小,但 block 输入输出 hidden size 不变.
- 剪枝前后 block 级 forward 正常.

### 8. Transformer / ViT blocks

对应剪枝:

- block pruning
- layer pruning

为什么能剪:

- 编码器通常由重复 block 堆叠构成.
- 删除整块可以直接缩短推理路径.

应该怎么做:

1. 对 block 粒度打分.
2. 生成 `keep_block_indices`.
3. 重建 `ModuleList` 或 `Sequential`.
4. 更新和 block 数相关的元数据.
5. 若有 layer-wise cache / hook / checkpoint path,同步更新命名映射.

至少要验证:

- block 删掉后 forward 顺序仍正确.
- state_dict 可加载策略明确.
- benchmark 确实反映层数变少后的 latency 下降.

### 9. CNN residual blocks / stages

对应剪枝:

- residual block pruning
- stage pruning

为什么能剪:

- CNN 的重复 residual block 或 stage 也是天然的深度结构.
- 当前已支持一种保守 stage pruning: `ModuleList` / `Sequential` stage 容器中每个 stage 输入/输出通道一致,且内部卷积不改变空间尺寸时,可以按 `keep_indices` 删除 stage 并重建容器.

应该怎么做:

1. 评估 block 或 stage 重要性.
2. 只在结构边界清晰的位置裁剪.
3. 对 shape-compatible stage,直接重建容器并保留下游输入规格.
4. 对会改变通道数或分辨率的 stage,更新 shortcut / downsample / stage transition.
5. 若 neck/head 依赖 stage 输出,同步更新其输入规格.

至少要验证:

- shape-compatible 多 stage CNN forward 正常.
- stage 删除后 benchmark 入口能跑通.
- 不破坏多尺度输出契约.
- detection / segmentation 模型的 neck 输入仍匹配.

### 10. Embedding width / hidden size

对应剪枝:

- hidden width pruning
- embedding width pruning

为什么能剪:

- 从理论上可剪,但这是全局主干维度重写,难度高.

应该怎么做:

1. 明确这是“全局窄化”而不是局部 helper.
2. 统一收集所有依赖 hidden size 的模块:
   - embedding
   - qkv/proj
   - MLP
   - layer norm
   - residual add
   - classifier head
3. 生成全局 rewrite plan.
4. 重建整个 block stack.

至少要验证:

- 所有 hidden size 相关模块已同步改写.
- 导出和 state_dict 兼容策略清楚.

当前建议:

- 这类先放 P2,不要作为首批结构化剪枝主线.

### 11. Experts / MoE branches

对应剪枝:

- expert pruning
- router-aware structured pruning

为什么能剪:

- 专家之间通常是并列结构,低使用率专家可以整体移除.
- 当前已支持 toy MoE 最小闭环: 模块暴露 `router: nn.Linear` 和 `experts: ModuleList/Sequential`,usage 可由 `expert_usage`,`expert_usage_counts`,`router_usage` 或 router 参数推导.

应该怎么做:

1. 统计每个 expert 的 usage / load / contribution.
2. 生成保留专家集合.
3. 重写 expert 列表和 router 输出映射.
4. 更新 capacity 或 top-k routing 相关配置.
5. 在 `topology_changes` 中记录 kept/pruned experts 和 usage 依据.

至少要验证:

- router 输出索引与 expert 列表一致.
- 推理路径不访问已删除 expert.
- 报告可明确列出被删除 expert,保留 expert 和 usage 依据.

当前建议:

- 不作为 XQT 首批范围.

### 12. 权重块或 N:M 模式

对应剪枝:

- block sparsity
- N:M structured sparsity

为什么能剪:

- 这类结构满足特定硬件或 kernel 约束.

应该怎么做:

1. 明确 block 大小或 N:M 规则.
2. 对局部块打分并生成规则化 mask.
3. 保留原 shape,但记录格式约束满足情况.
4. 报告“后端是否真的支持该格式”.

至少要验证:

- 规则满足率
- 后端 capability
- benchmark 是否真的收益

注意:

- 这类严格说更接近“规则化稀疏”,不是 shape 缩小.
- 需要和 channel/head/block pruning 的报告分开.

## 按结构给出“应该怎么做”的通用流程

无论剪哪种结构,工程流程都建议统一为:

1. `结构识别`
   - 识别当前模型里哪些模块属于可剪结构.
2. `重要性评估`
   - 以 channel/head/block/neuron 为粒度生成 score.
3. `选择策略`
   - ratio, threshold, top-k 或显式 keep indices.
4. `生成 plan`
   - 不直接改模型,先生成可审查的 `PruningPlan`.
5. `执行 rewrite`
   - 依据 plan 重写模块和依赖链.
6. `结构校验`
   - shape, 元数据, forward 路径, 参数量/FLOPs.
7. `行为校验`
   - output diff, 任务 metric, benchmark.
8. `产物落盘`
   - 报告, manifest, recipe snapshot.

## 每类结构建议的首选实现策略

| 结构 | 对应剪枝 | 首选实现策略 |
| --- | --- | --- |
| Conv2d out_channels | channel/filter pruning | importance + keep indices + Conv/BN/downstream rewrite |
| Conv2d in_channels | 联动式通道剪枝 | 跟随上游 keep indices 自动传播 |
| BatchNorm channels | BN-aware channel pruning | 同步裁剪 affine 和 running stats |
| Linear out_features | neuron pruning | 裁剪当前输出并同步下游输入 |
| Linear in_features | 联动式 width pruning | 作为上游输出裁剪的依赖传播 |
| Attention heads | head pruning | attention-aware qkv/proj rewrite |
| MLP hidden neurons | FFN pruning | `fc1/fc2` 成对改写 |
| Transformer blocks | block pruning | 重建 `ModuleList` 并更新元数据 |
| CNN stages | stage pruning | shape-compatible stage 容器重建,跨尺度场景再做 neck/head 联动 |
| Embedding / hidden size | width pruning | XDL ViT 全局 residual width rewrite,通用 auto student 后期实现 |
| Experts / branches | expert pruning | usage-aware branch removal + `topology_changes` report |
| 权重块 / N:M | block sparse / N:M sparse | 规则化 mask + backend-aware report |

## 按模型族看,结构化剪枝对应什么功能

### CNN / Conv-heavy 模型

核心分类:

- channel pruning
- filter pruning
- stage/block pruning

XQT 需要的功能:

- Conv-BN-Conv / Conv-BN-Linear 依赖链重写
- channel keep index 传播
- 剪枝后参数量/FLOPs 统计
- 导出后 ONNX shape 验证

### ViT / Transformer

核心分类:

- attention head pruning
- MLP neuron pruning
- block pruning
- 可选 hidden width pruning

XQT 需要的功能:

- block 内 qkv/proj/MLP 联动重写
- `num_heads`, `head_dim`, `hidden_features` 一致性检查
- block 列表裁剪和位置保持
- 剪枝前后输出 diff 与任务指标对比

### LLM

核心分类:

- head pruning
- MLP width pruning
- layer pruning
- N:M 或 block structured sparsity

XQT 需要的功能:

- 更大模型的 calibration/importance 采样接口
- 更关注一次性后训练结构裁剪
- 更强的 export/backend 限制表达

### Detection / Segmentation / Diffusion

核心分类:

- backbone channel pruning
- neck/head channel pruning
- 可选 block pruning

XQT 需要的功能:

- 多分支 shape 对齐
- neck/head 输入输出通道协同更新
- 任务指标不再只是 top1,而是 mAP / IoU / 生成质量

## XQT 中结构化剪枝应提供的能力清单

### 1. 剪枝对象描述

需要统一表达“要剪什么结构”.

建议增加:

- `prune.granularity`: `channel`, `filter`, `head`, `mlp_neuron`, `block`, `stage`, `hidden_width`, `embedding_width`, `expert`, `nm`, `block_sparse`
- `prune.scope`: `global`, `per_layer`, `per_stage`, `custom`
- `prune.selection`: `ratio`, `keep_indices`, `score_threshold`

作用:

- 避免把所有策略都塞进 `method` 一个字符串里.

### 2. 重要性评估

需要统一表达“为什么保留这些结构”.

首批建议支持:

- `l1`
- `l2`
- `bn_gamma`
- `activation_magnitude`
- `gradient_weight`
- `taylor_first_order`

对应功能:

- per-layer / per-structure score table
- 导出排序结果
- 支持和 sensitivity 联合排序

### 3. 结构改写计划

需要把“算出该删谁”与“如何安全改模型”分开.

建议增加:

- `PruningPlan`
- `LayerPruningAction`
- `ModuleRewritePlan`

对应功能:

- 记录每层保留索引
- 记录哪些后继模块需要同步改写
- 支持 dry-run 预览

### 4. 重写执行器

需要把 rewrite helper 从单层函数提升为模型级执行器.

对应功能:

- 根据 plan 批量重写模块
- 更新父模块引用
- 校验 shape 一致性
- 输出 rewrite summary

### 5. 剪枝后验证

结构化剪枝不能只看 sparsity.

至少要验证:

- 参数量变化
- 每层 shape 变化
- FLOPs 变化
- 前向可运行
- 输出 diff
- 任务指标
- 导出可行性

### 6. 剪枝报告

结构化剪枝报告应区别于非结构化剪枝报告.

至少包含:

- pruning granularity
- target modules
- keep/remove indices
- pre/post shapes
- params before/after
- FLOPs before/after
- model output diff
- task metric delta
- export validation result

## 配置需求

当前 `compression.prune` 只有:

- `enabled`
- `method`
- `target_sparsity`
- `schedule`
- `params`

这对结构化剪枝不够.

建议中期扩展为:

```yaml
compression:
  prune:
    enabled: true
    method: structured
    granularity: head
    scope: per_layer
    target_sparsity: 0.25
    importance:
      type: l1
      normalize: true
    rewrite:
      dry_run: false
      validate_shapes: true
    params:
      module_name_patterns: [blocks.*.attn]
      exclude_module_names: [head]
```

对 block pruning 建议支持:

```yaml
compression:
  prune:
    enabled: true
    method: structured
    granularity: block
    selection:
      keep_indices: [0, 1, 3, 5, 7, 9]
```

对 channel pruning 建议支持:

```yaml
compression:
  prune:
    enabled: true
    method: structured
    granularity: channel
    importance:
      type: bn_gamma
    params:
      module_name_patterns: [layer1.*, layer2.*]
```

## 与当前模块的映射

### 当前已有

- `masks.py`: 非结构化剪枝和稀疏率统计
- `importance.py`: importance 统计和候选排序
- `rewrite.py`: `Linear` / `Conv2d` / `BatchNorm` 的局部裁剪 helper
- `schedule.py`: prune + KD 调度

### 建议新增

- `structured.py`
  - 结构化 plan 生成
  - head / block / channel 的执行入口
- `cost.py`
  - params / FLOPs 估算
- `report.py`
  - 结构化剪枝专用报告
- `graph.py`
  - 可选的依赖传播和模块关联分析

## 分阶段需求

### P0 - 先把 CNN 结构化剪枝闭环做完整

目标:

- `channel/filter pruning` 真正可用

需求:

- `Conv2d + BatchNorm + downstream Conv/Linear` 联动改写
- params/FLOPs 报告
- ONNX 导出 shape 验证
- recipe: `cnn_structured_prune`

验收:

- 剪枝后模型不依赖 mask
- ONNX 图真实减少 channel/filter

### P1 - 打通 ViT / Transformer 结构化剪枝

目标:

- `head pruning + MLP neuron pruning + block pruning`

需求:

- attention block rewrite
- MLP width rewrite
- block keep/drop 计划
- 剪枝后 forward + metric + benchmark

验收:

- 至少一个 ViT recipe
- 至少一种 `head` 或 `block` 粒度真实改写

### P2 - 增加硬件约束型结构稀疏

目标:

- `N:M structured sparsity`

需求:

- mask 约束
- backend capability 标记
- 额外 benchmark 字段

验收:

- 明确区分“格式满足”与“后端真的加速”

## 非目标

当前不建议在首批混入:

- 动态稀疏训练
- 专门为每篇论文单独实现一套抽象
- 一上来就做 hidden size 全局重写
- 未验证后端支持的稀疏格式泛化承诺

## 对应源码改动建议

如果按本文推进,优先涉及:

- `xqt/core/schema.py`
- `xqt/core/config.py`
- `xqt/prune/structured.py` 新增
- `xqt/prune/rewrite.py`
- `xqt/prune/importance.py`
- `xqt/pipeline/passes.py`
- `xqt/recipes/`
- `tests/xqt/test_prune.py`
- `tests/xqt/test_runner.py`

## 参考资料

下面这些参考资料适合指导分类和能力边界,但不应直接照搬实现接口:

- Network Slimming, ICCV 2017: <https://arxiv.org/abs/1708.06519>
- Structured Pruning 相关综述: <https://arxiv.org/abs/2402.00559>
- Vision Transformer Pruning: <https://arxiv.org/abs/2104.08500>
