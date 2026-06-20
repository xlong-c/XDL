# XQT 复杂结构剪枝扩展需求文档

## 负责内容

- 定义 `xqt.prune` 在下一阶段扩展复杂结构化剪枝时的工程目标,对象边界和实现约束.
- 把“更复杂,更可扩展的剪枝”拆成可以落到 `structured.py`, `rewrite.py`, `pipeline`, `recipe` 和测试上的具体需求.
- 明确当前已支持范围和后续扩展范围,避免把局部 helper 误写成通用剪枝系统.

## 不负责内容

- 不把本文中的扩展项视为当前已实现事实.
- 不替代 [XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md) 的总需求说明.
- 不替代 [XQT_PRUNING_IMPLEMENTATION_PLAN.md](XQT_PRUNING_IMPLEMENTATION_PLAN.md) 的已完成能力和首轮实现拆解.
- 不展开量化,蒸馏,导出和扩散少步蒸馏的完整路线图.

## 当前实现基线

当前 `xqt.prune` 已有的结构化剪枝闭环是:

- CNN `channel/filter` pruning
- ResNet stage 输出宽度的 residual add-aware `channel` pruning,覆盖 BasicBlock / Bottleneck 的主分支,shortcut downsample 和下一 stage 输入联动
- 一般 grouped `Conv2d` 的 group-aligned `channel` pruning,覆盖自动 keep 选择,输入通道传播和 rewrite 校验
- shape-compatible CNN `stage` pruning,支持 `ModuleList` / `Sequential` 中输入/输出通道一致且不改变空间尺寸的 stage 删除
- ViT / Transformer `mlp_neuron` pruning
- Gated MLP / FFN `gate_proj/up_proj/down_proj` 中间宽度 pruning
- ViT / Transformer attention `head` pruning,包含 fused `qkv + proj` 和 split `q_proj/k_proj/v_proj/out_proj`
- `block` pruning
- XDL `VisionTransformer` 的 `hidden_width` / `embedding_width` 全局 residual width 缩减最小闭环
- toy MoE `expert` pruning,支持 `router + experts` 结构的 usage-aware expert 删除和 `topology_changes` 报告
- `nm_structured` 规则化稀疏
- `block_sparse` 固定二维块稀疏 mask 生成,层级报告和 runner 分支
- 结构化 plan/report 字段包含 `adapters`,`structure_families`,`blocked_modules`,`dependency_graph`,`export_status`,`benchmark_status`
- sparse capability 报告可区分 `pattern_present` 和 `speedup_verified`
- sparse benchmark recipe 可输出稀疏模式报告,capability 和 eager latency

关键代码位置:

- [xqt/prune/rewrite.py](/root/workspace/xdl/xqt/prune/rewrite.py:25)
- [xqt/prune/structured.py](/root/workspace/xdl/xqt/prune/structured.py:15)
- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py:440)

当前实现的真实边界也很明确:

- CNN 结构化剪枝支持三类路径: chain-like `Conv2d -> BatchNorm -> Conv2d/Linear` 顺序模型,ResNet 风格 stage 输出宽度剪枝,以及 shape-compatible stage 删除.
- residual 支持范围是 stage 级 output-channel pruning: `nn.Sequential` stage 内 child 必须同为 `BasicBlock` 或 `Bottleneck`,并且当前只自动枚举带 downsample 的 stage 边界.它会同步重写主分支输出,shortcut downsample 输出和下一 stage / head 输入.
- stage 删除支持范围是 `ModuleList` / `Sequential` 中每个 stage 输入/输出通道一致且内部 `Conv2d` stride=1,空间尺寸不变的 CNN stage.它会按 `selection.keep_indices` 重建容器,但不会自动处理会改变分辨率或通道数的 backbone stage,也不会自动重写 FPN/neck/head 的多尺度输入契约.
- residual 支持尚不等于任意 branch / merge 图支持: 已支持 toy concat branch 的 consumer input slice 重映射,但 FPN,UNet skip,任意 add 拓扑和跨尺度 stage 删除仍在后续 backlog.
- 一般 grouped convolution 目前支持 chain-like propagation 中的 group-aligned keep 选择和输入通道裁剪,并已支持 toy MBConv-like `pw expand -> dw -> pw project` 中间宽度联动剪枝,但更广的 depthwise-separable,MBConv 或 inverted bottleneck 真实变体仍未全面覆盖.
- attention head pruning 已支持 fused `qkv + proj`,split `q_proj/k_proj/v_proj/out_proj`,toy cross-attn `q_proj/k_proj/v_proj/out_proj` 和 toy GQA/MQA split projection attention,但尚未覆盖更广的 HF decoder attention 变体和更复杂的 grouped-query 实现差异.
- gated MLP pruning 支持 `gate_proj/up_proj/down_proj` 命名结构,并要求 gate/up 输出宽度与 down 输入宽度一致.其他 FFN 命名或更复杂专家路由仍需额外 adapter.
- hidden / embedding width pruning 当前只支持 XDL `VisionTransformer` 骨架,按同一组 keep indices 同步重写 patch embedding,`cls_token`,`pos_embed`,LayerNorm,attention,MLP 和 head,并要求保留宽度可被 `num_heads` 整除.这不是通用 HF Transformer 或任意 residual 主干的自动窄 student 生成器.
- expert pruning 当前支持 toy MoE 结构: 模块需要暴露 `router: nn.Linear` 和 `experts: ModuleList/Sequential`,usage 可来自 `expert_usage`,`expert_usage_counts`,`router_usage` 或 router bias/weight.剪枝会同步缩小 router 输出和 expert 容器,但不处理 shared expert,auxiliary loss,top-k capacity,expert parallel 和分布式 checkpoint.
- 结构化报告已新增 `topology_changes`,用于显式记录 branch/expert/block/stage 的 kept/pruned indices; expert 剪枝会额外记录 router,experts,kept/pruned experts 和 usage 依据.
- block pruning 当前只处理 `nn.ModuleList` 或 `nn.Sequential`,已支持由复合子模块组成的 heterogeneous block container 重建和 toy DiT 重复块 smoke,但还没有扩展到更广的 container adapter,层编号/cached metadata 校验和真实 diffusion UNet / DiT 结构语义.
- export / benchmark 状态只记录 XQT runner 中 export/benchmark pass 是否执行及其产物摘要,不替代真实部署后端的性能证明.
- sparse capability 目前只覆盖 `nm_structured` 和 `block_sparse`,并且默认不会把 pattern 存在误写成 speedup 已验证.
- sparse benchmark 目前是 PyTorch eager 延迟基线,用于把稀疏模式、capability 和 latency 放到同一份 recipe 里,不是后端 sparse kernel 的性能证明.

见:

- [xqt/prune/structured.py](/root/workspace/xdl/xqt/prune/structured.py:483)
- [xqt/prune/rewrite.py](/root/workspace/xdl/xqt/prune/rewrite.py:79)
- [xqt/prune/structured.py](/root/workspace/xdl/xqt/prune/structured.py:746)
- [xqt/prune/structured.py](/root/workspace/xdl/xqt/prune/structured.py:795)
- [tests/xqt/test_prune.py](/root/workspace/xdl/tests/xqt/test_prune.py:260)

这意味着当前系统更像是:

- 对一批清晰结构的剪枝闭环验证
- 对单链路拓扑和 ResNet stage residual 输出宽度的模型级 rewrite
- 对 ViT 风格模块的专项适配

而不是:

- 对任意 PyTorch 拓扑都可泛化的结构剪枝系统
- 对常见 CNN / Transformer 家族都自动适配的通用依赖传播框架

## 扩展目标

下一阶段复杂结构剪枝的目标,不是再增加几个局部 helper,而是把系统从“局部 rewrite”推进到“依赖感知的结构改写”.

需要同时满足四个目标:

1. 剪枝对象更复杂: 从单链路扩展到 residual, branch, grouped conv, generic attention, gated MLP.
2. 依赖传播更完整: 从 producer -> 单 consumer,扩展到多 consumer, merge op 和 stage 级联动.
3. 适配面更广: 从 toy CNN / ViT-like,扩展到常见 ResNet, MobileNet, timm / HF Transformer 变体.
4. 验证闭环更硬: 不只看参数量和 shape,还要看 forward,导出,benchmark 和剪枝后恢复训练.

## 为什么当前实现还不够

当前实现足以支撑:

- `channel/filter`
- `mlp_neuron`
- `head`
- `block`

这些剪枝粒度的最小可用闭环.

但在更真实的模型上,会立刻遇到下面几类问题:

- 残差相加两侧必须保持通道对齐,不能只剪主分支.
- concat / FPN / neck 多分支会把单一 `keep_indices` 传播问题变成多源多汇的 shape 约束问题.
- grouped conv 和 depthwise-separable block 的可剪单元不是普通 channel 独立集合.
- attention 并不总是 `qkv + proj` 这一个实现,还存在 split q/k/v, cross-attn, MQA, GQA.
- 很多 Transformer 的 FFN 不是简单 `fc1 -> act -> fc2`,而是 `gate_proj/up_proj/down_proj`.
- 更激进的 hidden width / embedding width 剪枝会触碰整个残差主干和 LayerNorm,本质上接近“自动生成窄 student”.

所以,接下来的扩展重点应该是“复杂拓扑和复杂模块的系统支持”,而不是继续在简单链路里堆更多粒度名词.

## 复杂结构扩展分类

## 1. Residual / Branch-aware CNN 剪枝

### 涉及结构

- ResNet BasicBlock / Bottleneck
- shortcut + main branch 相加
- downsample branch
- concat branch
- FPN / neck 多尺度分支

### 对应剪枝

- output-channel pruning
- residual block pruning
- stage pruning
- branch pruning

### 需要怎么做

- 从“按叶子模块顺序扫描”升级为“按图上的 producer / consumer / merge 关系建模”.
- 对 `add` 类 merge,要求参与相加的分支在 merge 点之前 shape 对齐.
- 对 `concat` 类 merge,允许每个分支独立剪枝,但要重写后继输入通道切片映射.
- 对带 `downsample` 的 residual block,主分支和 shortcut 分支的 keep plan 要协同生成.
- 对 stage pruning,要连同 stage 下游的 neck / head 输入通道假设一起重写.

### 最低验收

- ResNet basic block / bottleneck forward 正常.
- 剪枝后 shortcut 和主分支 shape 一致.
- ONNX 导出通过.
- 参数量,通道数和 latency 有真实变化.

## 2. Grouped / Depthwise / Separable Conv 剪枝

### 涉及结构

- grouped `Conv2d`
- depthwise + pointwise
- MobileNet inverted bottleneck
- ConvNeXt / MBConv 类 block

### 对应剪枝

- grouped channel pruning
- depthwise channel pruning
- pointwise width pruning
- stage-local structured width pruning

### 需要怎么做

- 对 grouped conv, keep indices 不能任意取,必须满足 group 对齐约束.
- 对 depthwise-separable block,需要联动 `pw expand -> dw -> pw project` 三段结构.
- 对 MBConv / inverted bottleneck,内部扩展维度和输出投影维度是不同层次的可剪对象,不能只按单层卷积处理.
- rewrite 时必须验证 `in_channels % groups == 0` 和 `out_channels % groups == 0`.

### 最低验收

- 一般 grouped conv 不再因为 helper 限制直接报错.
- MobileNet-like block 剪枝后 forward 正常.
- 导出图中 grouped / depthwise 权重 shape 正确缩小.

## 3. Generic Transformer Attention 剪枝

### 涉及结构

- fused `qkv + proj`
- split `q_proj / k_proj / v_proj / out_proj`
- self-attention
- cross-attention
- MQA
- GQA

### 对应剪枝

- attention head pruning
- kv-head pruning
- cross-attention head pruning
- decoder / encoder block attention pruning

### 需要怎么做

- 不再把 attention 识别写死为 `qkv + proj`.
- 引入 attention adapter / descriptor,先把模块归一化描述成:
  - q path
  - k path
  - v path
  - out projection
  - `num_heads`
  - `num_kv_heads`
  - `head_dim`
  - `embed_dim`
- head 选择要区分:
  - q/k/v 同步裁剪
  - 只裁 kv heads
  - self-attn 和 cross-attn 使用不同依赖规则
- 必须处理 MQA / GQA 里 `num_heads != num_kv_heads` 的情况.

### 最低验收

- timm ViT 风格和至少一种 HF split projection attention 都能走通.
- 剪枝后 attention 模块元数据正确更新.
- forward 和导出都正常.

## 4. Gated MLP / FFN 变体剪枝

### 涉及结构

- `fc1 -> act -> fc2`
- `gate_proj + up_proj -> elementwise gate -> down_proj`
- SwiGLU
- GEGLU

### 对应剪枝

- `mlp_neuron` pruning
- gated width pruning
- FFN intermediate width pruning

### 需要怎么做

- 当前 `fc1/fc2` 成对改写要扩展成“成组 linear 改写”.
- 对 gated MLP, gate 路径和 up 路径必须共享同一组中间神经元选择.
- 对 `down_proj`,需要联动裁剪输入维度.
- importance 统计不能只看单层 L1/L2,应支持组合得分.

### 最低验收

- 至少支持一种 gated FFN 结构.
- 剪枝前后中间宽度和下游输入维度保持一致.
- forward,导出和参数量统计正常.

## 5. Hidden Width / Embedding Width / Residual Width 剪枝

### 涉及结构

- token embedding width
- block 主干 hidden size
- residual path width
- classifier / projector 输入宽度

### 对应剪枝

- hidden width pruning
- embedding width pruning
- residual width pruning

### 需要怎么做

- 这不是单个 block 内局部缩宽,而是全局 shape 重写.
- 需要联动:
  - embedding
  - layernorm / batchnorm
  - q/k/v / proj
  - FFN
  - residual add 两侧
  - classifier / head
- 需要单独的 global width plan,不能复用当前局部 candidate 机制硬拼.
- 实际上这条路线通常要配合蒸馏,更接近“自动从 teacher 生成窄 student”.

### 最低验收

- 至少在单一 Transformer 骨架上支持固定 hidden width 缩减.
- 相关归一化层和 residual shape 全部一致.
- 剪枝后可继续做 KD 恢复训练.

## 6. Block / Stage 剪枝的复杂化扩展

### 当前边界

- block pruning 只处理 `nn.ModuleList` / `nn.Sequential`
- 只枚举由复合子模块组成的重复 block 容器
- 更像简单 container 重建

### 扩展方向

- CNN stage pruning
- heterogeneous block container pruning
- encoder / decoder 分层剪枝
- diffusion UNet / DiT 重复块剪枝

### 需要怎么做

- 对 block / stage 引入 container adapter,而不是只看 Python 容器类型.
- 支持 `drop_count`, `keep_indices`, `score_threshold` 三类选择模式.
- 对 Transformer block pruning,同步校验位置编码,cache,层编号等元数据.
- 对 CNN stage pruning,同步更新 neck / head 输入规格.

### 最低验收

- 不同 container 形态都能被统一枚举为 block target.
- 剪枝后容器元数据和 forward 路径可解释.

## 7. 稀疏模式扩展

### 当前边界

- 已有 `nm_structured`
- 已有 `block_sparse` 的固定二维块稀疏 mask 和报告闭环
- 还没有 runtime-aware 稀疏加速闭环

### 对应剪枝

- block sparse pruning
- semi-structured sparsity
- backend-aware sparse pruning

### 需要怎么做

- 区分“shape 缩小型结构化剪枝”和“稀疏模式型结构化剪枝”.
- 扩展 `block_sparse` 的后端能力声明.
- preflight / benchmark / export 需要知道某个 sparse pattern 是否真的可执行.
- 报告里要显式给出:
  - 稀疏模式
  - pattern 参数
  - backend capability
  - runtime speedup 是否已验证

### 最低验收

- `block_sparse` 至少能生成可复现报告.
- 对未支持 runtime 的环境,明确标记“有稀疏模式,无真实加速承诺”.

## 8. MoE / Expert / Router-aware 剪枝

### 涉及结构

- MoE experts
- router
- shared expert
- expert parallel 拓扑

### 对应剪枝

- expert pruning
- router-aware pruning
- low-usage branch pruning

### 需要怎么做

- 收集 expert usage,router probability,load balance 信息.
- 剪专家不是简单删除某个 `Linear`,还要维护 router 输出维度和 auxiliary loss 逻辑.
- 需要考虑分布式 expert parallel 的 checkpoint 和导出兼容.

### 最低验收

- 至少支持 usage-aware expert selection plan.
- 剪枝后 router 输出和 expert 列表一致.
- 报告明确列出 kept/pruned experts 和 usage 依据.

### 优先级判断

- 当前只完成 toy `router + experts` 结构的 P3 最小闭环,真实 MoE 产线支持仍应作为研究阶段继续推进.

## 跨模块工程需求

## 1. 剪枝依赖图

复杂结构剪枝不能再只靠“从某个模块往后找一个 consumer”.

需要明确一个轻量依赖图抽象,至少能表达:

- producer -> consumer
- one-to-many consumer
- many-to-one merge
- branch boundary
- stage boundary
- normalization dependency
- residual equivalence group

推荐输出对象:

- `PruningDependencyGraph`
- `DependencyNode`
- `DependencyEdge`
- `MergeConstraint`

这个图不一定要求完整 FX 图级别通用,但至少要够支撑常见 CNN / Transformer family.

## 2. 结构适配器

后续不建议继续把结构识别逻辑堆在 `structured.py` 的单个大函数里.

建议按结构族拆 adapter:

- `cnn_chain_adapter`
- `residual_cnn_adapter`
- `attention_adapter`
- `gated_mlp_adapter`
- `stage_adapter`

每个 adapter 负责:

- target discovery
- dependency annotation
- plan validation
- rewrite dispatch metadata

## 3. 统一选择接口

复杂剪枝至少要支持三种选择模式:

- `ratio`
- `keep_indices`
- `score_threshold`

并明确哪些粒度还需要:

- `drop_count`
- `keep_count`
- `pattern`
- `group_multiple`

不要再把所有粒度都硬塞进一个 `target_sparsity` 语义.

## 4. 剪枝后恢复训练

复杂结构剪枝如果没有恢复训练接口,实际可用性会很弱.

需要至少预留:

- prune-only
- prune + finetune
- prune + KD finetune
- width shrink + distill student recovery

这里和 `xqt.distill` 的接口要能配起来.

当前实现已具备两条恢复训练基线:

- 非结构化 `global_l1_unstructured` 的 `prune_finetune_cpu.yaml` schedule + KD smoke
- 结构化 `structured_prune_kd_cpu.yaml` 的 `structured prune -> KD finetune -> ONNX export` 最小闭环

更复杂的 width shrink student recovery 和多阶段恢复策略仍在后续 backlog.

## 5. 导出和 benchmark 闭环

复杂结构剪枝的验收不能只停在 PyTorch forward.

至少要覆盖:

- PyTorch forward
- 参数量变化
- shape 变化
- ONNX 导出
- benchmark latency
- 对无法真实加速的模式给出明确标记

## 6. 报告扩展

复杂结构剪枝的报告建议新增:

- `structure_family`
- `adapter`
- `dependency_groups`
- `merge_constraints`
- `rewritten_modules`
- `blocked_modules`
- `export_checked`
- `benchmark_checked`
- `speedup_verified`
- `recovery_recipe`

## 配置扩展建议

当前 `PruneConfig` 已支持:

- `method`
- `granularity`
- `scope`
- `importance`
- `selection`
- `rewrite`
- `params`

复杂结构扩展阶段,建议保留这个方向,不要再退回到散落在 `params` 里的隐式配置.

后续可考虑增加:

- `structure_family`
- `adapter`
- `constraints`
- `recovery`
- `validation`

但在真正实现前,不要先把 schema 扩成很大一片未使用字段.

## 优先级建议

建议扩展顺序:

1. residual / branch-aware CNN channel pruning
2. grouped / depthwise / separable conv support
3. generic Transformer attention adapter
4. gated MLP pruning
5. CNN stage pruning 和 generic block container pruning
6. block sparse report + runtime capability 闭环
7. hidden width / embedding width pruning
8. MoE expert pruning

这个顺序的原因是:

- 前四项最容易接到真实模型和现有 `structured.py` / `rewrite.py` 上.
- stage / block sparse 需要更强的依赖传播和后端信息.
- hidden width 和 MoE 剪到的是全局结构,风险和实现成本都明显更高.

## 和现有文档的关系

- 总体结构化剪枝分类和基础需求见 [XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md).
- 当前已完成能力和首轮实现拆解见 [XQT_PRUNING_IMPLEMENTATION_PLAN.md](XQT_PRUNING_IMPLEMENTATION_PLAN.md).
- 本文只负责“复杂结构和可扩展剪枝”的下一阶段需求边界.
