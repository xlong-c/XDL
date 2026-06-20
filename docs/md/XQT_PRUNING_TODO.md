# XQT 复杂结构剪枝 Todo

## 负责内容

- 把复杂结构剪枝的后续工作按优先级拆成可执行 backlog.
- 给每个 todo 标出目标,依赖,主要风险和最低验收.

## 不负责内容

- 不把 todo 视为当前已完成事实.
- 不替代 [XQT_PRUNING_EXTENSION_REQUIREMENTS.md](XQT_PRUNING_EXTENSION_REQUIREMENTS.md) 的需求边界说明.
- 不给出逐文件级别的完整实现细节.

## 使用方式

建议按 P0 -> P1 -> P2 -> P3 推进.

- P0: 先补基础设施,不然后面的复杂剪枝会继续堆 if/else.
- P1: 先打通最常见,最有部署价值的复杂结构.
- P2: 再做需要更强全局 rewrite 或后端闭环的路线.
- P3: 最后做 MoE 和研究型扩展.

## P0 - 基础设施

- [x] 建立 `PruningDependencyGraph` 轻量抽象
  - 目标: 支撑 one-to-many, many-to-one, merge 和 residual 对齐.
  - 依赖: 当前 `StructuredPruningAction` / `PruningTarget` 数据结构.
  - 风险: 图抽象过重会拖慢实现,过轻又支撑不了 residual.
  - 验收: 至少能描述 producer,consumer,merge,dependency_group 和 shape constraint.

- [x] 把 target discovery 拆成 adapter 机制
  - 目标: 不再把 CNN, attention, MLP, block 识别逻辑全部堆在单个 `structured.py`.
  - 依赖: 统一的 candidate / action / validation 接口.
  - 风险: adapter 边界不清会导致 rewrite 分发更乱.
  - 验收: 现有 `channel`, `mlp_neuron`, `head`, `block` 至少能迁移到 adapter 风格入口.

- [x] 增加 shape / dependency consistency validator
  - 目标: 在 rewrite 前明确报出 residual 对齐,group 约束,merge 约束错误.
  - 依赖: dependency graph 或 adapter metadata.
  - 风险: 错误只在 forward 时暴露,定位成本高.
  - 验收: 对 invalid keep plan 能在 apply 前给出结构化错误.

- [x] 扩展结构化报告字段
  - 目标: 区分 structure family,adapter,blocked modules,export/benchmark 状态.
  - 依赖: report schema 和 runner artifact writer.
  - 风险: 报告字段继续只适合简单链路模型.
  - 验收: 复杂剪枝报告能看出为什么成功,为什么失败,哪些模块被跳过.
  - 当前状态: 结构化 plan/report 已输出 `adapters`,`structure_families`,`blocked_modules`,`dependency_graph`,`export_status`,`benchmark_status`;runner 的 export/benchmark pass 会回填结构化剪枝报告中的运行状态.

## P1 - CNN 复杂拓扑

- [x] 支持 residual add-aware channel pruning
  - 目标: 支持 ResNet BasicBlock / Bottleneck 的主分支 + shortcut 联动剪枝.
  - 依赖: dependency graph,shape validator.
  - 风险: shortcut / downsample 分支与主分支 keep plan 不一致.
  - 验收: ResNet-like block 剪枝后 forward 正常,shortcut shape 对齐,ONNX 可导出.
  - 当前状态: 已支持 ResNet 风格 `nn.Sequential` stage 的输出宽度剪枝,覆盖 BasicBlock / Bottleneck,downsample 输出和下一 stage / head 输入联动.尚不覆盖任意 add/concat 拓扑和 stage 删除.

- [x] 支持 concat / branch-aware channel propagation
  - 目标: 处理 FPN,UNet,neck 等 concat 多分支输入.
  - 依赖: merge constraint 表达,consumer input slice 重映射.
  - 风险: concat 后下游输入通道切片映射错误.
  - 验收: 至少一个 concat branch toy model 剪枝后 forward 和导出通过.
  - 当前状态: 已支持 toy `branch1 + branch2 -> concat -> fuse` 场景的 channel pruning,两个 branch 可以独立生成 keep plan,并在 rewrite 时重建 concat 后 consumer 的输入切片映射.更广的 FPN,UNet skip 和多级 concat 图仍在后续 backlog.

- [x] 支持一般 grouped convolution 剪枝
  - 目标: 不再只支持 `groups=1` 或 depthwise.
  - 依赖: group-aligned keep rule.
  - 风险: `in_channels/out_channels/groups` 整除关系被破坏.
  - 验收: grouped conv model 剪枝后模块参数合法且 forward 正常.
  - 当前状态: 已支持 chain-like 模型中的一般 grouped `Conv2d` 自动 group-aligned keep 选择,输入通道传播和 forward 验证.不包含 depthwise-separable / MBConv 的多层联动语义.

- [x] 支持 depthwise-separable / MBConv 联动剪枝
  - 目标: 打通 `pw expand -> dw -> pw project` 结构.
  - 依赖: grouped conv support,多层联动 rewrite.
  - 风险: expand width 和 project width 两层语义混淆.
  - 验收: MobileNet-like block 剪枝后参数量和导出图 shape 正确变化.
  - 当前状态: 已支持 toy MBConv-like `expand_conv -> depthwise_conv -> project_conv` 中间宽度联动剪枝,会同步重写 expand 输出,depthwise in/out/groups 和 project 输入,并验证 forward 正常.更广的 inverted bottleneck 变体和真实 MobileNet stage 级依赖仍在后续 backlog.

- [x] 支持 CNN stage pruning
  - 目标: 从 block 剪枝扩展到 stage 级删减.
  - 依赖: residual / branch-aware graph.
  - 风险: stage 下游 neck / head 输入规格失配.
  - 验收: 至少一个多 stage CNN 可以按 `keep_indices` 完成 stage 裁剪并跑通 benchmark.
  - 当前状态: 已支持 `ModuleList` / `Sequential` 中 shape-compatible CNN stage 删除,要求每个 stage 输入/输出通道一致且内部 `Conv2d` 不改变空间尺寸.可以按 `selection.keep_indices` 重建 stage 容器,并已验证多 stage CNN forward 和 `benchmark_callable` 跑通.不覆盖会改变通道数/分辨率的 backbone stage 删除,也不会自动重写 FPN/neck/head 的多尺度输入契约.

## P1 - Transformer 通用化

- [x] 支持 split q/k/v attention adapter
  - 目标: 除 `qkv + proj` 外,支持 `q_proj/k_proj/v_proj/out_proj`.
  - 依赖: attention descriptor.
  - 风险: 不同实现的 bias,shape 和 forward 细节不一致.
  - 验收: 至少一种 HF attention 实现支持 head pruning.

- [x] 支持 cross-attention head pruning
  - 目标: 区分 self-attn 和 cross-attn 的 q / kv 来源.
  - 依赖: generic attention adapter.
  - 风险: encoder hidden states 和 decoder hidden states 宽度联动错误.
  - 验收: 至少一个 cross-attn toy block 可以完成 head pruning 并 forward 正常.
  - 当前状态: 已支持 split `q_proj/k_proj/v_proj/out_proj` 形式的 toy cross-attn head pruning,target metadata 会区分 `attention_role=self/cross`,apply 后会同步裁剪 `q/k/v` 输出和 `out_proj` 输入.更广的 HF decoder cross-attn,MQA,GQA 仍在后续 backlog.

- [x] 支持 MQA / GQA head 结构
  - 目标: 处理 `num_heads != num_kv_heads`.
  - 依赖: attention descriptor,约束校验.
  - 风险: 只按普通 MHA 逻辑裁剪会破坏 kv head 共享关系.
  - 验收: planner 能正确拒绝非法 plan,并支持合法 kv-head 裁剪.
  - 当前状态: 已支持 split projection 形式的 toy GQA/MQA target 发现,descriptor/report 会输出 `num_kv_heads` 和 `attention_variant`,planner 会把 keep indices 解释为 `kv_head` 选择并拒绝非法索引,apply 后会同步裁剪 q-group,k/v 输出和 `out_proj` 输入.更广的 HF grouped-query decoder attention 变体仍在后续 backlog.

- [x] 支持 gated MLP pruning
  - 目标: 支持 `gate_proj/up_proj/down_proj` 结构.
  - 依赖: 成组 linear rewrite.
  - 风险: gate 路径和 up 路径 neuron 选择不一致.
  - 验收: 至少一种 SwiGLU / GEGLU 风格 block 剪枝后 forward 和导出通过.
  - 当前状态: 已支持 `gate_proj/up_proj/down_proj` 命名结构的中间宽度剪枝,同一 keep plan 会同步裁剪 gate/up 输出和 down 输入.尚未扩展到任意 FFN 命名或 MoE expert 内部结构.

- [x] 扩展 generic block pruning adapter
  - 目标: 不再只依赖 `ModuleList` / `Sequential` + homogeneous children.
  - 依赖: container adapter.
  - 风险: block target 发现不稳定,误把非重复子模块当 block.
  - 验收: 至少一种非简单 container 的 Transformer block 集合可被枚举和裁剪.
  - 当前状态: 已支持 `ModuleList` / `Sequential` 中由复合子模块组成的 heterogeneous block container 枚举与重建,不会再要求 child type 完全一致.更广的 container adapter,`drop_count` / `score_threshold` 选择模式和层编号/cache 元数据校验仍在后续 backlog.

## P2 - 更强全局 rewrite

- [x] 支持 hidden width / embedding width pruning
  - 目标: 支持全局 residual width 缩减.
  - 依赖: global dependency graph,LayerNorm / Embedding rewrite.
  - 风险: 这条路线已经接近自动生成窄 student,复杂度很高.
  - 验收: 单一 Transformer 骨架能按固定 hidden width 缩减并继续训练.
  - 当前状态: 已支持 XDL `VisionTransformer` 的 `hidden_width` / `embedding_width` 最小闭环,会用同一组 keep indices 同步缩减 patch embedding 输出,`cls_token`,`pos_embed`,所有 block 的 LayerNorm,attention `qkv/proj`,MLP 输入/输出和分类 head 输入,并保持 `embed_dim % num_heads == 0`.已验证 XDL ViT forward 和一次优化器训练步可继续运行.这不是任意 Transformer/HF 模型的自动窄 student rewrite.

- [x] 打通 prune + KD recovery recipe
  - 目标: 对复杂结构剪枝提供恢复训练闭环.
  - 依赖: `xqt.distill`,新结构化报告字段.
  - 风险: 只有剪枝没有恢复,真实可用性差.
  - 验收: 至少一个复杂剪枝 recipe 支持 prune -> KD finetune -> export.
  - 当前状态: 已新增 `xqt/recipes/structured_prune_kd_cpu.yaml`,支持 `structured prune -> KD finetune -> ONNX export` 的最小闭环.runner 的 `prune` pass 已支持结构化剪枝后按 `kd_steps_per_prune` 调用 `xqt.distill` 恢复训练,并在 `tests/xqt/test_runner.py` 中验证 prune、KD 和 export 全链路.

- [x] 支持 diffusion UNet / DiT block pruning
  - 目标: 评估生成模型重复块剪枝的可行闭环.
  - 依赖: generic block adapter,cross-attn support.
  - 风险: 生成模型对结构改写更敏感,导出链也更复杂.
  - 验收: 至少一个 toy DiT / UNet block 级剪枝 smoke 可跑通.
  - 当前状态: 已支持 toy DiT-like `ModuleList` block pruning smoke,覆盖 time embedding + cross-attn + MLP 的多输入 diffusion-style forward 路径,并验证剪枝后 forward 正常.真实 diffusion UNet,更复杂 DiT 变体和导出链仍在后续 backlog.

## P2 - 稀疏模式和后端闭环

- [x] 实现 `block_sparse` pruning / report
  - 目标: 补齐 schema 已声明但未闭环的 `block_sparse`.
  - 依赖: sparse pattern schema,report 扩展.
  - 风险: 只有模式报告,没有 runtime 语义会误导用户.
  - 验收: 至少支持固定 block size 的稀疏模式生成和报告导出.
  - 当前状态: 已支持固定二维 `block_shape` 的 block-sparse mask 生成,层级报告和 runner 分支.报告只承诺稀疏模式存在,不承诺当前 runtime 有加速.

- [x] 增加 sparse runtime capability matrix
  - 目标: 明确某个 pattern 在当前 backend 是否有真实加速路径.
  - 依赖: preflight,benchmark,manifest.
  - 风险: 用户把稀疏率误当成已验证 speedup.
  - 验收: preflight / report / benchmark 都能显式区分 "pattern present" 和 "speedup verified".
  - 当前状态: `nm_structured` 和 `block_sparse` 已通过统一 capability helper 在 preflight / benchmark 中输出 `pattern_present`,`supported`,`speedup_verified`,`runtime`,`reason`.当前仍主要报告“模式存在但未验证加速”.

- [x] 增加 sparse benchmark recipe
  - 目标: 给 `nm_structured` 和 `block_sparse` 提供真实延迟验证入口.
  - 依赖: capability matrix.
  - 风险: 没有 benchmark,稀疏路线难以验收.
  - 验收: 至少能在支持环境输出 pattern,compliance 和 latency 对比.
  - 当前状态: 已新增 CPU sparse benchmark recipe,可输出 `block_sparse` 的 pruning report,benchmark latency 和 capability 字段.当前 benchmark 仍是 PyTorch eager 路径,不会把它误写成 sparse kernel speedup.

## P3 - MoE 和研究型扩展

- [x] 支持 usage-aware expert pruning
  - 目标: 基于 router usage / probability 裁掉低利用 expert.
  - 依赖: expert usage 统计,router-aware graph.
  - 风险: router 输出维度和 expert 列表失配.
  - 验收: 至少能生成 expert pruning plan,并校验 router / expert 一致性.
  - 当前状态: 已支持 `granularity=expert` 的 toy MoE 适配器,要求模块暴露 `router: nn.Linear` 和 `experts: ModuleList/Sequential`,并可从 `expert_usage` / `expert_usage_counts` / `router_usage` 或 router bias/weight 生成 usage-aware 分数.剪枝会同步重写 router 输出维度和 expert 容器,并校验 forward 正常.不覆盖 shared expert,aux loss,top-k capacity 或 expert parallel.

- [x] 支持 branch / expert 混合剪枝报告
  - 目标: 统一记录 experts,branches,router 相关变更.
  - 依赖: 报告 schema 扩展.
  - 风险: MoE 剪枝结果难追踪.
  - 验收: 报告可明确列出被删除 expert,保留 expert 和 usage 依据.
  - 当前状态: `StructuredPruningPlan` 和 `StructuredPruningReport` 已新增 `topology_changes`,会显式记录 expert/branch/block/stage 的 kept/pruned indices.对 expert pruning 会输出 router,experts,kept/pruned experts 和 usage 依据.

- [x] 预研 width pruning -> auto student rewrite
  - 目标: 探索把 hidden width 剪枝和 student 结构生成统一起来.
  - 依赖: global width pruning,KD recovery.
  - 风险: 复杂度极高,容易偏离当前 XQT 主线.
  - 验收: 只要求研究报告和最小原型,不作为主线阻塞项.
  - 当前状态: 已新增 [XQT_WIDTH_TO_AUTO_STUDENT_RESEARCH.md](XQT_WIDTH_TO_AUTO_STUDENT_RESEARCH.md),记录 XDL ViT hidden/embedding width rewrite 的最小原型,通用 auto student 的风险边界,推荐架构和后续验收口径.

## 建议先做的前三项

如果只按工程收益和当前代码基础排序,最值得先做的是:

1. `PruningDependencyGraph`
2. residual add-aware channel pruning
3. split q/k/v attention adapter

原因:

- 这三项会决定后续复杂结构剪枝是“继续堆特例”,还是“开始形成可扩展框架”.
- 它们也最贴近真实模型: ResNet,MobileNet,HF Transformer 都会直接受益.

## 暂时不要急着做的项

下面几项建议不要过早进入主线:

- hidden width / embedding width 全局剪枝
- MoE expert pruning
- diffusion 大模型真实产线级 block pruning

原因不是这些方向不重要,而是它们都强依赖:

- 更完整的 dependency graph
- 更强的恢复训练闭环
- 更稳的导出和 benchmark 基础设施

在这些基础能力没稳定前,过早推进会让剪枝模块很快失控.
