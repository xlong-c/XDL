# XQT Width Pruning To Auto Student Rewrite 预研

## 负责内容

- 记录 `hidden_width` / `embedding_width` pruning 向 auto student rewrite 扩展时的技术边界.
- 说明当前 XQT 已完成的最小原型和不能直接泛化的部分.
- 给后续实现提供分阶段路线,验收口径和风险清单.

## 不负责内容

- 不承诺任意 Transformer / LLM / HF 模型都可自动生成窄 student.
- 不替代 [XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md) 的结构化剪枝总需求.
- 不替代 [XQT_PRUNING_EXTENSION_REQUIREMENTS.md](XQT_PRUNING_EXTENSION_REQUIREMENTS.md) 的复杂结构剪枝边界.

## 当前最小原型

当前 `xqt.prune` 已支持 XDL `VisionTransformer` 的 `hidden_width` / `embedding_width` 最小闭环:

- 目标模块: XDL 自带 `VisionTransformer`.
- 剪枝单元: residual 主干 hidden dimension.
- 选择约束: keep count 必须可被 `num_heads` 整除.
- 同步重写:
  - `patch_embed.proj.out_channels`
  - `cls_token`
  - `pos_embed`
  - 每个 block 的 `norm1` / `norm2`
  - attention `qkv` 输入/输出和 `proj` 输入/输出
  - MLP `fc1.in_features` 和 `fc2.out_features`
  - 最终 `norm`
  - 分类 `head.in_features`
- 已验证:
  - target discovery
  - explicit `keep_indices`
  - forward shape
  - 一次 optimizer step 继续训练
  - XQT 配置校验接受 `hidden_width` / `embedding_width`

这已经是 auto student rewrite 的最小原型: 它不只是把权重置零,而是生成一个参数量更小、hidden width 更窄的可运行模型实例.

## 为什么还不是通用 auto student

通用 auto student 至少还需要解决下面问题:

- 模型结构差异: HF / timm / 自定义模型的 embedding,attention,MLP,LayerNorm 命名和 forward 语义不一致.
- 残差全局约束: hidden width 一旦变化,所有 residual 分支和 merge 点都必须一致.
- cache / rotary / position encoding: LLM decoder 常带 KV cache,RoPE,ALiBi 或相对位置偏置,这些状态也依赖 hidden/head 维度.
- attention 变体: MHA,GQA,MQA,cross-attn 的 q/k/v 维度和共享关系不同.
- checkpoint 兼容: 窄 student 的 state_dict shape 与 teacher 不一致,需要显式映射策略.
- 训练恢复: 宽度剪枝后通常需要 KD / fine-tune,不能只做 rewrite.
- 导出后端: ONNX,TensorRT,torch.export 对动态改写后的模块有不同约束.

## 推荐架构

后续不建议把 auto student 写成一个直接改任意模型的函数.更稳的路线是:

1. `StudentRewriteSpec`
   - 描述目标模型族,原始 hidden width,目标 hidden width,head 约束,保留索引和可改写模块清单.
2. `DependencyGraph`
   - 显式记录 residual group,producer,consumer,merge,shape constraint 和参数路径.
3. `FamilyAdapter`
   - 每个模型族单独实现 descriptor 和 rewrite:
     - `xdl_vit`
     - `timm_vit`
     - `hf_bert`
     - `hf_decoder_mha`
     - `hf_decoder_gqa`
4. `WeightMapper`
   - 把 teacher 权重映射到 student,并输出哪些权重可直接切片,哪些需要重新初始化.
5. `RecoveryRecipe`
   - 统一接入 KD / fine-tune / export / benchmark.

## 最小可行下一步

建议按下面顺序扩展:

1. 固化 XDL ViT adapter
   - 增加导出 smoke.
   - 增加 `state_dict` 映射说明.
   - 报告中输出 hidden width before/after.
2. 增加 timm ViT adapter
   - 覆盖常见 `patch_embed.proj`,`blocks`,`attn.qkv`,`attn.proj`,`mlp.fc1/fc2` 命名.
3. 增加 HF encoder-only adapter
   - 先从 BERT-like self-attention + FFN 做起.
4. 再考虑 decoder / LLM
   - 必须先处理 KV cache,RoPE,GQA/MQA 和 generation config.

## 验收口径

每新增一个模型族 adapter,至少要验证:

- plan 能输出完整 dependency graph.
- rewrite 后参数量真实下降.
- forward 与 logits shape 正常.
- 至少一个训练 step 可继续运行.
- export 或明确记录不支持原因.
- report 明确标出:
  - original hidden width
  - target hidden width
  - keep indices
  - affected modules
  - skipped modules
  - recovery recipe 是否执行

## 当前结论

XQT 当前已经有 auto student rewrite 的最小原型,但范围应严格限定为 XDL `VisionTransformer` 的 hidden width / embedding width 缩减.下一阶段可以逐步增加模型族 adapter,但不应把这条能力描述成通用自动模型压缩器.
