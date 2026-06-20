# XQT 量化需求文档

## 负责内容

- 定义 `xqt.quant` 在 XQT 工具链中的职责边界.
- 说明当前 `xqt` 量化主线支持什么,不支持什么.
- 把“哪些模块可以量化,怎么量化”翻译成按模型族划分的工程策略.
- 为视觉模型,大模型,扩散模型和异构/多模态模型提供量化需求基线.
- 为后续 `schema`, `quant` backend, `analysis`, `preflight`, recipe 和验收标准提供统一约束.

## 不负责内容

- 不把本文中的后续需求直接视为“当前已实现事实”.
- 不替代 [XQT.md](XQT.md) 的整体架构草案.
- 不重复展开剪枝,蒸馏和扩散少步蒸馏的完整规划.
- 不承诺首期一次支持所有量化算法,所有模型族和所有硬件后端.
- 不把第三方后端的实现细节复制进 XQT; XQT 只定义接入边界和验收方式.

## 背景

当前 `xqt` 量化线已经有一条最小可运行链路,但还没有一份单独的长期需求文档把边界讲清楚.

当前已存在的量化相关能力:

- `xqt/quant/policy.py`: 模块选择策略和候选枚举.
- `xqt/quant/torchao_backend.py`: `torchao` PyTorch runtime 量化 adapter.
- `xqt/quant/onnx_qdq.py`: ONNX Runtime static QDQ INT8 quantization adapter.
- `xqt/quant/calibration.py`: 激活统计和分布漂移分析.
- `xqt/quant/sensitivity.py`: 逐层输出误差,权重误差和高精度建议.
- `xqt/pipeline/passes.py`: 内置 `quant` pass,当前只接 `torchao` 和 `onnxruntime_qdq`.
- `xqt/pipeline/preflight.py`: 量化依赖,CUDA 和 calibration 数据预检查.
- `xqt/recipes/image_vit_torchao_fp8.yaml`: ViT torchao FP8 recipe.
- `xqt/recipes/image_resnet_onnx_qdq_int8.yaml`: ResNet ONNX QDQ INT8 recipe.
- `xqt/recipes/image_resnet_cifar100_qdq_cpu.yaml`: 本地 CIFAR-100 ONNX QDQ CPU recipe.

对应源码入口:

- [xqt/quant/torchao_backend.py](/root/workspace/xdl/xqt/quant/torchao_backend.py)
- [xqt/quant/onnx_qdq.py](/root/workspace/xdl/xqt/quant/onnx_qdq.py)
- [xqt/quant/calibration.py](/root/workspace/xdl/xqt/quant/calibration.py)
- [xqt/quant/sensitivity.py](/root/workspace/xdl/xqt/quant/sensitivity.py)
- [xqt/pipeline/passes.py](/root/workspace/xdl/xqt/pipeline/passes.py)

截至 2026-06-19,我额外核对了几类官方文档,用于约束本文对第三方后端的描述:

- `torchao` 官方 inference 文档把重点放在 `torch.nn.functional.linear` 的 dynamic 和 weight-only inference quantization.
- Hugging Face `Transformers` 量化文档当前列出 AWQ,GPTQ,bitsandbytes 和 torchao 集成.
- ONNX Runtime 官方文档当前把 static/dynamic quantization,QDQ 和量化调试作为主线.
- TensorRT 官方文档当前仍把 INT8/QAT/PTQ 和 QDQ ONNX export 作为核心部署路径之一.
- OpenVINO 官方 PTQ 文档当前仍强调 representative calibration dataset.

这意味着 XQT 的需求文档不能只写“算法名列表”,而必须同时回答 4 个问题:

1. 哪些对象是当前 XQT 真正要负责的.
2. 哪些模块在不同后端下真的可量化.
3. 哪些模块默认应该保高精度.
4. 不同模型族在量化目标,数据要求和验收指标上有什么不同.

## 量化模块在 XQT 中的边界

量化在 XQT 中不是一个单文件功能,而是一条跨 `data -> quant -> export -> eval -> benchmark -> manifest` 的工程链.

### `xqt.quant` 应负责什么

- 定义量化策略: backend, dtype, granularity, allowlist/denylist, skip list.
- 执行量化: 调用具体 backend adapter 产出量化模型或量化 artifact.
- 分析量化: 激活统计,逐层误差,高精度保留建议.
- 表达结果: 返回结构化 metadata,量化模块列表,校准摘要和产物路径.

### `xqt.quant` 不负责什么

- 不直接实现底层 kernel.
- 不自己重写 TensorRT,OpenVINO,ONNX Runtime 或 `torchao` 的量化逻辑.
- 不替代 `xqt.export` 的图导出和部署格式落盘.
- 不替代 `xqt.data` 的 calibration loader 构建.
- 不替代 `xqt.eval` 和 `xqt.benchmark` 的精度/性能验收.

### 邻接模块边界

| 模块 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `xqt.data` | calibration/train/validation/prompt 数据构建 | 不决定量化算法 |
| `xqt.quant` | 量化策略,backend adapter,量化分析 | 不落地部署格式 |
| `xqt.export` | ONNX,TensorRT,OpenVINO,`torch.export`,TorchScript 等导出 | 不决定哪些层跳过量化 |
| `xqt.eval` | 输出 diff,任务指标,报告表 | 不执行业务量化 |
| `xqt.benchmark` | latency,memory,性能阈值 | 不判断精度是否可接受 |
| `xqt.pipeline` | 串联 pass 和错误包装 | 不实现算法细节 |
| `xqt.prune` / `xqt.distill` | 与量化联动的前后处理 | 不替代量化 backend |

## XQT 量化需求应按哪些维度建模

如果只按“INT8 / FP8 / 4bit”记需求,文档会很快失真. XQT 需要同时记录下面几类维度.

### 1. 量化时机

- runtime dtype baseline: `fp32`, `bf16`, `fp16`.
- PTQ dynamic: 运行时按输入动态决定 activation quantization.
- PTQ static: 需要 calibration dataset,常见于 ONNX QDQ INT8.
- weight-only: 只量化权重,激活保高精度或更高精度.
- QAT: 训练中插 fake quant,当前不作为 XQT 首期内置主线.

### 2. 量化对象

- weights
- activations
- KV cache
- embedding tables
- attention projections
- MLP/FFN weights
- convolution kernels
- experts weights

### 3. 量化粒度

- per-tensor
- per-channel
- per-group
- block-wise
- token dynamic
- layer-wise mixed precision
- component-wise mixed precision

### 4. 承载 runtime

- PyTorch eager / `torch.compile`
- ONNX Runtime
- TensorRT
- OpenVINO
- 后续可选: vLLM,TensorRT-LLM,ExecuTorch,ncnn,MNN

### 5. 工程目标

- 降显存
- 降内存带宽压力
- 提升 tokens/s 或 images/s
- 降低 p50/p99 latency
- 生成部署友好的量化图
- 保持任务指标在可接受阈值内

## 当前 XQT 应明确承诺的量化路径

这里区分“当前代码已经接线”与“后续需求”.

### 当前已接线

| 路径 | 当前状态 | 主要对象 | 产物形态 | 备注 |
| --- | --- | --- | --- | --- |
| `backend: torchao` | 已有内置 pass | Linear-heavy 模型,ViT/Transformer 优先 | PyTorch runtime 量化模型 | 当前最适合 weight-only 和 FP8 dynamic 路径 |
| `backend: onnxruntime_qdq` | 已有内置 pass | CNN/ResNet,可导出 ONNX 的模型 | QDQ ONNX | 依赖 calibration 或 validation fallback |
| `analysis` + `quant` | 已接线 | 量化前后对比 | JSON/CSV/Markdown + manifest metric | 用于 skip list 和 mixed precision 建议 |

### 当前不应视为已内置

| 路径 | 当前状态 | 说明 |
| --- | --- | --- |
| PT2E quantization | 未内置 | `xqt/quant/pt2e_backend.py` 仍不存在 |
| ModelOpt / SmoothQuant | 未内置 | 只在架构草案中,还没有 backend adapter |
| GPTQ / AWQ / bitsandbytes | 未内置 | 可作为大模型后续 P1/P2 目标 |
| KV cache quantization | 未内置 | 需要单独 schema,benchmark 和服务侧验收 |
| QAT | 未内置 | 需要训练链接入,不应塞进当前轻量 `quant` pass |

## 哪些模块可以量化

“可以量化”必须区分 3 个层次:

1. 策略层候选: 选择器允许它进入候选集合.
2. backend 层覆盖: 具体 backend 真的能稳定处理它.
3. 业务层可接受: 量化后任务指标和延迟收益都可接受.

当前 `QuantizationPolicy` 默认:

- include: `Linear`, `Conv2d`, `MultiheadAttention`
- exclude: `LayerNorm`, `BatchNorm1d/2d/3d`, `Embedding`
- 默认按名字排除 `head`, `classifier`

但这只是“候选筛选器”,不是“backend 覆盖契约”. 尤其是 `torchao` 路径,官方文档当前重点说明的是 `linear` inference quantization,所以 XQT 不能把 `Conv2d` 和 `MultiheadAttention` 在 torchao 路径上写成已验证事实.

### 模块级量化矩阵

| 模块/对象 | 默认态度 | 首选路径 | 默认保高精度? | 说明 |
| --- | --- | --- | --- | --- |
| `Linear` | 第一优先级 | torchao,ONNX QDQ,GPTQ/AWQ/INT4 weight-only | 否 | 视觉 Transformer,LLM,DiT 的主战场 |
| `Conv2d` | 重要 | ONNX QDQ,OpenVINO,后续 QAT/PTQ | 否 | CNN/检测/分割更适合图量化,不是 torchao 首选主线 |
| `Gemm` / `MatMul` | 重要 | ONNX QDQ | 否 | 导出图里比模块名更重要 |
| `MultiheadAttention` 容器 | 谨慎 | 拆成 q/k/v/out `Linear` 后量化 | 是,容器本身不单独承诺 | 关键是 projection 和 softmax 路径分开处理 |
| `Embedding` | 默认保守 | 后续大模型 weight-only 专线 | 是 | 内存收益大,但精度和 backend 兼容性敏感 |
| `LayerNorm` / `RMSNorm` | 默认跳过 | mixed precision 保留高精度 | 是 | 小算子,敏感,延迟收益通常不大 |
| `BatchNorm` | 不单独量化 | 先 fuse 到 Conv | 是 | inference 部署优先 fusion,不是独立量化对象 |
| `Softmax` / attention score | 不作为低 bit 主对象 | 保持高精度累积 | 是 | 数值范围敏感 |
| `lm_head` / `classifier` / `head` | 默认跳过或后评估 | mixed precision | 通常是 | 最终输出层对指标敏感 |
| `KV cache` | 单独建模 | 后续 LLM 服务量化专线 | 视后端 | 不能和普通 activation quantization 混为一谈 |
| MoE router / gating | 默认跳过 | mixed precision | 是 | routing 错误会放大系统性误差 |
| experts weights | 候选 | weight-only / per-group | 否 | 需要覆盖稀疏路由分布 |
| cross-modal projector | 谨慎 | 先 BF16/FP16,再逐步下探 | 通常是 | 异构模型里经常是对齐瓶颈 |
| VAE decoder 输出头 | 默认保守 | mixed precision | 是 | 生成视觉质量敏感 |

### 默认高精度保留清单

除非某条 recipe 已经验证过,下面对象默认应保留在 `fp32/bf16/fp16`:

- `LayerNorm`, `RMSNorm`
- `Softmax` 和 attention score accumulation
- `Embedding`
- `lm_head`, `classifier`, `head`
- router/gating/top-k/selective dispatch
- VAE decoder 最末几层和图像输出头
- 跨模态 projector,alignment head
- 任何导致 shape/control-flow 改变的离散决策路径

## 视觉模型怎么量化

这里的“视觉模型”至少要拆成 4 类,不能用一条量化路线覆盖.

### 1. CNN 分类模型

典型对象:

- ResNet
- MobileNet
- ConvNeXt 的 Conv-heavy 部分

首选路线:

- `ONNX Runtime static QDQ INT8`
- 后续可衔接 TensorRT 或 OpenVINO

优先量化对象:

- `Conv2d`
- 导出图中的 `MatMul` / `Gemm`
- 最后的 `Linear`

默认高精度对象:

- fusion 前的 `BatchNorm` 先并入 Conv,不作为独立量化对象
- 首层和末层可作为 mixed precision 候选
- classification head 默认先保守

关键考量:

- calibration 数据必须覆盖真实图像尺寸,亮度,对比度和类别分布.
- 如果模型有大量 residual add,分支合流点要关注 scale 对齐.
- 对部署后端来说,Conv-BN-ReLU 这类结构先做 pre-export fusion 更稳定.
- 验证指标不能只看 output diff,还要看 top-1/top-5 或业务 metric.

### 2. ViT 和视觉 Transformer

典型对象:

- ViT
- DeiT
- Swin 的 Linear-heavy 部分

首选路线:

- 支持硬件上优先 `torchao` weight-only 或 FP8 dynamic
- 需要导出部署图时,再评估 ONNX QDQ 对 `MatMul/Gemm` 的覆盖

优先量化对象:

- patch projection 的 `Linear` 或等价投影
- q/k/v/out projections
- MLP `fc1` / `fc2`

默认高精度对象:

- `LayerNorm`
- `Softmax`
- class token/head
- 位置编码和最后分类头

关键考量:

- activation outlier 往往比 CNN 更明显,需要 sensitivity 和 activation drift.
- 真实收益高度依赖 GPU 是否支持 FP8 友好路径.
- sequence length 和 dynamic shape 会直接影响 runtime 选择.
- attention path 里即使量化 qkv,也常需要保留 score accumulation 的高精度.

### 3. 检测,分割,姿态等复杂视觉模型

典型对象:

- Faster R-CNN
- YOLO
- RT-DETR
- Mask2Former

首选路线:

- 以 `ONNX QDQ INT8` 为主
- 再根据目标机器接 TensorRT/OpenVINO

优先量化对象:

- backbone
- neck
- 大型卷积和线性层

默认高精度对象:

- NMS,top-k,decode,proposal generation
- 位置/几何后处理
- 很薄的输出头

关键考量:

- calibration 需要覆盖不同分辨率,长宽比和小目标/大目标分布.
- 不能只用分类指标验收,必须看 mAP,IoU,召回率和后处理一致性.
- 动态 shape 和多输出张量比分类模型更麻烦,需要更严格的 export 校验.
- 复杂后处理经常不是量化收益主来源,不要强行把所有节点都压到低精度.

### 4. 扩散和生成视觉模型

典型对象:

- Stable Diffusion / SDXL
- Flux / DiT
- 文生图 pipeline 中的 text encoder,transformer/UNet,VAE

首选路线:

- 组件拆分量化,不要整条 pipeline 一刀切
- 先量化 `text encoder` 和 `transformer/UNet` 中的 `Linear`
- `VAE` 和最终图像输出路径默认更保守

默认高精度对象:

- `VAE` decoder 输出头
- scheduler 数值逻辑
- prompt embedding 对齐路径
- 最容易引入色偏/细节劣化的首尾层

关键考量:

- 图像质量退化通常不会在线性 output diff 里充分暴露,还要看 fixed seed 样本对比.
- prompt adherence,构图稳定性,肤色/纹理和高频细节都要单独检查.
- 不同组件可以走不同 backend,例如 text encoder 保 BF16,DiT 走 FP8,VAE 保 FP16.
- 如果同时做少步蒸馏和量化,必须区分是谁引入了误差.

## 大模型怎么量化

这里的“大模型”主要指 LLM,以及以 Transformer block 为主的参数量巨大模型.

### 大模型量化目标和视觉模型不同

LLM 最常见的第一目标不是“把每个算子都变成 INT8”,而是:

- 让模型能装进目标显存.
- 降低 decode 阶段的内存带宽压力.
- 提升 tokens/s.
- 在可接受的 perplexity/任务精度损失下,把服务成本降下来.

### 大模型优先路线

#### 1. weight-only 8bit / 4bit

适用场景:

- 单机推理
- 显存受限
- 更关注可加载性和吞吐,而不是严格的全图静态部署

代表路线:

- `torchao`
- `bitsandbytes`
- AWQ
- GPTQ
- 后续可选 AQLM,HQQ,SpQR

优先量化对象:

- attention q/k/v/o projections
- MLP up/gate/down projections
- experts 的大矩阵

默认高精度对象:

- embeddings
- final norm
- `lm_head`
- router/gating
- KV cache 元信息和索引路径

#### 2. FP8

适用场景:

- Ada/Hopper 等支持 FP8 友好路径的 NVIDIA GPU
- 大型 Transformer/DiT/ViT

优点:

- 常比 INT8 更容易保住精度
- 对某些 GPU 路径有较好的吞吐潜力

限制:

- 高度依赖硬件和 backend 支持
- 不应写成“任何 GPU 都有收益”

#### 3. static INT8 / SmoothQuant 类路线

适用场景:

- 需要更稳定导出到 TensorRT/OpenVINO
- activation outlier 问题需要显式处理

限制:

- calibration 数据要求更严格
- 需要更强的图规整和前置融合

### 大模型量化的特殊考量

- prefill 和 decode 的瓶颈不同,不能只测单个 batch latency.
- `KV cache` 是独立压缩轴,不能附带在普通 activation quantization 里含糊处理.
- context length 会显著改变显存压力和误差表现.
- group size,per-channel/per-group 的选择会显著影响 4bit 精度.
- LoRA/PEFT 场景下,通常是“量化 base model + adapter 保高精度”,而不是把 adapter 一起压到最低 bit.
- MoE 场景下 expert 可以量化,router 默认保高精度,还要覆盖长尾 expert 路由分布.
- 只看输出 diff 不够,要看 perplexity,基准任务,长上下文稳定性和 hallucination 风险变化.

## 异构模型怎么量化

本文里的“异构模型”至少包括 4 类情况:

1. 结构异构: CNN + Transformer + MLP.
2. 模态异构: image encoder + projector + LLM decoder.
3. pipeline 异构: text encoder + DiT/UNet + VAE + scheduler.
4. runtime 异构: 同一个系统里一部分走 TensorRT,一部分走 PyTorch 或 ONNX Runtime.

### 异构模型的核心原则

- 先拆组件,再定策略,不要先定一个统一 bit-width.
- 先识别对齐边界,再做低 bit 下探.
- 允许不同组件使用不同 backend.
- 允许同一组件内部 mixed precision.

### 异构模型的推荐拆法

| 组件类型 | 常见例子 | 推荐优先级 | 默认策略 |
| --- | --- | --- | --- |
| vision encoder | ViT,ConvNet,SigLIP | 高 | 先按视觉模型策略独立量化 |
| text decoder / LLM core | Llama,Qwen,Mistral | 高 | 先 weight-only 或 FP8 |
| cross-modal projector | MLP,Linear bridge | 谨慎 | 默认 BF16/FP16,验证后再量化 |
| retrieval / rerank head | dual encoder,cross encoder | 中 | 单独评估,不要复用主干阈值 |
| router / modality selector | MoE,tool router,branch gate | 低 | 默认高精度 |
| postprocess | NMS,token selection,image decode | 低 | 默认不量化 |

### 典型异构场景

#### 1. 视觉语言模型

示例结构:

`vision encoder -> projector -> LLM decoder`

建议:

- vision encoder 按 ViT/CNN 路线单独评估.
- LLM decoder 按大模型路线单独评估.
- projector 默认先保高精度,因为它承担跨模态对齐.
- 如果需要低 bit,先从 decoder 开始,最后才动 projector.

#### 2. 多分支视觉模型

示例结构:

`backbone -> detection head`
`backbone -> segmentation head`

建议:

- backbone 可以共享量化策略.
- 不同 head 应有独立阈值和独立 metric.
- 一个 head 可接受的误差,不代表另一个 head 也可接受.

#### 3. MoE 大模型

建议:

- expert weights 可以优先 weight-only.
- router/gating 保高精度.
- calibration / eval 数据要覆盖 route 分布,不要只覆盖高频 expert.

#### 4. 多 backend 组合部署

示例:

- vision encoder 导出 ONNX QDQ -> TensorRT
- LLM decoder 保留 PyTorch + torchao
- projector 保 BF16

这类系统不是失败,而是 XQT 需要正面支持的现实部署形态. 文档,manifest 和 recipe 都要能表达它.

## 量化时要考量哪些方面

下面这些点应该作为量化需求和验收 checklist,不是“附加项”.

### 1. 模型结构

- 是 `Conv2d` 主导,还是 `Linear` 主导.
- 是否有大量 residual,skip connection,branch merge.
- 是否包含 `Embedding`,`Norm`,`Softmax`,`Router`.
- 是否有动态 shape,可变序列长度,多输入多输出.

### 2. 目标 runtime 和硬件

- CPU,GPU,NPU 的最优量化路线不同.
- 是否有 FP8 友好硬件.
- 是否目标就是 TensorRT/OpenVINO/ONNX Runtime 图部署.
- 是否允许混合 runtime.

### 3. 量化目标

- 追求的是可加载性,吞吐,延迟,还是部署格式.
- 是离线产物交付,还是研究试验.
- 是单 batch 低延迟,还是大 batch 吞吐.

### 4. 数据和 calibration

- calibration 数据是否代表真实输入分布.
- 是否覆盖长尾样本,长上下文,多尺度图像,复杂 prompt.
- 是否只用了 validation fallback.
- sample_limit 是否足够做稳定 calibration.

### 5. 精度风险点

- 首层和末层是否需要保高精度.
- `Norm`,`Softmax`,`Embedding`,`Router` 是否应跳过.
- 是否有 activation outlier.
- 是否需要 sensitivity 排序来决定 skip list.

### 6. 导出和图兼容

- 模型是否可稳定导出 ONNX.
- ONNX 图里具体是 `Conv`, `MatMul`, `Gemm` 还是复杂自定义算子.
- 前置融合是否会提升 QDQ 图质量.
- 是否存在 runtime 不支持的量化节点组合.

### 7. 服务期行为

- decode 阶段和 prefill 阶段是否分别测过.
- KV cache 是否单独处理.
- streaming 场景下的 tokens/s,TTFT,p99 是否达标.
- 视觉服务里是否要看 batch sweep 和多分辨率延迟.

### 8. 产物和可复现性

- manifest 是否记录 backend,dtype,policy,skip list,calibration summary.
- 是否记录依赖版本和硬件环境.
- 是否能复现同一量化产物.

### 9. 联动优化

- 是否已经做过 prune/distill.
- 量化前后是否需要再蒸馏恢复.
- 是否需要先做 pre-export fusion.
- 是否要允许“先剪枝再量化”和“先量化再蒸馏”两种顺序.

## 对 schema 和 recipe 的需求

当前 `QuantConfig` 只有:

- `enabled`
- `backend`
- `policy`

这足够做 P0 smoke,但对大模型和异构模型不够. 后续需求至少应支持下面几类表达能力.

### 1. 组件级策略

需要能表达:

- 哪个子模块走哪个 backend
- 哪些模块强制高精度
- 哪些模块允许更低 bit
- 哪些模块只做分析,暂不真正量化

建议字段方向:

- `component_policies`
- `keep_high_precision`
- `force_quantize`
- `skip_quantize`
- `analysis_only_modules`

### 2. 数据级策略

需要能表达:

- calibration split 名称
- calibration sample limit
- prefill/decode 专用评估数据
- 多模态各分支的代表性数据

### 3. backend 级策略

需要能表达:

- backend 名称
- strategy 名称
- weight/activation dtype
- op allowlist
- pre-export fusion
- target runtime

### 4. 验收级策略

需要能表达:

- output diff 阈值
- 任务 metric 最大下降
- latency / tokens/s / memory 阈值
- 组件级和整机级双重验收

### 组件级量化示意

下面是需求层面的示意,不是当前已实现 schema:

```yaml
compression:
  quant:
    enabled: true
    component_policies:
      - name: vision_encoder
        target: model.vision_tower
        backend: onnxruntime_qdq
        strategy: int8_static
        calibration_split: calibration_vision
        keep_high_precision: [head, norm]
      - name: projector
        target: model.mm_projector
        backend: none
        strategy: bf16
      - name: decoder
        target: model.language_model
        backend: torchao
        strategy: int4_weight_only
        keep_high_precision: [lm_head, norm]
```

## 验收标准

量化模块不能只以“跑完没有报错”验收.

### 通用验收

- 至少有 baseline 和 quantized 两套可比较输出.
- manifest 记录 backend,策略,产物路径和关键 metric.
- 量化产物可继续进入下游 runtime,或明确标记 `PyTorch runtime only`.
- 能回答“哪些层被量化了,哪些层被跳过了,为什么”.

### 视觉模型验收

- 分类: top-1/top-5 或业务指标下降在阈值内.
- 检测/分割: mAP,IoU,召回率下降在阈值内.
- 延迟和显存必须至少有一项真实收益.
- 多分辨率输入不出现明显 runtime 错误.

### 大模型验收

- perplexity 或核心 benchmark 下降在阈值内.
- 记录 prefill/decode tokens/s.
- 记录显存峰值和 context length 相关行为.
- `KV cache` 如果参与量化,必须单独报告.

### 异构模型验收

- 组件级 metric 和整机级 metric 都要有.
- 明确哪个组件贡献了收益,哪个组件带来了误差.
- 跨模态对齐任务要看端到端结果,不能只看单组件 diff.

## 分阶段范围

### P0

- 保持 `torchao` 和 `onnxruntime_qdq` 两条主线.
- 明确模块量化矩阵和默认高精度清单.
- 修正当前多输入 batch 和模块选择/分析边界问题.
- 强化 calibration summary,skip list 和 manifest 表达.

### P1

- 大模型 weight-only backend: AWQ,GPTQ,bitsandbytes 或 torchao HF integration.
- 异构模型组件级量化 schema.
- SmoothQuant / ModelOpt / OpenVINO NNCF 等更强 deployment path.
- `KV cache` 专项量化和 benchmark.

### P2

- QAT
- 更激进的 2bit/3bit/FP4/NF4 路线
- runtime 级混合 backend 调度
- 自动 mixed precision 推荐和闭环搜索

## 参考资料

以下外部资料已在 2026-06-19 核对过,可作为后续实现前的事实参考:

- torchao Quantized Inference: <https://docs.pytorch.org/ao/stable/workflows/inference.html>
- torchao Hugging Face Integration: <https://docs.pytorch.org/ao/stable/eager_tutorials/torchao_hf_integration.html>
- ONNX Runtime quantization: <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- NVIDIA TensorRT quantized types: <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html>
- Hugging Face Transformers quantization overview: <https://huggingface.co/docs/transformers/main_classes/quantization>
- Hugging Face Transformers quantization method selection: <https://huggingface.co/docs/transformers/quantization/selecting>
- Hugging Face Diffusers torchao quantization: <https://huggingface.co/docs/diffusers/quantization/torchao>
- OpenVINO PTQ basic quantization flow: <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>

## 相关文档

- [XQT.md](XQT.md)
- [XQT_DATA.md](XQT_DATA.md)
- [XQT_ANALYSIS.md](XQT_ANALYSIS.md)
- [XQT_PRE_EXPORT_FUSION.md](XQT_PRE_EXPORT_FUSION.md)
- [XQT_PRUNING_REQUIREMENTS.md](XQT_PRUNING_REQUIREMENTS.md)
