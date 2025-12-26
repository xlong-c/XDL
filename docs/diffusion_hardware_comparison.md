# 扩散模型部署硬件对比调研报告 (2025Q4)

## 1. 调研背景
针对扩散模型（Stable Diffusion XL, Flux.1, SD3.5 等）的部署需求，对比三款主流/新兴硬件：RTX 5060 Ti 16GB、RTX 5060 以及 Jetson Orin NX 16GB。

## 2. 核心参数与算力精细对比 (TFLOPS/TOPS)

| 维度 | RTX 5060 Ti 16GB | RTX 5060 | Jetson Orin NX 16GB |
| :--- | :--- | :--- | :--- |
| **架构** | Blackwell (Gen 5 Tensor) | Blackwell (Gen 5 Tensor) | Ampere (Gen 3 Tensor) |
| **FP8 Dense (FP16 Acc)** | **~380 TFLOPS** | **~307 TFLOPS** | 不支持 |
| **FP8 Sparse (AI TOPS)** | **~760 TFLOPS** | **~614 TOPS*** | 不支持 |

*\*注：614 为 NVIDIA 官方/泄露文档中常见的 AI TOPS 指标，通常对应 FP8 精度下的结构化稀疏（Sparse）峰值性能。*
| **FP8 Dense (FP32 Acc)** | ~190 TFLOPS | ~130 TFLOPS | 不支持 |
| **FP16 Dense (FP32 Acc)** | ~95 TFLOPS | ~65 TFLOPS | 3.76 TFLOPS |
| **FP16 Sparse (FP32 Acc)** | ~190 TFLOPS | ~130 TFLOPS | 7.52 TFLOPS |
| **显存带宽** | ~500 GB/s (GDDR7) | ~380 GB/s (GDDR7) | 102 GB/s (LPDDR5) |
| **显存容量** | 16GB | 8GB/12GB | 16GB (共享) |

## 3. 技术深度分析

### 3.1 累加器精度 (Accumulation Precision)
- **FP16 Accumulation**: 在 FP8 推理中，使用 FP16 作为累加器能最大化利用 Blackwell 架构的吞吐量。对于扩散模型图像生成，FP16 累加带来的精度损失通常是不可察觉的。
- **FP32 Accumulation**: 当任务涉及超长链式计算或对数值稳定性极度敏感时（如大规模训练），需切换至 FP32 累加。在消费级 Blackwell 卡上，这会导致吞吐量减半（下降 50%）。

### 3.2 结构化稀疏 (2:4 Sparsity)
- **原理**: NVIDIA 从 Ampere 开始支持 2:4 结构化稀疏技术。Blackwell 的第五代 Tensor Core 对此进行了增强。
- **扩散模型应用**: 扩散模型（如 SDXL）的权重矩阵具有高度冗余性。通过 TensorRT 对模型进行稀疏化压缩后，开启 Sparse 算力模式，RTX 5060 Ti 的推理效率理论上可以翻倍，实测通常有 1.4x-1.7x 的速度提升。

### 3.3 显存带宽与瓶颈分析
- **Compute Bound**: 在进行高 Step 推理或大 Batch 生成时，5060 Ti 的高 TFLOPS 优势明显。
- **Memory Bound**: 在低 Step（如 SDXL Turbo/Lightning）场景下，显存带宽是关键。5060 Ti 的 GDDR7 带宽比 Orin NX 高出近 5 倍，这决定了两者在实时生成任务中的量级差异。

### 3.2 显存容量对部署的影响
- **16GB (5060 Ti / Orin NX)**：可以流畅运行 Flux.1 (dev/schnell) 以及带有多个控制网络（ControlNet）的 SDXL 流程。
- **8GB/12GB (5060)**：在运行高性能扩散模型时经常需要进行“切片”处理（Tiled VAE）或使用量化版本，会显著降低生成超大分辨率图片的效率。

### 3.3 运行速度与能效比
- **绝对性能**：RTX 5060 Ti 16GB 是三者中的性能王者，适合作为生产力工具。
- **能效比**：Jetson Orin NX 在每瓦性能上完胜，适合 24/7 不间断运行的自动化边缘设备。

## 4. 结论与部署建议

- **个人开发者/工作站**：首选 **RTX 5060 Ti 16GB**。Blackwell 架构的 FP8 加速配合 16GB 大显存是目前部署生成式 AI 最理想的入门级配置。
- **预算敏感型**：若主要运行 SD1.5 或轻量级任务，**RTX 5060** 具有较高性价比，但需注意显存瓶颈。
- **工业/嵌入式场景**：若需要在无风扇环境、机器人或低功耗设备上部署，**Jetson Orin NX 16GB** 是唯一选择，需配合 TensorRT 进行深度优化。
