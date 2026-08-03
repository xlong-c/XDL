# SD3.5 Medium 量化工具

该脚本使用 `torchao` 对 Stable Diffusion 3.5 Medium 模型进行量化，以降低显存占用并提高推理速度。

## 功能特性

- 支持多种量化算法：
  - `int8_wo`: INT8 仅权重分析 (Weight-Only)。
  - `int4_wo`: INT4 仅权重分析。
  - `fp8`: FP8 动态激活与权重量化（需要 NVIDIA 40 系列及以上显卡）。
- 针对 SD3.5 Medium 的 `Transformer` 和 `T5 Text Encoder` 进行量化。
- 支持量化后直接进行推理测试。

## 环境依赖

确保已安装以下库：

```bash
pip install torch torchvision diffusers transformers accelerate torchao
```

## 使用方法

### 1. 基础量化 (INT8)

```bash
python learn/quant/sd35_quant.py --type int8_wo
```

### 2. INT4 量化并运行推理测试

```bash
python learn/quant/sd35_quant.py --type int4_wo --infer
```

### 3. FP8 量化 (推荐 4090/4080 用户)

```bash
python learn/quant/sd35_quant.py --type fp8 --infer
```

## 参数说明

- `--model_id`: HuggingFace 模型 ID，默认为 `stabilityai/stable-diffusion-3.5-medium`。
- `--type`: 量化类型，可选 `int8_wo`, `int4_wo`, `fp8`。
- `--infer`: 是否在量化后运行推理测试。

## 输出

量化后的模型权重和测试图片将保存在 `others/checkpoints/sd35_quantized` 目录下。
