# XDL 安装指南

XDL 是一个基于 PyTorch 的模块化深度学习框架，专为计算机视觉任务设计。

## 快速开始

### 1. 基础安装

使用 pip 安装核心依赖：

```bash
# 克隆仓库
git clone https://gitee.com/xlong_t/xdl.git
cd xdl

# 基础安装 (仅核心功能)
pip install -r requirements.txt

# 或者使用 pyproject.toml
pip install .
```

### 2. 完整安装

安装所有可选功能（推荐）：

```bash
# 完整功能安装
pip install -r requirements-full.txt

# 开发模式安装 (可编辑)
pip install -e .[all,dev]
```

### 3. 按需安装

根据你的需求选择安装：

#### 仅用于模型训练
```bash
pip install torch torchvision numpy pyyaml tqdm
```

#### 计算机视觉任务
```bash
pip install torch torchvision numpy pyyaml tqdm \
            opencv-python albumentations matplotlib
```

#### 实验管理和跟踪
```bash
pip install torch torchvision numpy pyyaml tqdm \
            tensorboard wandb
```

#### 模型优化和加速
```bash
pip install torch torchvision numpy pyyaml tqdm \
            accelerate safetensors
```

## 依赖说明

### 核心依赖 (必须)
- **torch>=1.12.0** - PyTorch 深度学习框架
- **torchvision>=0.13.0** - PyTorch 视觉库
- **numpy>=1.21.0** - 数值计算
- **pyyaml>=5.4.0** - 配置文件解析
- **tqdm>=4.62.0** - 进度条显示
- **typing-extensions>=4.0.0** - 类型提示支持

### 可选依赖

#### 计算机视觉
- **opencv-python>=4.5.0** - 图像处理和变换
- **albumentations>=1.0.0** - 强大的数据增强库
- **matplotlib>=3.5.0** - 数据可视化和绘图

#### 实验管理
- **tensorboard>=2.10.0** - Google TensorBoard
- **wandb>=0.13.0** - Weights & Biases 实验跟踪

#### 模型优化
- **accelerate>=0.12.0** - HuggingFace 训练加速
- **safetensors>=0.3.0** - 安全快速的模型权重格式

#### 高级优化 (需要GPU)
- **torchao>=0.1.0** - PyTorch 量化优化
- **triton>=2.0.0** - GPU 内核优化
- **tilelang>=0.1.0** - Tile 语言优化

### 开发依赖
- **pytest>=7.0.0** - 单元测试框架
- **black>=22.0.0** - 代码格式化
- **flake8>=5.0.0** - 代码检查
- **mypy>=0.991** - 类型检查

## 环境配置

### Conda 环境 (推荐)

```bash
# 创建环境
conda create -n xdl python=3.10
conda activate xdl

# 安装 PyTorch (根据你的CUDA版本)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装 XDL
pip install -r requirements-full.txt
```

### 虚拟环境

```bash
python -m venv xdl_env
source xdl_env/bin/activate  # Linux/Mac
# 或 xdl_env\Scripts\activate  # Windows

pip install -r requirements-full.txt
```

## 验证安装

```python
import torch
import xdl
from xdl.trainer import Trainer
from xdl.callbacks import ModelCheckpoint

print(f"PyTorch 版本: {torch.__version__}")
print(f"XDL 版本: {xdl.__version__}")
print("✅ 安装成功！")
```

## 常见问题

### 1. PyTorch 安装问题

如果遇到 PyTorch 安装问题，请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统的安装命令。

### 2. CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 可选依赖导入错误

如果不需要某些功能，可以忽略对应的导入错误，或者安装缺失的包：

```bash
# 例如，如果不需要 wandb
pip install tensorboard  # 仅安装 tensorboard
```

## 开发环境

```bash
# 克隆并安装开发版本
git clone https://gitee.com/xlong_t/xdl.git
cd xdl

# 安装所有依赖 (包括开发工具)
pip install -e .[all,dev]

# 安装 pre-commit 钩子
pre-commit install

# 运行测试验证
pytest tests/
```

## Docker 支持

如果需要 Docker 环境，可以创建 `Dockerfile`：

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY . .

RUN pip install -r requirements-full.txt
RUN pip install -e .

CMD ["python", "train.py"]
```

## 系统要求

- **Python**: 3.8+
- **PyTorch**: 1.12.0+
- **操作系统**: Linux, Windows, macOS
- **GPU**: 可选，推荐 NVIDIA GPU (CUDA 11.8+)

## 下一步

安装完成后，查看：

1. [README.md](../README.md) - 项目概述
2. [examples/](../examples/) - 示例代码
3. [docs/](../) - 其他文档

开始训练你的第一个模型：

```bash
python train.py --config config/vgg_cifar100.yaml
```