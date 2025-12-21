# 深度学习框架 README

## 项目概述

这是一个基于PyTorch的深度学习框架, 旨在提供模块化、可扩展的机器学习开发环境。框架支持模型、数据集、优化器、调度器等组件的灵活配置和管理。

## 目录结构

```
xdl/
├── config/          # 配置文件
├── data/            # 数据相关文件
├── docs/            # 文档
├── examples/        # 示例代码
├── logs/            # 日志文件
├── others/          # 实验相关文件(checkpoints、data、results等)
├── scripts/         # 执行脚本
├── xdl/             # 源代码
│   ├── dataset/     # 数据集定义
│   ├── logger/      # 日志系统
│   ├── model/       # 模型定义
│   ├── tests/       # 测试代码
│   ├── trainer/     # 训练器
│   └── utils/       # 工具函数
│       ├── components/  # 组件模块
│       └── ...
└── ...
```

## 核心特性

### 1. 统一日志系统

- 支持多后端：Console、File、TensorBoard
- 统一的日志接口, 便于监控和调试

### 2. 组件注册系统

- 全局注册表：MODEL、DATASET、OPTIMIZER、SCHEDULER、LOSS、METRIC
- 支持动态注册和构建组件

### 3. 模块化架构

- 清晰的模块划分, 易于扩展和维护
- 配置驱动的组件构建

## 快速开始

### 安装依赖

#### 方式一：使用 pip 安装（推荐）

```bash
# 基础安装（仅核心功能）
pip install -r requirements.txt

# 完整安装（所有功能）
pip install -r requirements-full.txt

# 开发环境安装
pip install -e .[all,dev]
```

#### 方式二：使用安装脚本

**Linux/Mac:**
```bash
# 基础安装
bash scripts/install.sh base

# 完整安装
bash scripts/install.sh full

# CPU 版本
bash scripts/install.sh cpu

# CUDA 版本
bash scripts/install.sh cuda
```

**Windows:**
```cmd
# 基础安装
scripts\install.bat base

# 完整安装
scripts\install.bat full

# CPU 版本
scripts\install.bat cpu

# CUDA 版本
scripts\install.bat cuda
```

#### 方式三：按需安装

```bash
# 仅核心依赖
pip install torch torchvision numpy pyyaml tqdm

# 计算机视觉任务
pip install torch torchvision numpy pyyaml tqdm opencv-python albumentations matplotlib

# 实验管理
pip install torch torchvision numpy pyyaml tqdm tensorboard wandb

# 模型优化
pip install torch torchvision numpy pyyaml tqdm accelerate safetensors
```

**详细安装指南请参考 [docs/INSTALL.md](docs/INSTALL.md)**

```bash
pip install -e .
```

### 运行示例

```bash
# 运行示例训练
python train.py

# 查看TensorBoard日志
tensorboard --logdir others/logs
```

### 配置文件

框架使用YAML格式的配置文件, 支持以下配置项：

- 模型配置
- 数据集配置
- 训练参数配置
- 优化器配置
- 评估指标配置

## 开发指南

### 代码规范

- 使用中文注释, 但符号使用英文的半角符号
- 专业术语保持英文(如loss、accuracy、epoch等)
- 遵循现有代码风格

### 组件开发

新组件需通过注册系统进行注册：

```python
from xdl.utils.registry import MODEL

@MODEL.register_module()
class MyModel(nn.Module):
    # 模型定义
```

## 文件说明

- `__pycache__/`: Python缓存文件
- `.git/`: Git版本控制文件
- `.gitignore`: Git忽略规则文件
- `config/`: 项目配置文件
- `data/`: 数据文件
- `logs/`: 日志文件
- `others/`: 存放日志文件、结果文件、缓存文件、模型权重等非代码文件
- `scripts/`: 执行脚本
- `xdl/`: 源代码目录
- `tests/`: 单元测试目录
- `config/`: 配置文件目录 (YAML)
- `tools/`: 数据处理工具
- `examples/`: 示例脚本
- `others/`: 实验数据、日志、检查点等非代码资产存放处
- `README.md`: 项目说明文件
- `QWEN.md`: 代码助手使用说明
- `AGENTS.md`: 智能体说明文件
- `CLAUDE.md`: Claude Code指导文件

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

[在此处添加许可证信息]