# XDL - 模块化深度学习框架

**Generated:** 2026-02-27
**Commit:** d3b4406
**Branch:** master


## 项目概述

XDL 是一个基于 PyTorch 的模块化深度学习框架，旨在提供可扩展、灵活的机器学习开发环境。框架支持模型、数据集、优化器、调度器等组件的灵活管理，采用组件注册系统实现模块化设计。

**技术栈**: Python 3.8+ / PyTorch 1.12+ / CUDA (可选)

## 架构说明

### 核心架构

```
xdl/
├── callbacks/      # 训练生命周期钩子系统
├── dataset/      # 数据集定义和数据加载
├── loss/         # 损失函数实现
├── metric/       # 评估指标
├── model/        # 模型架构（CNN、ViT、生成模型等）
├── optimizer/    # 优化器封装
├── scheduler/    # 学习率调度器
├── trainer/      # 训练器核心逻辑
└── utils/        # 工具函数和注册系统
```

### 关键设计模式

1. **组件注册系统**: 使用 `xdl.utils.registry` 实现全局注册表
   - 支持 MODEL、DATASET、OPTIMIZER、SCHEDULER、LOSS、METRIC 等类型
   - 通过装饰器 `@XXX.register_module()` 注册组件

2. **两种使用方式**:
   - **纯代码方式**（如 VAE/GAN 示例）: 直接手动实例化所有组件，适合快速实验和自定义逻辑
   - **配置文件方式**: 通过 YAML/JSON 配置文件构建组件，适合标准化实验和超参搜索

3. **回调系统**: 完整的训练生命周期钩子
   - 支持训练/验证/测试/预测各阶段
   - 内置日志、监控、检查点、早停等回调

## 开发规范

### 代码风格

- **语言**: 中文注释，符号使用英文半角
- **类型注解**: 所有函数添加类型注解
- **专业术语**: 保持英文（如 loss、accuracy、epoch）
- **代码格式化**: 使用 black、isort
- **静态检查**: 使用 mypy、flake8

### 组件开发规范

新组件必须通过注册系统注册：

```python
from xdl.utils.registry import MODEL

@MODEL.register_module()
class MyModel(nn.Module):
    def __init__(self, arg1: int, arg2: str):
        super().__init__()
        # 实现
```

### 使用方式示例

**方式一：纯代码方式**（如 VAE/GAN 示例）

```python
# 直接手动实例化所有组件
model = MyModel()
dataset = MyDataset()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainer = Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
```

**方式二：配置文件方式**（需自行实现配置解析）

参考 `config/vgg_cifar100.yaml` 的结构，自行编写配置解析逻辑：

```yaml
# config/example.yaml
training:
  device: "cuda"
  num_epochs: 100
  batch_size: 128

core_config:
  model:
    backbone:
      name: "vgg16_bn"
      params:
        num_classes: 100
  
  optimizer:
    main_optimizer:
      name: "SGD"
      params:
        lr: 0.01
```

**注意**: 框架提供了注册系统和配置文件示例，但配置文件的解析和加载逻辑需要根据具体需求自行实现。VAE/GAN 示例展示了纯代码方式的使用。

### 方式三：YAML 配置方式（推荐）

框架提供了 `setup_from_yaml` 函数，支持从 YAML 配置文件一键构建完整的训练配置：

```python
from xdl.config import setup_from_yaml

# 一行代码构建所有组件
setup = setup_from_yaml('config/vgg_cifar100.yaml')

# 直接访问各组件
model = setup.model
optimizer = setup.optimizer
scheduler = setup.scheduler
train_loader = setup.train_loader
val_loader = setup.val_loader
loss_fn = setup.loss_fn
metrics = setup.metrics

# 或使用 Trainer
from xdl.trainer import Trainer
trainer = Trainer(
    model=setup.model,
    train_dataloader=setup.train_loader,
    val_dataloader=setup.val_loader,
    optimizer=setup.optimizer,
    loss_fn=setup.loss_fn,
    metrics=setup.metrics,
    max_epochs=setup.num_epochs,
)
trainer.fit()
```

**返回类型**: `TrainSetup` dataclass，提供类型安全的属性访问。

## 常用命令

### 环境安装

```bash
# 基础安装
pip install -e .

# 完整安装（所有功能）
pip install -e ".[all]"

# 使用安装脚本
bash scripts/install.sh full
```

### 训练运行

```bash
# 运行VAE训练（纯代码方式，无需YAML配置）
python train_VAE.py

# 运行GAN训练（纯代码方式，无需YAML配置）
python train_GAN.py

# 使用DeepSpeed进行分布式训练（需要配置DeepSpeed配置文件）
deepspeed train_script.py --deepspeed config/deepspeed_config.json

# 指定GPU运行
CUDA_VISIBLE_DEVICES=0,1 python train_VAE.py
```

### 代码质量

```bash
# 格式化代码
black xdl/ tests/
isort xdl/ tests/

# 静态检查
mypy xdl/
flake8 xdl/

# 运行测试
pytest tests/ -v --cov=xdl
```

### 日志和可视化

```bash
# 启动TensorBoard
tensorboard --logdir others/logs

# 查看WandB实验
wandb sync others/wandb
```

## 目录说明

### 关键目录

| 目录 | 用途 | 说明 |
|------|------|------|
| `xdl/` | 核心框架 | callbacks, model, trainer, utils 等 |
| `config/` | 配置文件 | YAML 配置，DeepSpeed 配置 |
| `tools/` | 数据工具 | 图像处理、数据集下载 |
| `scripts/` | 安装脚本 | install.sh |
| `learn/` | 实验代码 | CUDA 内核、模型研究 |
| `sci_research/` | 研究工具 | DBLP 分析、论文工具 |
| `examples/` | 示例代码 | - |
| `tests/` | 单元测试 | **缺失**，需创建 |

### 入口脚本

| 脚本 | 用途 |
|------|------|
| `train_VAE.py` | VAE 训练 (MNIST) |
| `train_GAN.py` | GAN 训练 (MNIST) |
| `train_TwinFlow.py` | TwinFlow 生成模型 |
## 注意事项

1. **注册系统**: 组件必须通过 `@XXX.register_module()` 注册
2. **回调优先级**: 数值越小越先执行，默认 999
3. **GPU 内存**: 使用梯度累积、AMP、激活检查点节省显存
4. **分布式训练**: DeepSpeed/Accelerate 需同步保存检查点

## 子模块文档

- [xdl/callbacks/](xdl/callbacks/AGENTS.md) - 回调系统详解
- [xdl/model/](xdl/model/AGENTS.md) - 模型架构详解
