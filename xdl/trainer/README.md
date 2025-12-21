# 加速器模块

XDL 框架的分布式训练加速器模块, 提供对多种分布式训练框架的统一支持。

## 🚀 快速开始

```python
from xdl.trainer import Trainer
from xdl.trainer.accelerate_config import AcceleratorConfig

# 自动检测最佳策略
trainer = Trainer(
    max_epochs=10,
    accelerator_strategy='auto'
)

# 训练
trainer.fit(model, train_loader, val_loader)
```

## 📦 核心组件

### 1. AcceleratorConfig - 配置类
统一管理所有加速器配置。

```python
config = AcceleratorConfig(
    strategy='accelerate',
    mixed_precision='fp16',
    gradient_accumulation_steps=4,
    gradient_clipping=1.0
)
```

### 2. AcceleratorManager - 管理器
提供统一的加速器接口。

```python
from xdl.trainer.accelerate_manager import create_accelerator_manager

manager = create_accelerator_manager(config)
manager.initialize(trainer)
```

### 3. AcceleratorCallback - 回调
统一处理分布式训练逻辑。

```python
from xdl.callbacks.accelerate_callback import AcceleratorCallback

callback = AcceleratorCallback(config)
```

## 📚 文档

- [快速开始](../docs/ACCELERATOR_QUICKSTART.md) - 5 分钟上手指南
- [完整架构](../docs/ACCELERATOR_ARCHITECTURE.md) - 详细的架构设计文档
- [实现总结](../docs/IMPLEMENTATION_SUMMARY.md) - 实现的详细信息

## 🧪 测试

```bash
python -m pytest tests/test_accelerator.py -v
```

## 📝 示例

```bash
python examples/accelerate_usage_example.py
```

## 🤝 支持的加速器

- **Accelerate** - HuggingFace 的轻量级分布式训练库
- **DeepSpeed** - 微软的深度学习优化库, 适合大模型训练
- **FSDP** - PyTorch 的全参数分片训练
- **None** - 纯 CPU/GPU 训练

## 📄 许可证

遵循项目许可证。
