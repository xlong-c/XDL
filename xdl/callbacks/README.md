# XDL Callback 开发指南

本文档旨在说明 XDL 框架中 Callback 的设计逻辑、生命周期钩子以及编写新 Callback 时的最佳实践。

## 1. 核心设计逻辑

Callback 是 XDL 框架中用于非侵入式扩展训练逻辑的核心机制。它通过在训练生命周期的关键节点注入钩子（Hooks），实现诸如日志记录、模型保存、早停、进度条显示等功能，而无需修改 `Trainer` 或 `CoreModel` 的源代码。

### 设计原则：
- **非侵入性**：不改变核心训练流程的执行结果。
- **解耦性**：每个 Callback 负责单一功能。
- **可配置性**：通过初始化参数灵活控制行为。
- **安全性**：即使第三方库未安装，也要保证框架能正常运行。

## 2. 优先级系统

所有 Callback 继承自 `xdl.callbacks.base.Callback`。在初始化时可以设置 `priority`：

```python
def __init__(self, priority: int = 999):
    self.priority = priority
```

- **数值越小，优先级越高**（越先执行）。
- **默认优先级**：999。
- **推荐范围**：
    - 高优先级（0-200）：进度条 (`TqdmCallback`: 100)、日志初始化 (`WandbCallback`: 180, `TensorBoardCallback`: 150)。
    - 中优先级（200-500）：模型检查点 (`ModelCheckpoint`: 400)。
    - 低优先级（500+）：普通任务、后处理。

## 3. 生命周期钩子 (Lifecycle Hooks)

以下是常用的钩子函数及其触发时机：

| 钩子名称 | 触发时机 | 常用场景 |
| :--- | :--- | :--- |
| `setup` | 训练/验证/测试开始前的环境准备 | 初始化日志运行、检查目录 |
| `on_train_start` | 训练循环正式开始前 | 记录初始超参数、打印模型结构 |
| `on_train_batch_end` | 每个训练批次结束时 | 记录 Batch Loss、更新训练步数 |
| `on_train_epoch_end` | 每个训练周期结束时 | 记录 Epoch 指标、计算平均值 |
| `on_validation_epoch_end` | 验证周期结束时 | 比较 Best Metric、触发模型保存 |
| `teardown` | 任务完全结束时 | 关闭文件句柄、上传日志、清理内存 |

## 4. 处理第三方依赖的安全模式 (重要)

当 Callback 依赖可选的第三方库（如 `wandb`, `psutil`, `tensorboard`）时，必须遵循以下模式以确保框架的鲁棒性。

### 推荐模式：

```python
# 1. 在文件顶部使用 try-except
try:
    import some_library
    LIBRARY_AVAILABLE = True
except ImportError:
    some_library = None
    LIBRARY_AVAILABLE = False

class MyCallback(Callback):
    def setup(self, trainer, core_module, stage):
        # 2. 在运行关键逻辑前进行显式检查
        if some_library is None:
            print("警告: some_library 未安装，功能已降级")
            return
            
        # 3. 使用 self.run 等变量持有实例，并检查 None
        self.run = some_library.init()

    def on_train_batch_end(self, ...):
        # 4. 运行时保护：同时检查实例和库是否存在
        if self.run and some_library is not None:
            self.run.log({"metric": 1.0})
```

## 5. 标准 Callback 模板

```python
from typing import Any, Dict, Optional
from .base import Callback

class CustomCallback(Callback):
    """
    自定义 Callback 模板
    """
    def __init__(self, param1: float = 1.0, priority: int = 500):
        super().__init__(priority=priority)
        self.param1 = param1
        
    def setup(self, trainer, core_module, stage: str):
        # 初始化资源
        pass

    def on_train_epoch_end(self, trainer, core_module):
        # 执行逻辑
        # 提示：通常通过 core_module.last_epoch_avg 获取指标
        metrics = getattr(core_module, 'last_epoch_avg', {})
        pass

    def teardown(self, trainer, core_module, stage: str):
        # 清理资源
        pass
```

## 6. 现有 Callback 概览

- `ConsoleCallback`：控制台打印基础信息。
- `TqdmCallback`：美化的进度条显示。
- `ModelCheckpoint`：自动保存最佳和最新的模型权重。
- `EarlyStopping`：根据指标自动停止训练。
- `WandbCallback` / `TensorBoardCallback`：远程和本地日志可视化。
- `DeviceStatsMonitor`：监控 CPU/GPU 使用率。
- `LearningRateMonitor`：监控学习率变化。
