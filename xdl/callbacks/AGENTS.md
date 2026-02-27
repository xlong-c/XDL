# XDL Callbacks - 训练生命周期钩子系统

**Generated:** 2026-02-27
**Commit:** d3b4406
**Branch:** master

## 概述

XDL Callback 模块提供非侵入式训练扩展机制，通过生命周期钩子实现日志、检查点、监控等功能。

## 架构

```
xdl/callbacks/
├── base.py                 # Callback 基类，定义所有钩子接口
├── callback_list.py        # CallbackList 管理器，处理优先级排序
├── console_callback.py     # 控制台输出
├── tqdm_callback.py        # 进度条显示（最复杂，952行）
├── model_checkpoint.py     # 模型检查点保存
├── early_stopping.py       # 早停机制
├── tensorboard_callback.py # TensorBoard 日志
├── wandb_callback.py       # Weights & Biases 日志
├── layer_monitor.py        # 层级监控（梯度、激活统计）
├── learning_rate_monitor.py# 学习率监控
├── device_stats_monitor.py # 设备资源监控
├── timer.py                # 训练计时器
├── model_summary.py        # 模型结构摘要
├── lambda_callback.py      # 灵活的 lambda 钩子
├── logging_callback.py     # 通用日志回调
└── sampling_animation_callback.py # 生成模型采样动画
```

## 核心概念

### 优先级系统

```python
# 数值越小，优先级越高（越先执行）
TqdmCallback: 100       # 高优先级
TensorBoardCallback: 150
WandbCallback: 180
ModelCheckpoint: 400    # 中优先级
默认: 999               # 低优先级
```

### 生命周期钩子

| 钩子 | 触发时机 | 典型用途 |
|------|----------|----------|
| `setup` | 训练开始前 | 初始化资源 |
| `on_train_batch_end` | 每批次结束 | 记录 loss |
| `on_train_epoch_end` | 每周期结束 | 计算平均指标 |
| `on_validation_epoch_end` | 验证结束 | 比较 best metric |
| `teardown` | 训练结束后 | 清理资源 |

## 使用示例

```python
from xdl.callbacks import (
    TqdmCallback, ModelCheckpoint, EarlyStopping,
    TensorBoardCallback, WandbCallback
)

trainer = Trainer(
    callbacks=[
        TqdmCallback(),
        ModelCheckpoint(save_dir="checkpoints", monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=10),
        TensorBoardCallback(log_dir="logs"),
    ]
)
```

## 开发规范

### 必须遵循

1. **继承 Callback 基类**：所有回调必须继承 `xdl.callbacks.base.Callback`
2. **处理可选依赖**：使用 `try-except` 包裹第三方导入
3. **设置优先级**：通过 `__init__(priority=N)` 设置

### 可选依赖安全模式

```python
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

class MyCallback(Callback):
    def setup(self, trainer, core_module, stage):
        if wandb is None:
            print("警告: wandb 未安装，功能已降级")
            return
        # 实际逻辑
```

### 获取训练指标

```python
def on_train_epoch_end(self, trainer, core_module):
    # 通过 core_module 获取指标
    metrics = getattr(core_module, 'last_epoch_avg', {})
    avg_loss = metrics.get('loss', 0.0)
```

## 注意事项

- **不要在回调中修改训练状态**：回调应为只读观察者
- **优先级冲突**：同优先级按注册顺序执行
- **异常处理**：回调异常会中断训练，务必处理
- **内存管理**：大型回调（如 LayerMonitor）注意显存占用

## 文件复杂度

| 文件 | 行数 | 说明 |
|------|------|------|
| tqdm_callback.py | 952 | 最复杂，处理进度条、多阶段显示 |
| callback_list.py | 583 | 回调管理器 |
| layer_monitor.py | 538 | 层级监控 |
| wandb_callback.py | 356 | WandB 集成 |