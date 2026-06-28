# xdl/callbacks - 训练回调模块

## 目录职责

- 提供训练生命周期中的横切扩展点
- 承载日志,检查点,进度条,早停,设备监控和可视化等能力

## 当前内容

- `base.py`:`Callback` 基类
- `callback_list.py`:回调管理器与优先级排序
- `console_callback.py`:控制台日志
- `tqdm_callback.py`:进度条
- `model_checkpoint.py`:模型保存
- `early_stopping.py`:早停
- `tensorboard_callback.py`:TensorBoard
- `wandb_callback.py`:W&B
- `layer_monitor.py`:层激活与梯度统计
- `learning_rate_monitor.py`:学习率监控
- `device_stats_monitor.py`:设备资源监控
- `timer.py`:计时
- `torch_profiler.py`: PyTorch profiler trace 导出
- `model_summary.py`:模型结构摘要
- `lambda_callback.py`:轻量钩子封装
- `logging_callback.py`:通用日志回调
- `sampling_animation_callback.py`:采样动画
- `save_trainable_state.py`: 任务可训练状态保存
- `preview.py`: 任务预览采样触发
- `quantization.py`: QAT observer/fake-quant/BN lifecycle callback

当前 `xdl.callbacks.__all__` 导出 **25** 个名称.

## 核心约束

- 回调优先级数值越小越先执行
- 回调更适合作为观察者,不要把主要训练逻辑塞进回调
- 可选依赖必须优雅降级,避免导入时直接失败

## 修改约束

- 新回调必须继承 `Callback`
- 先看 `base.py` 和 `callback_list.py`,确认钩子时机和调用方式
- 回调不应直接接管模型训练逻辑或优化器步进
- 新增导出时同步更新 `xdl/callbacks/__init__.py`

## 使用方式

```python
from xdl.callbacks import (
    ModelCheckpoint,
    SaveTrainableStateCallback,
    TensorBoardCallback,
    TorchProfilerCallback,
    TqdmCallback,
)
from xdl.trainer import Trainer

trainer = Trainer(
    callbacks=[
        TqdmCallback(),
        ModelCheckpoint(dirpath="checkpoints", monitor="val_loss"),
        TensorBoardCallback(log_dir="logs"),
        TorchProfilerCallback(log_dir="logs/profiler"),
    ]
)
```

## 验证建议

- 检查优先级顺序是否符合预期
- 检查无可选依赖环境下是否能正常降级
- 对重型回调关注额外 I/O 和显存开销
