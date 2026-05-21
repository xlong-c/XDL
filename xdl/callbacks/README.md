# xdl.callbacks

`xdl/callbacks/` 是 XDL 的训练期扩展层，负责日志、检查点、进度条、早停、监控和可视化。

## 当前文件

- `base.py`：`Callback` 基类
- `callback_list.py`：回调管理器与优先级排序
- `console_callback.py`：控制台日志
- `tqdm_callback.py`：进度条
- `model_checkpoint.py`：模型保存
- `early_stopping.py`：早停
- `tensorboard_callback.py`：TensorBoard
- `wandb_callback.py`：W&B
- `layer_monitor.py`：层激活与梯度统计
- `learning_rate_monitor.py`：学习率监控
- `device_stats_monitor.py`：设备资源监控
- `timer.py`：计时
- `model_summary.py`：模型结构摘要
- `lambda_callback.py`：轻量钩子封装
- `logging_callback.py`：通用日志回调
- `sampling_animation_callback.py`：采样动画

## 核心规则

- 回调通过优先级排序，数值越小越先执行，默认值是 `999`
- 回调更适合作为观察者，不应接管训练主逻辑
- 可选依赖要优雅降级，不要让导入直接失败

## 当前导出

`xdl.callbacks.__all__` 现在导出 16 个名称：

`Callback`, `DeviceStatsMonitor`, `EarlyStopping`, `LambdaCallback`, `LayerMonitor`, `LearningRateMonitor`, `ModelCheckpoint`, `ModelSummary`, `SamplingAnimationCallback`, `ConsoleCallback`, `LoggingCallback`, `SystemStatsCallback`, `TensorBoardCallback`, `Timer`, `TqdmCallback`, `WandbCallback`

## 使用方式

```python
from xdl.callbacks import TqdmCallback, ModelCheckpoint, TensorBoardCallback
from xdl.trainer import Trainer

trainer = Trainer(
    callbacks=[
        TqdmCallback(),
        ModelCheckpoint(dirpath="checkpoints", monitor="val_loss"),
        TensorBoardCallback(log_dir="logs"),
    ]
)
```
