# xdl.callbacks

`xdl/callbacks/` 是 XDL 的训练期扩展层,负责日志,检查点,进度条,早停,监控和可视化.

## 当前文件

- `base.py`: `Callback` 基类
- `callback_list.py`: 回调管理器与优先级排序
- `console_callback.py`: 控制台日志
- `tqdm_callback.py`: 进度条
- `model_checkpoint.py`: 模型保存
- `early_stopping.py`: 早停
- `tensorboard_callback.py`: TensorBoard
- `wandb_callback.py`: W&B
- `layer_monitor.py`: 层激活与梯度统计
- `feature_capture.py`: 验证期中间特征抓取与落盘
- `attention_rollout.py`: 验证期 attention rollout 计算与落盘
- `learning_rate_monitor.py`: 学习率监控
- `device_stats_monitor.py`: 设备资源监控
- `timer.py`: 计时
- `torch_profiler.py`: PyTorch profiler trace 导出
- `model_summary.py`: 模型结构摘要
- `lambda_callback.py`: 轻量钩子封装
- `logging_callback.py`: 通用日志回调
- `sampling_animation_callback.py`: 采样动画
- `save_trainable_state.py`: 调用任务模型保存 LoRA/adapter 等可训练状态
- `preview.py`: 调用任务模型保存验证或训练预览

## 核心规则

- 回调通过优先级排序,数值越小越先执行,默认值是 `999`
- 回调实例上的 `priority` 会自动参与排序;`CallbackList.add_callback(..., priority=...)` 可显式覆盖
- 回调更适合作为观察者,不应接管训练主逻辑
- 可选依赖要优雅降级,不要让导入直接失败

## FeatureCaptureCallback

`FeatureCaptureCallback` 面向验证期中间特征采样, 关键参数:

- `module_names`: 需要抓取的模块名列表
- `output_dir`: 特征输出目录
- `every_n_epochs`: 每隔多少个验证 epoch 抓一次
- `first_val_batch_only`: 是否只抓第一个验证 batch
- `max_batches_per_epoch`: 每个验证 epoch 最多抓多少个 batch
- `save_format`: `pt` 或 `safetensors`

其中 `safetensors` 只保存 tensor 激活本体, 并额外写一个 JSON manifest 保存 shape 和批次元信息.

最小用法:

```python
from xdl.callbacks import FeatureCaptureCallback

callback = FeatureCaptureCallback(
    ["backbone.layer4", "head.proj"],
    output_dir="runs/features",
    first_val_batch_only=True,
    max_batches_per_epoch=1,
    save_format="pt",
)
```

默认文件名形如 `features_epoch0002_step00000007_batch0000.pt`.

## AttentionRolloutCallback

`AttentionRolloutCallback` 面向 ViT / CLIP 路线验证期 attention 聚合, 关键参数:

- `output_dir`: rollout 输出目录
- `batch_adapter`: 把验证 batch 映射成模型 `forward` 输入
- `every_n_epochs`: 每隔多少个验证 epoch 抓一次
- `first_val_batch_only` / `max_batches_per_epoch`: 控制每轮产物数量
- `save_format`: `pt` 或 `safetensors`

如果不传 `batch_adapter`, 当前默认支持这些 batch 形式:

- `{"x": tensor}`
- `{"image": tensor}`
- `{"pixel_values": tensor}`
- `{"inputs": (...), "forward_kwargs": {...}}`
- 直接传入单个 `Tensor`

最小用法:

```python
from xdl.callbacks import AttentionRolloutCallback

callback = AttentionRolloutCallback(
    output_dir="runs/attention",
    first_val_batch_only=True,
    max_batches_per_epoch=1,
    save_format="pt",
)
```

默认文件名形如 `attention_rollout_epoch0002_step00000005_batch0000.pt`. 默认只负责把 rollout 张量落盘, 不负责生成图像可视化.

## 当前导出

`xdl.callbacks.__all__` 现在导出 27 个名称:

`Callback`, `AttentionRolloutCallback`, `DeviceStatsMonitor`, `EarlyStopping`, `FeatureCaptureCallback`, `LambdaCallback`, `LayerMonitor`, `LearningRateMonitor`, `ModelCheckpoint`, `ModelSummary`, `PreviewCallback`, `QATLifecycleCallback`, `QATLifecycleState`, `SamplingAnimationCallback`, `SaveTrainableStateCallback`, `ConsoleCallback`, `LoggingCallback`, `SystemStatsCallback`, `TensorBoardCallback`, `Timer`, `TorchProfilerCallback`, `TqdmCallback`, `ModelMergeCallback`, `ReferenceModelCallback`, `RolloutBatch`, `RolloutCallback`, `WandbCallback`

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
