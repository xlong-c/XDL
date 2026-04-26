# P2-10: Trainer.setup_logger() **kwargs 吞掉未知参数

## 涉及文件

`xdl/trainer/trainer.py:110-179`

## 问题代码

```python
def setup_logger(
    self,
    experiment_name: str = "experiment",
    log_dir: str = "./others/logs",
    checkpoint_dir: str = "./others/checkpoints",
    monitor: Optional[str] = "val_loss",
    mode: str = "min",
    save_top_k: int = 1,
    log_every_n_steps: int = 50,
    every_n_epochs: int = 1,
    enable_tensorboard: bool = True,
    enable_checkpoint: bool = True,
    enable_tqdm: bool = True,
    enable_console: bool = False,
    enable_total_progress: bool = False,
    tqdm_metric_keys: Optional[List[str]] = None,
    **kwargs,  # ← 吞掉所有未知参数
):
    if enable_tqdm:
        self.callback_list.add_callback(
            TqdmCallback(log_frequency=log_every_n_steps, ...)
        )
    if enable_console:
        self.callback_list.add_callback(
            ConsoleCallback(log_frequency=kwargs.get("console_log_frequency", log_every_n_steps), ...)
        )
    if enable_tensorboard:
        self.callback_list.add_callback(
            TensorBoardCallback(log_frequency=kwargs.get("tensorboard_log_frequency", 1), ...)
        )
    if enable_checkpoint:
        self.callback_list.add_callback(
            ModelCheckpoint(
                ...
                save_last=kwargs.get("save_last", True),
            )
        )
```

## 问题说明

1. `**kwargs` 用于接收额外的可选配置（`console_log_frequency`、`tensorboard_log_frequency`、`save_last`），但没有任何校验。
2. 如果调用者拼写错误（例如 `save_lst=True` 或 `tenorboard_log_frequency=5`），这些参数被 **静默吸收**，不会报错，也不会生效。
3. 已知的额外参数只有 3 个，却使用了 `**kwargs`，这降低了接口的明确性。

## 修复建议

将已知的额外参数提升为显式参数：

```python
def setup_logger(
    self,
    ...,
    console_log_frequency: Optional[int] = None,     # 替代 kwargs.get("console_log_frequency")
    tensorboard_log_frequency: int = 1,              # 替代 kwargs.get("tensorboard_log_frequency")
    save_last: bool = True,                          # 替代 kwargs.get("save_last")
):
```

去掉 `**kwargs`，让拼写错误直接抛出 `TypeError`。
