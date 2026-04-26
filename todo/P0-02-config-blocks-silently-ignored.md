# P0-02: logging / checkpoint / accelerate 配置被静默丢弃

## 概述

YAML 配置文件中定义了 `logging`、`checkpoint`、`accelerate` 三个配置块，schema 中也定义了对应的 dataclass，但 `setup_from_yaml` 完全不处理这些块。用户在 YAML 中写的这些配置被静默忽略，不报错也不生效。

## 涉及文件

| 文件 | 关键位置 | 角色 |
|------|----------|------|
| `xdl/config/schema.py:91-99` | `LoggingConfig` | 定义了 log_dir, enable_console, enable_tensorboard, enable_wandb 等 |
| `xdl/config/schema.py:103-111` | `CheckpointConfig` | 定义了 save_last, save_top_k, monitor, mode 等 |
| `xdl/config/schema.py:14` | `AccelerateConfig` | 定义了 mixed precision, FSDP, DeepSpeed 等 |
| `xdl/config/schema.py:137-139` | `ConfigSchemaV1` | accelerate / logging / checkpoint 三个字段 |
| `xdl/config/setup.py:168-262` | `setup_from_yaml` | **未提取**这三块配置 |
| `xdl/config/dataclass.py:14-71` | `TrainSetup` | **没有**对应的字段 |
| `xdl/trainer/trainer.py:110-179` | `Trainer.setup_logger()` | 期望手动传入 logging 参数 |
| `xdl/trainer/trainer.py:48-96` | `Trainer.__init__()` | 期望在构造时传入 accelerate_config |

## 具体丢弃的字段

### logging 块（YAML 中写了，无效）

```yaml
logging:
  log_dir: "./others/logs"
  enable_console: true
  enable_tensorboard: true
  enable_wandb: false
  wandb_project: ~
  wandb_entity: ~
```

`Trainer.setup_logger()` 确实接受同名参数（`trainer.py:110-126`），但 `setup_from_yaml` 没有调用它，也没有把这些值传递过去。

### checkpoint 块（YAML 中写了，无效）

```yaml
checkpoint:
  dirpath: "./others"
  save_last: true
  save_top_k: 5
  monitor: "Accuracy"
  mode: "max"
  every_n_epochs: 5
```

`Trainer.setup_logger()` 内部在 `enable_checkpoint=True` 时会创建 `ModelCheckpoint` 回调（`trainer.py:169-179`），接受 `monitor`、`mode`、`save_top_k` 等参数。但这些参数也无法从 YAML 传递过来。

### accelerate 块（YAML 可以写，schema 定义了，完全无路径传递）

```yaml
accelerate:
  mixed_precision: "fp16"
  fsdp:
    ...
  deepspeed_config:
    ...
```

`Trainer.__init__` 接受 `accelerate_config: Optional[Dict[str, Any]]`（`trainer.py:58`），但 `setup_from_yaml` 没有提取 `accelerate` 块并传递。

## 影响

- 用户按 YAML 配置了日志目录，实际日志不会写到指定位置
- 用户配置了 checkpoint 策略（`save_top_k: 5`），但实际可能采用 Trainer 的默认值（`save_top_k: 1`）或根本不启用
- 分布式训练（DeepSpeed/FSDP）无法通过 YAML 配置，必须回到方式 1 手写代码
- 静默丢弃让用户以为配置生效了，调试困难

## 修复建议

1. 在 `TrainSetup` 中增加三个字段（或新增一个 `TrainerConfig` 包装字段）：
   ```python
   logging_config: Dict[str, Any] = field(default_factory=dict)
   checkpoint_config: Dict[str, Any] = field(default_factory=dict)
   accelerate_config: Optional[Dict[str, Any]] = None
   ```

2. 在 `setup_from_yaml` 中提取并填充这些字段（`setup.py:237` 附近）。

3. 提供 `setup.start_training()` 或 `Trainer.from_setup(setup)` 等便捷方法，自动将这些配置传递给 Trainer 的构造函数和 `setup_logger()`。

4. 或者不做 #3，而是在 `TrainSetup` 的文档中明确告知用户如何手动传入这些配置。
