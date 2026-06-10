# XDL Config 系统说明

本文档只讲 XDL 当前的配置系统，不重复介绍整个框架结构。

## 1. 配置系统解决什么问题

XDL 的 config 系统负责把 YAML 变成可运行组件集合。它解决的是：

- 固定实验配置的上层结构
- 用统一写法构建模型、数据集、优化器、调度器、loss、metrics
- 把配置解析和对象实例化从训练脚本里抽出来

它不负责替代 `Trainer`，也不试图把全部训练逻辑都塞进 YAML。

## 2. 当前主链路

当前推荐入口是：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

主链路如下：

```text
YAML
  -> load_config_with_schema()
  -> to_plain_dict()
  -> build_model / build_dataset / build_dataloader
  -> build_optimizer / build_scheduler / build_loss / build_metrics
  -> TrainSetup
```

返回结果是 `TrainSetup`，其中包含：

- `model`
- `train_loader`
- `val_loader`
- `test_loader`
- `optimizer`
- `scheduler`
- `loss_fn`
- `metrics`
- `logging_config`
- `checkpoint_config`
- `accelerate_config`
- `trainer_config` 以及 `precision` / `gradient_accumulation_steps` 等常用 trainer 字段

如果要直接交给 `Trainer.fit()`，可以继续：

```python
model = setup.create_model()
```

公共入口契约见 [API.md](API.md)。配置主链路推荐只依赖 `from xdl.config import setup_from_yaml, TrainSetup`，不要直接依赖 `xdl.config.setup` 内部辅助函数。

## 3. 模块分工

`xdl/config/` 主要由四部分组成：

- [schema.py](../xdl/config/schema.py)：固定顶层结构和默认值
- [resolver.py](../xdl/config/resolver.py)：schema merge、插值解析、普通 dict 转换
- [builder.py](../xdl/config/builder.py)：组件实例化
- [setup.py](../xdl/config/setup.py)：组装整条构建链路

职责边界如下：

- schema 决定“允许什么结构”
- resolver 决定“配置怎样被解析”
- builder 决定“对象怎样被构建”
- setup 决定“怎样把对象拼成 `TrainSetup`”

## 4. 顶层结构

当前 schema 版本为 `v1`，顶层字段包括：

- `config_version`
- `runtime`
- `trainer`
- `model`
- `task`
- `train_transforms` / `val_transforms` / `test_transforms`
- `train_dataset` / `val_dataset` / `test_dataset`
- `dataloader_defaults`
- `train_dataloader` / `val_dataloader` / `test_dataloader`
- `optimization`
- `loss`
- `metrics`
- `callbacks`
- `logging`
- `checkpoint`
- `accelerate`
- `deepspeed`
- `xdl`

其中最常用的几块是：

- `runtime`：设备、实验名、输出目录
- `trainer`：epoch、batch size、precision、梯度累积
- `model`：模型组件
- `task`: 可选的 `CoreModel` 任务组件, 适合大模型或手写训练逻辑
- `optimization`：优化器和调度器
- `loss` / `metrics`：训练目标和评估指标
- `callbacks`: 用 import path 配置化构建 callback

## 5. 统一组件写法

XDL 当前统一使用 `target + params`：

```yaml
model:
  target: "registry:simple_mlp"
  params:
    input_size: 784
    hidden_size: 128
    num_classes: 10
```

`target` 使用 `source:name` 形式。

常见来源：

- `registry:...`
- `torch.nn:...`
- `torch.optim:...`
- `torch.optim.lr_scheduler:...`
- `torchvision.transforms:...`

`builder.py` 会先解析 `target`，再实例化组件。

内置上下文字段由配置加载器注入:

```yaml
runtime:
  output_dir: ${xdl.abspath:${xdl.config_dir},outputs}
```

可用字段:

- `${xdl.config_path}`: 当前 YAML 文件的绝对路径
- `${xdl.config_dir}`: 当前 YAML 文件所在目录
- `${xdl.project_root}`: 调用 `setup_from_yaml()` 时的当前工作目录
- `${xdl.join_path:...}` / `${xdl.abspath:...}`: 路径拼接和绝对路径 resolver

## 6. transform、dataset、dataloader 的关系

配置系统把数据链路拆成三层：

1. transform
2. dataset
3. dataloader

示例：

```yaml
dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 0
  pin_memory: false

train_dataset:
  target: "registry:SyntheticClassificationDataset"
  params:
    num_samples: 500
    input_shape: [784]
    num_classes: 10
    seed: 42

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true
    drop_last: true
```

当前实现里：

- `dataloader_defaults` 负责公共默认值
- `train_dataloader.params` 负责局部覆盖
- `val_dataloader.dataset: ${train_dataset}` 这类写法可以复用已有 dataset 配置
- `trainer.batch_size` 可作为默认 batch size 来源
- `collate_fn` 支持 `None`、可调用对象或 `target + params`

常用内置 dataset 模板:

- `registry:ImageFolderClassificationDataset`: 读取 `root/class_name/image` 分类目录.
- `registry:ManifestClassificationDataset`: 从 JSONL/JSON/CSV manifest 读取 `image + label`.
- `registry:ManifestImageTextDataset`: 从 manifest 读取 `image + text/prompt/caption`, 适合图文微调.
- `registry:ManifestImageEditDataset`: 从 manifest 读取 source/target/reference/mask 多图编辑样本.
- `registry:ManifestRecordDataset`: 直接返回 manifest 里的 dict 记录.

## 7. loss 和 metrics 的写法

### 单个 loss

```yaml
loss:
  - target: "torch.nn:CrossEntropyLoss"
    params: {}
```

### 多个 loss

`build_loss()` 支持列表形式，多项时会构建 `WeightedLoss`：

```yaml
loss:
  - target: "torch.nn:MSELoss"
    params: {}
    weight: 1.0
  - target: "registry:DiceLoss"
    params: {}
    weight: 0.5
```

### metrics

```yaml
metrics:
  - target: "registry:Accuracy"
    params:
      num_classes: 10
```

`build_metrics()` 返回指标实例列表。

### callbacks

Callback 暂不新增 registry 类型, 使用 import path 构建:

```yaml
callbacks:
  - target: "xdl.callbacks:SaveTrainableStateCallback"
    params:
      dirpath: ${xdl.abspath:${xdl.config_dir},adapters}
      every_n_epochs: 1
```

`Trainer.from_setup(setup)` 会把这些 callback 加入训练器。

### CoreModel task

标准监督任务继续使用 `model + optimization + loss`. 如果任务本身继承
`CoreModel` 并在 `configure_optimizers()` 中构建优化器, 可以改用:

```yaml
task:
  target: "my_project.tasks:MyTask"
  params:
    lr: 0.0001
```

此时 `optimizer` 和 `loss` 可以省略, `setup.create_model()` 会直接返回该
`CoreModel` 实例。

## 8. 一个最小可运行示例

仓库里的 [config/unified_logger_example.yaml](../config/unified_logger_example.yaml) 是当前最合适的主路径样例。

最小消费方式：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
print(len(setup.train_loader.dataset))
```

## 9. 推荐扩展方式

当你想扩展配置系统时，先判断变化属于哪类：

- 新顶层字段：改 `schema.py`
- 新插值或 merge 规则：改 `resolver.py`
- 新组件构建方式：改 `builder.py`
- 新装配流程：改 `setup.py`

不要把所有变化都堆到 `setup_from_yaml()`。

训练入口里的轻量 dataclass 配置推荐使用:

```python
from xdl.config import load_structured_dataclass_config

config = load_structured_dataclass_config(MyConfig, "train.yaml")
```

该工具遵循 `structured dataclass 默认值 -> YAML 覆盖 -> overrides 覆盖`
的顺序, 并统一走 OmegaConf resolver 和插值解析。

## 10. 当前边界

当前 config 系统已经稳定支持：

- schema v1 顶层结构
- `target + params`
- transform / dataset / dataloader 构建
- optimizer / scheduler / loss / metrics 构建
- `TrainSetup` 返回

但仍有边界：

- 它不替代训练主循环
- 它不定义统一 CLI
- 它不自动覆盖所有任务特化逻辑
- 它更适合“配置化构建组件”，而不是“声明式描述整个实验世界”
- schema dataclass、resolver 和 builder 内部辅助函数仍属于演进中的 API；稳定入口以 [API.md](API.md) 为准
