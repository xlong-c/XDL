# XDL Config 系统说明

本文档说明 XDL 当前的配置系统是什么、解决什么问题、推荐怎样组织 YAML，以及配置如何从文本一路变成可运行的训练组件。

这份文档的定位是“说明文档”，不是阶段性改造记录。因此重点放在：

- 配置系统的职责边界
- 官方推荐结构
- `target + params` 的统一写法
- `${...}` 插值和 `OmegaConf` 的作用
- 数据链路、优化链路和 loss / metrics 的配置方式
- 如何在 Python 中消费配置
- 当前兼容策略和边界

## 1. Config 系统是什么

XDL 的 config 系统不是一个独立训练框架，也不是想把 Python 逻辑全部变成 YAML。它的核心目标只有两个：

1. 固定实验配置的上层结构，避免 YAML 在项目里自由漂移。
2. 用统一规则把模型、数据集、优化器、调度器、loss、metrics 等组件构建出来。

当前系统由四部分组成：

- [schema.py](/root/workspace/xdl/xdl/config/schema.py)
  - 用 dataclass 固定顶层结构和默认值。
- [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)
  - 用 `OmegaConf` 做 merge、类型校验和 `${...}` 插值解析。
- [builder.py](/root/workspace/xdl/xdl/config/builder.py)
  - 根据配置实例化模型、dataset、optimizer、scheduler、transform、loss、metrics。
- [setup.py](/root/workspace/xdl/xdl/config/setup.py)
  - 串起整条链路，最终返回 `TrainSetup`。

这套分工的原则很明确：

- schema 决定“允许什么结构”
- resolver 决定“配置怎样被解析”
- builder 决定“对象怎样被实例化”
- setup 决定“这些对象怎样被组装起来”

## 2. 它不负责什么

当前 config 系统的职责边界也需要说清楚：

- 它负责组件构建，不直接替代 `Trainer`
- 它负责配置解析，不替代 registry
- 它负责把 YAML 变成对象，不替代任务代码本身
- 它还没有把 `logging`、`checkpoint`、`accelerate` 全部打通到完整训练编排链路

所以更准确地说，XDL 当前已经有一个稳定的“配置解析 + 组件构建”系统，但还不是一个完全闭环的实验平台 DSL。

## 3. 配置是怎样工作的

XDL config 的主链路如下：

```text
YAML / dict
  -> load_config_with_schema(...)
  -> OmegaConf.structured(schema)
  -> OmegaConf.merge(schema_cfg, raw_cfg)
  -> OmegaConf.resolve(...)
  -> to_plain_dict(...)
  -> build_model / build_dataset / build_optimizer / build_scheduler / build_loss / build_metrics
  -> setup_from_yaml(...)
  -> TrainSetup
```

可以把它理解成两阶段：

### 第一阶段：配置对象阶段

这一阶段由 `dataclass + OmegaConf` 完成，负责：

- 顶层结构校验
- 默认值补齐
- `${...}` 引用展开
- 把 YAML 转成结构化配置对象

### 第二阶段：运行对象阶段

这一阶段由 `builder + setup` 完成，负责：

- 根据 `target` 找到组件类或函数
- 用 `params` 实例化组件
- 构建 transform、dataset、dataloader
- 构建 optimizer、scheduler、loss、metrics
- 返回统一的 `TrainSetup`

这也是 XDL 当前最重要的设计边界：

- `OmegaConf` 处理“配置对象”
- `builder + registry` 处理“运行对象”

## 4. 官方推荐结构

当前官方推荐的配置格式是 schema v1。

```yaml
config_version: 1

runtime:
  device: cuda
  seed: 42
  data_dir: ./data
  output_dir: ./others
  experiment_name: demo

trainer:
  max_epochs: 100
  batch_size: 128
  precision: "32"
  gradient_accumulation_steps: 1
  grad_clip_max_norm: 5.0

model:
  target: registry:vgg16_bn
  params:
    num_classes: 100

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 4
  pin_memory: true

train_transforms: ...
val_transforms: ...
train_dataset: ...
val_dataset: ...
train_dataloader: ...
val_dataloader: ...

optimization:
  optimizer: ...
  scheduler: ...

loss:
  target: torch.nn:CrossEntropyLoss
  params: {}

metrics:
  - target: registry:Accuracy
    params:
      num_classes: 100

logging:
  ...

checkpoint:
  ...

accelerate:
  ...
```

这里最重要的规则有五条：

- 顶层结构由 dataclass 固定，不再让官方 YAML 自由设计上级结构
- 组件统一使用 `target + params`
- 数据链路优先使用顶层紧凑别名，减少一层 `data` 缩进
- transform 推荐显式写 `Compose + params`
- 旧格式只保留兼容，不再作为官方推荐写法

## 5. Dataclass 固定了什么

当前 schema 固定的是“上层结构”和“骨架字段”，而不是每个组件的所有参数。

主要固定的顶层字段包括：

- `runtime`
- `trainer`
- `model`
- `train_transforms / val_transforms / test_transforms`
- `train_dataset / val_dataset / test_dataset`
- `dataloader_defaults`
- `train_dataloader / val_dataloader / test_dataloader`
- `data`
- `optimization`
- `loss`
- `metrics`
- `logging`
- `checkpoint`
- `accelerate`

这意味着：

- 顶层字段有没有，由 schema 决定
- 顶层字段类型是否正确，由 schema + `OmegaConf` 决定
- 顶层字段默认值是什么，由 dataclass 决定
- 上层结构稳定，底层组件参数仍保持灵活

保持灵活的内容主要在这些位置：

- `model.params`
- `train_dataset.params`
- `optimization.optimizer.params`
- `optimization.scheduler.params`
- transform / metric / loss 的组件参数

这样设计的原因是：

- 项目上层结构应该稳定
- 组件参数天然变化大，不适合全部硬编码进 schema

## 6. 统一组件写法：`target + params`

当前官方主格式是：

```yaml
some_component:
  target: source:name
  params:
    ...
```

其中：

- `target` 负责定位组件
- `params` 负责提供构造参数

例如：

```yaml
model:
  target: registry:vgg16_bn
  params:
    num_classes: 100
    dropout: 0.5
```

```yaml
optimization:
  optimizer:
    target: torch.optim:Adam
    params:
      lr: 0.001
      weight_decay: 0.0001
```

```yaml
loss:
  target: torch.nn:CrossEntropyLoss
  params: {}
```

### `target` 的含义

`target` 使用 `source:name` 形式。

常见来源包括：

- `registry`
  - 从 XDL 自己的 registry 中找组件
- `torch.nn`
  - 用于 loss 或部分模型模块
- `torch.optim`
  - 用于 optimizer
- `torch.optim.lr_scheduler`
  - 用于 scheduler
- `torchvision.datasets`
  - 用于 dataset
- `torchvision.transforms`
  - 用于 transform

例如：

- `registry:vgg16_bn`
- `registry:Accuracy`
- `torch.nn:CrossEntropyLoss`
- `torch.optim:SGD`
- `torch.optim.lr_scheduler:MultiStepLR`
- `torchvision.datasets:CIFAR100`
- `torchvision.transforms:Compose`

### 仍然兼容但不推荐的旧格式

当前 builder 仍兼容这些旧写法：

- `type + source + params`
- `name + from_library + params`

保留兼容只是为了迁移历史配置，不代表它们仍然是官方主格式。

## 7. 顶层紧凑别名与数据链路

XDL 当前对数据链路的推荐不是：

```yaml
data:
  transforms:
  datasets:
  dataloaders:
```

而是使用更紧凑的顶层别名：

- `train_transforms / val_transforms / test_transforms`
- `train_dataset / val_dataset / test_dataset`
- `train_dataloader / val_dataloader / test_dataloader`
- `dataloader_defaults`

推荐这样做，是为了：

- 减少 YAML 缩进层级
- 让 train / val / test 三条链路一眼可读
- 避免 `data.transforms.train` 这类层级嵌套变深

### 推荐写法

```yaml
runtime:
  data_dir: ./data

trainer:
  batch_size: 128

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 4
  pin_memory: true

train_transforms:
  target: torchvision.transforms:Compose
  params:
    transforms:
      - target: torchvision.transforms:Resize
        params:
          size: [224, 224]
      - target: torchvision.transforms:ToTensor
        params: {}

train_dataset:
  target: torchvision.datasets:CIFAR100
  params:
    root: ${runtime.data_dir}
    train: true
    download: true
    transform: ${train_transforms}

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true
    drop_last: true
```

### 默认值与覆盖规则

data loader 推荐遵循三层默认来源：

1. schema 默认值
2. `dataloader_defaults`
3. 具体 `train_dataloader / val_dataloader / test_dataloader.params`

推荐的理解方式是：

- `trainer.batch_size` 作为全局 batch size 默认来源
- `dataloader_defaults` 作为多个 dataloader 的公共参数来源
- 具体 dataloader 的 `params` 作为最终覆盖层

例如：

```yaml
trainer:
  batch_size: 128

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 4
  pin_memory: true

val_dataloader:
  dataset: ${val_dataset}
  params:
    batch_size: 256
    shuffle: false
```

这表示：

- train 默认使用 `128`
- val 明确覆盖为 `256`
- `num_workers` 和 `pin_memory` 由公共默认层统一控制

### 数据目录应该放在哪里

数据根目录推荐放在：

```yaml
runtime:
  data_dir: ./data
```

而不是混入 `output_dir`。两者语义不同：

- `data_dir` 是输入资源路径
- `output_dir` 是训练产物路径

## 8. Transform 为什么推荐显式 `Compose`

虽然 builder 支持 transform 的紧凑写法，例如字符串或列表，但官方推荐仍然是显式写出：

```yaml
train_transforms:
  target: torchvision.transforms:Compose
  params:
    transforms:
      - target: torchvision.transforms:RandomHorizontalFlip
        params:
          p: 0.5
      - target: torchvision.transforms:ToTensor
        params: {}
```

推荐显式 `Compose` 的原因是：

- 结构统一，transform 和其他组件的表达方式一致
- `params` 保留完整，不依赖额外约定
- 文档和示例更稳定，不容易出现“这个地方为什么突然是 list”的风格漂移

当前实现上，builder 仍支持这些简写：

- 单个字符串 target
- 仅由 transform 组成的列表

但这些更适合作为兼容或临时简写，不建议作为官方样板继续传播。

## 9. `${...}` 插值和 `OmegaConf`

### `OmegaConf` 在 XDL 里负责什么

当前 `OmegaConf` 只负责配置层：

- 从 YAML / dict 加载配置
- 把用户配置 merge 到 dataclass schema
- 做 `${...}` 插值解析
- 输出结构化配置对象

它不负责：

- 直接构建 PyTorch 对象
- 替代 XDL registry
- 替代 builder

### `${train_transforms}` 是怎样工作的

以下面配置为例：

```yaml
train_transforms:
  target: torchvision.transforms:Compose
  params:
    transforms:
      - target: torchvision.transforms:ToTensor
        params: {}

train_dataset:
  target: torchvision.datasets:CIFAR100
  params:
    root: ${runtime.data_dir}
    train: true
    transform: ${train_transforms}
```

它的执行过程分成两步：

1. 配置解析阶段
   - `OmegaConf.resolve(...)` 会把 `${train_transforms}` 展开成那一整段 transform 配置
   - 这一步之后，`train_dataset.params.transform` 已经不再是字符串，而是一个普通 dict 配置

2. 实例化阶段
   - `setup_from_yaml(...)` 会先收集 transform 配置并构建真实 transform 对象
   - 再构建 dataset
   - dataset 的 `transform` 参数最终拿到的是实例化后的 transform，而不是原始 YAML 文本

所以本质上是：

- `OmegaConf` 负责“引用展开”
- `builder` 负责“对象实例化”

## 10. 优化器、调度器、loss、metrics 怎么写

### Optimizer

```yaml
optimization:
  optimizer:
    target: torch.optim:SGD
    params:
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0005
```

当前 optimizer 构建默认使用模型全部参数。

如果需要更细控制，builder 还支持：

- `target_modules`
- `param_groups`

这意味着后续可以在配置中指定只优化某些子模块，或者给不同模块分不同超参数。

### Scheduler

```yaml
optimization:
  scheduler:
    target: torch.optim.lr_scheduler:MultiStepLR
    params:
      milestones: [30, 60, 80]
      gamma: 0.1
```

### Loss

单个 loss 推荐直接写单对象：

```yaml
loss:
  target: torch.nn:CrossEntropyLoss
  params: {}
```

多个 loss 可以写成列表：

```yaml
loss:
  - target: torch.nn:CrossEntropyLoss
    weight: 1.0
    params: {}
  - target: registry:SomeAuxLoss
    weight: 0.2
    params: {}
```

当前实现规则是：

- 单个 loss 直接构建为对应实例
- 多个 loss 会聚合成 `WeightedLoss`
- `weight` 只在多 loss 组合时生效

### Metrics

metrics 使用列表表达更自然，因为训练中通常会同时维护多个指标：

```yaml
metrics:
  - target: registry:Accuracy
    params:
      num_classes: 100
```

## 11. 如何在 Python 中使用配置

当前官方入口是：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

返回对象是 [TrainSetup](/root/workspace/xdl/xdl/config/dataclass.py)，主要包含：

- `model`
- `train_loader`
- `val_loader`
- `test_loader`
- `optimizer`
- `scheduler`
- `loss_fn`
- `metrics`
- `device`
- `num_epochs`
- `batch_size`
- `full_config`

典型用法是：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

model = setup.model
optimizer = setup.optimizer
loss_fn = setup.loss_fn
train_loader = setup.train_loader
```

如果你只想验证配置系统主链路是否正常，可以直接运行：

```bash
python - <<'PY'
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
print(len(setup.train_loader.dataset))
PY
```

## 12. 兼容策略

当前配置系统不是“只认新格式、不认旧格式”，而是采用双轨过渡策略：

- 官方主格式使用 schema v1
- 历史旧格式会先被归一化，再进入统一 schema / builder 链路

当前仍兼容的内容包括：

- `data.*` 层级写法
- `type + source + params`
- `name + from_library + params`
- 旧版 `training / core_config / data_config / save_config / logger_config` 风格

但兼容不等于推荐。对新文档、新示例、新配置文件，应该统一使用：

- 顶层紧凑别名
- `target + params`
- 显式 `Compose + params`

## 13. 当前边界

当前 config 系统已经稳定的部分是：

- schema v1 顶层结构
- `${...}` 插值解析
- model / transform / dataset / dataloader 构建
- optimizer / scheduler / loss / metrics 构建
- `TrainSetup` 返回

当前仍然需要明确的边界是：

- `logging` 进入了 schema，但还没有统一接成 logger / callback builder
- `checkpoint` 进入了 schema，但还没有完整接到 checkpoint callback 构建链路
- `accelerate` 已进入 schema，但还没有形成完整训练闭环
- 一些 dataset 或 transform 依赖可选三方库，环境缺失时不会注册

因此，当前最准确的定位仍然是：

- 一个已经稳定的配置解析与组件构建系统
- 一个尚未完全闭环的实验编排系统

## 14. 扩展这套配置系统时的原则

如果后续继续扩展 config，建议遵循下面几条原则：

### 什么时候加 schema 字段

当某个字段会成为多个实验共享的稳定上层结构时，再把它加入 schema。

不要先让示例 YAML 自由生长，再倒逼 schema 去追。

### 什么时候放进 `params`

当参数明显属于某个具体组件的构造参数，而且变化频繁时，优先放进 `params`，不要急着上提成顶层字段。

### 什么时候改 builder

只有当组件实例化规则真的发生变化时，才修改 builder。不要让 registry、schema、builder 混着承担同一件事。

### 什么时候保留兼容层

兼容层只服务于迁移，不服务于继续扩散旧风格。新示例和新文档应始终使用主格式。

## 15. 推荐阅读顺序

如果你要继续深入读代码，建议按下面顺序：

1. [schema.py](/root/workspace/xdl/xdl/config/schema.py)
2. [resolver.py](/root/workspace/xdl/xdl/config/resolver.py)
3. [builder.py](/root/workspace/xdl/xdl/config/builder.py)
4. [setup.py](/root/workspace/xdl/xdl/config/setup.py)
5. [config/unified_logger_example.yaml](/root/workspace/xdl/config/unified_logger_example.yaml)
6. [config/vgg_cifar100.yaml](/root/workspace/xdl/config/vgg_cifar100.yaml)
