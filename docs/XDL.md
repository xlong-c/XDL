# XDL 项目分析与使用指南

本文档不是文件清单，而是对 XDL 项目的结构性说明。重点回答四个问题：

1. XDL 到底是什么类型的项目。
2. 项目为什么要拆成现在这样的结构。
3. 这套结构带来什么收益。
4. 开发者应该怎样使用这些结构，而不是绕开这些结构写成脚本堆叠。

## 1. 项目定位

XDL 是一个基于 PyTorch 的模块化深度学习框架。它的目标不是替代 PyTorch，而是在 PyTorch 之上补一层“项目结构”和“实验组织”能力。

换句话说，XDL 解决的不是“怎么写一个网络层”，而是下面这些更接近真实项目的问题：

- 一个项目里有很多模型、很多损失、很多数据集，怎样统一组织。
- 训练代码怎样避免越来越像一次性脚本。
- 日志、检查点、进度条、监控这些横切逻辑，怎样避免塞满训练循环。
- 当项目开始走向配置化实验时，怎样把 YAML 和 Python 的边界划清。

因此，XDL 更适合：

- 需要频繁替换组件的研究型项目。
- 希望逐步把实验脚本收敛成稳定结构的工程型项目。

## 2. 整体结构

XDL 的核心可以看成五层：

```text
组件实现层
  model / dataset / loss / metric / optimizer / scheduler

组件发现层
  utils.registry

配置装配层
  config.schema / resolver / builder / setup

训练编排层
  trainer.CoreModel / trainer.Trainer / callbacks

基础工具层
  utils.checkpoint / utils.tools / utils.weight
```

这种分层是刻意设计出来的，不是代码自然长出来的结果。

### 对应目录

源码主目录是 [`xdl/`](../xdl/)：

- [`xdl/model/`](../xdl/model/)：模型定义和模型注册。
- [`xdl/dataset/`](../xdl/dataset/)：数据集定义和数据集注册。
- [`xdl/loss/`](../xdl/loss/)：损失函数定义和注册。
- [`xdl/metric/`](../xdl/metric/)：评估指标实现和注册。
- [`xdl/optimizer/`](../xdl/optimizer/)：优化器封装和注册。
- [`xdl/scheduler/`](../xdl/scheduler/)：学习率调度器封装和注册。
- [`xdl/trainer/`](../xdl/trainer/)：训练器、核心模型基类、训练状态。
- [`xdl/callbacks/`](../xdl/callbacks/)：训练生命周期扩展点。
- [`xdl/config/`](../xdl/config/)：配置 schema、插值解析、对象构建。
- [`xdl/utils/`](../xdl/utils/)：注册表、checkpoint 和其他基础工具。

项目根目录的配套结构：

- [`config/`](../config/)：官方配置文件示例。
- [`examples/`](../examples/)：示例脚本。
- [`tests/`](../tests/)：测试。
- [`scripts/`](../scripts/)：安装和辅助脚本。
- [`docs/`](./) ：项目文档。

## 3. 为什么要这样拆

### 3.1 组件定义层单独存在

模型、数据集、损失、指标、优化器、调度器本质上都属于“可替换组件”。如果它们和训练循环写死在一起，会出现几个典型问题：

- 替换一个模型时，需要去改训练脚本内部逻辑。
- 多个实验会复制粘贴大量相似代码。
- 配置化实验几乎不可能做干净。

把这些对象先收敛成独立组件，有两个直接好处：

- Python 代码的复用边界更明确。
- 配置系统只需要“找到并构建组件”，不用理解整个训练过程。

### 3.2 注册表只做名字到对象的映射

注册表定义在 [`xdl/utils/registry.py`](../xdl/utils/registry.py)。它刻意保持极简：

- 注册。
- 查找。
- 列出可用组件。

它不做下面这些事：

- 不解析 YAML。
- 不自动猜测参数。
- 不处理权重。
- 不负责实例化整个实验图。

为什么要这么做：

- 职责单一，出错时容易定位。
- 注册表不被配置细节污染。
- builder 可以演化，registry 仍然稳定。

这也是为什么 XDL 的 registry 是“框架基础设施”，不是“万能工厂”。

### 3.3 配置层和运行层分开

当前配置系统在 [`xdl/config/`](../xdl/config/) 下，主要有四类对象：

- `schema.py`：定义上层结构。
- `resolver.py`：负责 merge、默认值和 `${...}` 插值。
- `builder.py`：负责把配置构建成组件。
- `setup.py`：负责把整条链路串起来，返回 `TrainSetup`。

这层设计的核心思想是：

- 配置对象不等于运行对象。
- YAML 负责描述结构。
- builder 负责把描述变成实例。

这样做的好处是：

- 结构化配置可校验。
- 组件实例化逻辑可以集中维护。
- Python 和 YAML 的边界清晰。

### 3.4 `CoreModel`、`Trainer`、`Callback` 三者分工明确

XDL 的训练编排不是只有一个 `Trainer`。它实际上分成三类角色：

#### `CoreModel`

定义在 [`xdl/trainer/coreModel.py`](../xdl/trainer/coreModel.py)。

它承担的是“任务逻辑”：

- `training_step`
- `validation_step`
- `configure_optimizers`
- 指标记录
- checkpoint 相关能力

为什么需要它：

- 训练器不应该知道每个任务的具体前向与损失细节。
- 不同任务类型可以继承同一个抽象基类。
- 任务逻辑和训练循环本身可以解耦。

#### `Trainer`

定义在 [`xdl/trainer/trainer.py`](../xdl/trainer/trainer.py)。

它承担的是“训练循环编排”：

- epoch / step 循环
- 设备处理
- 梯度累积
- mixed precision
- 验证调度
- callback 调度

为什么需要它：

- 训练循环是基础设施，不应该在每个项目里重复手写。
- 统一的训练入口更利于监控、回调和后续扩展。

#### `Callback`

定义在 [`xdl/callbacks/`](../xdl/callbacks/)。

它承担的是“横切逻辑”：

- 日志
- 检查点
- 早停
- 进度条
- 设备统计
- 学习率监控

为什么要单独做 callback：

- 这些逻辑本质上不是模型本身的一部分。
- 如果放进 `Trainer`，训练器会迅速膨胀。
- 如果放进 `CoreModel`，任务逻辑会被大量副作用代码污染。

## 4. 这套结构的收益

### 4.1 可替换

模型和损失、数据集和调度器可以按组件替换，不需要改训练循环主逻辑。

### 4.2 可配置

配置系统可以围绕稳定结构工作，而不是围绕某个单一实验脚本硬编码。

### 4.3 可扩展

扩展新模型、新数据集、新 callback 时，只需要接入对应层，不必改整个框架。

### 4.4 可测试

builder、schema、setup、registry 都能单独测试，而不是只能通过完整训练脚本间接验证。

### 4.5 更适合项目演进

脚本式项目常见的问题是：

- 一开始很快。
- 三个月后所有逻辑互相缠绕。

XDL 的结构化拆分就是为了延缓这种退化，让项目能从“一个实验脚本”平滑过渡到“一个可维护代码库”。

## 5. 当前推荐使用方式

XDL 不要求所有人都从配置开始。当前推荐两条工作流并存。

### 5.1 纯代码方式

适合快速研究、验证想法、手工控制优化逻辑。

```python
from xdl.trainer import Trainer

model = ...
train_loader = ...
val_loader = ...

trainer = Trainer(
    max_epochs=10,
    device="cuda",
    gradient_accumulation_steps=1,
)
trainer.fit(model, train_loader, val_loader)
```

什么时候优先选这条路：

- 你还在快速改任务逻辑。
- 模型本身还没稳定。
- 训练步骤高度定制化。

### 5.2 配置驱动方式

适合标准实验、统一风格、多人协作和批量调参。

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

model = setup.model
optimizer = setup.optimizer
train_loader = setup.train_loader
```

什么时候优先选这条路：

- 组件已经相对稳定。
- 想统一实验结构和配置风格。
- 需要把训练脚本和实验参数解耦。

## 6. 当前配置结构应该怎么理解

当前配置系统的主格式是：

- 顶层固定结构由 dataclass 约束。
- 可构建组件统一使用 `target + params`。
- 数据链路推荐使用顶层紧凑别名。

一个简化示意：

```yaml
runtime:
  device: cuda
  data_dir: ./data

trainer:
  max_epochs: 100
  batch_size: 128

model:
  target: registry:vgg16_bn
  params:
    num_classes: 100

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 4
  pin_memory: true

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

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true
```

推荐阅读 [`CONFIG.md`](CONFIG.md) 了解完整细节。

## 7. 怎样正确扩展 XDL

### 新增模型

1. 在 [`xdl/model/`](../xdl/model/) 添加模型实现。
2. 在 [`xdl/model/__init__.py`](../xdl/model/__init__.py) 中注册。
3. 在代码或 YAML 中使用注册名。

示例：

```python
from xdl.utils.registry import register_model

@register_model("MyModel")
class MyModel(...):
    ...
```

### 新增数据集

1. 在 [`xdl/dataset/`](../xdl/dataset/) 添加数据集实现。
2. 在 [`xdl/dataset/__init__.py`](../xdl/dataset/__init__.py) 注册。
3. 如果依赖可选三方库，导入失败时要保证框架整体仍可用。

### 新增 callback

如果逻辑属于：

- 日志
- 监控
- 保存
- 训练控制

优先写 callback，而不是改 `Trainer` 主循环。

### 新增配置能力

如果某个字段会成为多个实验共享的稳定结构，应该先进入 schema，而不是先在 YAML 里自由生长。

这是因为：

- schema 决定了结构边界。
- builder 决定了构建行为。
- 示例 YAML 不应该比 schema 更“先进”。

## 8. 项目当前边界

为了让文档对项目状态保持诚实，这里明确当前边界：

1. XDL 已经具备稳定的模块化组件组织方式。
2. 配置系统已经能稳定完成 schema merge、`${...}` 插值、组件构建和 `TrainSetup` 返回。
3. callback、trainer、registry 的基础结构已经成形。
4. `logging`、`checkpoint`、`accelerate` 的“配置到训练编排”的全链路仍在继续收敛。
5. 因此，XDL 当前最稳的是“模块化框架 + 组件构建系统”，而不是“一切都已全自动闭环”的实验平台。

## 9. 推荐阅读路径

如果你是第一次进入 XDL，建议按下面顺序看代码：

1. [`xdl/utils/registry.py`](../xdl/utils/registry.py)
   - 先理解组件是怎么被发现的。
2. [`xdl/model/__init__.py`](../xdl/model/__init__.py)
   - 看模型是怎样接入 registry 的。
3. [`xdl/trainer/coreModel.py`](../xdl/trainer/coreModel.py)
   - 理解任务逻辑应该写在哪里。
4. [`xdl/trainer/trainer.py`](../xdl/trainer/trainer.py)
   - 理解训练循环是怎样组织的。
5. [`xdl/callbacks/README.md`](../xdl/callbacks/README.md)
   - 理解横切逻辑为什么走 callback。
6. [`xdl/config/setup.py`](../xdl/config/setup.py)
   - 理解配置怎样最终变成组件对象。
7. [`config/unified_logger_example.yaml`](../config/unified_logger_example.yaml)
   - 看一个最小配置示例。
8. [`config/vgg_cifar100.yaml`](../config/vgg_cifar100.yaml)
   - 看一个更完整的视觉分类配置示例。

## 10. 结论

XDL 的关键价值不在于“提供了多少模型”，而在于它给了一个比较清晰的项目骨架：

- 组件如何定义
- 组件如何发现
- 配置如何装配
- 训练如何编排
- 副作用如何扩展

只要这个骨架保持稳定，项目就能从脚本式实验逐步演进到结构化代码库，而不需要每次重构都推倒重来。
