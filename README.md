# XDL

XDL 是一个基于 PyTorch 的模块化深度学习框架。它把项目拆成四个核心层次：

1. 组件定义层：模型、数据集、损失、指标、优化器、调度器。
2. 组件发现层：注册表负责把名字映射到对象。
3. 配置装配层：配置系统负责把 YAML 解析为可构建组件。
4. 训练编排层：`CoreModel + Trainer + Callback` 负责训练循环和训练期扩展。

这套结构的目标不是追求“框架感”，而是让实验代码从一开始就具备可扩展、可复用、可迁移的形态。对快速实验，你可以直接写纯代码；对标准化实验，你可以用统一配置驱动组件构建。

## 项目定位

XDL 当前最适合下面两类场景：

- 研究型项目，需要频繁替换模型、损失、数据集和训练策略。
- 工程型项目，希望逐步把“脚本式训练”收敛成结构化组件和统一配置。

当前成熟度更接近 `alpha`：

- 模块划分、注册体系、训练器和回调系统已经成形。
- 配置系统已经能稳定完成组件构建和引用解析。
- logger / checkpoint / accelerate 的配置化编排仍在继续收敛，不应宣称为完全闭环的实验平台。

## 核心结构

源码主目录是 [`xdl/`](xdl/)：

- [`xdl/model/`](xdl/model/)：模型定义与注册入口。
- [`xdl/dataset/`](xdl/dataset/)：数据集定义与注册入口。
- [`xdl/loss/`](xdl/loss/)：损失函数定义与注册入口。
- [`xdl/metric/`](xdl/metric/)：指标实现与注册入口。
- [`xdl/optimizer/`](xdl/optimizer/)：优化器封装与注册入口。
- [`xdl/scheduler/`](xdl/scheduler/)：学习率调度器与注册入口。
- [`xdl/trainer/`](xdl/trainer/)：`CoreModel`、`Trainer`、训练状态管理。
- [`xdl/callbacks/`](xdl/callbacks/)：训练生命周期扩展点。
- [`xdl/config/`](xdl/config/)：配置 schema、解析、引用解析、组件构建。
- [`xdl/utils/`](xdl/utils/)：注册表、checkpoint、工具函数。

项目根目录的高频入口还有：

- [`config/`](config/)：官方 YAML 配置示例。
- [`examples/`](examples/)：脚本级示例。
- [`train_VAE.py`](train_VAE.py)、[`train_GAN.py`](train_GAN.py)、[`train_TwinFlow.py`](train_TwinFlow.py)：当前可直接运行的训练入口。
- [`docs/`](docs/)：项目文档。

## 为什么这么分层

### 1. 组件定义和训练编排分离

模型、损失、指标等对象本质上是“组件”；训练循环、回调、日志和检查点本质上是“编排逻辑”。把两者分开后：

- 组件可以独立复用。
- 训练器不需要知道每个模型的业务细节。
- 不同任务可以共享同一套训练编排能力。

### 2. 注册表只负责发现，不负责构建

XDL 的注册表在 [`xdl/utils/registry.py`](xdl/utils/registry.py) 中，职责被刻意收窄为：

- 注册名字。
- 通过名字查找对象。

它不负责 YAML 解析、不负责自动注入参数、不负责权重处理。这样做的好处是边界清晰，排错时也更容易定位问题。

### 3. 配置系统只负责“配置层”，不直接取代 Python

XDL 当前配置系统建立在 `dataclass + OmegaConf + builder` 上：

- `dataclass` 固定上层结构。
- `OmegaConf` 负责 merge、默认值和 `${...}` 插值。
- builder 负责把配置转成真实对象。

这样既保留了结构化配置的收益，也不会把项目做成一套难维护的小型 DSL。

### 4. Callback 把副作用从训练循环中抽离

日志、检查点、学习率监控、进度条这些逻辑都属于“训练期副作用”，不应该直接塞满 `Trainer`。Callback 系统的意义在于：

- 保持训练主循环可读。
- 给扩展留稳定挂点。
- 避免为了新增日志或监控去反复修改训练器主逻辑。

## 推荐使用方式

### 方式一：纯代码方式

适合快速实验、生成式模型原型和需要细粒度控制的任务。

```python
import torch
from xdl.trainer import Trainer

model = ...
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = ...
val_loader = ...

trainer = Trainer(
    max_epochs=10,
    device="cuda",
    gradient_accumulation_steps=1,
)
trainer.fit(model, train_loader, val_loader)
```

### 方式二：配置驱动方式

适合标准化实验、组件替换频繁的项目和需要统一配置风格的场景。

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

print(type(setup.model).__name__)
print(type(setup.optimizer).__name__)
print(type(setup.loss_fn).__name__)
```

配置系统当前返回 [`TrainSetup`](xdl/config/dataclass.py)，包含：

- `model`
- `train_loader / val_loader / test_loader`
- `optimizer`
- `scheduler`
- `loss_fn`
- `metrics`
- `device / num_epochs / batch_size`
- `full_config`

## 扩展项目的正确方式

### 新增模型

1. 在 [`xdl/model/`](xdl/model/) 添加模型实现。
2. 在 [`xdl/model/__init__.py`](xdl/model/__init__.py) 中注册到 `MODEL_REGISTRY`。
3. 在代码或配置中通过注册名使用。

### 新增数据集

1. 在 [`xdl/dataset/`](xdl/dataset/) 添加数据集实现。
2. 在 [`xdl/dataset/__init__.py`](xdl/dataset/__init__.py) 中注册。
3. 通过代码或 YAML 的 `target` 使用。

### 新增训练期功能

如果逻辑属于日志、监控、检查点、统计或训练控制，优先写成 Callback，而不是直接改 `Trainer`。

## 文档入口

优先阅读下面三份文档：

1. [`docs/XDL.md`](docs/XDL.md)：项目详细分析，说明结构、设计原因、收益和使用方式。
2. [`docs/INSTALL.md`](docs/INSTALL.md)：安装、验证与开发环境说明。
3. [`docs/CONFIG.md`](docs/CONFIG.md)：当前配置系统说明与官方写法。

## 快速开始

安装：

```bash
pip install -e .
```

完整安装：

```bash
pip install -e ".[all]"
```

开发环境：

```bash
pip install -e ".[all,dev]"
```

也可以使用安装脚本：

```bash
bash scripts/install.sh full
```

配置系统测试：

```bash
pytest tests/config -q
```

运行现有训练脚本：

```bash
python train_VAE.py
python train_GAN.py
python train_TwinFlow.py
```
