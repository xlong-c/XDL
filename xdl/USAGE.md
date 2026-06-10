# XDL 直接用法

这是一份随 `xdl` wheel 一起安装的单文件入口。目标是让人或大模型在只拿到已安装包时，也能快速知道 XDL 的真实使用方式，而不需要先探索整个仓库。

## 先读这个结论

- XDL 是 PyTorch 上层的训练组织层，不是新的张量框架。
- 两条主路径：`CoreModel + Trainer` 纯代码训练，或 `setup_from_yaml()` 配置化构建。
- `CoreModel.training_step()` 是手动优化模式，必须自己执行 `zero_grad()`、反向传播和 `step()`。
- 所有可配置组件优先通过 registry 注册，并在 YAML 中用 `target: "registry:Name"` 引用。
- 大模型、diffusers pipeline、PEFT LoRA 等重组件优先在 `CoreModel.setup(stage)` 中懒加载。

## 安装后怎么查看这份文件

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以直接读取：

```python
import xdl

print(xdl.get_usage_text())
```

## 路径一：纯代码训练

适合 VAE、GAN、自定义大模型微调、diffusers/PEFT 等训练逻辑明显不是标准分类流程的任务。

最小骨架：

```python
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from xdl.trainer import CoreModel, Trainer


class MyModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        optimizer = self.optimizers[0]

        optimizer.zero_grad()
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        self.manual_backward(loss)
        optimizer.step()

        self.log("loss", loss.detach(), prefix="train")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.net.parameters(), lr=1e-3)


x = torch.randn(128, 10)
y = torch.randint(0, 2, (128,))
train_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

model = MyModel()
trainer = Trainer(max_epochs=3, device="cuda" if torch.cuda.is_available() else "cpu")
trainer.fit(model, train_loader)
```

关键规则：

- `Trainer.fit(model, train_loader, val_loader=None, inference_data=None)` 是主入口。
- `Trainer.fit()` 会先调用 `model.setup("fit")`，再做设备、优化器和 callback setup。
- `configure_optimizers()` 在 `setup()` 之后被调用；依赖懒加载模块的优化器要在这里创建。
- Trainer 不会自动优化；`training_step()` 内要自己完成优化流程。
- 记录指标用 `self.log("name", value)`，`value` 支持 Python 数值或单元素 `torch.Tensor`，callback 和进度条会读取这些指标；`self.log("loss", value, prefix="train")` 会生成兼容旧脚本的 `train_loss` 键。
- 需要梯度裁剪时，在 `training_step()` 中调用 `self.clip_gradients(...)`。
- 手动梯度累积可以用 `self.accumulation_steps`、`self.is_accumulation_start`、`self.is_accumulation_boundary`、`self.should_optimizer_step`，这些 helper 基于 Trainer 的 `gradient_accumulation_steps`。

梯度累积写法：

```python
def training_step(self, batch: Any, batch_idx: int) -> None:
    x, y = batch
    optimizer = self.optimizers[0]

    logits = self.net(x)
    loss = self.loss_fn(logits, y)
    self.manual_optimization_step(
        loss,
        optimizer=optimizer,
        model=self.net,
        max_grad_norm=1.0,
    )

    self.log("loss", loss.detach(), prefix="train")
```

## 大模型训练骨架

大模型不要在脚本 import 阶段或 `__init__()` 里急切加载全部权重。推荐只保存配置，在 `setup()` 中懒加载，并把需要 Trainer 迁移设备或保存的 `nn.Module` 挂成 `self.xxx` 属性。

```python
from typing import Any, Dict

from torch.optim import AdamW

from xdl.trainer import CoreModel


class LargeModelFinetune(CoreModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.transformer = None
        self.vae = None
        self.text_encoder = None

    def setup(self, stage: str) -> None:
        if self.transformer is not None:
            return

        # 在这里加载 diffusers/transformers/PEFT 等重组件。
        # 如果有 pipeline，也要把底层 nn.Module 注册成 self 属性。
        # self.pipe = ...
        # self.transformer = self.pipe.transformer
        # self.vae = self.pipe.vae
        # self.text_encoder = self.pipe.text_encoder

    def training_step(self, batch: Any, batch_idx: int) -> None:
        optimizer = self.optimizers[0]
        optimizer.zero_grad()

        # Trainer 会递归迁移 Tensor / dict / list / tuple / dataclass。
        # 自定义第三方对象仍建议在这里显式处理设备。
        # loss = ...
        # self.manual_backward(loss)
        # optimizer.step()
        # self.log("loss", loss.detach(), prefix="train")

    def configure_optimizers(self) -> AdamW:
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return AdamW(trainable_params, lr=float(self.config["lr"]))
```

大模型注意事项：

- 标准非 Accelerate 路径只会迁移 `CoreModel.__dict__` 中的 `nn.Module` 属性；普通 pipeline 不是 `nn.Module`。
- 普通 pipeline 可通过 `configure_device_objects()` 返回 `{属性名: 对象}` 让 Trainer 调用 `.to(device)`.
- Trainer 会递归迁移 batch 里的 `Tensor / dict / list / tuple / dataclass`；第三方自定义对象仍建议在 `training_step()` 内显式 `.to(self.device)`。
- `Trainer` 构造参数里的 `gradient_accumulation_steps` 不会替代手动优化逻辑；自己实现累积时用 `self.accumulation_steps`、`self.is_accumulation_start`、`self.is_accumulation_boundary` 控制 backward/step。
- 使用 Accelerate 时仍保持手动优化语义, 避免在任务代码里再写一套与 Trainer 不一致的私有 step 计数.
- `on_train_step_start()` 会在 `training_step()` 前递增 `_total_train_steps`；新代码优先读 `self.micro_step`，不要直接依赖私有字段。
- diffusers/PEFT LoRA 权重保存优先写专用 callback，调用 `save_pretrained()` 或框架自己的保存 API。

## 路径二：YAML 配置训练

适合标准模型、数据集、优化器、loss、metric 都能通过配置描述的任务。

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

最小 YAML 结构：

```yaml
config_version: 1

runtime:
  device: "cuda"

trainer:
  max_epochs: 10
  batch_size: 32

model:
  target: "registry:simple_mlp"
  params:
    input_size: 784
    hidden_size: 128
    num_classes: 10

dataloader_defaults:
  batch_size: ${trainer.batch_size}
  num_workers: 0

train_dataset:
  target: "registry:SyntheticClassificationDataset"
  params:
    num_samples: 500
    input_shape: [784]
    num_classes: 10

train_dataloader:
  dataset: ${train_dataset}
  params:
    shuffle: true

optimization:
  optimizer:
    target: "torch.optim:AdamW"
    params:
      lr: 0.001

loss:
  - target: "torch.nn:CrossEntropyLoss"
    params: {}

metrics:
  - target: "registry:Accuracy"
    params:
      num_classes: 10

callbacks:
  - target: "xdl.callbacks:SaveTrainableStateCallback"
    params:
      dirpath: "./outputs/adapters"
```

YAML 规则：

- 组件格式是 `target + params`。
- `registry:Name` 表示从 XDL registry 查找。
- `torch.optim:AdamW` 这类格式表示 `module:name` 动态导入。
- `setup_from_yaml()` 返回 `TrainSetup`，包含 `model / optimizer / train_loader / val_loader / loss_fn / metrics / scheduler`。
- YAML 中可用 `${xdl.config_dir}`, `${xdl.project_root}`, `${xdl.join_path:...}` 和 `${xdl.abspath:...}` 组织路径.
- `callbacks` 使用 import path 构建, 会被 `Trainer.from_setup(setup)` 自动附加。
- `setup.create_model()` 会包装成 `TrainSetupModel`，直接适配 `Trainer.fit()`。

## 注册自定义组件

所有可复用组件优先通过注册系统接入。注册集中放在对应子包的 `__init__.py`。

```python
from xdl.utils.registry import register_model

from .my_model import MyModel

register_model("MyModel")(MyModel)
```

然后在 YAML 中引用：

```yaml
model:
  target: "registry:MyModel"
  params:
    hidden_size: 512
```

支持的 registry 类型：

- `MODEL`
- `DATASET`
- `OPTIMIZER`
- `SCHEDULER`
- `LOSS`
- `METRIC`
- `TRANSFORM`
- `COLLATE`

## Callback 和保存

Callback 用来处理日志、采样、checkpoint、LoRA 保存等横切逻辑。优先级数值越小越先执行，未指定默认 `999`。

```python
from typing import TYPE_CHECKING

from xdl.callbacks import Callback

if TYPE_CHECKING:
    from xdl.trainer import CoreModel, Trainer


class SaveLoraCallback(Callback):
    priority = 100

    def on_train_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        # core_module.pipe.save_lora_weights(...)
        pass
```

普通 `ModelCheckpoint` 保存 `CoreModel` 的 state_dict, optimizer 状态和 callback state; 第三方格式权重通常用 `SaveTrainableStateCallback` 调用任务模型自己的保存方法.

## 公共 API 边界

新代码优先依赖这些稳定入口：

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

历史文件级导入路径当前继续兼容，但不建议新代码继续扩散。完整 Stable / Provisional / Internal API 边界见源码仓库的 `docs/API.md`。

## 常见坑

- 不要期待 `training_step()` 自动优化；XDL 当前是手动优化模式。
- 不要把大模型权重加载放在模块 import 阶段。
- 不要只把 diffusers pipeline 挂到 `self.pipe` 就指望 Trainer 迁移设备；底层 `nn.Module` 也要挂到 `self`。
- 不要让第三方自定义 batch 对象完全依赖 Trainer 自动迁移；内置容器会递归迁移，但自定义对象需要在 `training_step()` 显式处理。
- 新训练脚本不要引入命令行参数解析库；优先 YAML 配置，或用环境变量选择 YAML 文件。
- 只安装 wheel 时，仓库里的 `examples/`、`docs/`、`config/` 不一定存在；以本文件和包内公开 API 为准。
