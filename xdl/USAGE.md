# XDL 极小使用手册

这是随 `xdl` wheel 分发的单文件说明. 目标是让别的项目只安装包, 也能快速接入 XDL 的训练能力.

## 安装后查看

```bash
xdl-usage
python -m xdl.usage
```

Python 内读取:

```python
import xdl

print(xdl.get_usage_text())
```

## XDL 是什么

- XDL 是 PyTorch 上层的训练组织层, 不是新的张量框架.
- 常用路径有两条: `CoreModel + Trainer` 纯代码训练, 或 `setup_from_yaml()` 配置化构建.
- `CoreModel.training_step()` 是手动优化模式, 需要自己调用 `zero_grad()`, `manual_backward()`, `optimizer.step()`.
- 多卡和混合精度通过 `torchrun + Trainer(precision=..., accelerate_config=...)` 使用.
- 横切能力优先用 Callback, 如 checkpoint, tqdm, TensorBoard, 自定义保存.

## 最小纯代码训练

```python
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from xdl.trainer import CoreModel, Trainer


class Task(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        loss = self.loss_fn(self.net(x), y)
        self.manual_optimization_step(loss, max_grad_norm=1.0)
        self.log("loss", loss.detach(), prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        loss = self.loss_fn(self.net(x), y)
        self.log("loss", loss.detach(), prefix="val")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.net.parameters(), lr=1e-3)


x = torch.randn(128, 10)
y = torch.randint(0, 2, (128,))
loader = DataLoader(TensorDataset(x, y), batch_size=32)

trainer = Trainer(max_epochs=3, device="cuda")
trainer.fit(Task(), loader, loader)
```

## 大模型接入要点

```python
from typing import Any

from xdl.trainer import CoreModel


class BigTask(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def setup(self, stage: str) -> None:
        if self.model is None:
            self.model = load_big_model_or_pipeline()

    def configure_device_objects(self) -> dict[str, Any]:
        return {"model": self.model}
```

- `Trainer.fit()` 会先调用 `model.setup("fit")`, 再设置设备和优化器.
- `configure_optimizers()` 在 `setup()` 之后调用, 优化器可以依赖懒加载模块.
- 标准 batch 设备迁移支持 `Tensor / dict / list / tuple / dataclass`.
- 第三方 pipeline 或非 `nn.Module` 对象, 可在 `configure_device_objects()` 中声明.

## 多卡和混合精度

单机 4 卡:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

代码里:

```python
from xdl.trainer import Trainer

trainer = Trainer(
    max_epochs=10,
    precision="bf16",
    gradient_accumulation_steps=4,
    accelerate_config={"mixed_precision": "bf16"},
)
```

FSDP:

```python
trainer = Trainer(max_epochs=10, fsdp=2, accelerate_config={"mixed_precision": "bf16"})
```

训练步里仍使用 XDL 的手动优化 helper:

```python
loss = compute_loss(batch)
stepped = self.manual_optimization_step(loss, max_grad_norm=1.0)
if stepped:
    scheduler.step()
```

## YAML 配置训练

适合模型, 数据集, 优化器, loss 都能配置化描述的任务.

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/train.yaml", device="cuda")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

YAML 中组件通常使用 `target + params`:

```yaml
model:
  target: "registry:MyModel"
  params:
    hidden_dim: 256

optimizer:
  target: "torch.optim.AdamW"
  params:
    lr: 1.0e-4
```

## 注册组件

别的项目可以在自己的包里注册组件, 然后在入口处 import 这个注册模块.

```python
from torch import nn

from xdl.utils.registry import register_model


@register_model("MyModel")
class MyModel(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Linear(hidden_dim, hidden_dim)
```

常用注册函数:

```python
from xdl.utils.registry import (
    register_collate,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_scheduler,
    register_transform,
)
```

## 回调和 checkpoint

```python
from xdl.callbacks import ModelCheckpoint, TqdmCallback
from xdl.trainer import Trainer

callbacks = [
    TqdmCallback(log_frequency=10),
    ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    ),
]

trainer = Trainer(max_epochs=10, callbacks=callbacks)
```

自定义 Callback:

```python
from typing import Any

from xdl.callbacks import Callback
from xdl.trainer import CoreModel, Trainer


class SavePreview(Callback):
    def on_validation_end(self, trainer: Trainer, core_module: CoreModel) -> None:
        if core_module.is_main_process():
            save_preview_images()
```

## 常用接口速查

稳定导入:

```python
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.config import TrainSetup, setup_from_yaml
from xdl.callbacks import Callback, ModelCheckpoint, TqdmCallback
from xdl.utils import resolve_dtype, save_yaml, seed_everything
```

`CoreModel` 常用方法和属性:

- `setup(stage)`: 懒加载大模型, tokenizer, pipeline.
- `training_step(batch, batch_idx)`: 手动训练逻辑.
- `validation_step(batch, batch_idx)`: 验证逻辑.
- `configure_optimizers()`: 返回 optimizer 或 `([optimizers], [schedulers])`.
- `manual_backward(loss)`: 兼容单卡和 Accelerate 的反向传播.
- `manual_optimization_step(loss, max_grad_norm=...)`: 推荐的单优化器手动优化模板.
- `log(name, value, prefix="train")`: 记录单个指标, 如 `train_loss`.
- `log_metrics(metrics, prefix="val")`: 批量记录指标.
- `is_main_process()`: 分布式下判断主进程.
- `accumulation_steps`, `is_accumulation_start`, `is_accumulation_boundary`, `should_optimizer_step`: 梯度累积 helper.

`Trainer` 常用参数:

- `max_epochs`
- `device`
- `precision`
- `gradient_accumulation_steps`
- `grad_clip_max_norm`
- `callbacks`
- `accelerate_config`
- `fsdp`

## 推荐给外部项目的最小入口

```python
import os

from xdl.trainer import Trainer

from my_project.config import load_config
from my_project.data import build_dataloaders
from my_project.model import MyTask


def main() -> int:
    cfg = load_config(os.environ["XDL_CONFIG"])
    train_loader, val_loader = build_dataloaders(cfg.data)
    model = MyTask(cfg.model)

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        accelerate_config={"mixed_precision": cfg.train.mixed_precision},
    )
    trainer.fit(model, train_loader, val_loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行:

```bash
XDL_CONFIG=configs/train.yaml torchrun --standalone --nproc_per_node=4 train.py
```
