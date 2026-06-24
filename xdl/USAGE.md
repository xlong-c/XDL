# XDL 直接用法

这是随 `xdl` wheel 安装的单文件入口. 目标是让人或大模型在只拿到已安装包时,也能快速知道 XDL 的真实用法.

## 先读结论

- XDL 是 PyTorch 上层的训练组织层,不是新的张量框架.
- 两条主路径: `CoreModel + Trainer` 纯代码训练,或 `setup_from_yaml()` 配置化构建.
- `CoreModel.training_step()` 是手动优化模式,必须自己执行 `zero_grad()`,反向传播和 `step()`.
- 可配置组件优先通过 registry 注册,并在 YAML 中用 `target: "registry:Name"` 引用.

## 安装后查看

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以直接读取:

```python
import xdl

print(xdl.get_usage_text())
```

## 路径一: 纯代码训练

适合需要手写训练逻辑的任务.

```python
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

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
        loss = self.loss_fn(self.net(x), y)
        self.manual_backward(loss)
        optimizer.step()
        self.log("loss", loss.detach(), prefix="train")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.net.parameters(), lr=1e-3)
```

## 路径二: YAML 配置训练

适合标准模型,数据集,优化器和 loss 都能通过配置描述的任务.

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 核心规则

- `Trainer.fit()` 先调用 `model.setup("fit")`,再做设备和优化器 setup.
- `configure_optimizers()` 在 `setup()` 之后被调用.
- 手动梯度累积优先用 `self.accumulation_steps`, `self.is_accumulation_start`, `self.is_accumulation_boundary`, `self.should_optimizer_step`.
- 大模型、diffusers pipeline、PEFT LoRA 等重组件优先在 `CoreModel.setup(stage)` 中懒加载.
- 只安装 wheel 时,仓库里的 `examples/`, `docs/`, `config/` 不一定存在;以本文件和包内公开 API 为准.
