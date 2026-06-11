# XDL

XDL 是一个基于 PyTorch 的模块化深度学习框架。它提供三条核心能力：

- 组件注册：模型、数据集、loss、metric、optimizer、scheduler 统一注册
- 配置构建：用 YAML 把组件装配成 `TrainSetup`
- 训练编排：`CoreModel + Trainer + Callback` 管理训练生命周期

它不是新的张量框架，也不是完整实验平台，更像一套可扩展的项目骨架和训练组织层。

## 适用场景

- 研究型项目：频繁替换模型、损失、数据集和训练策略
- 工程型项目：把脚本式训练逐步收敛成结构化组件和统一配置

## 两条主路径

### 纯代码路径

参考入口：

- [train_VAE.py](train_VAE.py)
- [train_GAN.py](train_GAN.py)
- [train_TwinFlow.py](train_TwinFlow.py)

典型写法：

```python
from xdl.trainer import CoreModel, Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

`CoreModel.training_step()` 是手动优化模式；指标可用 `self.log("loss", value, prefix="train")` 记录为 `train_loss`，梯度累积可用 `is_accumulation_boundary` 等 helper 控制 step 时机。

### YAML 配置路径

典型写法：

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 目录入口

- [xdl/](xdl/)：框架源码
- [config/](config/)：YAML 配置示例
- [examples/](examples/)：脚本级示例
- [docs/](docs/): 长期文档,包含 MD 工作文档和 HTML 阅读版
- [tests/](tests/)：测试
- [tools/](tools/)：工具脚本

## 文档入口

给用户看的阅读版:

1. [docs/html/index.html](docs/html/index.html)
2. [docs/html/dataset-structure.html](docs/html/dataset-structure.html)

给 Codex 和开发者改代码前看的高密度 MD:

1. [docs/README.md](docs/README.md)
2. [docs/INSTALL.md](docs/INSTALL.md)
3. [docs/XDL.md](docs/XDL.md)
4. [docs/CONFIG.md](docs/CONFIG.md)
5. [docs/DATASET.md](docs/DATASET.md)
6. [docs/API.md](docs/API.md)
7. [docs/xdl-functional-boundary.md](docs/xdl-functional-boundary.md)

## 稳定公共入口

新代码优先依赖这些入口：

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

完整公共 API 边界见 [docs/API.md](docs/API.md)。历史文件级导入路径仍保持兼容，但推荐逐步迁移到子包入口。

## 快速开始

安装：

```bash
pip install -e .
```

完整安装：

```bash
pip install -e ".[all]"
```

验证配置主链路：

```bash
pytest tests/config -q
```

运行现有训练脚本：

```bash
python train_VAE.py
python train_GAN.py
python train_TwinFlow.py
```

## Wheel 安装后的单文件入口

如果只有安装后的 `xdl` 包，没有源码仓库，可以直接查看随 wheel 分发的用法摘要：

```bash
python -m xdl.usage
xdl-usage
```

Python 内可用：

```python
import xdl

print(xdl.get_usage_text())
```

这份单文件入口也会随 wheel 分发，可作为没有源码仓库时的快速使用说明。
