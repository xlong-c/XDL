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
from xdl.trainer import Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

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
- [docs/](docs/)：长期文档
- [tests/](tests/)：测试
- [tools/](tools/)：工具脚本

## 文档入口

优先阅读：

1. [docs/INSTALL.md](docs/INSTALL.md)
2. [docs/XDL.md](docs/XDL.md)
3. [docs/CONFIG.md](docs/CONFIG.md)
4. [docs/xdl-functional-boundary.md](docs/xdl-functional-boundary.md)

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
