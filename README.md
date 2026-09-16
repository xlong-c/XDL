# XDL

XDL 是一个基于 PyTorch 的模块化深度学习框架.它提供三条核心能力:

- 组件注册:模型,数据集,loss,metric,optimizer,scheduler 统一注册
- 配置构建:用 YAML 把组件装配成 `TrainSetup`
- 训练编排:`CoreModel + Trainer + Callback` 管理训练生命周期

它不是新的张量框架,也不是完整实验平台,更像一套可扩展的项目骨架和训练组织层.

## 适用场景

- 研究型项目:频繁替换模型,损失,数据集和训练策略
- 工程型项目:把脚本式训练逐步收敛成结构化组件和统一配置

## 两条主路径

### 纯代码路径

参考入口:

- [train_VAE.py](train/pretrain/train_VAE.py)
- [train_GAN.py](train/pretrain/train_GAN.py)
- [train_TwinFlow.py](train/pretrain/train_TwinFlow.py)

典型写法:

```python
from xdl.trainer import CoreModel, Trainer

model = MyCoreModel()
trainer = Trainer(max_epochs=10, device="cuda")
trainer.fit(model, train_loader, val_loader)
```

`CoreModel.training_step()` 是手动优化模式;指标可用 `self.log("loss", value, prefix="train")` 记录为 `train_loss`,梯度累积可用 `is_accumulation_boundary` 等 helper 控制 step 时机.

### YAML 配置路径

典型写法:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 目录入口

- [xdl/](xdl/):框架源码
- [config/](config/):YAML 配置示例
- [examples/](examples/):脚本级示例
- [docs/md/](docs/md/): 给 agents 和开发者写代码前看的 MD 工作文档
- [docs/html/assets/](docs/html/assets/): `research/` HTML 页面共用的样式与主题资产
- [tests/](tests/):测试
- [tools/](tools/):工具脚本

## 文档入口

文档第一规则: `docs/md/` 是给 agents 和开发者写代码前看的工作文档,也是行为,字段,API 和兼容边界的事实源. `research/` 下的 HTML 页面只做面向人类的调研阅读层,不定义契约.

给 Codex 和开发者改代码前看的 MD 先从新总入口进入,兼容页只在需要旧链接时再看:

1. 总入口: [docs/md/index.md](docs/md/index.md)
2. 兼容摘要入口: [docs/md/README_SUMMARY.md](docs/md/README_SUMMARY.md)
3. 兼容详细事实源: [docs/md/README.md](docs/md/README.md)

先按新结构进入时,可以按这个顺序阅读:

1. [Markdown 总入口](docs/md/index.md)
2. [XDL 架构正文](docs/md/architecture/xdl.md)
3. [XDL 概念说明](docs/md/explanation/xdl-concepts.md)
4. [安装与验证](docs/md/usage/xdl-install-and-verify.md)
5. [配置工作流](docs/md/usage/xdl-config-workflows.md)
6. [XDL 工作流](docs/md/usage/xdl-workflows.md)

需要兼容旧结构或旧锚点时,再看:

1. [安装与验证摘要](docs/md/README_SUMMARY.md#安装与验证摘要)
2. [框架与训练摘要](docs/md/README_SUMMARY.md#框架与训练摘要)
3. [配置系统摘要](docs/md/README_SUMMARY.md#配置系统摘要)
4. [Dataset 摘要](docs/md/README_SUMMARY.md#dataset-摘要)
5. [API 与模块边界摘要](docs/md/README_SUMMARY.md#api-与模块边界摘要)
6. [文档与 HTML 规范摘要](docs/md/README_SUMMARY.md#文档与-html-规范摘要)
7. [优化方向摘要](docs/md/README_SUMMARY.md#优化方向摘要)

## 稳定公共入口

新代码优先依赖这些入口:

```python
from xdl.config import setup_from_yaml, TrainSetup
from xdl.trainer import CoreModel, Trainer, TrainSetupModel
from xdl.callbacks import Callback
```

完整公共 API 边界见 [docs/md/README.md#xdl-api-稳定边界](docs/md/README.md#xdl-api-稳定边界).历史文件级导入路径仍保持兼容,但推荐逐步迁移到子包入口.

## 快速开始

安装:

```bash
pip install -e .
```

完整安装:

```bash
pip install -e ".[all]"
```

验证配置主链路:

```bash
pytest tests/config -q
```

运行现有训练脚本:

```bash
python train/pretrain/train_VAE.py
python train/pretrain/train_GAN.py
python train/pretrain/train_TwinFlow.py
```

## Wheel 安装后的单文件入口

如果只有安装后的 `xdl` 包,没有源码仓库,可以直接查看随 wheel 分发的用法摘要:

```bash
python -m xdl.usage
xdl-usage
```

Python 内可用:

```python
import xdl

print(xdl.get_usage_text())
```

这份单文件入口也会随 wheel 分发,可作为没有源码仓库时的快速使用说明.
