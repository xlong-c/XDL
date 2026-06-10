# XDL 安装与验证

本文档只讲当前仓库里真实可用的安装方式、验证方式和运行入口，不重复解释框架结构。

## 1. 环境要求

- Python `>=3.8`
- PyTorch `>=1.12`
- Linux / macOS / Windows 均可，当前仓库主要在 Linux 环境下维护
- GPU 不是必需；如果要跑 CUDA 训练，先按本机 CUDA 版本安装对应 PyTorch wheel

## 2. 推荐安装方式

### 基础安装

适合阅读代码、运行配置构建链路和最小实验：

```bash
git clone https://gitee.com/xlong_t/xdl.git
cd xdl
pip install -e .
```

### 完整安装

适合需要日志、增强和其他可选能力的环境：

```bash
pip install -e ".[all]"
```

### 开发环境安装

适合要跑测试和开发检查：

```bash
pip install -e ".[all,dev]"
```

### 使用仓库脚本

```bash
bash scripts/install.sh base
bash scripts/install.sh full
bash scripts/install.sh dev
```

## 3. PyTorch / CUDA 安装建议

如果本机还没有合适的 PyTorch，先按你的 CUDA 或 CPU 环境安装 wheel，再安装 XDL。

示例：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[all]"
```

```bash
# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

这一步以你的驱动和 CUDA 环境为准，不建议机械照搬固定版本。

## 4. 安装后怎样验证

### 验证包导入

```bash
python - <<'PY'
import importlib.metadata
import xdl

print(importlib.metadata.version("xdl"))
print("import xdl ok")
PY
```

### 查看安装包内用法说明

如果是从 wheel 安装，源码仓库里的 `docs/`、`examples/`、`config/` 不一定存在。安装包内固定提供一份单文件入口：

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以读取：

```python
import xdl

print(xdl.get_usage_text())
```

### 验证配置构建主链路

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

如果能正常输出模型、优化器、loss 和数据集长度，说明 schema、resolver、builder 和 `TrainSetup` 主链路已经打通。

### 跑现有测试

```bash
pytest tests/config -q
```

如果你改动了 registry、dataset、loss、metric，也应额外跑对应子目录测试。

### 跑开发检查

开发环境使用 Ruff 负责格式化、import 排序和基础 lint, 使用 Pyright/Pylance 负责类型检查:

```bash
ruff check xdl tests config examples infer train research/diffusion-models-survey-2025/sana_hair_lora
pyright
```

需要统一格式时, 再按目录运行 `ruff format <paths>`。

## 5. 当前可直接运行的入口

仓库里当前清晰可见的训练入口主要有：

```bash
python train_VAE.py
python train_GAN.py
python train_TwinFlow.py
```

这些入口代表纯代码路径。

如果要走 YAML 配置路径，当前推荐直接在 Python 中调用：

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 6. 可选依赖与现实边界

一些能力依赖可选包：

- `tensorboard`
- `wandb`
- `accelerate`
- `opencv-python`
- `albumentations`
- `loguru`

如果你只做最小验证，不需要一次装满全部依赖；如果要用完整日志或增强链路，优先用：

```bash
pip install -e ".[all]"
```

或：

```bash
bash scripts/install.sh full
```

## 7. 下一步阅读

安装和验证完成后，建议继续看：

1. [README.md](../README.md)
2. [XDL.md](XDL.md)
3. [CONFIG.md](CONFIG.md)
4. [API.md](API.md)
