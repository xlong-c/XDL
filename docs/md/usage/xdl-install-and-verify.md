# XDL 安装与验证

本文汇总 `XDL` 当前真实可用的安装方式, 验证方式和运行入口.

## 负责什么

- 说明环境要求和安装方式.
- 给出安装后的验证命令.
- 指向当前可直接运行的训练入口.

## 不负责什么

- 不重复解释框架整体架构.
- 不承载完整配置系统说明.
- 不列出所有扩展工作流.

## 环境要求

- Python `>=3.12`
- PyTorch `>=1.12`
- Linux / macOS / Windows 均可, 当前仓库主要在 Linux 环境下维护
- GPU 不是必需; 如果要跑 CUDA 训练, 先按本机 CUDA 版本安装对应 PyTorch wheel

## 推荐安装方式

### 基础安装

```bash
git clone https://gitee.com/xlong_t/xdl.git
cd xdl
pip install -e .
```

### 完整安装

```bash
pip install -e ".[all]"
```

### 开发环境安装

```bash
pip install -e ".[all,dev]"
```

### 使用仓库脚本

```bash
bash scripts/install.sh base
bash scripts/install.sh full
bash scripts/install.sh dev
```

## 安装后怎样验证

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

```bash
python -m xdl.usage
xdl-usage
```

Python 内也可以读取:

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

如果能正常输出模型, 优化器, loss 和数据集长度, 说明 schema, resolver, builder 和 `TrainSetup` 主链路已经打通.

### 跑现有测试

```bash
pytest tests/config -q
```

### 跑开发检查

```bash
ruff check xdl tests config examples infer train
pyright
```

## 当前可直接运行的入口

仓库里当前清晰可见的训练入口主要有:

```bash
python train/pretrain/train_VAE.py
python train/pretrain/train_GAN.py
python train/pretrain/train_TwinFlow.py
```

如果要走 YAML 配置路径, 当前推荐直接在 Python 中调用:

```python
from xdl.config import setup_from_yaml
from xdl.trainer import Trainer

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
model = setup.create_model()
trainer = Trainer.from_setup(setup)
trainer.fit(model, setup.train_loader, setup.val_loader)
```

## 继续阅读

- [../architecture/xdl.md](../architecture/xdl.md)
- [../architecture/api-boundary.md](../architecture/api-boundary.md)
- [index.md](index.md)
