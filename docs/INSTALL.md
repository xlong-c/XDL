# XDL 安装与验证

本文档只说明当前仓库里真实可用的安装方式、验证方式和已知边界，不再保留已经失效的 `requirements.txt`、`train.py` 或 `xdl.__version__` 之类旧入口说明。

## 1. 环境要求

- Python `>=3.8`
- PyTorch `>=1.12`
- Linux / macOS / Windows 均可，当前仓库主要在 Linux 环境下维护
- GPU 不是必需，但如果要跑 CUDA 训练，建议先按本机 CUDA 版本安装对应 PyTorch wheel

如果你的机器还没有合适的 PyTorch，先参考 PyTorch 官方安装页安装，再继续下面的步骤。

## 2. 推荐安装方式

### 基础安装

适合阅读代码、运行核心配置构建链路、做最小实验。

```bash
git clone https://gitee.com/xlong_t/xdl.git
cd xdl

pip install -e .
```

### 完整安装

适合需要视觉增强、日志、加速和其他可选能力的环境。

```bash
pip install -e ".[all]"
```

### 开发环境安装

适合要跑测试、格式化和静态检查的开发环境。

```bash
pip install -e ".[all,dev]"
```

### 使用安装脚本

仓库内提供了安装脚本：

```bash
bash scripts/install.sh base
bash scripts/install.sh full
bash scripts/install.sh dev
```

其中：

- `base` 对应基础安装
- `full` / `all` 对应完整安装
- `dev` 对应开发环境安装

## 3. CUDA / CPU 安装建议

如果需要显式安装 CUDA 或 CPU 版 PyTorch，可以先安装对应 wheel，再安装 XDL：

```bash
# 示例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[all]"
```

```bash
# 示例：CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

这一步要以你的实际驱动和 CUDA 环境为准，不建议直接照搬固定版本。

## 4. 安装后如何验证

### 验证包已正确安装

```bash
python - <<'PY'
import importlib.metadata
import xdl

print(importlib.metadata.version("xdl"))
print("import xdl ok")
PY
```

当前包版本可以通过 `importlib.metadata.version("xdl")` 获取，不应再使用不存在的 `xdl.__version__`。

### 验证配置系统主链路

这是当前最稳妥的仓库级 smoke test：

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

如果输出类似：

- `SimpleMLP`
- `Adam`
- `CrossEntropyLoss`
- `500`

说明 schema 解析、`${...}` 引用解析、builder 构建和 dataloader 装配都已经打通。

### 运行配置测试

```bash
pytest tests/config -q
```

当前仓库里已经存在的测试主要集中在配置系统：

- `tests/config/test_builder.py`
- `tests/config/test_schema.py`
- `tests/config/test_setup.py`

## 5. 当前可直接运行的入口

仓库当前可见的训练脚本入口是：

```bash
python train_VAE.py
python train_GAN.py
python train_TwinFlow.py
```

这些脚本代表“纯代码方式”的使用路径，不是统一的 CLI 框架入口。也就是说，当前项目并不存在统一的 `train.py --config ...` 官方入口，不应在文档中继续这样描述。

如果你要使用 YAML 配置方式，当前推荐直接在 Python 中调用：

```python
from xdl.config import setup_from_yaml

setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")
```

## 6. 可选依赖与已知边界

XDL 的模块化结构已经比较清晰，但安装层仍有一些现实边界，文档需要明确写出来：

- `tensorboard`、`wandb`、`accelerate` 等功能依赖可选包
- 一些数据集或增强链路依赖 `opencv-python`、`albumentations` 等三方库
- 日志回调相关代码会使用 `loguru`，如果你的环境里没有它，需要手动安装：

```bash
pip install loguru
```

也可以直接使用：

```bash
bash scripts/install.sh full
```

安装脚本会额外补装 `loguru`。

## 7. 常用开发命令

```bash
black xdl tests
isort xdl tests
flake8 xdl
mypy xdl
pytest tests/config -q
```

## 8. 下一步阅读

安装完成后，建议按下面顺序继续：

1. [README.md](../README.md)
2. [XDL.md](XDL.md)
3. [CONFIG.md](CONFIG.md)
