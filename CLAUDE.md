# XDL — 模块化深度学习框架

PyTorch 深度学习框架，组件注册系统 + 回调生命周期。Python 3.8+, PyTorch 1.12+, CUDA 可选。

## 行为准则

- **语言**：所有推理和回答均使用中文
- 做非 trivial 改动前，先用 `grep` / `find` / Agent 搜索代码库，理解现有模式再动手
- 多文件重构或架构级变更时，先进入规划模式制定方案
- 不确定文件位置或调用关系时，优先搜索而非猜测
- 涉及第三方库用法、API 变更、最佳实践等外部知识时，先查最新文档
- **脚本参数**：Python 脚本不要使用命令行参数解析库（如 `argparse` / `args`），优先使用 YAML 配置或代码内显式配置

## 架构

```text
xdl/
├── callbacks/   # 训练生命周期钩子
├── dataset/     # 数据集定义
├── loss/        # 损失函数
├── metric/      # 评估指标
├── model/       # 模型架构
├── optimizer/   # 优化器
├── scheduler/   # 学习率调度器
├── trainer/     # 训练器核心
└── utils/       # 注册系统与基础工具
```

其他目录：`config/`、`docs/`、`examples/`、`learn/`、`tests/`、`tools/`。

## 核心约束

- 所有组件通过注册系统注册，并在对应 `__init__.py` 中集中调用 `register_*("Name")(Class)`
- 当前注册类型包括：MODEL、DATASET、OPTIMIZER、SCHEDULER、LOSS、METRIC、TRANSFORM、COLLATE
- 回调优先级数值越小越先执行，默认值为 `999`
- 所有函数必须加类型注解

## 训练接入备忘

编写训练入口时，优先先看：

- `train_VAE.py`
- `train_TwinFlow.py`
- `xdl/trainer/trainer.py`
- `xdl/trainer/coreModel.py`

关键点：

- `Trainer.fit()` 会先调用 `model.setup("fit")`
- `CoreModel.training_step()` 是手动优化模式，不会自动 `zero_grad/backward/step`
- `configure_optimizers()` 在 `setup()` 之后调用
- 标准非 Accelerate 路径下，Trainer 只会自动迁移 `nn.Module` 属性到设备
- 复杂 batch 结构建议在 `training_step()` 内显式 `.to(self.device)`
- 自定义保存优先通过 callback 集成

## 子模块文档

### 顶层目录

- [config/CLAUDE.md](config/CLAUDE.md) — 训练与运行配置目录
- [data/CLAUDE.md](data/CLAUDE.md) — 本地数据目录
- [docs/CLAUDE.md](docs/CLAUDE.md) — 仓库文档目录
- [downloads/CLAUDE.md](downloads/CLAUDE.md) — 下载产物目录
- [examples/CLAUDE.md](examples/CLAUDE.md) — 示例脚本目录
- [infer/CLAUDE.md](infer/CLAUDE.md) — 推理脚本目录
- [learn/CLAUDE.md](learn/CLAUDE.md) — 学习与实验目录
- [others/CLAUDE.md](others/CLAUDE.md) — 其他资产目录
- [research/CLAUDE.md](research/CLAUDE.md) — 研究资料目录
- [scripts/CLAUDE.md](scripts/CLAUDE.md) — 仓库维护脚本目录
- [tests/CLAUDE.md](tests/CLAUDE.md) — 测试目录
- [third_party/CLAUDE.md](third_party/CLAUDE.md) — 第三方代码与资产目录
- [tools/CLAUDE.md](tools/CLAUDE.md) — 数据与工程工具目录
- [train/CLAUDE.md](train/CLAUDE.md) — 训练入口目录
- [xdl/CLAUDE.md](xdl/CLAUDE.md) — XDL 框架源码根目录
- [xqt/CLAUDE.md](xqt/CLAUDE.md) — 量化与实验脚本目录

### XDL 核心子模块

- [xdl/callbacks/CLAUDE.md](xdl/callbacks/CLAUDE.md) — 回调系统
- [xdl/config/CLAUDE.md](xdl/config/CLAUDE.md) — 配置构建子模块
- [xdl/dataset/CLAUDE.md](xdl/dataset/CLAUDE.md) — 数据集子模块
- [xdl/loss/CLAUDE.md](xdl/loss/CLAUDE.md) — 损失函数子模块
- [xdl/metric/CLAUDE.md](xdl/metric/CLAUDE.md) — 评估指标子模块
- [xdl/model/CLAUDE.md](xdl/model/CLAUDE.md) — 模型架构子模块
- [xdl/optimizer/CLAUDE.md](xdl/optimizer/CLAUDE.md) — 优化器子模块
- [xdl/scheduler/CLAUDE.md](xdl/scheduler/CLAUDE.md) — 学习率调度器子模块
- [xdl/trainer/CLAUDE.md](xdl/trainer/CLAUDE.md) — 训练器核心子模块
- [xdl/utils/CLAUDE.md](xdl/utils/CLAUDE.md) — 注册系统与工具子模块
