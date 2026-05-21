# XDL 包 — 框架源码根目录

这里是 XDL 的核心 Python 包实现，负责组件注册、配置构建、训练编排和基础设施。

## 进入此目录时优先关注

- 改动属于组件层、配置层、训练层还是基础设施层
- 是否会影响 registry、`setup_from_yaml()` 或 `Trainer.fit()` 主链路
- 当前实现是否已经在对应子目录有局部文档

## 关键结构

- `config/`：YAML 到 `TrainSetup`
- `trainer/`：`CoreModel`、`Trainer`、`TrainSetupModel`
- `callbacks/`：日志、检查点、监控、进度条等回调
- `model/`：模型与工厂函数
- `dataset/`：数据集、transform、collate
- `loss/`：损失函数
- `metric/`：评估指标
- `optimizer/`：优化器
- `scheduler/`：学习率调度器
- `utils/`：registry、checkpoint、tiling 与通用工具

## 本地约束

- 所有组件通过对应 `__init__.py` 集中注册
- 生命周期相关改动先回看 `trainer/` 和 `callbacks/`
- 配置相关改动先回看 `config/schema.py`、`resolver.py`、`builder.py`、`setup.py`
- 基础设施层改动优先考虑兼容性，而不是局部重写
