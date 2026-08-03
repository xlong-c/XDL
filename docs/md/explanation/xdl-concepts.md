# XDL 概念说明

本文解释 `XDL` 的核心概念和理解路径. 它帮助人类和 agent 先建立认知, 再进入架构或使用文档.

## 这是什么

`XDL` 是一个基于 PyTorch 的模块化深度学习框架. 它不替代 PyTorch, 而是在其上提供更稳定的项目组织能力.

## 为什么需要

训练项目往往会同时遇到这些问题:

- 组件多, 命名和接入方式容易漂移
- 配置, 代码和训练生命周期容易各走各路
- 训练脚本越写越厚, 难以扩展和交接

`XDL` 的目标就是把这些问题收拢到统一结构里:

- 注册系统
- 配置构建
- 训练编排
- callback 扩展

## 核心概念

### 1. 组件注册

模型, 数据集, loss, metric, optimizer, scheduler, transform, collate 都通过 registry 暴露. 这保证了组件发现和配置构建有统一入口.

### 2. 配置构建

`xdl.config` 负责把 YAML 或结构化配置组装成 `TrainSetup`. 它解决的是"如何把组件和参数拼起来", 不是"如何替你完成训练任务".

### 3. 训练编排

`xdl.trainer` 负责训练生命周期. 其中:

- `CoreModel` 承载任务逻辑
- `Trainer` 承载循环与调度
- `TrainSetupModel` 承载配置流到任务逻辑的桥接

### 4. 手动优化

`XDL` 当前默认采用手动优化模式. 这意味着训练步的反向传播, 梯度裁剪和优化器步进由任务逻辑显式控制.

### 5. 文档分层

`XDL` 的长期文档现在分成三层:

- 架构层: 讲边界和系统结构
- 说明层: 讲概念和理解路径
- 使用层: 讲命令, workflow 和交接

## 与相近概念的区别

- 它不是新的张量计算框架
- 它不是统一 CLI 平台
- 它不是自动优化平台
- 它不是把所有实验流程都包办掉的系统

## 常见误区

- 不要把 registry 当成配置解析器
- 不要把 callback 当成主要训练逻辑容器
- 不要假设 YAML 路径一定比显式代码路径更高级
- 不要只看目录名猜生命周期, 先看真实训练入口

## 推荐理解路径

1. [../architecture/xdl.md](../architecture/xdl.md)
2. [../usage/xdl-install-and-verify.md](../usage/xdl-install-and-verify.md)
3. [../usage/xdl-workflows.md](../usage/xdl-workflows.md)
4. [../architecture/api-boundary.md](../architecture/api-boundary.md)
