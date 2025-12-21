<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于PyTorch的深度学习框架项目, 主要用于计算机视觉任务。框架采用模块化设计, 包含完整的训练、评估和部署流程。

请全部使用UTF-8编码

### 模块化架构

```
xdl/
├── model/          # 模型定义(ResNet、VGG、ViT等)
├── dataset/        # 数据集处理
├── trainer/        # 训练逻辑
│   ├── trainer.py           # 主训练器类
│   ├── coreModel.py         # 核心模型基类
│   ├── trainer_state.py     # 训练状态管理
│   └── accelerate_config.py # 加速器配置
├── optimizer/      # 优化器
├── scheduler/      # 学习率调度器
├── loss/           # 损失函数
├── metric/         # 评估指标
├── callbacks/      # 回调函数(Checkpoint、EarlyStopping等)
└── utils/          # 工具函数
    ├── registry.py    # 注册表系统
    ├── config.py      # 配置管理
    └── tools.py       # 通用工具
```

### 实验管理

- **配置文件驱动**: 所有实验参数通过YAML文件管理
- **日志系统**: 统一日志到TensorBoard、文件和控制台
- **检查点**: 自动保存最佳模型和训练状态到`others/checkpoints/`
- **结果存储**: 非代码文件统一存放到`others/`目录

## 代码规范

- 使用中文注释, 但符号使用英文的半角符号(如冒号(:)、逗号(,)、句号(.)等)
- 专业术语保持英文(如loss、accuracy、epoch等)
- 遵循现有代码风格
- 如果没有显性要求, 尽量不要写兼容代码
- 修改时保持最小修改原则

## 目录结构说明

- `others/` - 非代码文件存放目录(checkpoints、data、results、logs等)
- `docs/` - 文档相关文件
- `scripts/` - 执行脚本
- `config/` - 配置文件
- `tools/` - 图像处理和数据预处理工具脚本
- `examples/` - 示例代码
- `tests/` - 测试代码

## 重要文件

- `train_mnist.py` - MNIST训练完整示例
- `examples/final_accelerate_test.py` - 加速器集成测试
- `config/vgg_cifar100.yaml` - 配置文件示例
- `xdl/utils/registry.py` - 注册表系统核心
- `xdl/trainer/trainer.py` - 训练器实现
- `README.md` - 项目概述文档
- `AGENTS.md` - OpenSpec规范文档