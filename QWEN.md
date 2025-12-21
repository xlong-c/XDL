# CLAUDE.md

This file provides guidance to coding when working with code in this repository.

## 项目概述

- 这是一个深度学习相关的项目,主要是进行各种计算机视觉任务
- 在回答时候使用中文,代码编写的时候注释使用中文,专业的特有的名词无需翻译成中文
- 所有文件使用UTF-8编码,不使用其他编码

## 项目架构

### 核心组件

1. **Registry系统** (`xdl/utils/registry.py`)
   - 核心机制：基于注册表的模块动态加载
   - 应用范围：模型、数据集、优化器、调度器、损失函数和评估指标

### 源代码目录结构 (xdl/)

- `xdl/model/` - 模型定义
- `xdl/dataset/` - 数据集处理
- `xdl/trainer/` - 训练逻辑
- `xdl/config/` - 配置管理
- `xdl/utils/` - 工具函数


### 代码规范
- 使用中文注释, 但是符号使用英文的半角符号,比如冒号(:)、逗号(,)、句号(.)等
- 专业术语保持英文(如loss、accuracy、epoch等)
- 遵循现有代码风格
- 如果没有显性要求, 尽量不要写兼容代码

## 目录结构说明

- `others/` - 实验相关文件(checkpoints、data、results等)
- `scripts/` - 执行脚本
- `config/` - 配置文件
- `others/` - 在产生日志文件,结果文件,缓存文件,模型权重等非代码文件都放到这个目录之下