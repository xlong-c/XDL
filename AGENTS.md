## 项目概述

- 这是一个深度学习相关的项目,主要是进行各种计算机视觉任务
- 在回答时候使用中文,代码编写的时候注释使用中文,专业的特有的名词无需翻译成中文
- 所有文件使用UTF-8编码,不使用其他编码

## 项目架构

### 核心组件

1. **Registry系统** (`xdl/utils/registry.py`)
   - 核心机制：装饰器驱动的模块化映射
   - 包含：MODEL, DATASET, OPTIMIZER, SCHEDULER, LOSS, METRIC

### 模块结构规范

- `xdl/model/` - 模型定义
- `xdl/dataset/` - 数据集处理
- `xdl/trainer/` - 训练逻辑
- `xdl/config/` - 配置管理
- `xdl/utils/` - 工具函数
- `xdl/callbacks` - 回调函数


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