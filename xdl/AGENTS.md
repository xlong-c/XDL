# xdl - 框架源码根目录

## 目录职责

- 承载 XDL 的核心 Python 包实现
- 提供配置化构建,训练生命周期,注册系统与基础组件

## 子模块

- `config/`:YAML 配置构建
- `trainer/`:训练器与 `CoreModel`
- `callbacks/`:训练回调
- `model/`:模型架构
- `dataset/`:数据集与 collate
- `loss/`:损失函数
- `metric/`:评估指标
- `optimizer/`:优化器
- `scheduler/`:学习率调度器
- `utils/`:registry,checkpoint 与基础工具

## 核心约束

- 所有组件通过注册系统集中注册
- 修改公共模块前先搜索调用面,评估兼容性
- 配置,训练,回调三条主链路要保持契约稳定
- 所有函数必须保留类型注解
- **模块文件命名**: 一律 snake_case (全小写 + 下划线, 遵循 PEP 8),
  例如 `core_model.py` / `callback_list.py` / `trainer_state.py`;
  类名保持 CapWords (`CoreModel` / `CallbackList` / `TrainerState`).

## 开发建议

- 改动前优先阅读对应子目录的局部文档
- 涉及生命周期和构建链时,优先联动检查 `xdl/config/`,`xdl/trainer/`,`xdl/utils/`
