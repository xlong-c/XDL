# xdl/utils — 工具与注册系统子模块

## 目录职责

- 提供 registry、checkpoint、tiling、权重与通用工具函数
- 为整个 XDL 包提供基础设施层支持

## 核心文件

- `registry.py`：注册系统与 `build_*` / `register_*`
- `checkpoint.py`：检查点保存与加载
- `tiling.py`：分块推理
- `tools.py`：通用辅助工具
- `weight.py`：权重处理工具

## 核心约束

- 所有组件注册都依赖这里的 registry 能力
- 注册名应稳定、明确、可追踪
- checkpoint 逻辑变更要特别谨慎，涉及兼容性和恢复能力

## 修改约束

- 改动 registry 前必须全局评估影响面
- 工具函数优先通用、确定、低副作用
- 结构化能力优先，不把业务逻辑混到基础设施层

## 验证建议

- registry 相关改动优先跑 `tests/utils/test_registry.py`
- checkpoint 相关改动至少验证保存与恢复主路径
