# xdl/optimizer — 优化器子模块

## 目录职责

- 封装自定义优化器并注册到 XDL registry
- 为训练脚本和 YAML 配置提供统一优化器构建能力

## 当前内容

- `muon.py`
- `soap.py`

## 修改约束

- 新增优化器后必须在 `__init__.py` 中集中注册
- 接口保持兼容 PyTorch Optimizer 习惯
- 参数组、状态初始化、device/dtype 行为要清楚
- 若依赖特定训练模式或模型结构，需在文档中注明

## 验证建议

- 修改后至少做最小 step 验证
- 涉及 config 构建时检查 `xdl/config/builder.py` 兼容性
