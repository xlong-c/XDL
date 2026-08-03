# tests — 测试目录

## 目录职责

- 存放 XDL 核心模块的单元测试与集成级构建测试
- 验证 registry、config、dataset、loss、metric 等基础行为

## 当前内容

- `config/`：配置系统测试
- `dataset/`：数据集测试
- `loss/`：损失函数测试
- `metric/`：指标测试
- `utils/`：注册系统与工具测试
- `conftest.py`：共享 fixture

## 修改约束

- 测试应优先覆盖行为边界，而不是实现细节
- 新增核心功能时同步补测试，至少覆盖主路径和明显错误路径
- 保持测试可重复执行，避免依赖私有路径、随机外部资源和网络
- 重型 GPU 或大模型验证不要直接塞进默认单测集

## 运行建议

- 优先跑受影响子目录
- 配置相关改动先看 `tests/config/`
- 注册系统改动记得覆盖 `tests/utils/test_registry.py`
