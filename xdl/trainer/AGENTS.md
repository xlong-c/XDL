# xdl/trainer — 训练器核心子模块

## 目录职责

- 提供 XDL 训练生命周期、状态管理与 `CoreModel` 抽象
- 负责 callback 驱动、设备管理、训练/验证/推理循环

## 核心文件

- `trainer.py`：`Trainer` 主类
- `coreModel.py`：`CoreModel` 基类
- `trainer_state.py`：状态管理
- `trainSetupModel.py`：把外部组件包装成可训练模型
- `example_tasks.py`：官方配置样例使用的轻量 `CoreModel` task 集合

## 核心约束

- `Trainer.fit()` 会先调用 `model.setup("fit")`
- `CoreModel.training_step()` 采用手动优化模式，需自行 `zero_grad/backward/step`
- `CoreModel.log(..., prefix="train")` 会生成 `train_loss` 这类兼容指标名
- `micro_step`、`accumulation_steps`、`is_accumulation_start`、`is_accumulation_boundary`、`should_optimizer_step` 是手动梯度累积的公开 helper
- 标准路径会迁移 `CoreModel.__dict__` 中的 `nn.Module` 属性，并递归迁移常见 batch 容器；第三方自定义对象需显式处理
- callback 生命周期与优先级行为是这里的核心契约

## 修改约束

- 改动这里前先全局搜索调用面，评估对训练入口和 callback 的影响
- 公共行为变更要优先考虑兼容性和错误提示
- 避免把模型特定逻辑写进 Trainer 主循环
- 状态字段命名与生命周期触发点要保持稳定

## 验证建议

- 先跑受影响测试
- 至少用一个真实训练入口做主路径检查
