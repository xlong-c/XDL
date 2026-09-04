# xdl/task/pretrain - 预训练算法任务

## 目录职责

- 承载从零训练或基于通用模型组件的预训练算法任务.
- 任务实现继承 `xdl.trainer.CoreModel`, 训练循环仍由 `Trainer` 编排.

## 当前内容

- `tbsm.py`: TBSM one-step scattering 训练任务.

## 边界

- 不放通用模型组件, 相关实现归 `xdl/model/`.
- 不放后训练任务, 偏好优化,LoRA 微调和蒸馏归 `xdl/task/posttrain/`.
