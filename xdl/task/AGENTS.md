# xdl/task - 训练算法任务

## 目录职责

- 承载具体训练算法对 `CoreModel` 的任务实现.
- 按阶段划分为 `pretrain/` 和 `posttrain/`.

## 边界

- 通用训练循环放在 `xdl/trainer/`.
- 通用模型组件放在 `xdl/model/`.
- 算法任务可以组合模型,loss 和 callback, 但不应把算法特例加入通用 Trainer.
