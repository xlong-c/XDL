# xdl/post_training - 后训练子模块

## 目录职责

- 收拢基于 pretrain checkpoint 的后训练能力, 服务四类工作流:
  - SFT/LoRA 微调支撑: adapter 状态保存, SFT checkpoint 合并
  - 偏好优化与 RL: DPO/STPO/GRPO 损失, rollout 与冻结参考模型回调
  - 蒸馏: 通用 KD 损失与扩散模型 few-step 蒸馏损失
- 从零训练 (预训练) 的通用损失在 `xdl/loss/`, 不经过本子模块

## 当前内容

- `losses/`: 后训练纯 loss 的分层入口, 包含偏好优化, 通用蒸馏和扩散步数蒸馏.
- `callbacks/`: rollout, 冻结参考模型和 adapter 状态保存等训练期扩展.
- `checkpoint/`: checkpoint 合并等不依赖 Trainer 的产物处理入口.
- `lora.py`: LoRA target modules, rank, alpha 和 dropout 的共享规范化/校验协议.
- 根目录旧模块 (`preference_loss.py` 等) 作为兼容实现路径保留, 新代码优先从上述分层入口导入.

## 修改约束

- loss 的 LOSS_REGISTRY 注册集中在 `_registry.py` 的
  `register_post_training_losses()`, registry 名保持稳定 (YAML 引用)
- `xdl/utils/registry.py` 的 LOSS bootstrap 会引导导入本子模块, 不要在
  其他组件包内反向 import 本子模块
- `RolloutCallback` 的 `rollout_ref_logps` 必须由 `ref_logp_fn` 真实计算,
  不允许零占位 (会静默破坏 importance ratio)
- 回调继承 `xdl.callbacks.base.Callback`, 遵守回调不接管训练主逻辑的约束

## 验证建议

- 修改后跑 `tests/loss/test_preference_loss.py`,
  `tests/loss/test_diffusion_distillation_loss.py`, `tests/loss/test_losses.py`
- 回调改动跑 `tests/callbacks/`, batch 替换契约见
  `tests/callbacks/test_callback_batch_replacement.py`
- 端到端链路用 `train/posttrain/train_GRPO.py` 玩具入口闭环验证
