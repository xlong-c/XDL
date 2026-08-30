# xdl/post_training - 后训练子模块

## 目录职责

- 收拢基于 pretrain checkpoint 的后训练能力, 服务四类工作流:
  - SFT/LoRA 微调支撑: adapter 状态保存, SFT checkpoint 合并
  - 偏好优化与 RL: DPO/STPO/GRPO 损失, rollout 与冻结参考模型回调
  - 蒸馏: 通用 KD 损失与扩散模型 few-step 蒸馏损失
- 从零训练 (预训练) 的通用损失在 `xdl/loss/`, 不经过本子模块

## 当前内容

- `preference_loss.py`: `dpo_loss`, `stpo_loss`, `grpo_loss` 及 breakdown
- `distillation_loss.py`: 通用 KD 损失 (response/feature/relation)
- `diffusion_distillation_loss.py`: 扩散模型步数蒸馏损失 (`tdm_loss`)
- `rollout.py`: `RolloutCallback` - K 路 rollout, 奖励打分, 参考 logps,
  注入 `RolloutBatch` 替换训练 batch
- `reference_model.py`: `ReferenceModelCallback` - 冻结参考模型 + EMA 同步
- `model_merge.py`: `ModelMergeCallback` - SFT checkpoint 线性插值合并
- `save_trainable_state.py`: `SaveTrainableStateCallback` - LoRA/adapter
  可训练状态保存

## 修改约束

- loss 的 LOSS_REGISTRY 注册集中在 `__init__.py` 的
  `_register_post_training_losses()`, registry 名保持稳定 (YAML 引用)
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
