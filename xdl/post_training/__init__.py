"""
后训练 (post-training) 子模块 - 收拢基于 pretrain checkpoint 的训练能力.

覆盖后训练四类工作流:
- SFT / LoRA 微调支撑: adapter 状态保存, SFT checkpoint 合并
- 偏好优化与 RL: DPO / STPO / GRPO 损失, rollout 与冻结参考模型回调
- 蒸馏: 通用 KD 损失与扩散模型 few-step 蒸馏损失

从零训练 (预训练) 使用 `xdl.loss` 通用损失与 `xdl.trainer` 生命周期,
不经过本子模块.
"""

from ..utils.registry import register_loss

# 偏好优化损失 (DPO / STPO / GRPO)
from .preference_loss import (
    GRPOLossBreakdown,
    STPOLossBreakdown,
    dpo_loss,
    grpo_loss,
    stpo_loss,
)

# 知识蒸馏损失
from .distillation_loss import (
    DistillationLossBreakdown,
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)

# 扩散模型步数蒸馏损失
from .diffusion_distillation_loss import tdm_loss, tdm_loss_weighted

# 后训练回调
from .model_merge import ModelMergeCallback
from .reference_model import ReferenceModelCallback
from .rollout import RolloutBatch, RolloutCallback
from .save_trainable_state import SaveTrainableStateCallback


def _register_post_training_losses() -> None:
    """统一注册后训练损失到 LOSS_REGISTRY (registry 名与迁移前一致)"""

    # 知识蒸馏损失
    register_loss("kl_divergence_with_temperature")(kl_divergence_with_temperature)
    register_loss("distillation_loss")(distillation_loss)
    register_loss("feature_distillation_loss")(feature_distillation_loss)
    register_loss("relation_distillation_loss")(relation_distillation_loss)

    # 扩散模型步数蒸馏损失
    register_loss("tdm_loss")(tdm_loss)
    register_loss("tdm_loss_weighted")(tdm_loss_weighted)

    # 偏好优化损失
    register_loss("dpo_loss")(dpo_loss)
    register_loss("stpo_loss")(stpo_loss)
    register_loss("grpo_loss")(grpo_loss)


_register_post_training_losses()

__all__ = [
    "DistillationLossBreakdown",
    "GRPOLossBreakdown",
    "STPOLossBreakdown",
    "ModelMergeCallback",
    "ReferenceModelCallback",
    "RolloutBatch",
    "RolloutCallback",
    "SaveTrainableStateCallback",
    "dpo_loss",
    "stpo_loss",
    "grpo_loss",
    "distillation_loss",
    "feature_distillation_loss",
    "kl_divergence_with_temperature",
    "relation_distillation_loss",
    "tdm_loss",
    "tdm_loss_weighted",
]
