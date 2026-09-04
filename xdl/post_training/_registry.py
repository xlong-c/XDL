"""后训练 loss 的 registry 注册入口.

只导入纯 loss 实现, 不导入 callback 或 checkpoint 合并逻辑. 该模块由
``xdl.utils.registry`` 按需加载.
"""

from ..utils.registry import register_loss
from .losses.diffusion_distillation_loss import tdm_loss, tdm_loss_weighted
from .losses.distillation_loss import (
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)
from .losses.preference_loss import dpo_loss, grpo_loss, stpo_loss


def register_post_training_losses() -> None:
    """将后训练 loss 注册到全局 LOSS registry."""
    registrations = {
        "kl_divergence_with_temperature": kl_divergence_with_temperature,
        "distillation_loss": distillation_loss,
        "feature_distillation_loss": feature_distillation_loss,
        "relation_distillation_loss": relation_distillation_loss,
        "tdm_loss": tdm_loss,
        "tdm_loss_weighted": tdm_loss_weighted,
        "dpo_loss": dpo_loss,
        "stpo_loss": stpo_loss,
        "grpo_loss": grpo_loss,
    }
    for name, loss in registrations.items():
        register_loss(name)(loss)


register_post_training_losses()

__all__ = ["register_post_training_losses"]
