"""
学习率调度器模块 - 包含常见的学习率调度策略
"""

from ..utils.registry import register_scheduler

# CosineAnnealingLR调度器
from .cosine_annealing_lr import CosineAnnealingLR, cosine_annealing_lr

# CosineAnnealingWarmRestarts调度器
from .cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestarts,
    cosine_annealing_warm_restarts,
)

# ExponentialLR调度器
from .exponential_lr import ExponentialLR, exponential_lr

# MultiStepLR调度器
from .multi_step_lr import MultiStepLR, multi_step_lr

# StepLR调度器
from .step_lr import StepLR, step_lr


def _register_schedulers():
    """统一注册所有学习率调度器到SCHEDULER_REGISTRY"""

    # 注册StepLR系列
    register_scheduler("StepLR")(StepLR)
    register_scheduler("step_lr")(step_lr)

    # 注册MultiStepLR系列
    register_scheduler("MultiStepLR")(MultiStepLR)
    register_scheduler("multi_step_lr")(multi_step_lr)

    # 注册ExponentialLR系列
    register_scheduler("ExponentialLR")(ExponentialLR)
    register_scheduler("exponential_lr")(exponential_lr)

    # 注册CosineAnnealingLR系列
    register_scheduler("CosineAnnealingLR")(CosineAnnealingLR)
    register_scheduler("cosine_annealing_lr")(cosine_annealing_lr)

    # 注册CosineAnnealingWarmRestarts系列
    register_scheduler("CosineAnnealingWarmRestarts")(CosineAnnealingWarmRestarts)
    register_scheduler("cosine_annealing_warm_restarts")(cosine_annealing_warm_restarts)


# 自动执行调度器注册
_register_schedulers()

__all__ = [
    # StepLR
    "StepLR",
    "step_lr",
    # MultiStepLR
    "MultiStepLR",
    "multi_step_lr",
    # ExponentialLR
    "ExponentialLR",
    "exponential_lr",
    # CosineAnnealingLR
    "CosineAnnealingLR",
    "cosine_annealing_lr",
    # CosineAnnealingWarmRestarts
    "CosineAnnealingWarmRestarts",
    "cosine_annealing_warm_restarts",
]
