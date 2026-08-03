"""
CosineAnnealingWarmRestarts学习率调度器
"""

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as TorchCosineAnnealingWarmRestarts


class CosineAnnealingWarmRestarts(TorchCosineAnnealingWarmRestarts):
    """
    带重启的余弦退火学习率调度,适用于SGDR (Stochastic Gradient Descent with Warm Restarts)

    Args:
        optimizer: 优化器实例
        T_0: 第一次重启前的epoch数
        T_mult: 重启间隔的倍增因子,默认为1
        eta_min: 最小学习率,默认为0
        last_epoch: 上一个epoch的索引,用于恢复训练,默认为-1
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


def cosine_annealing_warm_restarts(
    optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1
):
    """
    创建CosineAnnealingWarmRestarts调度器的便捷函数

    Args:
        optimizer: 优化器实例
        T_0: 第一次重启前的epoch数
        T_mult: 重启间隔的倍增因子,默认为1
        eta_min: 最小学习率,默认为0
        last_epoch: 上一个epoch的索引,默认为-1

    Returns:
        CosineAnnealingWarmRestarts调度器实例
    """
    return CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch,
    )
