"""
StepLR学习率调度器
"""

from torch.optim.lr_scheduler import StepLR as TorchStepLR


class StepLR(TorchStepLR):
    """
    每隔step_size个epoch,学习率衰减gamma倍

    Args:
        optimizer: 优化器实例
        step_size: 学习率衰减的步长,单位为epoch
        gamma: 学习率衰减的系数,默认为0.1
        last_epoch: 上一个epoch的索引,用于恢复训练,默认为-1
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )


def step_lr(optimizer, step_size, gamma=0.1, last_epoch=-1):
    """
    创建StepLR调度器的便捷函数

    Args:
        optimizer: 优化器实例
        step_size: 学习率衰减的步长,单位为epoch
        gamma: 学习率衰减的系数,默认为0.1
        last_epoch: 上一个epoch的索引,默认为-1

    Returns:
        StepLR调度器实例
    """
    return StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch,
    )
