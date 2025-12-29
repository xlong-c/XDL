"""
ExponentialLR学习率调度器
"""

from torch.optim.lr_scheduler import ExponentialLR as TorchExponentialLR


class ExponentialLR(TorchExponentialLR):
    """
    每个epoch学习率按指数衰减

    Args:
        optimizer: 优化器实例
        gamma: 学习率衰减的系数,每个epoch学习率乘以gamma
        last_epoch: 上一个epoch的索引,用于恢复训练,默认为-1
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        super().__init__(optimizer=optimizer, gamma=gamma, last_epoch=last_epoch, verbose=verbose)


def exponential_lr(optimizer, gamma, last_epoch=-1, verbose=False):
    """
    创建ExponentialLR调度器的便捷函数

    Args:
        optimizer: 优化器实例
        gamma: 学习率衰减的系数,通常取值在0.95-0.99之间
        last_epoch: 上一个epoch的索引,默认为-1
        verbose: 是否输出调度信息,默认为False

    Returns:
        ExponentialLR调度器实例
    """
    return ExponentialLR(optimizer=optimizer, gamma=gamma, last_epoch=last_epoch, verbose=verbose)
