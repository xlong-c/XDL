"""
CosineAnnealingLR学习率调度器
"""

from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR


class CosineAnnealingLR(TorchCosineAnnealingLR):
    """
    余弦退火学习率调度,学习率按余弦函数曲线变化

    Args:
        optimizer: 优化器实例
        T_max: 最大epoch数,余弦函数的半周期
        eta_min: 最小学习率,默认为0
        last_epoch: 上一个epoch的索引,用于恢复训练,默认为-1
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )


def cosine_annealing_lr(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
    """
    创建CosineAnnealingLR调度器的便捷函数

    Args:
        optimizer: 优化器实例
        T_max: 最大epoch数,余弦函数的半周期
        eta_min: 最小学习率,默认为0
        last_epoch: 上一个epoch的索引,默认为-1
        verbose: 是否输出调度信息,默认为False

    Returns:
        CosineAnnealingLR调度器实例
    """
    return CosineAnnealingLR(
        optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose
    )
