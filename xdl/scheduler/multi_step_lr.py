"""
MultiStepLR学习率调度器
"""

import torch
from torch.optim.lr_scheduler import MultiStepLR as TorchMultiStepLR


class MultiStepLR(TorchMultiStepLR):
    """
    在指定的epoch节点,学习率衰减gamma倍
    
    Args:
        optimizer: 优化器实例
        milestones: 衰减节点的epoch列表
        gamma: 学习率衰减的系数,默认为0.1
        last_epoch: 上一个epoch的索引,用于恢复训练,默认为-1
    """
    
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose
        )


def multi_step_lr(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
    """
    创建MultiStepLR调度器的便捷函数
    
    Args:
        optimizer: 优化器实例
        milestones: 衰减节点的epoch列表,例如[30, 60, 90]
        gamma: 学习率衰减的系数,默认为0.1
        last_epoch: 上一个epoch的索引,默认为-1
        verbose: 是否输出调度信息,默认为False
    
    Returns:
        MultiStepLR调度器实例
    """
    return MultiStepLR(
        optimizer=optimizer,
        milestones=milestones,
        gamma=gamma,
        last_epoch=last_epoch,
        verbose=verbose
    )