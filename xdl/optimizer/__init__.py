"""
优化器模块 - 包含常见的深度学习优化器
"""

from xdl.utils.registry import register_optimizer

# SGD优化器
from .sgd import SGD, sgd

# Adam优化器
from .adam import Adam, adam

# AdamW优化器
from .adamw import AdamW, adamw

# RMSprop优化器
from .rmsprop import RMSprop, rmsprop


def _register_optimizers():
    """统一注册所有优化器到OPTIMIZER_REGISTRY"""
    
    # 注册SGD系列
    register_optimizer("SGD")(SGD)
    register_optimizer("sgd")(sgd)
    
    # 注册Adam系列
    register_optimizer("Adam")(Adam)
    register_optimizer("adam")(adam)
    
    # 注册AdamW系列
    register_optimizer("AdamW")(AdamW)
    register_optimizer("adamw")(adamw)
    
    # 注册RMSprop系列
    register_optimizer("RMSprop")(RMSprop)
    register_optimizer("rmsprop")(rmsprop)


# 自动执行优化器注册
_register_optimizers()

__all__ = [
    # SGD
    'SGD', 'sgd',
    
    # Adam
    'Adam', 'adam',
    
    # AdamW
    'AdamW', 'adamw',
    
    # RMSprop
    'RMSprop', 'rmsprop'
]