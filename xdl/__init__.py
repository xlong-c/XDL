"""
xdl 框架初始化模块
"""

# 初始化各个子模块
from . import loss, metric, model, optimizer, scheduler, trainer, utils

__all__ = [
    "loss",
    "metric",
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "utils",
]
