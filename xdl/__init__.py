"""
xdl 框架初始化模块
"""

# 初始化各个子模块
from . import dataset, loss, metric, model, optimizer, scheduler, trainer, utils

__all__ = [
    "dataset",
    "loss",
    "metric",
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "utils",
]
