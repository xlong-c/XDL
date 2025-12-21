"""
xdl 框架初始化模块
"""

# 初始化各个子模块
from . import loss
from . import metric
from . import model
from . import optimizer
from . import scheduler
from . import trainer
from . import utils

__all__ = [
    "loss",
    "metric",
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "utils",
]
