"""
Trainer模块初始化文件
"""

from .core_model import CoreModel
from .trainer import Trainer
from .tbsm_model import TBSMCoreModel, TBSMModel

__all__ = [
    "Trainer",
    "CoreModel",
    "TBSMCoreModel",
    "TBSMModel",
    "TrainSetupModel",
]


def __getattr__(name: str):
    """惰性再导出 ``TrainSetupModel`` (定义在 ``xdl.config.train_setup_model``).

    保持 ``from xdl.trainer import TrainSetupModel`` 这一稳定入口, 同时避免
    config -> trainer -> config 的包级循环导入.
    """
    if name == "TrainSetupModel":
        from xdl.config.train_setup_model import TrainSetupModel

        return TrainSetupModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
