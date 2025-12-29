"""
优化器模块 - 包含常见的深度学习优化器
"""

from xdl.utils.registry import register_optimizer

# Muon优化器
from .muon import Muon, MuonWithAuxAdam, SingleDeviceMuon, SingleDeviceMuonWithAuxAdam


def _register_optimizers():
    """统一注册所有优化器到OPTIMIZER_REGISTRY"""

    # 注册Muon系列
    register_optimizer("Muon")(Muon)
    register_optimizer("SingleDeviceMuon")(SingleDeviceMuon)
    register_optimizer("MuonWithAuxAdam")(MuonWithAuxAdam)
    register_optimizer("SingleDeviceMuonWithAuxAdam")(SingleDeviceMuonWithAuxAdam)


# 自动执行优化器注册
_register_optimizers()

__all__ = [
    # Muon
    "Muon",
    "SingleDeviceMuon",
    "MuonWithAuxAdam",
    "SingleDeviceMuonWithAuxAdam",
]
