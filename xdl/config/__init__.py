"""
XDL 配置解析模块
提供从 YAML 配置文件构建训练组件的功能
"""

from .setup import setup_from_yaml
from .dataclass import TrainSetup

__all__ = [
    "setup_from_yaml",
    "TrainSetup",
]
