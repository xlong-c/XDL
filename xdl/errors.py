"""XDL 统一异常层次。

所有框架级异常继承自 XDLError，便于统一捕获和处理。
标准 Python 异常 (ValueError, TypeError 等) 仍用于参数校验等通用场景。
"""


class XDLError(Exception):
    """XDL 框架根异常。"""


class RegistryError(XDLError):
    """注册表查找/注册失败。"""


class DataError(XDLError):
    """数据加载或预处理失败。"""


class ModelError(XDLError):
    """模型构建或前向传播失败。"""


class TrainingError(XDLError):
    """训练过程异常。"""


__all__ = [
    "XDLError",
    "RegistryError",
    "DataError",
    "ModelError",
    "TrainingError",
]
