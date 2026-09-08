"""
xdl 框架初始化模块。

顶层包只提供轻量入口，具体子模块按需导入。这样安装 wheel 后，查看
使用说明或访问元信息时不会立刻加载训练栈和可选依赖。
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import analysis as analysis
    from . import callbacks as callbacks
    from . import config as config
    from . import dataset as dataset
    from . import loss as loss
    from . import metric as metric
    from . import model as model
    from . import optimizer as optimizer
    from . import post_training as post_training
    from . import scheduler as scheduler
    from . import task as task
    from . import trainer as trainer
    from . import utils as utils

_LAZY_SUBMODULES = {
    "analysis",
    "callbacks",
    "config",
    "dataset",
    "loss",
    "metric",
    "model",
    "optimizer",
    "post_training",
    "scheduler",
    "task",
    "trainer",
    "utils",
}


def get_usage_text() -> str:
    """返回随 wheel 分发的 XDL 单文件使用说明。"""
    from .usage import get_usage_text as _get_usage_text

    return _get_usage_text()


def get_usage_payload() -> Any:
    """返回结构化的 usage 内容, 便于 agent 或外部工具直接消费。"""
    from .usage import get_usage_payload as _get_usage_payload

    return _get_usage_payload()


def get_usage_questions() -> Any:
    """返回编写功能前建议先确认的问答清单。"""
    from .usage import get_usage_questions as _get_usage_questions

    return _get_usage_questions()


def print_usage() -> None:
    """打印随 wheel 分发的 XDL 单文件使用说明。"""
    from .usage import print_usage as _print_usage

    _print_usage()


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "analysis",
    "callbacks",
    "config",
    "dataset",
    "loss",
    "metric",
    "model",
    "optimizer",
    "post_training",
    "scheduler",
    "task",
    "trainer",
    "utils",
    "get_usage_text",
    "get_usage_payload",
    "get_usage_questions",
    "print_usage",
]
