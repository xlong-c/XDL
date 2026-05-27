"""
xdl 框架初始化模块。

顶层包只提供轻量入口，具体子模块按需导入。这样安装 wheel 后，查看
使用说明或访问元信息时不会立刻加载训练栈和可选依赖。
"""

from importlib import import_module
from typing import Any

_LAZY_SUBMODULES = {
    "callbacks",
    "config",
    "dataset",
    "loss",
    "metric",
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "utils",
}


def get_usage_text() -> str:
    """返回随 wheel 分发的 XDL 单文件使用说明。"""
    from .usage import get_usage_text as _get_usage_text

    return _get_usage_text()


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
    "callbacks",
    "config",
    "dataset",
    "loss",
    "metric",
    "model",
    "optimizer",
    "scheduler",
    "trainer",
    "utils",
    "get_usage_text",
    "print_usage",
]
