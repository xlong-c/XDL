"""安装后查看 XDL 直接用法的轻量入口。"""

from importlib import resources
from typing import Optional, Sequence

_USAGE_RESOURCE = "USAGE.md"


def get_usage_text() -> str:
    """返回随 wheel 分发的 XDL 单文件使用说明。"""
    return resources.read_text("xdl", _USAGE_RESOURCE, encoding="utf-8")


def print_usage() -> None:
    """打印随 wheel 分发的 XDL 单文件使用说明。"""
    print(get_usage_text(), end="")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：不解析参数，只打印固定说明文件。"""
    del argv
    print_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
