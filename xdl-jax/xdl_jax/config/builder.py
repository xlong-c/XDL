"""配置组件 builder 薄封装."""

from __future__ import annotations

from typing import Any

from ..registry import build_component


def build_task(config: dict[str, Any]) -> Any:
    """构建 JAX task."""

    return build_component(config, kind="task")


def build_dataset(config: dict[str, Any]) -> Any:
    """构建 JAX data source."""

    return build_component(config, kind="dataset")


def build_callback(config: dict[str, Any]) -> Any:
    """构建 JAX callback."""

    return build_component(config, kind="callback")
