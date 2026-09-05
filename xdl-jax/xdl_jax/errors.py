"""xdl-jax 异常类型, 保持自包含与清晰的错误层次."""

from __future__ import annotations


class XdlJaxError(Exception):
    """xdl-jax 异常基类."""


class ConfigurationError(XdlJaxError):
    """配置无效或组件无法构建."""


class DataValidationError(XdlJaxError):
    """输入 batch 不符合固定结构或 dtype/shape 契约."""


class TrainingError(XdlJaxError):
    """训练运行失败."""


class CheckpointError(XdlJaxError):
    """checkpoint 保存或恢复失败."""


class CallbackError(XdlJaxError):
    """callback 执行失败."""
