"""
配置系统异常定义。
"""

from typing import Optional

from xdl.errors import XDLError


class ConfigError(XDLError):
    """配置系统基础异常。"""


class ConfigValidationError(ConfigError):
    """配置结构或字段值非法。"""

    def __init__(self, message: str, *, field_path: Optional[str] = None):
        self.field_path = field_path
        full_message = f"{field_path}: {message}" if field_path else message
        super().__init__(full_message)


class ConfigInterpolationError(ConfigError):
    """配置插值或引用解析失败。"""

    def __init__(self, message: str, *, field_path: Optional[str] = None):
        self.field_path = field_path
        full_message = f"{field_path}: {message}" if field_path else message
        super().__init__(full_message)


class ComponentResolutionError(ConfigError):
    """组件解析失败。"""

    def __init__(
        self,
        component_type: str,
        component_name: str,
        *,
        source: Optional[str] = None,
        field_path: Optional[str] = None,
    ):
        self.component_type = component_type
        self.component_name = component_name
        self.source = source
        self.field_path = field_path

        source_hint = f" from '{source}'" if source else ""
        message = f"Failed to resolve {component_type} '{component_name}'{source_hint}"
        if field_path:
            message = f"{field_path}: {message}"
        super().__init__(message)


class UnusedConfigWarning(UserWarning):
    """配置中存在未消费字段。"""


__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "ConfigInterpolationError",
    "ComponentResolutionError",
    "UnusedConfigWarning",
]
