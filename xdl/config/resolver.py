"""
配置加载、合并和引用解析。
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .errors import ConfigInterpolationError, ConfigValidationError
from .schema import ConfigSchemaV1, create_structured_config

ConfigInput = Union[str, Path, Mapping[str, Any], DictConfig]


def register_default_resolvers() -> None:
    """注册 XDL 默认 resolver。"""

    if not OmegaConf.has_resolver("xdl.join_path"):
        OmegaConf.register_new_resolver(
            "xdl.join_path",
            lambda *parts: str(Path(*[str(part) for part in parts])),
        )

    if not OmegaConf.has_resolver("xdl.abspath"):
        OmegaConf.register_new_resolver(
            "xdl.abspath",
            lambda *parts: str(Path(*[str(part) for part in parts]).expanduser().resolve()),
        )


def _ensure_mapping_config(cfg: Union[DictConfig, ListConfig]) -> DictConfig:
    if isinstance(cfg, ListConfig):
        raise ConfigValidationError("Top-level config must be a mapping")
    return cfg


def _context_for_path(config_path: Optional[Path]) -> Dict[str, str]:
    cwd = Path.cwd().resolve()
    if config_path is None:
        return {
            "config_path": "",
            "config_dir": str(cwd),
            "project_root": str(cwd),
        }

    resolved = config_path.expanduser().resolve()
    return {
        "config_path": str(resolved),
        "config_dir": str(resolved.parent),
        "project_root": str(cwd),
    }


def _inject_xdl_context(
    cfg: DictConfig,
    *,
    config_path: Optional[Path],
) -> DictConfig:
    data = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
    if not isinstance(data, dict):
        raise ConfigValidationError("Top-level config must be a mapping")

    context = _context_for_path(config_path)
    existing = data.get("xdl")
    if isinstance(existing, Mapping):
        data["xdl"] = {**context, **dict(existing)}
    else:
        data["xdl"] = context
    return _ensure_mapping_config(OmegaConf.create(data))


def _to_dict_config(config: ConfigInput) -> DictConfig:
    if isinstance(config, DictConfig):
        return _inject_xdl_context(config, config_path=None)

    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise ConfigValidationError("Config file not found", field_path=str(config_path))
        try:
            return _inject_xdl_context(
                _ensure_mapping_config(OmegaConf.load(config_path)),
                config_path=config_path,
            )
        except OmegaConfBaseException as exc:
            raise ConfigValidationError(
                f"Failed to load config file: {exc}",
                field_path=str(config_path),
            ) from exc

    if isinstance(config, Mapping):
        try:
            return _inject_xdl_context(
                _ensure_mapping_config(OmegaConf.create(dict(config))),
                config_path=None,
            )
        except OmegaConfBaseException as exc:
            raise ConfigValidationError(f"Failed to create config from mapping: {exc}") from exc

    raise ConfigValidationError(f"Unsupported config input type: {type(config).__name__}")


def _normalize_before_merge(raw_cfg: DictConfig) -> DictConfig:
    data = OmegaConf.to_container(raw_cfg, resolve=False, enum_to_str=True)
    if not isinstance(data, dict):
        raise ConfigValidationError("Top-level config must be a mapping")

    if isinstance(data.get("loss"), dict):
        data["loss"] = [data["loss"]]
    if isinstance(data.get("metrics"), dict):
        data["metrics"] = [data["metrics"]]

    return _ensure_mapping_config(OmegaConf.create(data))


def merge_with_schema(
    config: ConfigInput,
    *,
    schema: Optional[ConfigSchemaV1] = None,
) -> DictConfig:
    """将原始配置合并到 dataclass schema。"""

    register_default_resolvers()
    base_cfg = create_structured_config(schema)
    raw_cfg = _normalize_before_merge(_to_dict_config(config))

    try:
        merged_cfg = OmegaConf.merge(base_cfg, raw_cfg)
    except OmegaConfBaseException as exc:
        raise ConfigValidationError(f"Failed to merge config with schema: {exc}") from exc

    return cast(DictConfig, merged_cfg)


def resolve_config(config: DictConfig) -> DictConfig:
    """解析 `${...}` 插值并返回新的配置对象。"""

    try:
        cfg_copy = OmegaConf.create(
            OmegaConf.to_container(config, resolve=False, enum_to_str=True)
        )
        cfg_copy = _ensure_mapping_config(cfg_copy)
        OmegaConf.resolve(cfg_copy)
        return cfg_copy
    except OmegaConfBaseException as exc:
        raise ConfigInterpolationError(f"Failed to resolve config interpolation: {exc}") from exc


def to_plain_dict(config: DictConfig, *, resolve: bool = True) -> Dict[str, Any]:
    """将 DictConfig 转为普通 dict。"""

    try:
        data = OmegaConf.to_container(config, resolve=resolve, enum_to_str=True)
    except OmegaConfBaseException as exc:
        raise ConfigInterpolationError(f"Failed to convert config to dict: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError("Resolved config must be a mapping")
    return cast(Dict[str, Any], data)


def load_config_with_schema(
    config: ConfigInput,
    *,
    schema: Optional[ConfigSchemaV1] = None,
    resolve: bool = False,
) -> DictConfig:
    """加载配置并合并到 schema，可选执行插值解析。"""

    merged_cfg = merge_with_schema(config, schema=schema)
    return resolve_config(merged_cfg) if resolve else merged_cfg


__all__ = [
    "ConfigInput",
    "register_default_resolvers",
    "merge_with_schema",
    "resolve_config",
    "to_plain_dict",
    "load_config_with_schema",
]
