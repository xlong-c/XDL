"""Structured dataclass config loading helpers."""

from pathlib import Path
from typing import Any, Mapping, Optional, Type, TypeVar, Union, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .errors import ConfigValidationError
from .resolver import register_default_resolvers

T = TypeVar("T")


def load_structured_dataclass_config(
    config_type: Type[T],
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    resolve: bool = True,
) -> T:
    """Load a dataclass config using structured defaults then YAML overrides."""

    register_default_resolvers()
    config_nodes = [OmegaConf.structured(config_type)]

    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise ConfigValidationError("Config file not found", field_path=str(path))
        config_nodes.append(OmegaConf.load(path))

    if overrides:
        config_nodes.append(OmegaConf.create(dict(overrides)))

    try:
        merged = OmegaConf.merge(*config_nodes)
        if resolve:
            OmegaConf.resolve(merged)
        return cast(T, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        raise ConfigValidationError(
            f"Failed to load structured dataclass config: {exc}"
        ) from exc
