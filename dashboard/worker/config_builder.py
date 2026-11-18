"""Template configuration building backed by OmegaConf and XDL validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from xdl.config import load_config_with_schema


def build_template_config(
    template_path: str | Path,
    overrides: Mapping[str, Any],
    output_path: str | Path,
    *,
    allowed_paths: Sequence[str],
    system_overrides: Mapping[str, Any] | None = None,
    use_accelerate: bool = False,
) -> Path:
    """Apply allowlisted dot-path overrides and save a validated YAML config."""
    template = OmegaConf.load(str(template_path))
    if not isinstance(template, DictConfig):
        raise ValueError("dashboard template must be a mapping")

    allowed = {str(path) for path in allowed_paths}
    merged = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
    for path, value in overrides.items():
        if path not in allowed:
            raise ValueError(f"Override path is not allowed: {path}")
        if not OmegaConf.select(merged, path, default=None) and not _path_exists(
            merged, path
        ):
            raise ValueError(f"Template does not define override path: {path}")
        OmegaConf.update(merged, path, value, merge=False)

    for path, value in (system_overrides or {}).items():
        if not _path_exists(merged, path):
            raise ValueError(f"Template does not define system path: {path}")
        OmegaConf.update(merged, path, value, merge=False)
    if use_accelerate and not _path_exists(merged, "accelerate"):
        OmegaConf.update(merged, "accelerate", {"cpu": False}, merge=False)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=merged, f=str(destination))
    load_config_with_schema(destination, resolve=True)
    return destination


def _path_exists(config: DictConfig, path: str) -> bool:
    """Check a dot path without treating a falsey value as missing."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


__all__ = ["build_template_config"]
