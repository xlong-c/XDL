"""OmegaConf 解析辅助函数."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..errors import ConfigurationError


def load_config_with_schema(path: str | Path) -> DictConfig:
    """以结构化 schema 合并 YAML, 并拒绝未知字段."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    schema = OmegaConf.create(
        {
            "config_version": 1,
            "backend": "jax",
            "runtime": {"seed": 42, "platform": "cpu"},
            "trainer": {
                "max_epochs": 1,
                "gradient_accumulation_steps": 1,
                "drop_incomplete_accumulation": True,
                "precision": "32",
                "jit": True,
                "grad_clip_max_norm": None,
                "fail_on_callback_error": False,
                "nan_patience": 0,
                "validate_every_n_epochs": 1,
                "max_train_steps": None,
            },
            "task": None,
            "train_data": None,
            "val_data": None,
            "optimization": {},
            "callbacks": [],
            "checkpoint": {},
        }
    )
    loaded = OmegaConf.load(config_path)
    try:
        merged = OmegaConf.merge(schema, loaded)
    except Exception as exc:
        raise ConfigurationError(f"invalid xdl-jax YAML schema: {exc}") from exc
    return cast(DictConfig, merged)


def to_plain_dict(config: DictConfig) -> dict[str, Any]:
    """将 resolved DictConfig 转换成普通字典."""

    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise ConfigurationError("xdl-jax root config must be a mapping")
    return cast(dict[str, Any], value)
