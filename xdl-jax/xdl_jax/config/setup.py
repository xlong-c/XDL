"""从 YAML 构建 JAX 训练 setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..callbacks import Callback, CheckpointCallback
from ..errors import ConfigurationError
from ..task import JaxTask
from ..trainer import TrainerConfig
from .builder import build_callback, build_dataset, build_task
from .resolver import load_config_with_schema, to_plain_dict


@dataclass(frozen=True)
class JaxTrainSetup:
    task: JaxTask
    train_data: Any
    val_data: Any
    optimizer_config: dict[str, Any] | None
    trainer_config: TrainerConfig
    callbacks: list[Callback]
    checkpoint_config: dict[str, Any]
    full_config: dict[str, Any]


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _require_mapping(config: Any, name: str) -> dict[str, Any]:
    if not isinstance(config, (dict, DictConfig)):
        raise ConfigurationError(f"{name} must be a mapping")
    return dict(_plain(config))


def _validate_fields(
    values: dict[str, Any],
    *,
    allowed: set[str],
    name: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(
            f"unknown fields in {name}: {sorted(unknown)}"
        )


def _build_checkpoint_callback(
    config: dict[str, Any],
) -> CheckpointCallback | None:
    """将 YAML checkpoint 配置显式转换成标准 callback."""

    if not config:
        return None
    _validate_fields(
        config,
        allowed={
            "enabled",
            "directory",
            "every_n_epochs",
            "max_to_keep",
            "asynchronous",
        },
        name="checkpoint",
    )
    enabled = bool(config.get("enabled", True))
    directory = config.get("directory")
    if not enabled:
        if directory is not None:
            raise ConfigurationError(
                "checkpoint.directory cannot be set when checkpoint.enabled is false"
            )
        return None
    if not isinstance(directory, str) or not directory:
        raise ConfigurationError(
            "checkpoint.directory is required when checkpoint is enabled"
        )
    max_to_keep = config.get("max_to_keep")
    if max_to_keep is not None:
        max_to_keep = int(max_to_keep)
        if max_to_keep < 1:
            raise ConfigurationError("checkpoint.max_to_keep must be positive")
    try:
        return CheckpointCallback(
            directory,
            every_n_epochs=int(config.get("every_n_epochs", 1)),
            max_to_keep=max_to_keep,
            asynchronous=bool(config.get("asynchronous", True)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid checkpoint configuration: {exc}") from exc


def setup_from_yaml(config_path: str | Path) -> JaxTrainSetup:
    """加载严格的 xdl-jax schema, 不做旧字段或路径的隐式迁移."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    config = to_plain_dict(load_config_with_schema(path))
    allowed = {
        "config_version",
        "backend",
        "runtime",
        "trainer",
        "task",
        "train_data",
        "val_data",
        "optimization",
        "callbacks",
        "checkpoint",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ConfigurationError(f"unknown xdl-jax config fields: {sorted(unknown)}")
    if config.get("config_version", 1) != 1:
        raise ConfigurationError("only config_version=1 is supported")
    if config.get("backend", "jax") != "jax":
        raise ConfigurationError("xdl-jax config requires backend: jax")
    if not config.get("task"):
        raise ConfigurationError("task config cannot be empty")
    if not config.get("train_data"):
        raise ConfigurationError("train_data config cannot be empty")

    task = build_task(_require_mapping(config["task"], "task"))
    train_data = build_dataset(
        _require_mapping(config["train_data"], "train_data"),
    )
    val_config = config.get("val_data")
    val_data = (
        build_dataset(_require_mapping(val_config, "val_data"))
        if val_config
        else None
    )
    trainer_values = _require_mapping(config.get("trainer", {}), "trainer")
    runtime_values = _require_mapping(config.get("runtime", {}), "runtime")
    _validate_fields(
        runtime_values,
        allowed={"seed", "platform"},
        name="runtime",
    )
    _validate_fields(
        trainer_values,
        allowed={
            "max_epochs",
            "gradient_accumulation_steps",
            "drop_incomplete_accumulation",
            "precision",
            "jit",
            "grad_clip_max_norm",
            "nan_patience",
            "fail_on_callback_error",
            "validate_every_n_epochs",
            "max_train_steps",
        },
        name="trainer",
    )
    trainer_config = TrainerConfig(
        seed=int(runtime_values.get("seed", 42)),
        max_epochs=int(trainer_values.get("max_epochs", 1)),
        gradient_accumulation_steps=int(
            trainer_values.get("gradient_accumulation_steps", 1)
        ),
        drop_incomplete_accumulation=bool(
            trainer_values.get("drop_incomplete_accumulation", True)
        ),
        precision=str(trainer_values.get("precision", "32")),
        jit=bool(trainer_values.get("jit", True)),
        grad_clip_max_norm=(
            None
            if trainer_values.get("grad_clip_max_norm") is None
            else float(trainer_values["grad_clip_max_norm"])
        ),
        nan_patience=int(trainer_values.get("nan_patience", 0)),
        fail_on_callback_error=bool(
            trainer_values.get("fail_on_callback_error", False)
        ),
        validate_every_n_epochs=int(
            trainer_values.get("validate_every_n_epochs", 1)
        ),
        max_train_steps=trainer_values.get("max_train_steps"),
        platform=str(runtime_values.get("platform", "cpu")),
    )
    callbacks = [
        build_callback(_require_mapping(item, "callback"))
        for item in config.get("callbacks", [])
    ]
    checkpoint_config = _require_mapping(
        config.get("checkpoint", {}),
        "checkpoint",
    )
    checkpoint_callback = _build_checkpoint_callback(checkpoint_config)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    if not all(isinstance(callback, Callback) for callback in callbacks):
        raise ConfigurationError("all configured callbacks must inherit Callback")
    full_config = config
    optimization = _require_mapping(
        config.get("optimization", {}),
        "optimization",
    )
    optimizer_config = optimization.get("optimizer")
    if optimizer_config is not None and not isinstance(optimizer_config, dict):
        raise ConfigurationError("optimization.optimizer must be a mapping")
    if optimizer_config is not None:
        optimizer_target = str(optimizer_config.get("target", ""))
        if (
            ":" not in optimizer_target
            or optimizer_target.split(":", 1)[0] != "optax"
        ):
            raise ConfigurationError(
                "xdl-jax optimization.optimizer target must use 'optax:name'"
            )
    return JaxTrainSetup(
        task=task,
        train_data=train_data,
        val_data=val_data,
        optimizer_config=dict(optimizer_config) if optimizer_config else None,
        trainer_config=trainer_config,
        callbacks=callbacks,
        checkpoint_config=checkpoint_config,
        full_config=full_config,
    )
