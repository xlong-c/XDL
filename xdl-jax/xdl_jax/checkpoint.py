"""Orbax checkpoint 薄封装."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import orbax.checkpoint as ocp

from ._version import __version__
from .errors import CheckpointError
from .types import JaxTrainState


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class CheckpointMetadata:
    """checkpoint 中可读的运行元数据."""

    format_version: int = 1
    xdl_jax_version: str = __version__
    jax_version: str = ""
    flax_version: str = ""
    optax_version: str = ""
    orbax_version: str = ""
    config_hash: str | None = None
    precision: str = "32"
    loop: Mapping[str, int] | None = None
    restore_capabilities: Mapping[str, bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "format_version": self.format_version,
                "xdl_jax_version": self.xdl_jax_version,
                "jax_version": self.jax_version,
                "flax_version": self.flax_version,
                "optax_version": self.optax_version,
                "orbax_version": self.orbax_version,
                "config_hash": self.config_hash,
                "precision": self.precision,
                "loop": self.loop or {},
                "restore_capabilities": self.restore_capabilities
                or {
                    "exact_data_resume": False,
                    "exact_rng_resume": True,
                },
            }
        )


@dataclass(frozen=True)
class CheckpointReport:
    saved: bool
    step: int
    directory: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RestoredCheckpoint:
    state: JaxTrainState
    data_state: dict[str, Any]
    callback_state: dict[str, Any]
    metadata: dict[str, Any]
    step: int


class JaxCheckpointManager:
    """保存完整 JAX train state 和可选 JSON 状态."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_to_keep: int | None = None,
        save_interval_steps: int = 1,
        asynchronous: bool = True,
    ) -> None:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=asynchronous,
        )
        self.directory = Path(directory)
        self._manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state", "data", "callback", "metadata"),
            options=options,
        )

    @property
    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    @property
    def steps(self) -> list[int]:
        return list(self._manager.all_steps())

    def save(
        self,
        step: int,
        state: JaxTrainState,
        *,
        data_state: Mapping[str, Any] | None = None,
        callback_state: Mapping[str, Any] | None = None,
        metadata: CheckpointMetadata | Mapping[str, Any] | None = None,
        wait: bool = True,
    ) -> CheckpointReport:
        if step < 0:
            raise CheckpointError("checkpoint step must be non-negative")
        if metadata is None:
            metadata_dict = CheckpointMetadata(
                jax_version=_package_version("jax"),
                flax_version=_package_version("flax"),
                optax_version=_package_version("optax"),
                orbax_version=_package_version("orbax-checkpoint"),
            ).to_dict()
        elif isinstance(metadata, CheckpointMetadata):
            metadata_dict = metadata.to_dict()
        else:
            try:
                metadata_dict = _jsonable(metadata)
            except TypeError as exc:
                raise CheckpointError(str(exc)) from exc
        try:
            saved = self._manager.save(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    data=ocp.args.JsonSave(_jsonable(data_state or {})),
                    callback=ocp.args.JsonSave(_jsonable(callback_state or {})),
                    metadata=ocp.args.JsonSave(metadata_dict),
                ),
                custom_metadata=metadata_dict,
                force=True,
            )
            if wait:
                self._manager.wait_until_finished()
        except Exception as exc:
            raise CheckpointError(f"failed to save checkpoint step {step}: {exc}") from exc
        return CheckpointReport(
            saved=bool(saved),
            step=step,
            directory=str(self.directory),
            metadata=metadata_dict,
        )

    def restore(
        self,
        step: int | None = None,
        *,
        mode: str = "exact",
        target: JaxTrainState | None = None,
    ) -> RestoredCheckpoint:
        if mode not in {"exact", "weights_only", "model_and_optimizer"}:
            raise CheckpointError(
                "restore mode must be exact, weights_only or model_and_optimizer"
            )
        if target is None:
            raise CheckpointError("restore requires a target JaxTrainState")
        selected_step = self.latest_step if step is None else step
        if selected_step is None:
            raise CheckpointError("no checkpoint is available")
        try:
            restored = self._manager.restore(
                selected_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(target),
                    data=ocp.args.JsonRestore(),
                    callback=ocp.args.JsonRestore(),
                    metadata=ocp.args.JsonRestore(),
                ),
            )
        except Exception as exc:
            raise CheckpointError(
                f"failed to restore checkpoint step {selected_step}: {exc}"
            ) from exc
        state = restored["state"]
        if mode == "weights_only":
            state = target.replace(model_state=state.model_state)
        elif mode == "model_and_optimizer":
            state = target.replace(
                model_state=state.model_state,
                optimizer_state=state.optimizer_state,
            )
        return RestoredCheckpoint(
            state=state,
            data_state=dict(restored.get("data", {})),
            callback_state=dict(restored.get("callback", {})),
            metadata=dict(restored.get("metadata", {})),
            step=int(selected_step),
        )

    def restore_metadata(self, step: int | None = None) -> dict[str, Any]:
        """只读取 checkpoint metadata, 不构建或恢复数组 state."""

        selected_step = self.latest_step if step is None else step
        if selected_step is None:
            raise CheckpointError("no checkpoint is available")
        try:
            restored = self._manager.restore(
                selected_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(None),
                    data=ocp.args.JsonRestore(),
                    callback=ocp.args.JsonRestore(),
                    metadata=ocp.args.JsonRestore(),
                ),
            )
        except Exception as exc:
            raise CheckpointError(
                f"failed to restore metadata at step {selected_step}: {exc}"
            ) from exc
        return dict(restored.get("metadata", {}))

    def close(self) -> None:
        self._manager.wait_until_finished()
        self._manager.close()

    def __enter__(self) -> "JaxCheckpointManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
