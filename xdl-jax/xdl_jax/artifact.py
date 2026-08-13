"""JAX 模型侧产物导出.

该模块只负责把训练 state 中的模型变量 materialize 成 XQT 或其他模型侧
工具可以消费的 host arrays. 不保存 optimizer/RNG/data state, 也不调用 XQT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from ._version import __version__
from .errors import CheckpointError
from .model import ModelAdapter
from .types import ModelState, PyTree


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _path_key(path: tuple[Any, ...]) -> str:
    try:
        return jax.tree_util.keystr(path)
    except Exception:
        return "/".join(str(item) for item in path)


def _materialize_tree(tree: PyTree) -> dict[str, np.ndarray]:
    if tree is None:
        return {}
    result: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        result[_path_key(path)] = np.asarray(jax.device_get(leaf))
    return result


def _tree_metadata(tree: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        path: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for path, array in tree.items()
    }


def _sharding_metadata(tree: PyTree) -> dict[str, Any]:
    if tree is None:
        return {}
    result: dict[str, Any] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        sharding = getattr(leaf, "sharding", None)
        if sharding is None:
            continue
        result[_path_key(path)] = str(sharding)
    return result


@dataclass(frozen=True)
class ModelArtifact:
    """已经 materialize 的模型侧参数产物."""

    parameters: dict[str, np.ndarray]
    mutable: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelArtifactReport:
    """导出目录和模型侧统计."""

    directory: str
    parameter_count: int
    mutable_count: int
    metadata_path: str
    weights_path: str


def materialize_model_artifact(
    model_state: ModelState,
    *,
    adapter: ModelAdapter | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelArtifact:
    """把模型 state 转成 host numpy arrays 和 JSON metadata."""

    parameters = _materialize_tree(model_state.params)
    mutable = _materialize_tree(model_state.mutable)
    artifact_metadata: dict[str, Any] = {
        "format_version": 1,
        "xdl_jax_version": __version__,
        "jax_version": _package_version("jax"),
        "flax_version": _package_version("flax"),
        "parameter_count": len(parameters),
        "mutable_count": len(mutable),
        "parameters": _tree_metadata(parameters),
        "mutable": _tree_metadata(mutable),
        "sharding": {
            "parameters": _sharding_metadata(model_state.params),
            "mutable": _sharding_metadata(model_state.mutable),
        },
        "restore_scope": "model_state_only",
    }
    if adapter is not None:
        artifact_metadata["state_spec"] = adapter.state_spec(model_state)
    if metadata:
        artifact_metadata["user_metadata"] = dict(metadata)
    return ModelArtifact(
        parameters=parameters,
        mutable=mutable,
        metadata=artifact_metadata,
    )


def export_model_artifact(
    directory: str | Path,
    model_state: ModelState,
    *,
    adapter: ModelAdapter | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelArtifactReport:
    """导出 `weights.npz` 和 `metadata.json`."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = materialize_model_artifact(
        model_state,
        adapter=adapter,
        metadata=metadata,
    )
    weights: dict[str, np.ndarray] = {}
    for path, array in artifact.parameters.items():
        weights[f"params::{path}"] = array
    for path, array in artifact.mutable.items():
        weights[f"mutable::{path}"] = array
    weights_path = output_dir / "weights.npz"
    metadata_path = output_dir / "metadata.json"
    np.savez(str(weights_path), **weights)  # pyright: ignore[reportArgumentType]
    metadata_path.write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ModelArtifactReport(
        directory=str(output_dir),
        parameter_count=len(artifact.parameters),
        mutable_count=len(artifact.mutable),
        metadata_path=str(metadata_path),
        weights_path=str(weights_path),
    )


def read_model_artifact_metadata(
    directory: str | Path,
) -> dict[str, Any]:
    """读取模型侧 metadata, 不恢复训练 state."""

    path = Path(directory) / "metadata.json"
    if not path.exists():
        raise CheckpointError(f"model artifact metadata not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid model artifact metadata: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointError("model artifact metadata must be a mapping")
    return value
