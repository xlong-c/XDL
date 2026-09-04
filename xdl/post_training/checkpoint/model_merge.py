"""Model merging callback for SFT domain-specific checkpoints.

After training domain-specific SFT checkpoints, this callback merges them
into a single generalist checkpoint using linear interpolation (first
stage; TIES / DARE planned for later extensions).

Ref: Krea 2 Technical Report (2026), SFT section.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from xdl.callbacks.base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


def merge_checkpoints(
    checkpoint_paths: list[str | Path],
    merge_weights: list[float] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """线性合并多个 state dict checkpoint, 返回输出路径.

    该函数不依赖 Trainer 或 CoreModel, 可独立用于后训练产物处理.
    输入可以是裸 state dict, 也可以是包含 ``model_state_dict`` 的 checkpoint.
    """
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) < 2:
        raise ValueError("At least 2 checkpoint paths required for merging")
    weights = [1.0] * len(paths) if merge_weights is None else merge_weights
    if len(weights) != len(paths):
        raise ValueError(
            f"merge_weights length ({len(weights)}) must match "
            f"checkpoint_paths length ({len(paths)})"
        )
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("merge_weights must not sum to zero")

    state_dicts: list[dict[str, Any]] = []
    for path in paths:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint {path} does not contain a state dict")
        state_dicts.append(state_dict)

    reference_keys = set(state_dicts[0])
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict) != reference_keys:
            raise ValueError(
                f"Checkpoint key mismatch: {paths[0]} vs {paths[index]}"
            )

    normalized_weights = [weight / total_weight for weight in weights]
    merged: dict[str, Any] = {}
    for key in reference_keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise TypeError(f"Checkpoint value for '{key}' is not a tensor")
        merged[key] = sum(
            weight * tensor.to(dtype=torch.float32)
            for weight, tensor in zip(normalized_weights, tensors)
        ).to(dtype=tensors[0].dtype)

    destination = (
        Path(output_path)
        if output_path is not None
        else paths[0].with_name(f"{paths[0].stem}_merged{paths[0].suffix}")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, destination)
    return destination


class ModelMergeCallback(Callback):
    """Merge multiple model checkpoints via linear interpolation.

    Called once in ``on_fit_end`` (or ``on_fit_start`` if
    ``merge_at_start=True``).  The merged checkpoint is saved to
    ``output_path`` (or ``checkpoint_paths[0]`` with a suffix if not
    specified).

    Usage::

        callback = ModelMergeCallback(
            checkpoint_paths=[
                "checkpoints/photorealism.pt",
                "checkpoints/illustration.pt",
                "checkpoints/3d_render.pt",
            ],
            merge_weights=[1.0, 0.8, 0.6],
            output_path="checkpoints/generalist.pt",
        )
    """

    def __init__(
        self,
        checkpoint_paths: list[str],
        merge_weights: list[float] | None = None,
        output_path: str | None = None,
        method: Literal["linear", "ties", "dare"] = "linear",
        merge_at_start: bool = False,
        priority: int = 999,
    ) -> None:
        """Args:
            checkpoint_paths: Paths to checkpoint files to merge.
            merge_weights: Per-checkpoint interpolation weights.
                ``None`` means equal weight (1.0 for each).
            output_path: Where to save the merged checkpoint.  Defaults to
                ``checkpoint_paths[0]_merged.pt``.
            method: Merge algorithm.  Only ``"linear"`` is implemented
                in the first stage; ``"ties"`` and ``"dare"`` raise
                ``NotImplementedError``.
            merge_at_start: If ``True``, merge in ``on_fit_start``
                (useful for loading merged checkpoint into a new model).
                Default ``False`` (merge in ``on_fit_end``).
            priority: Callback priority.
        """
        super().__init__(priority=priority)
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.merge_weights: list[float] = (
            [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
            if merge_weights is None
            else merge_weights
        )
        self.output_path = (
            Path(output_path)
            if output_path
            else self.checkpoint_paths[0].with_suffix("").with_name(
                self.checkpoint_paths[0].stem + "_merged.pt"
            )
        )
        self.method = method
        self.merge_at_start = merge_at_start

        if len(self.checkpoint_paths) < 2:
            raise ValueError("At least 2 checkpoint paths required for merging")
        if len(self.merge_weights) != len(self.checkpoint_paths):
            raise ValueError(
                f"merge_weights length ({len(self.merge_weights)}) must match "
                f"checkpoint_paths length ({len(self.checkpoint_paths)})"
            )
        if sum(self.merge_weights) == 0:
            raise ValueError("merge_weights must not sum to zero")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if not self.merge_at_start:
            return
        self._merge_and_save(trainer, core_module)

    def on_fit_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if self.merge_at_start:
            return
        self._merge_and_save(trainer, core_module)

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge_and_save(
        self, _trainer: "Trainer", _core_module: "CoreModel"
    ) -> None:
        if self.method == "linear":
            merge_checkpoints(
                self.checkpoint_paths,
                self.merge_weights,
                self.output_path,
            )
        elif self.method == "ties":
            raise NotImplementedError("TIES merge not yet implemented")
        elif self.method == "dare":
            raise NotImplementedError("DARE merge not yet implemented")
        else:
            raise ValueError(f"Unknown merge method: {self.method}")

        self._state["merged_checkpoint"] = str(self.output_path)

    def _linear_merge(self) -> dict[str, Any]:
        """Return the linear merge result without writing it to disk.

        This method is retained for compatibility and testing.  New code that
        needs a persisted artifact should call :func:`merge_checkpoints`.
        """
        total_w = sum(self.merge_weights)
        if total_w == 0:
            raise ValueError("merge_weights must not sum to zero")
        state_dicts: list[dict[str, Any]] = []
        for path in self.checkpoint_paths:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            state_dicts.append(state_dict)

        reference_keys = set(state_dicts[0].keys())
        for i, state_dict in enumerate(state_dicts[1:], start=1):
            if set(state_dict.keys()) != reference_keys:
                raise ValueError(
                    f"Checkpoint key mismatch: {self.checkpoint_paths[0]} vs "
                    f"{self.checkpoint_paths[i]}"
                )

        normalized_weights = [weight / total_w for weight in self.merge_weights]
        merged: dict[str, Any] = {}
        for key in reference_keys:
            tensors = [state_dict[key] for state_dict in state_dicts]
            merged[key] = sum(
                weight * tensor.to(dtype=torch.float32)
                for weight, tensor in zip(normalized_weights, tensors)
            ).to(dtype=tensors[0].dtype)
        return merged

    def _save_checkpoint(self, state_dict: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, self.output_path)


__all__ = [
    "ModelMergeCallback",
    "merge_checkpoints",
]
