"""Validation-time attention rollout capture callback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import torch

from xdl.analysis.attention import attention_rollout_for_model

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


BatchAdapter = Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]]


class AttentionRolloutCallback(Callback):
    """Run attention rollout during validation and save the result."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        batch_adapter: Optional[BatchAdapter] = None,
        every_n_epochs: int = 1,
        first_val_batch_only: bool = True,
        max_batches_per_epoch: Optional[int] = None,
        file_prefix: str = "attention_rollout",
        save_format: str = "pt",
        priority: int = 320,
    ) -> None:
        super().__init__(priority=priority)
        self.output_dir = Path(output_dir)
        self.batch_adapter = batch_adapter
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.first_val_batch_only = bool(first_val_batch_only)
        self.max_batches_per_epoch = (
            max(1, int(max_batches_per_epoch)) if max_batches_per_epoch is not None else None
        )
        self.file_prefix = file_prefix
        self.save_format = save_format.lower()
        self._captured_batches = 0

        if self.save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format: {save_format}")

    def on_validation_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        del core_module
        if self._should_capture_epoch(trainer):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._captured_batches = 0

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, dataloader_idx
        if not self._should_capture_epoch(trainer):
            return
        if self.first_val_batch_only and batch_idx != 0:
            return
        if self.max_batches_per_epoch is not None and self._captured_batches >= self.max_batches_per_epoch:
            return
        if not self._is_main_process(trainer, core_module):
            return

        model = self._resolve_model(core_module)
        forward_args, forward_kwargs = self._adapt_batch(batch)
        rollout = attention_rollout_for_model(model, *forward_args, forward_kwargs=forward_kwargs)
        payload = {
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "batch_idx": int(batch_idx),
            "rollout": rollout.detach().cpu(),
        }
        filename = (
            f"{self.file_prefix}_epoch{payload['epoch']:04d}"
            f"_step{payload['global_step']:08d}_batch{batch_idx:04d}"
        )
        saved_path = self._save_payload(payload, filename)
        self._state["last_saved_file"] = str(saved_path)
        self._state["last_epoch"] = payload["epoch"]
        self._state["last_step"] = payload["global_step"]
        self._captured_batches += 1

    def _adapt_batch(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.batch_adapter is not None:
            return self.batch_adapter(batch)
        if isinstance(batch, Mapping):
            if "inputs" in batch and isinstance(batch["inputs"], (tuple, list)):
                return tuple(batch["inputs"]), dict(batch.get("forward_kwargs", {}))
            if "x" in batch:
                return (batch["x"],), {}
            if "image" in batch:
                return (batch["image"],), {}
            if "pixel_values" in batch:
                return (), {"pixel_values": batch["pixel_values"]}
        if isinstance(batch, torch.Tensor):
            return (batch,), {}
        raise TypeError(
            "Could not adapt validation batch to model inputs. Provide batch_adapter explicitly."
        )

    def _should_capture_epoch(self, trainer: "Trainer") -> bool:
        epoch = getattr(trainer, "current_epoch", 0)
        return int(epoch) % self.every_n_epochs == 0

    @staticmethod
    def _resolve_model(core_module: Any) -> Any:
        model = getattr(core_module, "model", None)
        return model if model is not None else core_module

    @staticmethod
    def _is_main_process(trainer: "Trainer", core_module: "CoreModel") -> bool:
        if hasattr(core_module, "is_main_process"):
            return bool(core_module.is_main_process())
        if hasattr(trainer, "is_main_process"):
            return bool(trainer.is_main_process())
        return True

    def _save_payload(self, payload: dict[str, Any], stem: str) -> Path:
        if self.save_format == "pt":
            path = self.output_dir / f"{stem}.pt"
            torch.save(payload, path)
            return path
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "save_format='safetensors' requires the safetensors package"
            ) from exc

        tensor_path = self.output_dir / f"{stem}.safetensors"
        save_file({"rollout": payload["rollout"].contiguous()}, str(tensor_path))
        manifest_path = self.output_dir / f"{stem}.json"
        manifest = {
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "batch_idx": payload["batch_idx"],
            "rollout_shape": list(payload["rollout"].shape),
            "tensor_file": tensor_path.name,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._state["last_saved_manifest"] = str(manifest_path)
        return tensor_path


__all__ = [
    "AttentionRolloutCallback",
]
