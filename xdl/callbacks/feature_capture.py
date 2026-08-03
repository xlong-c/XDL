"""Validation-time feature capture callback."""

from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch

from xdl.analysis.activation import ActivationCapture

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class FeatureCaptureCallback(Callback):
    """Capture intermediate activations during validation and write them to disk."""

    def __init__(
        self,
        module_names: Sequence[str],
        *,
        output_dir: str | Path,
        every_n_epochs: int = 1,
        first_val_batch_only: bool = True,
        max_batches_per_epoch: Optional[int] = None,
        file_prefix: str = "features",
        save_format: str = "pt",
        to_cpu: bool = True,
        detach: bool = True,
        unwrap_tuple: bool = True,
        tensor_transform: Optional[TensorTransform] = None,
        priority: int = 310,
    ) -> None:
        super().__init__(priority=priority)
        self.module_names = list(module_names)
        self.output_dir = Path(output_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.first_val_batch_only = bool(first_val_batch_only)
        self.max_batches_per_epoch = (
            max(1, int(max_batches_per_epoch)) if max_batches_per_epoch is not None else None
        )
        self.file_prefix = file_prefix
        self.save_format = save_format.lower()
        self.to_cpu = to_cpu
        self.detach = detach
        self.unwrap_tuple = unwrap_tuple
        self.tensor_transform = tensor_transform
        self._capture: Optional[ActivationCapture] = None
        self._captured_batches = 0

        if self.save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format: {save_format}")

    def on_validation_epoch_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        if not self._should_capture_epoch(trainer, core_module):
            return
        model = self._resolve_model(core_module)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._captured_batches = 0
        self._capture = ActivationCapture(
            model,
            self.module_names,
            to_cpu=self.to_cpu,
            unwrap_tuple=self.unwrap_tuple,
            detach=self.detach,
            tensor_transform=self.tensor_transform,
        )
        self._capture.__enter__()

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch, dataloader_idx
        if self._capture is None:
            return
        if self.max_batches_per_epoch is not None and self._captured_batches >= self.max_batches_per_epoch:
            self._teardown_capture()
            return
        if self.first_val_batch_only and batch_idx != 0:
            return
        self._write_records(trainer, core_module, batch_idx)
        self._captured_batches += 1
        if self.first_val_batch_only:
            self._teardown_capture()

    def on_validation_epoch_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        del trainer, core_module
        self._teardown_capture()

    def _should_capture_epoch(self, trainer: "Trainer", core_module: "CoreModel") -> bool:
        if not self._is_main_process(trainer, core_module):
            return False
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

    def _write_records(self, trainer: "Trainer", core_module: "CoreModel", batch_idx: int) -> None:
        assert self._capture is not None
        payload = {
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "batch_idx": int(batch_idx),
            "modules": {
                name: {
                    "shape": record.shape,
                    "value": record.value,
                }
                for name, record in self._capture.records.items()
            },
        }
        filename = (
            f"{self.file_prefix}_epoch{payload['epoch']:04d}"
            f"_step{payload['global_step']:08d}_batch{batch_idx:04d}"
        )
        saved_path = self._save_payload(payload, filename)
        self._state["last_saved_file"] = str(saved_path)
        self._state["last_saved_modules"] = list(self._capture.records)
        self._state["last_epoch"] = payload["epoch"]
        self._state["last_step"] = payload["global_step"]
        self._state["captured_batches_in_epoch"] = self._captured_batches + 1

    def _save_payload(self, payload: dict[str, Any], stem: str) -> Path:
        if self.save_format == "pt":
            path = self.output_dir / f"{stem}.pt"
            torch.save(payload, path)
            return path

        tensors = self._flatten_tensor_payload(payload["modules"])
        metadata = {
            "epoch": str(payload["epoch"]),
            "global_step": str(payload["global_step"]),
            "batch_idx": str(payload["batch_idx"]),
            "module_names": json.dumps(list(payload["modules"].keys())),
        }
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "save_format='safetensors' requires the safetensors package"
            ) from exc

        tensor_path = self.output_dir / f"{stem}.safetensors"
        save_file(tensors, str(tensor_path), metadata=metadata)
        manifest_path = self.output_dir / f"{stem}.json"
        manifest = {
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "batch_idx": payload["batch_idx"],
            "modules": {
                name: {"shape": record["shape"]}
                for name, record in payload["modules"].items()
            },
            "tensor_file": tensor_path.name,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._state["last_saved_manifest"] = str(manifest_path)
        return tensor_path

    @staticmethod
    def _flatten_tensor_payload(modules: dict[str, Any]) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for module_name, record in modules.items():
            value = record["value"]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"safetensors format only supports tensor activations, got {type(value)!r} for module {module_name}"
                )
            key = module_name.replace(".", "__")
            tensors[key] = value.contiguous()
        return tensors

    def _teardown_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.__exit__(None, None, None)
        self._capture = None


__all__ = [
    "FeatureCaptureCallback",
]
