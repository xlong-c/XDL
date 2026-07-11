"""Activation capture helpers for model interpretation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import nn


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ActivationRecord:
    """One captured module activation."""

    name: str
    value: Any
    shape: Optional[tuple[int, ...]]


def _resolve_module_lookup_name(name: str) -> str:
    return "" if name == "<root>" else name


def _normalize_output(
    output: Any,
    *,
    unwrap_tuple: bool,
    to_cpu: bool,
    detach: bool,
    tensor_transform: Optional[TensorTransform],
) -> Any:
    if unwrap_tuple and isinstance(output, (tuple, list)):
        output = output[0] if output else output
    if isinstance(output, torch.Tensor):
        tensor = output.detach() if detach else output
        if tensor_transform is not None:
            tensor = tensor_transform(tensor)
        return tensor.cpu() if to_cpu else tensor
    return output


class ActivationCapture(AbstractContextManager["ActivationCapture"]):
    """Context manager that records outputs from named submodules."""

    def __init__(
        self,
        model: nn.Module,
        module_names: Sequence[str],
        *,
        to_cpu: bool = True,
        unwrap_tuple: bool = True,
        detach: bool = True,
        tensor_transform: Optional[TensorTransform] = None,
    ) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.to_cpu = to_cpu
        self.unwrap_tuple = unwrap_tuple
        self.detach = detach
        self.tensor_transform = tensor_transform
        self.records: Dict[str, ActivationRecord] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationCapture":
        modules = dict(self.model.named_modules())
        missing = [
            name
            for name in self.module_names
            if _resolve_module_lookup_name(name) not in modules
        ]
        if missing:
            raise KeyError(f"Modules not found: {missing}")

        for name in self.module_names:
            module = modules[_resolve_module_lookup_name(name)]
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            normalized = _normalize_output(
                output,
                unwrap_tuple=self.unwrap_tuple,
                to_cpu=self.to_cpu,
                detach=self.detach,
                tensor_transform=self.tensor_transform,
            )
            shape: Optional[tuple[int, ...]] = None
            if isinstance(normalized, torch.Tensor):
                shape = tuple(int(dim) for dim in normalized.shape)
            self.records[name] = ActivationRecord(name=name, value=normalized, shape=shape)

        return hook

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()


def capture_activations(
    model: nn.Module,
    module_names: Sequence[str],
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    to_cpu: bool = True,
    unwrap_tuple: bool = True,
    detach: bool = True,
    tensor_transform: Optional[TensorTransform] = None,
    use_grad: bool = False,
) -> Dict[str, ActivationRecord]:
    """Run a model once and capture outputs from the named modules."""

    kwargs = dict(forward_kwargs or {})
    with ActivationCapture(
        model,
        module_names,
        to_cpu=to_cpu,
        unwrap_tuple=unwrap_tuple,
        detach=detach,
        tensor_transform=tensor_transform,
    ) as capture:
        if use_grad:
            model(*forward_args, **kwargs)
        else:
            with torch.no_grad():
                model(*forward_args, **kwargs)
    return capture.records


__all__ = [
    "ActivationCapture",
    "ActivationRecord",
    "capture_activations",
]
