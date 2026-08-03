"""xdl.utils.memory 工具函数测试."""

import torch

from xdl.utils.memory import (
    activation_offload_context,
    enable_gradient_checkpointing,
    global_grad_norm,
)


def test_enable_gradient_checkpointing_hf_style() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def gradient_checkpointing_enable(self) -> None:
            self.called = True

    model = Model()
    assert enable_gradient_checkpointing(model) is True
    assert model.called is True


def test_enable_gradient_checkpointing_fallback_setter() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = False

        def set_gradient_checkpointing(self, value: bool) -> None:
            self.value = value

    model = Model()
    assert enable_gradient_checkpointing(model) is True
    assert model.value is True


def test_enable_gradient_checkpointing_unsupported() -> None:
    model = torch.nn.Linear(2, 2)
    assert enable_gradient_checkpointing(model) is False


def test_activation_offload_context_keeps_cpu_tensors_untouched() -> None:
    tensor = torch.randn(4, 4)
    with activation_offload_context(min_bytes=1):
        assert tensor.device.type == "cpu"


def test_global_grad_norm_computes_finite_norm() -> None:
    model = torch.nn.Linear(4, 2)
    model(torch.randn(3, 4)).square().mean().backward()
    norm = global_grad_norm(model)
    assert norm is not None
    assert torch.isfinite(norm).item()
