"""显存与梯度相关的通用工具.

三个函数都是无生命周期、低副作用的工具, 上层 (callback / Trainer / 任务代码)
按需显式调用:

- ``enable_gradient_checkpointing``: 打开模型梯度检查点;
- ``activation_offload_context``: 反向用的大激活同步卸载到 pinned CPU;
- ``global_grad_norm``: 计算全局梯度范数 (兼容 DDP/FSDP 集合语义).
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

import torch

logger = logging.getLogger(__name__)
_warned_grad_norm_fallback = False


def enable_gradient_checkpointing(model: torch.nn.Module) -> bool:
    """开启模型的梯度检查点 (激活重算).

    优先调用 ``gradient_checkpointing_enable()`` (HF 风格), 其次尝试
    ``set_gradient_checkpointing(True)``. 两者都没有时返回 False.
    """
    for method_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
        method = getattr(model, method_name, None)
        if callable(method):
            method()
            return True
    setter = getattr(model, "set_gradient_checkpointing", None)
    if callable(setter):
        setter(True)
        return True
    return False


@contextmanager
def activation_offload_context(
    min_bytes: int = 32 << 20,
    pinned: bool = True,
) -> Iterator[None]:
    """把反向用的大 CUDA 激活张量临时卸载到 CPU 的上下文管理器.

    Args:
        min_bytes: 超过该字节数的 CUDA 张量才卸载; 小张量 (mask/freqs_cis 等)
            留在 GPU, 避免 CPU 往返改变 stride 后触发算子的对齐检查.
        pinned: 是否使用 pinned CPU 内存.

    卸载使用同步 D2H/H2D 拷贝: 非阻塞拷贝配合 Python GC 存在竞态窗口,
    反向重算可能读到损坏数据.
    """
    hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
    if hooks is None:
        logger.warning(
            "当前 PyTorch 不支持 saved_tensors_hooks, 激活卸载被跳过"
        )
        yield
        return

    min_bytes = max(0, int(min_bytes))

    def pack(tensor: Any) -> Any:
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cuda"
            and tensor.numel() * tensor.element_size() >= min_bytes
        ):
            cpu_copy = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=pinned,
                memory_format=torch.contiguous_format,
            )
            cpu_copy.copy_(tensor, non_blocking=False)
            return (cpu_copy, tensor.device)
        return tensor

    def unpack(payload: Any) -> Any:
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], torch.Tensor)
        ):
            cpu_copy, device = payload
            return cpu_copy.to(device, non_blocking=False)
        return payload

    with hooks(pack, unpack):
        yield


def global_grad_norm(
    parameters: Union[torch.nn.Module, Any],
    accelerator: Optional[Any] = None,
) -> Optional[torch.Tensor]:
    """计算全局梯度范数, 不修改任何梯度.

    分布式路径使用 ``accelerator.clip_grad_norm_(..., float("inf"))``,
    兼容 DDP/FSDP 的集合语义; 失败时返回 None (调用方按未检测处理).
    """
    global _warned_grad_norm_fallback
    params = (
        list(parameters.parameters())
        if isinstance(parameters, torch.nn.Module)
        else list(parameters)
    )
    if not params:
        return None
    try:
        if accelerator is not None:
            return accelerator.clip_grad_norm_(params, float("inf"))
        return torch.nn.utils.clip_grad_norm_(params, float("inf"))
    except Exception as exc:  # 集合语义不支持时降级为不检测
        if not _warned_grad_norm_fallback:
            _warned_grad_norm_fallback = True
            logger.warning("全局梯度范数检测失败, 已跳过 (后续不再提示): %s", exc)
        return None
