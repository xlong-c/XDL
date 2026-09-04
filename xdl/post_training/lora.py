"""后训练 LoRA 参数的共享校验协议."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LoRAParameters:
    """PEFT LoRA 的框架级参数, 与具体 pipeline 无关."""

    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float


def normalize_lora_parameters(
    target_modules: Any,
    rank: Any,
    alpha: Any,
    dropout: Any,
) -> LoRAParameters:
    """规范化并校验 LoRA 参数, 返回不可变参数对象."""
    if isinstance(target_modules, str):
        modules = tuple(
            item.strip() for item in target_modules.split(",") if item.strip()
        )
    elif isinstance(target_modules, Sequence):
        modules = tuple(str(item).strip() for item in target_modules if str(item).strip())
    else:
        raise TypeError("target_modules must be a string or sequence of strings")
    if not modules:
        raise ValueError("target_modules must not be empty")

    normalized_rank = int(rank)
    normalized_alpha = int(alpha)
    normalized_dropout = float(dropout)
    if normalized_rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if normalized_alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not 0.0 <= normalized_dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")

    return LoRAParameters(
        target_modules=modules,
        rank=normalized_rank,
        alpha=normalized_alpha,
        dropout=normalized_dropout,
    )


__all__ = ["LoRAParameters", "normalize_lora_parameters"]
