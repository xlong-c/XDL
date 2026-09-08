"""后训练组件的轻量公开入口.

本模块只保存公开符号映射, 不在导入阶段加载 loss 或 callback 实现.
具体实现通过模块级 ``__getattr__`` 按需导入, 以保持 registry 和纯 loss
路径的依赖边界清晰.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DistillationLossBreakdown": (".losses", "DistillationLossBreakdown"),
    "GRPOLossBreakdown": (".losses", "GRPOLossBreakdown"),
    "STPOLossBreakdown": (".losses", "STPOLossBreakdown"),
    "ModelMergeCallback": (".checkpoint", "ModelMergeCallback"),
    "merge_checkpoints": (".checkpoint", "merge_checkpoints"),
    "ReferenceModelCallback": (".callbacks", "ReferenceModelCallback"),
    "RolloutBatch": (".callbacks", "RolloutBatch"),
    "RolloutCallback": (".callbacks", "RolloutCallback"),
    "SaveTrainableStateCallback": (".callbacks", "SaveTrainableStateCallback"),
    "LoRAParameters": (".lora", "LoRAParameters"),
    "normalize_lora_parameters": (".lora", "normalize_lora_parameters"),
    "dpo_loss": (".losses", "dpo_loss"),
    "stpo_loss": (".losses", "stpo_loss"),
    "grpo_loss": (".losses", "grpo_loss"),
    "distillation_loss": (".losses", "distillation_loss"),
    "feature_distillation_loss": (".losses", "feature_distillation_loss"),
    "kl_divergence_with_temperature": (".losses", "kl_divergence_with_temperature"),
    "relation_distillation_loss": (".losses", "relation_distillation_loss"),
    "tdm_loss": (".losses", "tdm_loss"),
    "tdm_loss_weighted": (".losses", "tdm_loss_weighted"),
    "normalize_lora_parameters": (".lora", "normalize_lora_parameters"),
}

__all__: list[str] = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    """按需加载后训练公开符号."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
