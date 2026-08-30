"""Reference model management callback for preference optimization.

Manages a frozen reference model used during DPO / STPO / GRPO training.
The reference model is a snapshot of the policy at the start of training
(or periodically updated via EMA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

from ..callbacks.base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


class ReferenceModelCallback(Callback):
    """Freeze a reference model copy and optionally sync via EMA.

    Expected usage in a user ``CoreModel.__init__``::

        self.policy = MyDiffusionModel(...)
        self.ref_model = copy.deepcopy(self.policy)
        # ref_model is registered as an nn.Module attribute, so Trainer
        # migrates it to the device automatically.

    The callback freezes ``ref_model`` in ``on_fit_start``.  If
    ``ema_sync_every_n_steps`` is set, the reference model weights are
    periodically updated toward the policy weights via:

        θ_ref ← (1 - ema_decay) * θ_policy + ema_decay * θ_ref
    """

    _REF_MODEL_ATTR: str = "ref_model"
    _POLICY_ATTR: str = "policy"

    def __init__(
        self,
        ema_sync_every_n_steps: int | None = None,
        ema_decay: float = 0.999,
        ref_attr: str = _REF_MODEL_ATTR,
        policy_attr: str = _POLICY_ATTR,
        priority: int = 100,
    ) -> None:
        """Args:
            ema_sync_every_n_steps: If set, sync reference weights toward
                policy weights every N training steps (global step count).
                ``None`` means no sync after initial freeze.
            ema_decay: EMA decay factor (closer to 1 = slower update).
            ref_attr: Attribute name on ``core_module`` holding the
                reference model.  Default: ``"ref_model"``.
            policy_attr: Attribute name on ``core_module`` holding the
                policy model.  Default: ``"policy"``.
            priority: Callback priority (lower = earlier).
        """
        super().__init__(priority=priority)
        self.ema_sync_every_n_steps = ema_sync_every_n_steps
        self.ema_decay = ema_decay
        self.ref_attr = ref_attr
        self.policy_attr = policy_attr
        # Internal tracking
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        ref_model = self._get_ref(core_module)
        self._freeze(ref_model)
        self._state["ref_frozen"] = True

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._step_count += 1

        if self.ema_sync_every_n_steps is None:
            return
        if self._step_count % self.ema_sync_every_n_steps != 0:
            return

        policy = self._get_policy(core_module)
        ref = self._get_ref(core_module)
        self._ema_sync(policy, ref)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_ref(self, core_module: "CoreModel") -> nn.Module:
        ref = getattr(core_module, self.ref_attr, None)
        if ref is None:
            raise AttributeError(
                f"CoreModel has no attribute '{self.ref_attr}'. "
                f"Please set self.{self.ref_attr} in __init__."
            )
        if not isinstance(ref, nn.Module):
            raise TypeError(
                f"'{self.ref_attr}' must be an nn.Module, got {type(ref)}"
            )
        return ref

    def _get_policy(self, core_module: "CoreModel") -> nn.Module:
        policy = getattr(core_module, self.policy_attr, None)
        if policy is None:
            raise AttributeError(
                f"CoreModel has no attribute '{self.policy_attr}'. "
                f"Please set self.{self.policy_attr} in __init__."
            )
        return policy

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    def _ema_sync(self, policy: nn.Module, ref: nn.Module) -> None:
        """θ_ref ← (1 - decay) * θ_policy + decay * θ_ref."""
        with torch_no_grad():
            for r_param, p_param in zip(ref.parameters(), policy.parameters()):
                r_param.data.mul_(self.ema_decay).add_(
                    p_param.data, alpha=1.0 - self.ema_decay
                )


def torch_no_grad():
    """Lazy import-safe no_grad context."""
    import torch

    return torch.no_grad()


__all__ = [
    "ReferenceModelCallback",
]
