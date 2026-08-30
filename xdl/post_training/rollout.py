"""Rollout callback for GRPO-style reinforcement learning.

Generates K rollouts per prompt, scores them with reward models, and
injects the augmented batch into the training step.  Designed for the
RL stage of diffusion model post-training (ref: Krea 2 Technical Report).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
import torch.nn as nn

from ..callbacks.base import Callback

if TYPE_CHECKING:
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer


@dataclass
class RolloutBatch:
    """Augmented batch injected by :class:`RolloutCallback`.

    The original batch is preserved as ``original_batch``.  Rollout
    fields are populated before ``training_step`` consumes them.
    """

    original_batch: Any = None
    rollout_latents: torch.Tensor | None = None  # (B, K, ...)
    rollout_rewards: torch.Tensor | None = None  # (B, K)
    rollout_ref_logps: torch.Tensor | None = None  # (B, K)


class RolloutCallback(Callback):
    """Generate rollouts and score with reward models for GRPO training.

    Lifecycle per batch (inside ``on_train_batch_start``):

    1. Sample prompts from the prompt pool.
    2. For each prompt, generate *K* rollouts with the policy model
       (no-grad inference).
    3. Compute reference log-probabilities for each rollout.
    4. Score each rollout with all configured reward models.
    5. Pack everything into a :class:`RolloutBatch` and replace the
       batch argument so ``training_step`` receives it.

    The user's ``CoreModel`` must expose:

    * ``core_module.policy`` - the trainable policy model.
    * ``core_module.ref_model`` (optional) - reference model for
      computing ``rollout_ref_logps``.  Falls back to ``policy`` if
      not present (degenerate on-policy mode).

    Reference log-probabilities are computed by ``ref_logp_fn`` under
    ``torch.no_grad()``; without it the callback raises instead of
    returning zero placeholders, which would silently corrupt the
    GRPO/DPO importance ratio.

    Reward models are provided as a ``dict[str, nn.Module]`` where each
    module's ``forward`` returns a scalar reward per sample.  Rewards
    from all models are summed (with optional per-model weights).
    """

    _POLICY_ATTR: str = "policy"
    _REF_ATTR: str = "ref_model"

    def __init__(
        self,
        reward_models: dict[str, nn.Module],
        prompt_pool: list[str] | None = None,
        num_rollouts_per_prompt: int = 4,
        rollout_fn: Callable[..., torch.Tensor] | None = None,
        ref_logp_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
        reward_fn: Callable[..., dict[str, torch.Tensor]] | None = None,
        reward_weights: dict[str, float] | None = None,
        rollout_every_n_steps: int = 1,
        prompt_selection_strategy: Literal["uniform", "adaptive"] = "uniform",
        policy_attr: str = _POLICY_ATTR,
        ref_attr: str = _REF_ATTR,
        priority: int = 50,
    ) -> None:
        """Args:
            reward_models: Name → reward model mapping.  Each model must
                accept a batch of images and return scalar rewards.
            prompt_pool: List of prompt strings.  Also accepts ``None``
                when the callback is configured purely for scoring (the
                user's training loop provides rollouts externally).
            num_rollouts_per_prompt: Number of rollouts (K) per prompt.
            rollout_fn: Function ``(policy, prompt, num_rollouts) -> latents``
                that runs diffusion inference.  If ``None``, the callback
                only handles reward scoring (user must inject rollouts).
            ref_logp_fn: Function ``(ref_model, rollout_latents) -> Tensor(B, K)``
                computing frozen-reference log-probabilities for each
                rollout; invoked under ``torch.no_grad()``.  Required for
                GRPO/DPO-style training - without it the callback raises
                instead of writing zero placeholders into
                ``rollout_ref_logps``.
            reward_fn: Function ``(reward_models, rollout_latents, prompts)
                -> dict[str, Tensor(N, K)]``.  If ``None``, each reward
                model is called independently with ``.forward(latents)``.
            reward_weights: Per-model weights for aggregating rewards.
                Default: equal weight 1.0 for each model.
            rollout_every_n_steps: Generate rollouts every N training steps.
            prompt_selection_strategy: ``"uniform"`` (random sampling) or
                ``"adaptive"`` (prioritize prompts with learning signal).
            policy_attr: Attribute name on ``core_module`` for the policy.
            ref_attr: Attribute name on ``core_module`` for the reference.
            priority: Callback priority.
        """
        super().__init__(priority=priority)
        self.reward_models = reward_models
        self.prompt_pool = prompt_pool or []
        self.num_rollouts_per_prompt = num_rollouts_per_prompt
        self._rollout_fn = rollout_fn
        self._ref_logp_fn = ref_logp_fn
        self._reward_fn = reward_fn
        self.reward_weights = reward_weights or {
            name: 1.0 for name in reward_models
        }
        self.rollout_every_n_steps = rollout_every_n_steps
        self.prompt_selection_strategy = prompt_selection_strategy
        self.policy_attr = policy_attr
        self.ref_attr = ref_attr

        # Adaptive prompt selection state
        self._prompt_stats: dict[str, dict[str, float]] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        self._step_count += 1

        if self._step_count % self.rollout_every_n_steps != 0:
            return batch

        if not self.prompt_pool:
            return batch

        # 1. Select prompts
        selected = self._select_prompts(len(batch) if has_len(batch) else 1)

        # 2. Generate rollouts
        rollout_latents = self._generate_rollouts(core_module, selected)

        # 3. Score with reward models
        rewards = self._score_rollouts(rollout_latents, selected)

        # 4. Compute ref logps
        ref_logps = self._compute_ref_logps(core_module, rollout_latents, selected)

        # 5. Pack augmented batch
        return RolloutBatch(
            original_batch=batch,
            rollout_latents=rollout_latents,
            rollout_rewards=rewards,
            rollout_ref_logps=ref_logps,
        )

    # ------------------------------------------------------------------
    # Prompt selection
    # ------------------------------------------------------------------

    def _select_prompts(self, batch_size: int) -> list[str]:
        pool = self.prompt_pool
        if not pool:
            return []

        if self.prompt_selection_strategy == "uniform":
            return random.choices(pool, k=batch_size)

        # Adaptive: skip saturated / noisy prompts
        eligible = []
        for p in pool:
            stats = self._prompt_stats.get(p)
            if stats is None:
                eligible.append(p)
                continue
            mean_r = stats.get("mean_reward", 0.5)
            std_r = stats.get("std_reward", 0.0)
            # Keep prompts that still provide learning signal
            if 0.1 < mean_r < 0.95 and std_r > 0.05:
                eligible.append(p)

        if not eligible:
            eligible = pool  # fallback

        return random.choices(eligible, k=batch_size)

    # ------------------------------------------------------------------
    # Rollout generation (delegated to user-provided function)
    # ------------------------------------------------------------------

    def _generate_rollouts(
        self, core_module: "CoreModel", prompts: list[str]
    ) -> torch.Tensor:
        """Run diffusion inference to generate K rollouts per prompt.

        Returns ``(B, K, ...)`` tensor.
        """
        if self._rollout_fn is None:
            raise RuntimeError(
                "RolloutCallback requires a rollout_fn for generation. "
                "Set rollout_fn when constructing the callback."
            )
        policy = getattr(core_module, self.policy_attr)
        return self._rollout_fn(policy, prompts, self.num_rollouts_per_prompt)

    def _compute_ref_logps(
        self,
        core_module: "CoreModel",
        rollout_latents: torch.Tensor,
        prompts: list[str],
    ) -> torch.Tensor:
        del prompts  # reserved for future log-prob computation hooks
        """Compute reference log-probabilities for each rollout.

        Returns ``(B, K)``.
        """
        if self._ref_logp_fn is None:
            raise RuntimeError(
                "RolloutCallback requires ref_logp_fn to fill "
                "rollout_ref_logps. Zero placeholders would silently "
                "corrupt the GRPO/DPO importance ratio. Pass "
                "ref_logp_fn=(ref, latents) -> Tensor(B, K), or compute "
                "reference log-probabilities in training_step."
            )
        ref = getattr(core_module, self.ref_attr, None)
        if ref is None:
            ref = getattr(core_module, self.policy_attr)
        with torch.no_grad():
            return self._ref_logp_fn(ref, rollout_latents)

    # ------------------------------------------------------------------
    # Reward scoring
    # ------------------------------------------------------------------

    def _score_rollouts(
        self, rollout_latents: torch.Tensor, prompts: list[str]
    ) -> torch.Tensor:
        """Score each rollout with all reward models.

        Returns ``(B, K)`` aggregated reward per rollout.
        """
        if self._reward_fn is not None:
            raw = self._reward_fn(self.reward_models, rollout_latents, prompts)
            return self._aggregate_rewards(raw)

        # Default path: call each reward model independently
        B, K = rollout_latents.shape[0], rollout_latents.shape[1]
        total = torch.zeros(
            B,
            K,
            device=rollout_latents.device,
            dtype=rollout_latents.dtype,
        )

        for name, rm in self.reward_models.items():
            # Reshape (B, K, ...) → (B*K, ...)
            flat = rollout_latents.reshape(B * K, *rollout_latents.shape[2:])
            with torch.no_grad():
                scores = rm(flat)
            scores = scores.reshape(B, K).to(
                device=rollout_latents.device,
                dtype=rollout_latents.dtype,
            )
            total = total + self.reward_weights.get(name, 1.0) * scores

            # Update adaptive stats
            self._update_prompt_stats(prompts, scores)

        return total

    @staticmethod
    def _aggregate_rewards(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum reward tensors across models.  Override in subclass for
        custom aggregation."""
        values = list(raw.values())
        if not values:
            return torch.tensor(0.0)
        result = values[0]
        for v in values[1:]:
            result = result + v
        return result

    def _update_prompt_stats(
        self, prompts: list[str], scores: torch.Tensor
    ) -> None:
        """Update running statistics for adaptive prompt selection."""
        for i, p in enumerate(prompts):
            row = scores[i]  # (K,)
            if p not in self._prompt_stats:
                self._prompt_stats[p] = {"mean_reward": 0.0, "std_reward": 0.0, "n": 0}
            stats = self._prompt_stats[p]
            n = stats["n"]
            old_mean = stats["mean_reward"]
            new_mean = row.mean().item()
            stats["mean_reward"] = (old_mean * n + new_mean) / (n + 1)
            stats["n"] = n + 1
            # Running std (Welford-style simplified)
            if n > 0:
                stats["std_reward"] = (
                    stats["std_reward"] * (n - 1) + (new_mean - old_mean) ** 2
                ) / n
            stats["std_reward"] = max(stats["std_reward"], 1e-8)


def has_len(obj: Any) -> bool:
    """Check if an object has a ``__len__``."""
    try:
        len(obj)  # type: ignore[arg-type]
        return True
    except (TypeError, NotImplementedError):
        return False


__all__ = [
    "RolloutBatch",
    "RolloutCallback",
]
