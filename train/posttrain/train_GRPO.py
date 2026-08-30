"""GRPO 后训练玩具参考入口.

用最小合成场景验证 XDL 后训练链路全接通:

    ReferenceModelCallback (冻结参考模型)
        + RolloutCallback (K 路 rollout -> 奖励打分 -> 参考 logps -> 注入 RolloutBatch)
        + grpo_loss (clipped surrogate + KL penalty)
        + CoreModel 手动优化 + Trainer 生命周期

玩具设定: policy 是一个可学习的对角高斯 N(mu, sigma), "latent" 是 1x8x8
张量; 奖励是负 MSE (latent 越接近固定目标图案分越高). prompt 仅作分组
标识, 不参与生成. 入口只检查 loss 有限, 不对随机 reward 趋势作收敛断言. 与真实扩
散后训练的差距: 真实场景中 rollout 是多步扩散采样, logps 是轨迹似然,
reward 是reward model 栈; 本入口只负责验证框架链路, 不负责算法保真.

运行:

    python train/posttrain/train_GRPO.py
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from xdl.post_training import (
    RolloutBatch,
    RolloutCallback,
    ReferenceModelCallback,
    grpo_loss,
)
from xdl.trainer import CoreModel, Trainer

# ---------------------------------------------------------------------------
# 显式配置 (无 CLI, 按仓库规范用代码内常量)
# ---------------------------------------------------------------------------

SEED: int = 0
LATENT_SHAPE: Tuple[int, int, int] = (1, 8, 8)
NUM_PROMPTS: int = 4          # 每个 batch 的 prompt 数 (B)
NUM_ROLLOUTS: int = 4         # 每个 prompt 的 rollout 数 (K)
DATASET_SIZE: int = 8         # 数据集大小 = 2 个 batch, 即每 epoch 2 步
MAX_EPOCHS: int = 60
LEARNING_RATE: float = 0.03
CLIP_EPSILON: float = 0.2
KL_BETA: float = 0.04
MAX_GRAD_NORM: float = 1.0

PROMPT_POOL: List[str] = ["alpha", "beta", "gamma", "delta"]


# ---------------------------------------------------------------------------
# 玩具组件: policy / reward
# ---------------------------------------------------------------------------


class ToyPolicy(nn.Module):
    """可学习的对角高斯 policy, 支持 sample 与 log_prob."""

    def __init__(self, latent_shape: Tuple[int, int, int] = LATENT_SHAPE) -> None:
        super().__init__()
        self.mu = nn.Parameter(0.1 * torch.randn(latent_shape))
        self.log_sigma = nn.Parameter(torch.full(latent_shape, -1.0))

    def sample(self, num_samples: int) -> torch.Tensor:
        """从 N(mu, sigma) 采样, 返回 (N, *latent_shape)."""
        with torch.no_grad():
            eps = torch.randn(num_samples, *self.mu.shape)
            return self.mu + torch.exp(self.log_sigma) * eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """对角高斯 log 概率密度, 返回 (N,)."""
        sigma2 = torch.exp(2.0 * self.log_sigma)
        per_dim = (
            -0.5 * ((x - self.mu) ** 2) / sigma2
            - self.log_sigma
            - 0.5 * math.log(2.0 * math.pi)
        )
        return per_dim.flatten(1).sum(1)


def make_target_pattern(latent_shape: Tuple[int, int, int]) -> torch.Tensor:
    """固定目标图案: 8x8 网格上的中心高斯凸包, 值域 (0, 1)."""
    h, w = latent_shape[1], latent_shape[2]
    ys = torch.arange(h).float() - (h - 1) / 2.0
    xs = torch.arange(w).float() - (w - 1) / 2.0
    grid = ys.view(-1, 1) ** 2 + xs.view(1, -1) ** 2
    pattern = torch.exp(-grid / 4.0)
    return pattern.view(1, h, w)


class ToyReward(nn.Module):
    """奖励 = 负 MSE, latent 越接近目标图案分越高 (上限 0)."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (N, *latent_shape) -> (N,) 每样本标量奖励."""
        mse = ((latents - self.target) ** 2).flatten(1).mean(1)
        return -mse


# ---------------------------------------------------------------------------
# rollout / ref logps 回调函数
# ---------------------------------------------------------------------------


def toy_rollout_fn(
    policy: nn.Module, prompts: List[str], num_rollouts: int
) -> torch.Tensor:
    """对每个 prompt 采样 K 个 latent, 返回 (B, K, *latent_shape)."""
    latents = policy.sample(len(prompts) * num_rollouts)
    return latents.view(len(prompts), num_rollouts, *latents.shape[1:])


def toy_ref_logp_fn(ref_model: nn.Module, rollout_latents: torch.Tensor) -> torch.Tensor:
    """冻结参考模型对 rollout 的 logps, 返回 (B, K), 回调内部 no_grad 调用."""
    b, k = rollout_latents.shape[0], rollout_latents.shape[1]
    flat = rollout_latents.reshape(b * k, *rollout_latents.shape[2:])
    return ref_model.log_prob(flat).view(b, k)


# ---------------------------------------------------------------------------
# CoreModel
# ---------------------------------------------------------------------------


class _IndexDataset(Dataset[int]):
    """仅提供 batch 长度的索引数据集, 语义由 rollout 回调接管."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class GRPOToyModel(CoreModel):
    """GRPO 后训练玩具任务: policy 对角高斯逼近奖励最高的 latent 区域."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.policy = ToyPolicy()
        # 注册为 nn.Module 属性, Trainer 自动迁移设备; ReferenceModelCallback 冻结
        self.ref_model = copy.deepcopy(self.policy)
        # reward model 也注册到 CoreModel, 保证与 policy 同设备
        self.reward_model = ToyReward(target)
        self._reward_history: List[float] = []
        self._loss_history: List[float] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if not isinstance(batch, RolloutBatch):
            raise RuntimeError(
                "GRPOToyModel 期望 RolloutCallback 注入的 RolloutBatch, "
                f"实际收到 {type(batch).__name__}"
            )

        latents: torch.Tensor = batch.rollout_latents
        rewards: torch.Tensor = batch.rollout_rewards
        ref_logps: torch.Tensor = batch.rollout_ref_logps
        if latents is None or rewards is None or ref_logps is None:
            raise RuntimeError("RolloutBatch 字段不完整, 检查 RolloutCallback 配置")

        b, k = latents.shape[0], latents.shape[1]
        flat = latents.reshape(b * k, *latents.shape[2:])
        # 带梯度的 policy logps (梯度流向 self.policy 的 mu/log_sigma)
        policy_logps = self.policy.log_prob(flat).view(b, k)

        # 组内 (每个 prompt 的 K 个 rollout) 归一化 advantage
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (
            rewards.std(dim=1, keepdim=True) + 1e-4
        )

        breakdown = grpo_loss(
            policy_logps,
            advantages,
            ref_logps,
            clip_epsilon=CLIP_EPSILON,
            kl_beta=KL_BETA,
        )

        self.manual_optimization_step(breakdown.total, max_grad_norm=MAX_GRAD_NORM)

        self.log("reward", float(rewards.mean()), prefix="train")
        self.log("advantage", breakdown.advantage.item(), prefix="train")
        self.log("kl_penalty", breakdown.kl_penalty.item(), prefix="train")
        self._reward_history.append(float(rewards.mean()))
        self._loss_history.append(float(breakdown.total.detach()))
        return breakdown.total

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        return None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    target = make_target_pattern(LATENT_SHAPE)
    model = GRPOToyModel(target)

    rollout_callback = RolloutCallback(
        reward_models={"target_bump": model.reward_model},
        prompt_pool=PROMPT_POOL,
        num_rollouts_per_prompt=NUM_ROLLOUTS,
        rollout_fn=toy_rollout_fn,
        ref_logp_fn=toy_ref_logp_fn,
    )
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        device="cpu",
        callbacks=[
            ReferenceModelCallback(),
            rollout_callback,
        ],
    )

    dataloader: DataLoader[int] = DataLoader(
        _IndexDataset(DATASET_SIZE), batch_size=NUM_PROMPTS, shuffle=False
    )
    trainer.fit(model, dataloader)

    steps_per_epoch = DATASET_SIZE // NUM_PROMPTS
    first_epoch = sum(model._reward_history[:steps_per_epoch]) / steps_per_epoch
    last_epoch = sum(model._reward_history[-steps_per_epoch:]) / steps_per_epoch
    print("GRPO toy 后训练完成")
    print(f"  mean reward: 首个 epoch {first_epoch:.4f} -> 末个 epoch {last_epoch:.4f}")
    # The toy run is a wiring smoke test, not a convergence benchmark.
    # Validate the actual optimization signal without making a fragile
    # assertion about stochastic reward trends.
    if not model._loss_history or not all(math.isfinite(v) for v in model._loss_history):
        raise RuntimeError("GRPO toy loss history contains non-finite values")


if __name__ == "__main__":
    main()
