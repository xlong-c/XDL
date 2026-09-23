"""Bounded first-four-slot ablation using official DSpark and XDL Trainer.

DSPARK_FIRST4_CONFIG selects a structured OmegaConf YAML. Checkpoints are new
directories. Official block construction and CE labels are unchanged. Only slot
weights differ between matched experiments. FP32 trainable master parameters,
BF16 autocast and frozen embedding/head avoid small-update BF16 weight rounding.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer


@dataclass
class ExperimentConfig:
    workspace: str = "/root/workspace/xdl"
    cache_dir: str = ""
    output_dir: str = ""
    slot_weights: list[float] = field(default_factory=lambda: [1.0] * 7)
    seed: int = 42
    learning_rate: float = 3e-5
    updates: int | None = 128
    epochs: int | None = None
    accumulation: int = 4
    anchors: int = 32
    save_every: int = 64
    initial_checkpoint: str | None = None
    training_head: str | None = None
    prefix_loss_weight: float = 0.0
    residual_rnn_head: bool = False
    head_learning_rate: float = 0.001
    causal_draft: bool = False
    draft_layers: int | None = None
    eval_cache_dir: str | None = None
    eval_limit: int = 32
    l1_loss_weight: float = 0.0


def epoch_orders(size: int, epochs: int, seed: int) -> list[list[int]]:
    """Every epoch contains each sample exactly once, without padding or dropping."""
    if size < 1 or epochs < 1:
        raise ValueError("Positive dataset size and epoch count required")
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        indices = list(range(size))
        rng.shuffle(indices)
        result.append(indices)
    return result


def accumulation_window(
    batch_idx: int, size: int, accumulation: int
) -> tuple[bool, bool, int]:
    """Use the actual tail window length when an epoch is not divisible by accumulation."""
    if not 0 <= batch_idx < size or accumulation < 1:
        raise ValueError("Invalid accumulation window")
    start = batch_idx // accumulation * accumulation
    length = min(accumulation, size - start)
    return batch_idx == start, batch_idx + 1 == start + length, length


def weighted_slot_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean CE over valid anchor/slot pairs with normalized nonnegative weights."""
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    valid_weights = mask.float() * weights
    return (per_token * valid_weights).sum() / valid_weights.sum().clamp_min(1)


def expected_prefix_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Negative expected matched prefix length under teacher-forced sampling.

    This is a differentiable training surrogate, not measured greedy acceptance.
    Invalid slots terminate the prefix and have no gradient contribution.
    """
    log_p = -F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    valid_prefix = mask.to(log_p.dtype).cumprod(dim=-1)
    survival = log_p.cumsum(dim=-1).exp() * valid_prefix
    blocks = mask[..., 0].sum().clamp_min(1)
    return -survival.sum() / blocks


def acceptance_proxy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy teacher-forced match per slot and accepted prefix length.

    Slot j is credited only when every earlier slot in the same block also
    matched, mirroring the prefix property of the deployed verifier, so the mean
    accepted length is the quantity that bounds chain acceptance. Returns the
    raw sums (slot hits, slot totals, accepted total, block total) so callers
    can aggregate across microbatches or files before dividing. This is an
    offline surrogate on cached trajectories and is not emitted acceptance.
    """
    hit = ((logits.argmax(dim=-1) == targets) & mask).to(torch.float32)
    accepted = hit.cumprod(dim=-1)
    return hit.sum(dim=(0, 1)), mask.sum(dim=(0, 1)), accepted.sum(), mask[..., 0].sum()


def aligned_distribution_l1(
    logits: torch.Tensor, target_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1 distance between draft and target next-token distributions per slot.

    Matches the official aligned-target L1 term: it pushes the whole draft
    distribution toward the deployed target's distribution rather than only the
    realised token, which is what the acceptance criterion compares. Requires
    cached target_last_hidden_states so the teacher logits are the target's own.
    """
    draft_probs = logits.float().softmax(dim=-1)
    target_probs = target_logits.detach().float().softmax(dim=-1)
    distance = (draft_probs - target_probs).abs().sum(dim=-1)
    valid = mask.float()
    return (distance * valid).sum() / valid.sum().clamp_min(1)


def residual_rnn_step(
    head: torch.nn.Module,
    state: torch.Tensor,
    prev_embeddings: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix-conditioned residual over the pretrained Markov correction."""
    joined = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
    gate, candidate, output = head.joint_proj(joined).chunk(3, dim=-1)
    gate = gate.sigmoid()
    new_state = gate * state + (1 - gate) * candidate.tanh()
    bias = head.project_bias(prev_embeddings + output.tanh())
    return new_state, bias


def causal_noise_embed(
    embed_tokens: torch.nn.Module,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    """Input at slot j is token anchor+j; its label remains anchor+j+1."""
    positions = anchor_positions[..., None] + torch.arange(
        block_size, device=input_ids.device
    )
    valid = block_keep_mask[..., None] & (positions < input_ids.shape[1])
    selected = input_ids.gather(
        1, positions.clamp(max=input_ids.shape[1] - 1).flatten(1)
    )
    selected = torch.where(valid.flatten(1), selected, mask_token_id)
    return embed_tokens(selected)


def causal_mask_predicate(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> Any:
    def mask(
        b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        del h
        block = q // block_size
        context = (kv < seq_len) & (kv < anchor_positions[b, block])
        draft = (
            (kv >= seq_len)
            & ((kv - seq_len) // block_size == block)
            & (kv - seq_len <= q)
        )
        return (context | draft) & block_keep_mask[b, block]

    return mask


def causal_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> Any:
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        causal_mask_predicate(anchor_positions, block_keep_mask, seq_len, block_size),
        B=anchor_positions.shape[0],
        H=None,
        Q_LEN=anchor_positions.shape[1] * block_size,
        KV_LEN=seq_len + anchor_positions.shape[1] * block_size,
        device=device,
    )


class CacheSequence(Dataset):
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.files = sorted(Path(cfg.cache_dir).glob("*.pt"))
        if not self.files:
            raise ValueError("No training cache files")
        self.orders = epoch_orders(len(self.files), cfg.epochs or 1, cfg.seed)
        if cfg.epochs is not None:
            self.order = self.orders[0]
            return
        rng = random.Random(cfg.seed)
        self.order: list[int] = []
        count = cfg.updates * cfg.accumulation
        while len(self.order) < count:
            indices = list(range(len(self.files)))
            rng.shuffle(indices)
            self.order.extend(indices)
        self.order = self.order[:count]

    def set_epoch(self, epoch: int) -> None:
        self.order = self.orders[epoch - 1]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = torch.load(
            self.files[self.order[index]],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        keys = ["input_ids", "loss_mask", "target_hidden_states"]
        if "target_last_hidden_states" in sample:
            keys.append("target_last_hidden_states")
        return {k: sample[k] for k in keys}


class FirstFourCoreModel(CoreModel):
    def __init__(
        self, cfg: ExperimentConfig, dataset: CacheSequence | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.drafter = None
        self.update_count = 0
        self.supervised_pairs = 0
        self.started = time.perf_counter()
        self.dataset = dataset
        self.seen_samples = 0
        self.epoch_loss_sum = 0.0
        self.metric_hits: torch.Tensor | None = None
        self.metric_slots: torch.Tensor | None = None
        self.metric_chain = 0.0
        self.metric_blocks = 0.0

    def reset_metrics(self) -> None:
        self.metric_hits = None
        self.metric_slots = None
        self.metric_chain = 0.0
        self.metric_blocks = 0.0

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.epoch_loss_sum = 0.0
        if self.cfg.epochs is not None:
            self.dataset.set_epoch(self.trainer.current_epoch)

    def on_epoch_end(self) -> None:
        if self.cfg.epochs is None:
            return
        epoch = self.trainer.current_epoch
        self.save_snapshot(f"epoch_{epoch:02d}")
        record = {
            "epoch": epoch,
            "updates": self.update_count,
            "samples_this_epoch": len(self.dataset),
            "samples_seen": self.seen_samples,
            "mean_microbatch_loss": self.epoch_loss_sum / len(self.dataset),
            "prefix_loss_weight": self.cfg.prefix_loss_weight,
            "l1_loss_weight": self.cfg.l1_loss_weight,
        }
        if self.cfg.eval_cache_dir:
            record["held_out"] = self.evaluate_acceptance()
        with (Path(self.cfg.output_dir) / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    @torch.no_grad()
    def evaluate_acceptance(self) -> dict[str, Any]:
        """Held-out greedy match per slot and accepted prefix length.

        Runs the same surrogate as the training monitor on a paper-disjoint
        cache so checkpoint selection follows held-out acceptance rather than
        training loss. When the cache also carries the target's post-final-norm
        state, the teacher-referenced acceptance is reported alongside: it is the
        criterion of probabilistic speculative sampling and does not depend on
        the greedy tie-breaking. Offline, teacher-forced, not emitted acceptance.
        """
        files = sorted(Path(self.cfg.eval_cache_dir).glob("*.pt"))[
            : self.cfg.eval_limit
        ]
        if not files:
            raise ValueError(f"No evaluation cache files in {self.cfg.eval_cache_dir}")
        device = next(self.drafter.parameters()).device
        was_training = self.drafter.training
        self.drafter.eval()
        hits = None
        slots = None
        tv_sums = None
        chain = 0.0
        tv_chain = 0.0
        blocks = 0.0
        ce_sum = 0.0
        ce_count = 0.0
        for path in files:
            sample = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            batch = {
                k: sample[k].unsqueeze(0).to(device)
                for k in ("input_ids", "loss_mask", "target_hidden_states")
            }
            if sample.get("target_last_hidden_states") is not None:
                batch["target_last_hidden_states"] = (
                    sample["target_last_hidden_states"].unsqueeze(0).to(device)
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.drafter(**batch)
            logits = outputs.draft_logits.float()
            mask = outputs.eval_mask
            batch_hits, batch_slots, batch_accepted, batch_blocks = acceptance_proxy(
                logits, outputs.target_ids, mask
            )
            hits = batch_hits if hits is None else hits + batch_hits
            slots = batch_slots if slots is None else slots + batch_slots
            chain += float(batch_accepted)
            blocks += float(batch_blocks)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                outputs.target_ids.reshape(-1),
                reduction="none",
            ).reshape_as(mask)
            ce_sum += float((per_token * mask).sum())
            ce_count += float(mask.sum())
            if outputs.aligned_target_logits is not None:
                draft_probs = logits.softmax(dim=-1)
                target_probs = outputs.aligned_target_logits.float().softmax(dim=-1)
                accept = (1.0 - 0.5 * (draft_probs - target_probs).abs().sum(-1)).clamp(
                    0.0, 1.0
                ) * mask
                slot_sum = accept.sum(dim=(0, 1))
                tv_sums = slot_sum if tv_sums is None else tv_sums + slot_sum
                tv_chain += float(accept.cumprod(dim=-1).sum())
                del draft_probs, target_probs, accept
            del outputs, logits, per_token
        self.drafter.train(was_training)
        return {
            "files": len(files),
            "blocks": int(blocks),
            "ce": ce_sum / max(ce_count, 1.0),
            "match": [float(hits[i] / slots[i].clamp_min(1)) for i in range(len(hits))],
            "chain_match": chain / max(blocks, 1.0),
            "tv_accept_rate": (
                [float(tv_sums[i] / slots[i].clamp_min(1)) for i in range(len(hits))]
                if tv_sums is not None
                else None
            ),
            "tau_probabilistic": (
                tv_chain / max(blocks, 1.0) if tv_sums is not None else None
            ),
        }

    def setup(self, stage: str = "fit") -> None:
        if self.drafter is not None:
            return
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import Qwen3Config

        root = Path(self.cfg.workspace)
        sys.path.insert(0, str(root / "third_party"))
        from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

        torch.manual_seed(self.cfg.seed)
        checkpoint = (
            Path(self.cfg.initial_checkpoint)
            if self.cfg.initial_checkpoint
            else root / "downloads/MiniCPM5-2B-DSpark"
        )
        self.payload = json.loads((checkpoint / "config.json").read_text())
        if self.payload.get("residual_rnn_head") and not self.cfg.residual_rnn_head:
            raise ValueError("Residual RNN checkpoint requires residual_rnn_head=true")
        self.payload["rope_theta"] = self.payload["rope_parameters"]["rope_theta"]
        self.payload["num_anchors"] = self.cfg.anchors
        if self.cfg.draft_layers is not None:
            if not 1 <= self.cfg.draft_layers <= self.payload["num_hidden_layers"]:
                raise ValueError(
                    "draft_layers must select a nonempty prefix of source layers"
                )
            self.payload["num_hidden_layers"] = self.cfg.draft_layers
            self.payload["layer_types"] = self.payload["layer_types"][
                : self.cfg.draft_layers
            ]
        if self.cfg.causal_draft:
            self.payload["causal_draft"] = True
        elif self.payload.get("causal_draft"):
            raise ValueError("Causal checkpoint requires causal_draft=true")
        if self.cfg.residual_rnn_head:
            self.payload["markov_head_type"] = "rnn"
            self.payload["residual_rnn_head"] = True
        config = Qwen3Config.from_dict(self.payload)
        config._attn_implementation = "flex_attention"
        self.drafter = Qwen3DSparkModel(config)
        state = load_file(str(checkpoint / "model.safetensors"))
        if self.cfg.draft_layers is not None:
            state = {
                k: v
                for k, v in state.items()
                if not k.startswith("layers.")
                or int(k.split(".")[1]) < self.cfg.draft_layers
            }
        missing, unexpected = self.drafter.load_state_dict(state, strict=False)
        allowed_missing = {"embed_tokens.weight", "lm_head.weight"}
        if self.cfg.residual_rnn_head:
            allowed_missing |= {
                "markov_head.joint_proj.weight",
                "markov_head.joint_proj.bias",
            }
        assert set(missing) <= allowed_missing and not unexpected
        if self.cfg.causal_draft:
            import types

            original = type(self.drafter).forward
            namespace = dict(original.__globals__)
            namespace.update(
                create_noise_embed=causal_noise_embed,
                create_dspark_attention_mask=causal_attention_mask,
            )
            forward = types.FunctionType(
                original.__code__,
                namespace,
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            self.drafter.forward = types.MethodType(forward, self.drafter)
        if self.cfg.residual_rnn_head:
            import types

            head = self.drafter.markov_head
            if "markov_head.joint_proj.weight" in missing:
                # Preserve the pretrained vanilla correction exactly at step zero.
                with torch.no_grad():
                    head.joint_proj.weight[2 * head.markov_rank :].zero_()
                    head.joint_proj.bias[2 * head.markov_rank :].zero_()
            head._rnn_step = types.MethodType(residual_rnn_step, head)
        with safe_open(
            str(root / "downloads/MiniCPM5-2B-bf16/model-00000-of-00001.safetensors"),
            framework="pt",
        ) as handle:
            for module, key in [
                (self.drafter.embed_tokens, "model.embed_tokens.weight"),
                (self.drafter.lm_head, "lm_head.weight"),
            ]:
                module.to(dtype=torch.bfloat16)
                module.weight.data.copy_(handle.get_tensor(key))
                module.requires_grad_(False)
        if self.cfg.training_head:
            weight = torch.load(
                self.cfg.training_head, map_location="cpu", weights_only=True
            )
            if weight.shape != self.drafter.lm_head.weight.shape:
                raise ValueError("Training head shape does not match draft vocabulary")
            self.drafter.lm_head.weight.data.copy_(weight)
        self.drafter.confidence_head.requires_grad_(False)
        self.probe_before = self.drafter.fc.weight.detach().clone()

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        params = [p for p in self.drafter.parameters() if p.requires_grad]
        if self.cfg.residual_rnn_head:
            special = list(self.drafter.markov_head.joint_proj.parameters())
            special_ids = {id(p) for p in special}
            params = [
                {"params": [p for p in params if id(p) not in special_ids]},
                {
                    "params": special,
                    "lr_scale": self.cfg.head_learning_rate / self.cfg.learning_rate,
                },
            ]
        return [
            torch.optim.AdamW(
                params,
                lr=self.cfg.learning_rate,
                weight_decay=0.0,
            )
        ]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        optimizer = self.optimizers[0]
        if not self.cfg.l1_loss_weight:
            batch.pop("target_last_hidden_states", None)
        if self.cfg.epochs is not None:
            start, boundary, divisor = accumulation_window(
                batch_idx, len(self.dataset), self.cfg.accumulation
            )
        else:
            start, boundary, divisor = (
                self.is_accumulation_start,
                self.is_accumulation_boundary,
                self.cfg.accumulation,
            )
        if start:
            optimizer.zero_grad(set_to_none=True)
            self.reset_metrics()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.drafter(**batch)
            weights = torch.tensor(
                self.cfg.slot_weights, device=outputs.draft_logits.device
            )
            loss = weighted_slot_ce(
                outputs.draft_logits, outputs.target_ids, outputs.eval_mask, weights
            )
            if self.cfg.prefix_loss_weight:
                loss = loss + self.cfg.prefix_loss_weight * expected_prefix_loss(
                    outputs.draft_logits, outputs.target_ids, outputs.eval_mask
                )
            if self.cfg.l1_loss_weight:
                assert outputs.aligned_target_logits is not None, (
                    "l1_loss_weight requires cached target_last_hidden_states"
                )
                loss = loss + self.cfg.l1_loss_weight * aligned_distribution_l1(
                    outputs.draft_logits,
                    outputs.aligned_target_logits,
                    outputs.eval_mask,
                )
        with torch.no_grad():
            hits, slots, accepted, blocks = acceptance_proxy(
                outputs.draft_logits.detach().float(),
                outputs.target_ids,
                outputs.eval_mask,
            )
            self.metric_hits = (
                hits if self.metric_hits is None else self.metric_hits + hits
            )
            self.metric_slots = (
                slots if self.metric_slots is None else self.metric_slots + slots
            )
            self.metric_chain += float(accepted)
            self.metric_blocks += float(blocks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        self.manual_backward(loss / divisor)
        self.seen_samples += 1
        self.epoch_loss_sum += float(loss.detach())
        self.supervised_pairs += int(outputs.eval_mask.sum())
        if boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.drafter.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm")
            self.update_count += 1
            warmup = min(self.update_count / 16, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = self.cfg.learning_rate * warmup * group.get("lr_scale", 1)
            optimizer.step()
            record = {
                "update": self.update_count,
                "epoch": self.trainer.current_epoch,
                "samples_seen": self.seen_samples,
                "loss_last_microbatch": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "supervised_pairs": self.supervised_pairs,
                "elapsed_seconds": time.perf_counter() - self.started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "chain_match": self.metric_chain / max(self.metric_blocks, 1.0),
                "match": [
                    float(self.metric_hits[i] / self.metric_slots[i].clamp_min(1))
                    for i in range(len(self.metric_hits))
                ],
            }
            with (Path(self.cfg.output_dir) / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if self.update_count % 8 == 0:
                print(json.dumps(record), flush=True)
            if self.cfg.epochs is None and self.update_count % self.cfg.save_every == 0:
                self.save_snapshot(f"step_{self.update_count:04d}")
        self.log("loss", float(loss.detach()), prefix="train")

    def save_snapshot(self, name: str) -> None:
        from safetensors.torch import save_file

        out = Path(self.cfg.output_dir) / name
        out.mkdir(exist_ok=False)
        state = {
            k: v.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for k, v in self.drafter.state_dict().items()
            if not k.startswith(("embed_tokens.", "lm_head."))
        }
        save_file(state, str(out / "model.safetensors"))
        (out / "config.json").write_text(json.dumps(self.payload, indent=2))
        delta = (self.drafter.fc.weight.detach().cpu() - self.probe_before).abs()
        (out / "update_evidence.json").write_text(
            json.dumps(
                {
                    "updates": self.update_count,
                    "fc_max_abs_change": float(delta.max()),
                    "fc_fraction_changed": float((delta > 0).float().mean()),
                    "supervised_pairs": self.supervised_pairs,
                },
                indent=2,
            )
        )


def main() -> None:
    path = Path(
        os.environ.get("DSPARK_FIRST4_CONFIG", str(Path(__file__).with_suffix(".yaml")))
    )
    cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.load(path))
    )
    if len(cfg.slot_weights) != 7 or any(w <= 0 for w in cfg.slot_weights):
        raise ValueError("Exactly seven positive slot weights required")
    if cfg.prefix_loss_weight < 0:
        raise ValueError("prefix_loss_weight must be nonnegative")
    if cfg.l1_loss_weight < 0:
        raise ValueError("l1_loss_weight must be nonnegative")
    if cfg.l1_loss_weight and not cfg.training_head:
        raise ValueError("l1_loss_weight requires the deployed training head")
    if cfg.eval_limit < 1:
        raise ValueError("eval_limit must be positive")
    if cfg.learning_rate <= 0 or cfg.head_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if (cfg.updates is None) == (cfg.epochs is None):
        raise ValueError("Specify exactly one of updates or epochs")
    if (cfg.updates or cfg.epochs) < 1 or cfg.accumulation < 1:
        raise ValueError("Positive training budget and accumulation required")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.structured(cfg), out / "config.yaml")
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    dataset = CacheSequence(cfg)
    (out / "data_order.json").write_text(
        json.dumps(
            {
                "files": [str(p) for p in dataset.files],
                "order": dataset.order,
                "epoch_orders": dataset.orders if cfg.epochs is not None else None,
                "driver_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    model = FirstFourCoreModel(cfg, dataset)
    trainer = Trainer(
        max_epochs=cfg.epochs or 1,
        device="cuda",
        precision=None,
        gradient_accumulation_steps=cfg.accumulation,
    )
    trainer.fit(model, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    expected_updates = (
        cfg.updates
        if cfg.epochs is None
        else cfg.epochs * ((len(dataset) + cfg.accumulation - 1) // cfg.accumulation)
    )
    assert model.update_count == expected_updates
    expected_samples = len(dataset) * (cfg.epochs or 1)
    assert model.seen_samples == expected_samples
    if cfg.epochs is not None:
        assert model.current_epoch == cfg.epochs
        assert len(list(out.glob("epoch_*/model.safetensors"))) == cfg.epochs
    (out / "complete.json").write_text(
        json.dumps(
            {
                "updates": model.update_count,
                "supervised_pairs": model.supervised_pairs,
                "epochs": cfg.epochs,
                "samples_seen": model.seen_samples,
                "expected_samples": expected_samples,
            }
        )
    )


if __name__ == "__main__":
    main()
