"""微调轨迹评测: 官方权重 + 每个 epoch 快照, 在留出文档上量接受长度.

口径: teacher-forced 逐槽 argmax 命中 —— 给定真实前缀, 块内第 k 槽的预测是否
等于目标模型在该位置的 argmax. 这是 greedy exact verification 下"该槽会不会被
接受"的直接对应物, 与训练 loss 无关 (组合 loss 里 confidence 项占大头, 它下降
不代表 argmax 变准).

主指标只用 val 两份留出文档: 439 条 train 文档全部在微调里见过, 拿它们报数只能
说明背下来了, 所以 train 抽样只作"拟合程度"的旁证.

用法 (在 xdl/xdl 目录下):
    PYTHONPATH=.:../third_party python3 tools/drafter/eval_finetune_trajectory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "third_party")
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.modeling.dspark.qwen3.config import build_draft_config  # noqa: E402

TARGET_MODEL = "downloads/MiniCPM5-2B-bf16"
TARGET_LAYER_IDS = [1, 10, 20, 30, 39]
BLOCK_SIZE = 7
MASK_TOKEN_ID = 75982
# 评测用官方值: >= 候选位置数 (约 511), 等价于全位置覆盖, 与训练时的 128 无关.
NUM_ANCHORS = 512
# 运行时 e2e 只生成 128 个 token, 接受率只被"回复开头"这一段决定.
EARLY_WINDOW = 128
CACHE = Path("data/drafter_translation/cache")

OFFICIAL = Path("downloads/MiniCPM5-2B-DSpark/model.safetensors")
EPOCH_DIR = Path("artifacts/translation_drafter/dspark/epochs")
TRAIN_STRIDE = 100  # train 抽样间隔


def draft_config():
    target_config = AutoConfig.from_pretrained(TARGET_MODEL)
    model_args = OmegaConf.create(
        {
            "num_draft_layers": 5,
            "target_layer_ids": TARGET_LAYER_IDS,
            "mask_token_id": MASK_TOKEN_ID,
            "num_anchors": NUM_ANCHORS,
            "block_size": BLOCK_SIZE,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "confidence_head_alpha": 1.0,
            "confidence_head_with_markov": True,
        }
    )
    return build_draft_config(target_config=target_config, model_args=model_args)


def build(ckpt: Path, cfg, embed, head) -> Qwen3DSparkModel:
    model = Qwen3DSparkModel(cfg)
    state = load_file(str(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise AssertionError(f"{ckpt}: 意外张量 {list(unexpected)[:5]}")
    if set(missing) - {"embed_tokens.weight", "lm_head.weight"}:
        raise AssertionError(f"{ckpt}: 缺非冻结张量 {sorted(set(missing))[:5]}")
    model.initialize_embeddings_and_head(embed_tokens=embed, lm_head=head, freeze=True)
    return model.to("cuda").to(torch.bfloat16).eval()


@torch.no_grad()
def stats(model, sample) -> dict:
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to("cuda"),
        "loss_mask": sample["loss_mask"].unsqueeze(0).long().to("cuda"),
        "target_hidden_states": sample["target_hidden_states"].unsqueeze(0).to("cuda"),
        "target_last_hidden_states": sample["target_last_hidden_states"]
        .unsqueeze(0)
        .to("cuda"),
    }
    out = model(**inputs)
    eval_mask = out.eval_mask
    hits = (out.draft_logits.argmax(-1) == out.target_ids) & eval_mask
    n_pos = int(eval_mask.any(dim=-1).sum().item())
    slot_hits = []
    for k in range(BLOCK_SIZE):
        denom = int(eval_mask[..., k].sum().item())
        slot_hits.append(
            float(hits[..., k].sum().item()) / denom if denom else float("nan")
        )
    accepted = hits.to(torch.int32).cumprod(dim=-1).sum(dim=-1).to(torch.float32)
    block_ok = eval_mask.any(dim=-1)
    tau = float(accepted[block_ok].mean().item()) if n_pos else float("nan")

    # 位置剖面: 运行时的接受率只覆盖"回复开头"这一小段, 而离线 τ 是整段 511 个
    # 位置的平均. 这里单独给出前 EARLY_WINDOW 个回复位置上的 τ, 才是与 e2e 可比
    # 的那一段 (anchor_positions 缺失时跳过).
    early = float("nan")
    early_n = 0
    anchors = getattr(out, "anchor_positions", None)
    if anchors is not None:
        lm = sample["loss_mask"].to(anchors.device)
        nz = (lm > 0).nonzero()
        if nz.numel():
            offset = anchors[0].to(torch.long) - int(nz[0].item())
            sel = block_ok[0] & (offset < EARLY_WINDOW)
            early_n = int(sel.sum().item())
            if early_n:
                early = float(accepted[0][sel].mean().item())
    ce = (
        float(
            F.cross_entropy(
                out.draft_logits[eval_mask].float(), out.target_ids[eval_mask]
            )
        )
        if int(eval_mask.sum().item())
        else float("nan")
    )
    result = {
        "tau": tau,
        "ce": ce,
        "slot_hits": slot_hits,
        "n_pos": n_pos,
        "tau_early": early,
        "n_early": early_n,
    }
    del inputs, out, eval_mask, hits, accepted
    return result


def main() -> None:
    ckpts = {"official": OFFICIAL}
    if EPOCH_DIR.exists():
        for d in sorted(EPOCH_DIR.glob("epoch_*")):
            ckpts[d.name] = d / "model.safetensors"
    absent = [k for k, v in ckpts.items() if not v.exists()]
    if absent:
        print(f"跳过不存在的权重: {absent}")

    val_files = sorted((CACHE / "val").glob("*.pt"))
    train_files = sorted((CACHE / "train").glob("*.pt"))[::TRAIN_STRIDE]
    pairs = [("val:" + p.stem, p) for p in val_files]
    pairs += [("train:" + p.stem, p) for p in train_files]
    loaded = [
        (name, torch.load(p, map_location="cpu", weights_only=False, mmap=True))
        for name, p in pairs
    ]
    print(f"样本: {[n for n, _ in loaded]}")
    print("(val 是留出文档, train 抽样只作拟合旁证)\n")

    cfg = draft_config()
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.bfloat16)
    embed, head = target.get_input_embeddings(), target.get_output_embeddings()

    val_names = {n for n, _ in loaded if n.startswith("val:")}
    header = (
        f"{'ckpt':<12}{'τ(val)':>9}{'τ(val,前128)':>13}{'CE(val)':>9}{'Δτ(val)':>9}   "
        f"{'每槽命中 0..6 (val)':<44}{'τ(train)':>10}"
    )
    print(header)
    print("-" * len(header))
    base = None
    for name, ckpt in ckpts.items():
        model = build(ckpt, cfg, embed, head)
        rows = []
        for sample_name, sample in loaded:
            st = stats(model, sample)
            st["name"] = sample_name
            rows.append(st)
        del model
        torch.cuda.empty_cache()

        val = [r for r in rows if r["name"] in val_names]
        tr = [r for r in rows if r["name"] not in val_names]
        tau_v = sum(r["tau"] for r in val) / len(val)
        ce_v = sum(r["ce"] for r in val) / len(val)
        tau_t = sum(r["tau"] for r in tr) / len(tr) if tr else float("nan")
        early = [r["tau_early"] for r in val if r["n_early"]]
        tau_e = sum(early) / len(early) if early else float("nan")
        slots = " ".join(
            f"{sum(r['slot_hits'][k] for r in val) / len(val):.3f}"
            for k in range(BLOCK_SIZE)
        )
        if base is None:
            base = tau_v
        print(
            f"{name:<12}{tau_v:>9.3f}{tau_e:>13.3f}{ce_v:>9.3f}{tau_v - base:>+9.3f}   "
            f"{slots:<44}{tau_t:>10.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
