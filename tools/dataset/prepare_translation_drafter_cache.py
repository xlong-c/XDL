"""翻译 Drafter 的 target 缓存生成脚本.

为什么存在: DSpark / DFlash / Eagle3 三种时序草稿模型都只在「给定目标模型的
隐藏状态」这个条件下学习预测未来 token, 因此训练循环里不需要反复前向 2B 的
目标模型. 官方 DeepSpec 的做法是把目标模型的表征预先算好落盘
(scripts/data/prepare_target_cache.py), 本脚本复刻同一语义, 但做了两处
面向本项目的有意偏离, 都记录在 manifest 里:

1. **本地 HF 贪心生成**, 而不是官方的 OpenAI 兼容 sglang server + temperature=0.7.
   本项目的验收口径是 greedy exact verification, 训练语料必须来自同一条
   解码路径, 否则草稿学到的是采样分布的众数而非 argmax 目标.
2. **每条样本一个 .pt**, 而不是官方的二进制 shard + mmap 索引.
   本项目语料只有几百条量级, 单卡训练也不需要分布式分片协议, 简单格式
   更容易断点续跑和逐条核对.

严格复刻的部分 (这些是承重语义, 改动会静默改变训练目标):

- ``target_hidden_states`` = 逐层 forward hook 抓取的 post-layer 输出, 按
  ``target_layer_ids`` 升序拼接. 这与参考实现 ``extract_context_feature`` 的
  ``hidden_states[layer_id + 1]`` 完全等价, 而不是 HF ``output_hidden_states``
  里按 index 数的同一批张量 (两者数值相同, 但前者不需要保留全部 43 层).
- ``target_last_hidden_states`` = final norm **之后**的表征, 等价于 HF
  ``CausalLMOutput.last_hidden_state``. 必须是 norm 后, 因为 L1 蒸馏项要求
  ``lm_head(target_last_hidden_states)`` 就是目标模型的真实分布.
- ``loss_mask`` 只在生成的回复区间为 1, prompt 区间为 0. 终止 token 计入
  loss, 否则草稿学不会在该停的地方停.

产物写到 ``data/drafter_translation/cache/{split}/``, 输入是
``build_translation_drafter_data.py`` 产出的 prompt jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 显式配置 (按 XDL 规范不使用 argparse, 采用代码内显式常量)
# ---------------------------------------------------------------------------
MODEL_PATH: str = "downloads/MiniCPM5-2B-bf16"
PROMPTS_DIR: Path = Path("data/drafter_translation")
CACHE_DIR: Path = PROMPTS_DIR / "cache"

# 与 downloads/MiniCPM5-2B-DSpark/config.json 完全一致, 不能改.
TARGET_LAYER_IDS: Tuple[int, ...] = (1, 10, 20, 30, 39)
MAX_NEW_TOKENS: int = 512
EOS_TOKEN_IDS: Tuple[int, ...] = (1, 130073)
PAD_TOKEN_ID: int = 1

# 分批大小按 bucket 给: 同一 bucket 内的 prompt 长度完全相同 (1528 / 3576 /
# 7672), 所以批量生成不需要给 prompt 补 padding, 只有回复长度的差异需要裁剪.
# 实测 2k 序列上 bs=8 的总吞吐是 bs=1 的 12 倍 (393.8 vs 32.7 tok/s), 而显存
# 只多出 1GB 量级; 8k 档的 KV cache 是 2k 档的四倍, 所以降到 4.
BATCH_SIZES: Dict[str, int] = {"2k": 8, "4k": 8, "8k": 4}
DEFAULT_BATCH_SIZE: int = 4

DEVICE: str = "cuda"
DEVICE_MAP: Optional[str] = None  # 单卡时保持 None, 显式 .to(DEVICE)
DTYPE: torch.dtype = torch.bfloat16
ATTN_IMPLEMENTATION: str = "sdpa"  # 与官方 prepare_target_cache 一致
RESUME: bool = True


def _get_backbone(model: Any) -> Any:
    """返回承载 ``layers`` / ``norm`` / ``embed_tokens`` 的骨干模块."""

    return getattr(model, "model", model)


def _hook_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"不支持的 hook 输出类型: {type(output)!r}")


def _capture_hidden_states(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """一次前向同时抓逐层表征与 final norm 之后的表征.

    返回 ``(target_hidden_states, target_last_hidden_states)``, 形状分别是
    ``[B, S, len(layer_ids) * hidden]`` 与 ``[B, S, hidden]``.
    """

    backbone = _get_backbone(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def capture(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer_id] = _hook_tensor(output).detach()

        return hook

    def capture_last(_module: Any, _inputs: Any, output: Any) -> None:
        # final norm 的输出就是 HF 的 last_hidden_state; 用 hook 而不是
        # output_hidden_states=True, 后者会保留全部 43 层, 在 8k 序列上白白
        # 多占约 1.5GB.
        captured[-2] = _hook_tensor(output).detach()

    try:
        for layer_id in target_layer_ids:
            handles.append(backbone.layers[layer_id].register_forward_hook(capture(layer_id)))
        handles.append(backbone.norm.register_forward_hook(capture_last))
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        stacked = torch.cat([captured[i] for i in target_layer_ids], dim=-1)
        last = captured[-2]
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()

    return stacked, last


def _truncate_at_eos(sequence: torch.Tensor, n_prompt: int) -> int:
    """返回回复区间的真实长度 (含终止 token).

    ``generate`` 在批模式下会把不足批内最长的样本补成 ``pad_token_id``, 而
    MiniCPM5 的 pad 与 EOS 同为 id=1, 无法从数值上区分「生成的终止符」和
    「批填充」. 因此取回复区间里第一个 EOS 即停 -- 自回归实际会生成的内容也
    正是这一段.
    """

    response = sequence[n_prompt:]
    eos_hits = torch.isin(
        response, torch.tensor(EOS_TOKEN_IDS, device=sequence.device)
    )
    if bool(eos_hits.any()):
        return int(eos_hits.to(torch.int32).argmax().item()) + 1
    return int(response.shape[0])


def _greedy_generate_batch(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """非思考模式下的批量贪心续写, 返回 ``[B, prompt + response]``."""

    with torch.no_grad():
        return model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=list(EOS_TOKEN_IDS),
            pad_token_id=PAD_TOKEN_ID,
            use_cache=True,
        )


def build_batch(
    *,
    model: Any,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """批量生成回复并批量抓取目标表征.

    同 bucket 的 prompt 等长, 所以生成阶段完全不需要给 prompt 补 padding; 只有
    回复长度不一, 抓表征时补到批内等长并用 attention_mask 屏蔽, 抓完逐条裁回
    真实长度.
    """

    # 同一个 bucket 的 prompt 通常等长, 但不保证: 论文池耗尽时最后一条文档会短于
    # 目标 token 数 (实测 2k 档 284 条里有 1 条只有 724). 直接 stack 会抛
    # "expected sequence of length N at dim 1", 而且会让整批一起失败. 所以这里
    # 仍然补 padding -- 补在右侧并用 attention_mask 屏蔽; 生成时所有序列都从
    # max_prompt_width 之后续写, 因此回复的起点是统一偏移.
    n_prompts = [len(record["prompt_ids"]) for record in records]
    max_prompt_width = max(n_prompts)
    prompt_ids = torch.full(
        (len(records), max_prompt_width), PAD_TOKEN_ID, dtype=torch.long, device=DEVICE
    )
    prompt_attention = torch.zeros(
        (len(records), max_prompt_width), dtype=torch.long, device=DEVICE
    )
    for index, record in enumerate(records):
        width = n_prompts[index]
        prompt_ids[index, :width] = torch.tensor(
            record["prompt_ids"], dtype=torch.long, device=DEVICE
        )
        prompt_attention[index, :width] = 1

    generated = _greedy_generate_batch(
        model=model, prompt_ids=prompt_ids, attention_mask=prompt_attention
    )

    sequences: List[torch.Tensor] = []
    for index, n_prompt in enumerate(n_prompts):
        n_response = _truncate_at_eos(generated[index], max_prompt_width)
        if n_response <= 0:
            raise RuntimeError(f"{records[index]['doc_id']}: 目标模型没有生成任何 token")
        sequences.append(
            torch.cat(
                [
                    prompt_ids[index, :n_prompt],
                    generated[index, max_prompt_width : max_prompt_width + n_response],
                ]
            ).clone()
        )

    widths = [int(sequence.shape[0]) for sequence in sequences]
    max_width = max(widths)
    padded = torch.full(
        (len(sequences), max_width), PAD_TOKEN_ID, dtype=torch.long, device=DEVICE
    )
    pad_mask = torch.zeros(
        (len(sequences), max_width), dtype=torch.long, device=DEVICE
    )
    for index, sequence in enumerate(sequences):
        padded[index, : widths[index]] = sequence
        pad_mask[index, : widths[index]] = 1

    target_hidden, target_last = _capture_hidden_states(
        model=model,
        input_ids=padded,
        attention_mask=pad_mask,
        target_layer_ids=TARGET_LAYER_IDS,
    )

    samples: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        n_prompt, width = n_prompts[index], widths[index]
        loss_mask = torch.zeros(width, dtype=torch.uint8, device=DEVICE)
        # 终止 token 也计入 loss: 草稿必须学会在该停的位置停.
        loss_mask[n_prompt:] = 1
        ended_with_eos = int(sequences[index][-1].item()) in EOS_TOKEN_IDS
        samples.append(
            {
                "doc_id": record["doc_id"],
                "bucket": record["bucket"],
                "target_layer_ids": list(TARGET_LAYER_IDS),
                "input_ids": sequences[index].to("cpu"),
                "attention_mask": pad_mask[index, :width].to("cpu"),
                "loss_mask": loss_mask.to("cpu"),
                "target_hidden_states": target_hidden[index, :width].to("cpu"),
                "target_last_hidden_states": target_last[index, :width].to("cpu"),
                "meta": {
                    "prompt_tokens": n_prompt,
                    "response_tokens": width - n_prompt,
                    "total_tokens": width,
                    "ended_with_eos": ended_with_eos,
                    "hit_max_new_tokens": (width - n_prompt >= MAX_NEW_TOKENS and not ended_with_eos),
                    "source_ids": record["source_ids"],
                    "source_count": record["source_count"],
                    "target_model": MODEL_PATH,
                    "decoding": "greedy",
                    "non_thinking": True,
                    "batch_size": len(records),
                },
            }
        )
    return samples


def load_records(split: str) -> List[Dict[str, Any]]:
    path = PROMPTS_DIR / f"prompts_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"缺少 prompt 文件: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_split(
    *,
    model: Any,
    split: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """按 bucket 分批跑完一个 split.

    必须按 bucket 分组而不是按文件顺序直接切批: 批量生成要求批内 prompt 等长,
    而 bucket 正是「prompt 长度相同」这一等价类.
    """

    records = load_records(split)
    if limit is not None:
        records = records[:limit]
    out_dir = CACHE_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_bucket.setdefault(str(record["bucket"]), []).append(record)

    done = 0
    skipped = 0
    failed: List[Dict[str, str]] = []
    response_tokens: List[int] = []
    total_tokens: List[int] = []
    started = time.time()
    total = len(records)

    for bucket in sorted(by_bucket):
        group = by_bucket[bucket]
        batch_size = BATCH_SIZES.get(bucket, DEFAULT_BATCH_SIZE)
        pending = [
            record
            for record in group
            if not (RESUME and (out_dir / f"{record['doc_id']}.pt").exists())
        ]
        skipped += len(group) - len(pending)
        print(
            f"  [{split}/{bucket}] {len(group)} 条, 待处理 {len(pending)}, "
            f"batch_size={batch_size}",
            flush=True,
        )

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            try:
                samples = build_batch(model=model, records=chunk)
            except Exception as exc:  # noqa: BLE001 - 一批失败不应中断整体
                failed.extend(
                    {"doc_id": record["doc_id"], "error": repr(exc)} for record in chunk
                )
                print(
                    f"  [fail] {[r['doc_id'] for r in chunk]}: {exc}", flush=True
                )
                torch.cuda.empty_cache()
                continue

            for sample in samples:
                torch.save(sample, out_dir / f"{sample['doc_id']}.pt")
                done += 1
                response_tokens.append(int(sample["meta"]["response_tokens"]))
                total_tokens.append(int(sample["meta"]["total_tokens"]))

            processed = done + skipped
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{split}] {processed}/{total} "
                f"(saved={done} skipped={skipped} failed={len(failed)}) "
                f"resp_tok_mean={sum(response_tokens) / max(len(response_tokens), 1):.0f} "
                f"{rate:.2f} samples/s  已用 {elapsed / 60:.1f} 分钟",
                flush=True,
            )

    summary = {
        "split": split,
        "records": len(records),
        "saved": done,
        "skipped_existing": skipped,
        "failed": failed,
        "batch_sizes": {b: BATCH_SIZES.get(b, DEFAULT_BATCH_SIZE) for b in by_bucket},
        "response_tokens_mean": (
            sum(response_tokens) / len(response_tokens) if response_tokens else 0.0
        ),
        "total_tokens_mean": (
            sum(total_tokens) / len(total_tokens) if total_tokens else 0.0
        ),
        "elapsed_seconds": time.time() - started,
        "target_layer_ids": list(TARGET_LAYER_IDS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "non_thinking": True,
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    torch.set_float32_matmul_precision("high")
    print(f"[*] 加载目标模型 {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    del tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    if DEVICE_MAP:
        model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)
    model.eval()
    print(f"[*] 模型就绪, 抓取层: {list(TARGET_LAYER_IDS)}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in ("val", "train"):
        print(f"[*] 处理 {split} ...", flush=True)
        summaries.append(run_split(model=model, split=split))
        torch.cuda.empty_cache()

    manifest = {
        "schema": "xdl.translation_drafter.cache.v1",
        "model_path": MODEL_PATH,
        "target_layer_ids": list(TARGET_LAYER_IDS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "non_thinking": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "splits": summaries,
        "notes": [
            "target_hidden_states 是逐层 hook 的 post-layer 输出, 等价于参考实现 "
            "extract_context_feature 的 hidden_states[layer_id + 1].",
            "target_last_hidden_states 取 final norm 之后, 因此 "
            "lm_head(target_last_hidden_states) 就是目标模型的真实 logits.",
            "response 由目标模型贪心生成; prompt 区间 loss_mask=0, 回复区间 "
            "loss_mask=1 (含终止 token).",
        ],
    }
    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[✓] 完成", flush=True)
    for summary in summaries:
        print(
            f"  {summary['split']}: saved={summary['saved']} "
            f"skipped={summary['skipped_existing']} failed={len(summary['failed'])} "
            f"resp_tok_mean={summary['response_tokens_mean']:.0f} "
            f"({summary['elapsed_seconds']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
