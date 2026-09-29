"""Clean body prose, group duplicates before splitting, cache actual AR states.

Local input/output paths are explicit. DSPARK_DATA_PHASE selects prepare/cache.
No Zotero originals are modified; existing output directories are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/root/workspace/xdl")
OUT = Path(
    os.environ.get("DSPARK_DATA_OUT", str(ROOT / "data/dspark_restart/first4_v1"))
)


def prepare_wide() -> None:
    """Re-select body prose with a wider length window, keeping every other filter.

    The v1 pilot window of 70..160 words dropped 5210 of 10553 candidate
    paragraphs for length alone. This phase re-runs the identical cleaning and
    paper-level grouping with a wider window, and asserts that every paper keeps
    the v1 split so held-out papers stay held out.
    """
    min_words = int(os.environ.get("DSPARK_MIN_WORDS", "40"))
    max_words = int(os.environ.get("DSPARK_MAX_WORDS", "400"))
    source = ROOT / "data/dspark_restart/first4_v1"
    body_source = Path(
        os.environ.get(
            "DSPARK_BODY_SOURCE",
            str(ROOT / "data/dspark_restart/body_v1/body_candidates.jsonl"),
        )
    )
    OUT.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in body_source.read_text().splitlines()]
    parent = {r["source_sha256"]: r["source_sha256"] for r in rows}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen: dict[str, str] = {}
    for row in rows:
        source_key = row["source_sha256"]
        keys = [
            "family:" + row["paper_family_candidate"],
            "text:" + digest(row["text"].lower()),
        ]
        words = re.findall(r"[a-z]+", row["text"].lower())
        keys.extend(
            "span:" + digest(" ".join(words[i : i + 30]))
            for i in range(0, len(words) - 29, 15)
        )
        for key in keys:
            if key in seen:
                union(source_key, seen[key])
            else:
                seen[key] = source_key
    v1_rows = [
        json.loads(line)
        for line in (source / "cleaned_body.jsonl").read_text().splitlines()
    ]
    v1_group_of_source = {r["source_sha256"]: r["group_id"] for r in v1_rows}
    v1_split_of_source = {r["source_sha256"]: r["split"] for r in v1_rows}
    group_of_source = {s: find(s) for s in parent}
    members_of_group: dict[str, list[str]] = defaultdict(list)
    for source_key, group in group_of_source.items():
        members_of_group[group].append(source_key)
    reserved = json.loads(
        (
            ROOT / "data/dspark_restart/body_v1/reserved_development_sources.json"
        ).read_text()
    )
    development_groups = {
        group_of_source[s] for s in reserved["source_sha256"] if s in parent
    }
    # Splits are inherited from the frozen v1 paper assignment, never re-derived: a
    # wider candidate pool can merge two v1 families through a newly found paragraph,
    # and re-hashing a merged group would silently move held-out papers into train.
    # Groups that merge papers from two different v1 splits are dropped instead.
    splits: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for group, members in members_of_group.items():
        inherited = {v1_split_of_source[m] for m in members if m in v1_split_of_source}
        if len(inherited) > 1:
            conflicts.append(
                {
                    "group": group,
                    "v1_splits": sorted(inherited),
                    "members": len(members),
                    "v1_papers": sum(1 for m in members if m in v1_split_of_source),
                }
            )
            continue
        if inherited:
            splits[group] = inherited.pop()
            continue
        n = int(digest("42:" + group)[:8], 16) / 2**32
        splits[group] = (
            "development"
            if group in development_groups
            else (
                "train"
                if n < 0.75
                else "development"
                if n < 0.85
                else "calibration"
                if n < 0.90
                else "final_test"
            )
        )
    for row in v1_rows:
        group = group_of_source[row["source_sha256"]]
        if group in splits:
            assert splits[group] == row["split"], (
                f"Paper split drifted for {row['group_id']}"
            )
    kept = []
    rejected = Counter()
    unique = set()
    for row in rows:
        text = row["text"]
        reason = None
        if group_of_source[row["source_sha256"]] not in splits:
            reason = "group_merges_two_v1_splits"
        elif not min_words <= row["word_count"] <= max_words:
            reason = "outside_wide_prose_length"
        elif re.search(r"[=∑∫∏√∈≤≥⊙→←]|\b[A-Za-z]\s+\d+[A-Za-z]|\bhttps?://", text):
            reason = "math_or_url_layout_review"
        elif not re.match(r"[A-Z][A-Za-z]", text):
            reason = "fragment_or_non_prose_start"
        elif (
            re.search(r"\b(?:Figure|Table|Algorithm)\s+\d", text)
            and text.count(";") > 4
        ):
            reason = "possible_caption_table"
        elif digest(text.lower()) in unique:
            reason = "duplicate_paragraph"
        if reason:
            rejected[reason] += 1
            continue
        unique.add(digest(text.lower()))
        row = dict(row, group_id=group_of_source[row["source_sha256"]])
        row["split"] = splits[row["group_id"]]
        row["quality_status"] = "conservative_prose_automated_wide"
        kept.append(row)
    with (OUT / "cleaned_body.jsonl").open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    selections = {
        s: [r for r in kept if r["split"] == s] for s in ("train", "development")
    }
    assert not (
        {r["group_id"] for r in selections["train"]}
        & {r["group_id"] for r in selections["development"]}
    )
    for split, selected in selections.items():
        (OUT / f"full_{split}.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2)
        )
    report = {
        "source": str(body_source),
        "length_window_words": [min_words, max_words],
        "input_paragraphs": len(rows),
        "cleaned_paragraphs": len(kept),
        "selection": {
            s: {"paragraphs": len(v), "groups": len({r["group_id"] for r in v})}
            for s, v in selections.items()
        },
        "split_paragraph_counts": dict(Counter(r["split"] for r in kept)),
        "rejected": dict(rejected),
        "split_inheritance": {
            "strategy": "frozen v1 paper split, inherited by v2 groups; conflicts dropped",
            "conflicting_groups": conflicts,
            "v1_papers_preserved": len(v1_rows),
            "groups_without_v1_paper": sum(
                1
                for group, members in members_of_group.items()
                if not any(m in v1_split_of_source for m in members)
            ),
        },
        "v1_paper_split_preserved": True,
        "v1_paragraphs_recovered": sum(
            1 for row in kept if not (70 <= row["word_count"] <= 160)
        ),
        "held_out_untouched": {
            s: sum(1 for r in kept if r["split"] == s)
            for s in ("calibration", "final_test")
        },
        "seed": 42,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "selection_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def prepare_full() -> None:
    """Reuse the frozen paper split; select every retained training/dev paragraph."""
    import shutil

    source = ROOT / "data/dspark_restart/first4_v1"
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ("cleaned_body.jsonl", "manifest.json"):
        shutil.copyfile(source / name, OUT / name)
    rows = [
        json.loads(line)
        for line in (OUT / "cleaned_body.jsonl").read_text().splitlines()
    ]
    selections = {
        s: [r for r in rows if r["split"] == s] for s in ("train", "development")
    }
    assert not (
        {r["group_id"] for r in selections["train"]}
        & {r["group_id"] for r in selections["development"]}
    )
    reused = Counter()
    for split, selected in selections.items():
        (OUT / f"full_{split}.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2)
        )
        destination = OUT / "cache" / split
        destination.mkdir(parents=True)
        for row in selected:
            old = source / "cache" / split / f"{row['paragraph_id']}.pt"
            if old.exists():
                os.link(old, destination / old.name)
                reused[split] += 1
    (OUT / "selection_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "selection": "all retained train/development paragraphs",
                "paragraphs": {s: len(v) for s, v in selections.items()},
                "groups": {
                    s: len({r["group_id"] for r in v}) for s, v in selections.items()
                },
                "reused_cache_files": dict(reused),
                "split_sha256": hashlib.sha256(
                    (source / "cleaned_body.jsonl").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(line)
        for line in (ROOT / "data/dspark_restart/body_v1/body_candidates.jsonl")
        .read_text()
        .splitlines()
    ]
    parent = {r["source_sha256"]: r["source_sha256"] for r in rows}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen: dict[str, str] = {}
    for row in rows:
        source = row["source_sha256"]
        keys = [
            "family:" + row["paper_family_candidate"],
            "text:" + digest(row["text"].lower()),
        ]
        # Shared long 30-word spans also group near-duplicate versions conservatively.
        words = re.findall(r"[a-z]+", row["text"].lower())
        keys.extend(
            "span:" + digest(" ".join(words[i : i + 30]))
            for i in range(0, len(words) - 29, 15)
        )
        for key in keys:
            if key in seen:
                union(source, seen[key])
            else:
                seen[key] = source
    reserved = json.loads(
        (
            ROOT / "data/dspark_restart/body_v1/reserved_development_sources.json"
        ).read_text()
    )
    development_groups = {find(s) for s in reserved["source_sha256"] if s in parent}
    splits: dict[str, str] = {}
    for key in sorted({find(s) for s in parent}):
        n = int(digest("42:" + key)[:8], 16) / 2**32
        splits[key] = (
            "development"
            if key in development_groups
            else (
                "train"
                if n < 0.75
                else "development"
                if n < 0.85
                else "calibration"
                if n < 0.90
                else "final_test"
            )
        )
    kept = []
    rejected = Counter()
    unique = set()
    for row in rows:
        text = row["text"]
        reason = None
        if not 70 <= row["word_count"] <= 160:
            reason = "outside_pilot_prose_length"
        elif re.search(r"[=∑∫∏√∈≤≥⊙→←]|\b[A-Za-z]\s+\d+[A-Za-z]|\bhttps?://", text):
            reason = "math_or_url_layout_review"
        elif not re.match(r"[A-Z][A-Za-z]", text):
            reason = "fragment_or_non_prose_start"
        elif (
            re.search(r"\b(?:Figure|Table|Algorithm)\s+\d", text)
            and text.count(";") > 4
        ):
            reason = "possible_caption_table"
        elif digest(text.lower()) in unique:
            reason = "duplicate_paragraph"
        if reason:
            rejected[reason] += 1
            continue
        unique.add(digest(text.lower()))
        row = dict(row, group_id=find(row["source_sha256"]))
        row["split"] = splits[row["group_id"]]
        row["quality_status"] = "conservative_prose_pilot_automated"
        kept.append(row)
    with (OUT / "cleaned_body.jsonl").open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pilot = {}
    for split, budget in [("train", 96), ("development", 16)]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept:
            if row["split"] == split:
                grouped[row["group_id"]].append(row)
        groups = sorted(grouped, key=lambda k: digest("pilot42:" + k))
        chosen = []
        for group in groups:
            candidates = sorted(grouped[group], key=lambda r: digest(r["paragraph_id"]))
            chosen.append(candidates[0])
            if len(chosen) == budget:
                break
        assert len(chosen) == budget, (split, len(chosen))
        pilot[split] = chosen
        (OUT / f"pilot_{split}.json").write_text(
            json.dumps(chosen, ensure_ascii=False, indent=2)
        )
    assert not (
        {r["group_id"] for r in pilot["train"]}
        & {r["group_id"] for r in pilot["development"]}
    )
    report = {
        "input_paragraphs": len(rows),
        "cleaned_paragraphs": len(kept),
        "split_paragraph_counts": dict(Counter(r["split"] for r in kept)),
        "split_group_counts": dict(Counter(splits.values())),
        "rejected": dict(rejected),
        "pilot_counts": {s: len(v) for s, v in pilot.items()},
        "seed": 42,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Automatic conservative prose filtering is not full formula/PDF quality certification.",
            "Paper families combine existing metadata and shared long spans; unidentified revisions can remain.",
            "No final-test model evaluation performed.",
        ],
        "source_groups": {s: find(s) for s in parent},
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_groups"}, indent=2),
        flush=True,
    )


def cache() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "xqt"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xqt.model.minicpm5_chat import render_translation_input_ids

    cache_root = OUT / "cache"
    resume = os.environ.get("DSPARK_DATA_RESUME", "0") == "1"
    selection = os.environ.get("DSPARK_DATA_SELECTION", "pilot")
    cache_root.mkdir(exist_ok=resume)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "downloads/MiniCPM5-2B-bf16")
    model = (
        AutoModelForCausalLM.from_pretrained(
            ROOT / "downloads/MiniCPM5-2B-bf16",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    records = []
    with torch.inference_mode():
        for split in ["development", "train"]:
            destination = cache_root / split
            destination.mkdir(exist_ok=resume)
            samples = json.loads((OUT / f"{selection}_{split}.json").read_text())
            for index, sample in enumerate(samples):
                target = destination / f"{sample['paragraph_id']}.pt"
                quarantine = destination / f"{sample['paragraph_id']}.quarantine.json"
                if resume and target.exists():
                    cached = torch.load(
                        target, map_location="cpu", weights_only=False, mmap=True
                    )
                    if cached["sample"] != sample:
                        raise ValueError(f"Cache source mismatch: {target}")
                    records.append(dict(cached["meta"], status="saved"))
                    del cached
                    continue
                if resume and quarantine.exists():
                    records.append(json.loads(quarantine.read_text()))
                    continue
                ids = torch.tensor(
                    [render_translation_input_ids(tokenizer, sample["text"])],
                    device="cuda",
                )
                generated = model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=384,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=[1, 130073],
                    pad_token_id=1,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                sequence = generated.sequences[0]
                prompt = ids.shape[1]
                response = sequence[prompt:]
                text = tokenizer.decode(response, skip_special_tokens=True)
                completed = int(response[-1]) in (1, 130073)
                meta = {
                    "paragraph_id": sample["paragraph_id"],
                    "split": split,
                    "group_id": sample["group_id"],
                    "prompt_tokens": prompt,
                    "response_tokens": len(response),
                    "ended_with_eos": completed,
                    "translation": text,
                }
                if not completed or len(response) < 32:
                    meta["status"] = "quarantine_incomplete_or_short"
                    records.append(meta)
                    quarantine.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2)
                    )
                    print(
                        split, index + 1, "/", len(samples), meta["status"], flush=True
                    )
                    del generated
                    continue
                terminal = model(
                    sequence[-1:].reshape(1, 1),
                    past_key_values=generated.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                steps = list(generated.hidden_states) + [terminal.hidden_states]
                selected = torch.cat(
                    [
                        torch.cat([s[i + 1] for i in (1, 10, 20, 30, 39)], dim=-1)[0]
                        for s in steps
                    ],
                    dim=0,
                )
                last = torch.cat([s[-1][0] for s in steps], dim=0)
                assert len(sequence) == selected.shape[0] == last.shape[0]
                mask = torch.zeros(len(sequence), dtype=torch.long)
                mask[prompt:] = 1
                payload = {
                    "input_ids": sequence.cpu(),
                    "loss_mask": mask,
                    "target_hidden_states": selected.cpu(),
                    "target_last_hidden_states": last.cpu(),
                    "sample": sample,
                    "meta": meta,
                    "provenance": "actual incremental BF16 SDPA generate, batch=1; terminal EOS forward appended",
                }
                temporary = target.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                temporary.replace(target)
                meta["status"] = "saved"
                records.append(meta)
                if index % 25 == 0:
                    temporary_manifest = cache_root / "generation_manifest.tmp"
                    temporary_manifest.write_text(
                        json.dumps(records, ensure_ascii=False, indent=2)
                    )
                    temporary_manifest.replace(cache_root / "generation_manifest.json")
                (cache_root / "progress.json").write_text(
                    json.dumps(
                        {
                            "split": split,
                            "index": index + 1,
                            "selected": len(samples),
                            "processed_records": len(records),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                )
                print(
                    split,
                    index + 1,
                    "/",
                    len(samples),
                    meta["response_tokens"],
                    flush=True,
                )
                del generated, terminal, selected, last, steps, payload
    (cache_root / "generation_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    (cache_root / "complete.json").write_text(
        json.dumps(
            {
                "records": len(records),
                "status_counts": dict(Counter(r["status"] for r in records)),
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    phase = os.environ.get("DSPARK_DATA_PHASE", "prepare")
    if phase == "prepare":
        prepare()
    elif phase == "prepare_full":
        prepare_full()
    elif phase == "prepare_wide":
        prepare_wide()
    elif phase == "cache":
        cache()
    else:
        raise ValueError(phase)
