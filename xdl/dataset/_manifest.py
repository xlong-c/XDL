"""Internal helpers for manifest-backed datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise TypeError(f"JSONL record must be a mapping: {path}")
        records.append(dict(item))
    return records


def _read_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise TypeError(f"JSON manifest must be a list or mapping: {path}")

    if not all(isinstance(item, dict) for item in records):
        raise TypeError(f"JSON manifest records must be mappings: {path}")
    return [dict(item) for item in records]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_manifest_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a jsonl, json, or csv manifest."""

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(manifest_path)
    if suffix == ".json":
        return _read_json(manifest_path)
    if suffix == ".csv":
        return _read_csv(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def resolve_path(value: Any, base_dir: Path) -> Path:
    """Resolve a manifest path against the manifest directory."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path
