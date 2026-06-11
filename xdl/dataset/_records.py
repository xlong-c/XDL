"""Internal helpers for record-backed datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]
Record = Dict[str, Any]


def _read_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise TypeError(f"JSONL record must be a mapping: {path}")
        records.append(dict(item))
    return records


def _read_json(path: Path) -> List[Record]:
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


def _read_csv(path: Path) -> List[Record]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_manifest_records(path: PathLike) -> List[Record]:
    """Load records from a jsonl, json, or csv manifest file."""

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(manifest_path)
    if suffix == ".json":
        return _read_json(manifest_path)
    if suffix == ".csv":
        return _read_csv(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def load_manifest_context(
    manifest_path: PathLike,
    *,
    base_dir: Optional[PathLike] = None,
) -> Tuple[Path, Path, List[Record]]:
    """Resolve manifest path, base dir, and non-empty records."""

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    records = load_manifest_records(resolved_manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {resolved_manifest_path}")

    resolved_base_dir = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else resolved_manifest_path.parent
    )
    return resolved_manifest_path, resolved_base_dir, records


def resolve_path(value: Any, base_dir: Path) -> Path:
    """Resolve a record path against the configured base directory."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def require_record_keys(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    """Validate that a record contains the required keys."""

    missing = [
        key
        for key in keys
        if key not in record or (not allow_empty and record.get(key) in (None, ""))
    ]
    if missing:
        if len(missing) == 1:
            raise KeyError(f"Manifest record requires key '{missing[0]}'")
        raise KeyError(f"Manifest record requires keys {missing}")


def resolve_record_path(record: Mapping[str, Any], key: str, base_dir: Path) -> Path:
    """Resolve a required path-like field from a record."""

    require_record_keys(record, (key,))
    return resolve_path(record[key], base_dir)


def first_present_value(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    """Return the first non-empty value from a record for the given keys."""

    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def build_sample_id(
    record: Mapping[str, Any],
    *,
    index: int,
    sample_id_key: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """Build a stable sample identifier from common record fields."""

    if sample_id_key and record.get(sample_id_key) not in (None, ""):
        return str(record[sample_id_key])
    for key in ("sample_id", "id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    if fallback not in (None, ""):
        return str(fallback)
    return str(index)
