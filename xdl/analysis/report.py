"""Small report helpers for XDL analysis results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested mappings into dotted keys."""

    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        flat_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(
                flatten_mapping(value, prefix=flat_key, separator=separator)
            )
        else:
            flattened[flat_key] = value
    return flattened


def normalize_record(record: object) -> dict[str, Any]:
    """Convert a mapping or dataclass-like object into a dictionary."""

    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
    elif isinstance(record, Mapping):
        raw = record
    else:
        raise TypeError("record must be a mapping or expose to_dict()")
    if not isinstance(raw, Mapping):
        raise TypeError("record.to_dict() must return a mapping")
    return dict(raw)


def records_to_rows(records: list[object]) -> list[dict[str, Any]]:
    """Normalize analysis records into flat row dictionaries."""

    return [flatten_mapping(normalize_record(record)) for record in records]


def write_json_report(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a mapping as a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def write_csv_report(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write row dictionaries to a CSV report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output_path


def write_markdown_summary(
    title: str,
    sections: Mapping[str, Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write a simple Markdown summary from sectioned metrics."""

    lines = [f"# {title}", ""]
    for section_name, values in sections.items():
        lines.extend([f"## {section_name}", "", "| Metric | Value |", "| --- | --- |"])
        for key, value in values.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_analysis_bundle(
    output_dir: str | Path,
    *,
    report_name: str,
    summary: Mapping[str, Any],
    records: list[object] | None = None,
    markdown_sections: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write a small bundle of JSON/CSV/Markdown analysis artifacts."""

    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    outputs["json"] = write_json_report(summary, bundle_dir / f"{report_name}.json")
    if records:
        outputs["csv"] = write_csv_report(records_to_rows(records), bundle_dir / f"{report_name}.csv")
    if markdown_sections:
        outputs["md"] = write_markdown_summary(
            report_name,
            markdown_sections,
            bundle_dir / f"{report_name}.md",
        )
    return outputs


__all__ = [
    "flatten_mapping",
    "normalize_record",
    "records_to_rows",
    "write_analysis_bundle",
    "write_csv_report",
    "write_json_report",
    "write_markdown_summary",
]
