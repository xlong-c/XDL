#!/usr/bin/env python3
"""Normalize common full-width punctuation to ASCII punctuation.

Usage:
    python scripts/normalize_punctuation.py

Environment variables:
    XDL_PUNCT_PATHS:
        Comma-separated files/directories to scan.
        Defaults to "docs,research".
    XDL_PUNCT_CHECK:
        Set to "1" to report files that need changes without writing them.
    XDL_PUNCT_EXTS:
        Comma-separated suffix allowlist. Defaults to common text/source files.

The script intentionally uses a fixed punctuation map instead of NFKC so it does
not rewrite Chinese text, full-width digits, units, math symbols, or model names.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_PATHS = "docs"
DEFAULT_EXTS = ".md,.txt,.rst,.html,.css,.js,.ts,.tsx,.py,.yaml,.yml,.toml"
IGNORE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "downloads",
    "third_party",
}

PUNCTUATION_MAP = {
    "，": ",",
    "。": ".",
    "、": ",",
    "；": ";",
    "：": ":",
    "！": "!",
    "？": "?",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "《": "<",
    "》": ">",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    "％": "%",
    "＃": "#",
    "＠": "@",
    "＆": "&",
    "＊": "*",
    "＋": "+",
    "－": "-",
    "＝": "=",
    "＾": "^",
    "＿": "_",
    "｀": "`",
    "｜": "|",
    "／": "/",
    "＼": "\\",
    "＜": "<",
    "＞": ">",
    "～": "~",
    "…": "...",
    "—": "-",
}


@dataclass(frozen=True)
class NormalizationResult:
    path: Path
    replacement_count: int


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_inside_repo(path: Path) -> None:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"拒绝扫描仓库外路径: {resolved}")


def is_outside_repo(path: Path) -> bool:
    """判断路径是否解析到仓库外 (例如指向兄弟仓库的软链)."""
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    return resolved != root and root not in resolved.parents


def normalize_text(text: str) -> tuple[str, int]:
    replacement_count = 0
    normalized_chars: list[str] = []
    for char in text:
        replacement = PUNCTUATION_MAP.get(char)
        if replacement is None:
            normalized_chars.append(char)
            continue
        normalized_chars.append(replacement)
        replacement_count += 1
    return "".join(normalized_chars), replacement_count


def should_skip_dir(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def discover_files(paths: list[str], suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        raw = REPO_ROOT / item
        if raw.is_symlink() and is_outside_repo(raw):
            print(f"[normalize_punctuation] skip out-of-repo symlink: {item}")
            continue
        path = raw.resolve()
        ensure_inside_repo(path)
        if not path.exists():
            print(f"[normalize_punctuation] skip missing path: {item}")
            continue
        if path.is_file():
            if path.suffix in suffixes:
                files.append(path)
            continue
        for candidate in path.rglob("*"):
            if candidate.is_symlink() and is_outside_repo(candidate):
                continue
            if should_skip_dir(candidate.relative_to(REPO_ROOT)):
                continue
            if candidate.is_file() and candidate.suffix in suffixes:
                files.append(candidate)
    return sorted(set(files))


def normalize_file(path: Path, *, check: bool) -> NormalizationResult | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[normalize_punctuation] skip non-utf8: {path.relative_to(REPO_ROOT)}")
        return None
    normalized, replacement_count = normalize_text(text)
    if replacement_count == 0:
        return None
    relative = path.relative_to(REPO_ROOT)
    if check:
        print(
            "[normalize_punctuation] needs change: "
            f"{relative} replacements={replacement_count}"
        )
    else:
        path.write_text(normalized, encoding="utf-8")
        print(
            "[normalize_punctuation] changed: "
            f"{relative} replacements={replacement_count}"
        )
    return NormalizationResult(path=path, replacement_count=replacement_count)


def main() -> int:
    paths = env_list("XDL_PUNCT_PATHS", DEFAULT_PATHS)
    suffixes = set(env_list("XDL_PUNCT_EXTS", DEFAULT_EXTS))
    check = env_bool("XDL_PUNCT_CHECK", False)
    files = discover_files(paths, suffixes)
    results: list[NormalizationResult] = []
    for path in files:
        result = normalize_file(path, check=check)
        if result is not None:
            results.append(result)
    replacement_count = sum(result.replacement_count for result in results)
    mode = "check" if check else "write"
    print(
        "[normalize_punctuation] "
        f"mode={mode} files={len(results)} replacements={replacement_count}"
    )
    return 1 if check and results else 0


if __name__ == "__main__":
    raise SystemExit(main())
