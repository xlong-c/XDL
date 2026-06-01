#!/usr/bin/env python3
"""Build the XDL wheel package.

Usage:
    python scripts/build_wheel.py

Environment variables:
    XDL_BUILD_OUT_DIR: output directory, defaults to "dist".
    XDL_BUILD_CLEAN: set to "1" to remove build/ and *.egg-info first.
    XDL_BUILD_CLEAN_DIST: set to "1" with XDL_BUILD_CLEAN to remove output dir.
    XDL_BUILD_BACKEND: "auto", "build", or "pip"; defaults to "auto".
    XDL_BUILD_CHECK: set to "0" to skip basic wheel integrity checks.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = "dist"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def ensure_inside_repo(path: Path) -> None:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"拒绝操作仓库外路径: {resolved}")


def remove_path(path: Path) -> None:
    ensure_inside_repo(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"[build_wheel] removed {path.relative_to(REPO_ROOT)}")


def clean_artifacts(out_dir: Path) -> None:
    remove_path(REPO_ROOT / "build")
    for egg_info in REPO_ROOT.glob("*.egg-info"):
        remove_path(egg_info)
    if env_bool("XDL_BUILD_CLEAN_DIST", False):
        remove_path(out_dir)


def run_command(command: Sequence[str]) -> None:
    printable = " ".join(command)
    print(f"[build_wheel] running: {printable}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def wheel_files(out_dir: Path) -> Set[Path]:
    if not out_dir.exists():
        return set()
    return {path.resolve() for path in out_dir.glob("*.whl")}


def choose_backend() -> str:
    backend = env_text("XDL_BUILD_BACKEND", "auto").lower()
    if backend not in {"auto", "build", "pip"}:
        raise RuntimeError("XDL_BUILD_BACKEND 只能是 auto、build 或 pip")
    if backend == "auto":
        return "build" if has_module("build") else "pip"
    return backend


def build_wheel(out_dir: Path) -> None:
    backend = choose_backend()
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "build":
        run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(out_dir),
            ]
        )
        return

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(out_dir),
        ]
    )


def newest_wheel(paths: Iterable[Path]) -> Path:
    wheels = list(paths)
    if not wheels:
        raise RuntimeError("没有找到生成的 whl 文件")
    return max(wheels, key=lambda path: path.stat().st_mtime)


def inspect_wheel(wheel_path: Path) -> None:
    print(f"[build_wheel] checking: {wheel_path.relative_to(REPO_ROOT)}")
    with zipfile.ZipFile(wheel_path) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"wheel 压缩包损坏: {bad_file}")

        names = set(archive.namelist())
        required_suffixes = [
            "xdl/__init__.py",
            "xdl/USAGE.md",
            ".dist-info/METADATA",
            ".dist-info/WHEEL",
        ]
        missing: List[str] = []
        for suffix in required_suffixes:
            if not any(name.endswith(suffix) for name in names):
                missing.append(suffix)
        if missing:
            raise RuntimeError(f"wheel 缺少必要文件: {', '.join(missing)}")


def main() -> None:
    out_dir = (REPO_ROOT / env_text("XDL_BUILD_OUT_DIR", DEFAULT_OUT_DIR)).resolve()
    ensure_inside_repo(out_dir)

    if env_bool("XDL_BUILD_CLEAN", False):
        clean_artifacts(out_dir)

    before = wheel_files(out_dir)
    build_wheel(out_dir)
    after = wheel_files(out_dir)
    generated = after - before
    wheel_path = newest_wheel(generated or after)

    if env_bool("XDL_BUILD_CHECK", True):
        inspect_wheel(wheel_path)

    print(f"[build_wheel] done: {wheel_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
