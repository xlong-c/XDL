#!/usr/bin/env python3
"""Hugging Face 缓存管理工具 - pywebview + FastAPI 单文件版.

用法:
    python tools/gui/hf_manager.py

环境变量:
    XDL_HF_MANAGER_PORT=8767   端口(默认 8767)
    HF_HUB_CACHE / HF_HOME     缓存根目录(标准 HF 环境变量)
    HF_TOKEN                   Token 显示时会自动遮蔽中间

行为说明:
    - 监听 127.0.0.1(不绑 0.0.0.0,避免无意暴露在容器/SSH 转发里)
    - 业务函数全部独立,可单测:collect_summary / build_env_instructions / scan_cache_manual
    - 删除操作走后台 task,前端 setInterval 轮询进度
    - 剪贴板走浏览器原生 navigator.clipboard.writeText,pywebview 内置 webkit2gtk / WebView2 都支持
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import stat
import sys
import threading
import time
import uuid
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# CONFIG
# ============================================================

HOST = os.environ.get("XDL_HF_MANAGER_HOST", "127.0.0.1")
PORT = int(os.environ.get("XDL_HF_MANAGER_PORT", "8767") or "8767")
TITLE = "Hugging Face Manager"
PAGE_SIZE = 25  # 与旧版 NiceGUI 表格分页一致
ENV_FILE_TYPES_LIMIT = 8
TOP_REPOS_LIMIT = 5
STATE_FILE = Path.home() / ".config" / "xdl-hf-manager-state.json"

HF_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_ENDPOINT",
    "HF_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
)

FILE_TYPE_LABELS = {
    ".safetensors": "SafeTensors",
    ".bin": "PyTorch Bin",
    ".json": "JSON Config",
    ".txt": "Text File",
    ".md": "Markdown",
    ".msgpack": "MessagePack",
    ".lock": "Lock File",
    ".py": "Python Script",
    ".h5": "HDF5 Model",
    ".pth": "PyTorch Weights",
    ".ckpt": "Checkpoint",
    ".incomplete": "未完成下载",
    ".so": "Shared Library",
    "(noext)": "No Extension",
}

REPO_TYPE_LABELS = {
    "model": "Model",
    "dataset": "Dataset",
    "space": "Space",
}


# ============================================================
# 业务函数(纯 Python,可单测)
# ============================================================


def format_size(size_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_time(timestamp: float) -> str:
    if timestamp <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def get_dir_size(path: Path, follow_symlinks: bool = True) -> int:
    """递归计算目录大小.follow_symlinks=False 时跳过符号链接(得到真实磁盘占用)."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                fp = os.path.join(dirpath, filename)
                try:
                    if follow_symlinks:
                        total += os.path.getsize(fp)
                    else:
                        st = os.lstat(fp)
                        if not stat.S_ISLNK(st.st_mode):
                            total += st.st_size
                except OSError:
                    pass
    except (PermissionError, OSError):
        return 0
    return total


def get_dir_mtime(path: Path) -> float:
    latest = 0.0
    try:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                try:
                    mtime = os.path.getmtime(os.path.join(dirpath, filename))
                    latest = max(latest, mtime)
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    return latest


def get_hf_cache_dir() -> Path:
    """返回默认缓存目录(向后兼容,单目录场景仍用此函数)."""
    for key in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser()
    return Path.home() / ".cache" / "huggingface"


def get_hf_cache_dirs() -> list[Path]:
    """返回所有要扫描的缓存目录.

    读取 XDL_HF_CACHE_DIRS 环境变量(逗号分隔绝对路径),
    未设置时退回默认单目录.
    """
    dirs_env = os.environ.get("XDL_HF_CACHE_DIRS", "").strip()
    if dirs_env:
        dirs: list[Path] = []
        for d in dirs_env.split(","):
            d = d.strip()
            if d:
                dirs.append(Path(d).expanduser().resolve())
        if dirs:
            return dirs
    return [get_hf_cache_dir()]


def get_hf_env_vars() -> dict[str, str]:
    result: dict[str, str] = {}
    for key in HF_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        if key == "HF_TOKEN" and len(value) > 8:
            result[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
        else:
            result[key] = value
    return result


def get_system_info() -> dict[str, str]:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
    }
    try:
        import huggingface_hub

        info["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        info["huggingface_hub"] = "Not installed"
    return info


def parse_repo_name(dir_name: str) -> tuple[str, str] | None:
    parts = dir_name.split("--")
    if len(parts) >= 3 and parts[0] in ("models", "datasets", "spaces"):
        return parts[0].rstrip("s"), "/".join(parts[1:])
    return None


def scan_cache_manual(cache_dir: Path) -> list[dict[str, Any]]:
    hub_path = cache_dir / "hub"
    repos: list[dict[str, Any]] = []
    if not hub_path.exists():
        return repos

    for item in sorted(hub_path.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        parsed = parse_repo_name(item.name)
        if parsed is None:
            continue
        repo_type, repo_name = parsed
        repos.append(
            {
                "type": repo_type,
                "name": repo_name,
                "size": get_dir_size(item),
                "files": sum(1 for child in item.rglob("*") if child.is_file()),
                "last_modified": get_dir_mtime(item),
                "path": str(item.resolve()),
            }
        )
    return repos


def scan_cache_file_types(cache_dir: Path) -> dict[str, int]:
    """统计缓存目录中各文件后缀的总大小.

    跳过 blobs/ 子目录 - 其中是 SHA256 哈希命名的无扩展名原始数据,
    与 snapshots/ 下的有扩展名文件通过硬链接共享同一物理存储,
    纳入统计会导致两倍虚高.用户应看 snapshots/ 层的文件类型分布.
    """
    extensions: dict[str, int] = defaultdict(int)
    try:
        for dirpath, _, filenames in os.walk(cache_dir):
            # 跳过 blob 存储目录(内部 SHA256 哈希文件,无扩展名)
            parts = dirpath.split(os.sep)
            if "blobs" in parts:
                # 但仍单独统计 .incomplete 未完成下载(浪费的空间)
                for filename in filenames:
                    if filename.endswith(".incomplete"):
                        try:
                            extensions[".incomplete"] += os.path.getsize(
                                os.path.join(dirpath, filename)
                            )
                        except OSError:
                            pass
                continue
            for filename in filenames:
                suffix = Path(filename).suffix.lower() or "(noext)"
                try:
                    extensions[suffix] += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
    except (PermissionError, OSError):
        pass
    return dict(extensions)


def get_repo_list(cache_dir: Path) -> list[dict[str, Any]]:
    """扫描缓存目录中的仓库列表.

    优先使用 huggingface_hub.scan_cache_dir,失败时回退到手动扫描.
    注意:部分版本的 scan_cache_dir 接受缓存根目录,部分版本只接受 hub/ 子目录;
    这里统一尝试二者,避免因版本差异静默返回 0.
    """
    if not cache_dir.exists():
        return []
    try:
        from huggingface_hub import scan_cache_dir as hf_scan

        # 兼容不同版本的 API:优先传 hub/ 子目录(新版本行为),
        # 如果不存在则传缓存根目录
        hub_dir = cache_dir / "hub"
        for candidate in (hub_dir, cache_dir):
            if not candidate.exists():
                continue
            try:
                result = hf_scan(candidate)
                repos: list[dict[str, Any]] = []
                for repo in result.repos:
                    repos.append(
                        {
                            "type": repo.repo_type,
                            "name": repo.repo_id,
                            "size": repo.size_on_disk,
                            "files": repo.nb_files,
                            "last_modified": repo.last_modified,
                            "path": str(repo.repo_path),
                        }
                    )
                if repos:
                    return repos
            except Exception:
                continue

        # 如果 hf_scan 没找到仓库,回退手动扫描
        return scan_cache_manual(cache_dir)
    except Exception:
        return scan_cache_manual(cache_dir)


def build_env_instructions(settings: dict[str, str]) -> dict[str, Any]:
    shell = os.path.basename(os.environ.get("SHELL", "bash"))
    rc_file = {
        "bash": "~/.bashrc",
        "zsh": "~/.zshrc",
        "fish": "~/.config/fish/config.fish",
    }.get(shell, "~/.bashrc")

    export_lines = ["# Hugging Face configuration"]
    for key, value in settings.items():
        export_lines.append(f'export {key}="{value}"')

    windows_commands = [f'setx {key} "{value}"' for key, value in settings.items()]
    return {
        "shell": shell,
        "rc_file": rc_file,
        "unix_script": "\n".join(export_lines),
        "windows_commands": windows_commands,
    }


def collect_summary() -> dict[str, Any]:
    cache_dirs = get_hf_cache_dirs()

    all_repos: list[dict[str, Any]] = []
    merged_file_types: dict[str, int] = defaultdict(int)
    dir_infos: list[dict[str, Any]] = []

    for cache_dir in cache_dirs:
        repos = get_repo_list(cache_dir)
        # 标记来源目录,前端展示用
        for r in repos:
            r["cache_root"] = str(cache_dir)
        all_repos.extend(repos)

        # 仓库数据大小(来自 scan_cache_dir,去重后的实际模型数据)
        dir_repo_size = sum(r["size"] for r in repos)

        # hub/ 目录物理占用(不跟随符号链接,真实磁盘占用)
        hub_path = cache_dir / "hub"
        raw_hub_size = get_dir_size(hub_path, follow_symlinks=False) if hub_path.exists() else 0

        # .incomplete 统计
        incomplete_count = 0
        incomplete_size = 0
        if hub_path.exists():
            for dirpath, _, filenames in os.walk(hub_path):
                for fn in filenames:
                    if fn.endswith(".incomplete"):
                        incomplete_count += 1
                        try:
                            incomplete_size += os.path.getsize(os.path.join(dirpath, fn))
                        except OSError:
                            pass

        dir_infos.append(
            {
                "path": str(cache_dir),
                "exists": cache_dir.exists(),
                "repo_count": len(repos),
                "repo_size": dir_repo_size,
                "repo_size_text": format_size(dir_repo_size),
                "raw_hub_size": raw_hub_size,
                "raw_hub_size_text": format_size(raw_hub_size),
                "incomplete_count": incomplete_count,
                "incomplete_size": incomplete_size,
                "incomplete_size_text": format_size(incomplete_size),
                # 快照冗余 = hub 物理占用 - 模型数据 - 未完成
                "overhead_size": max(0, raw_hub_size - dir_repo_size - incomplete_size),
                "overhead_size_text": format_size(
                    max(0, raw_hub_size - dir_repo_size - incomplete_size)
                ),
            }
        )

        # 合并文件类型统计
        if cache_dir.exists():
            ft = scan_cache_file_types(cache_dir)
            for suffix, size in ft.items():
                merged_file_types[suffix] = merged_file_types.get(suffix, 0) + size

    total_repo_size = sum(r["size"] for r in all_repos)

    top_file_types = [
        {
            "suffix": suffix,
            "label": FILE_TYPE_LABELS.get(suffix, suffix),
            "size": size,
            "size_text": format_size(size),
        }
        for suffix, size in sorted(
            merged_file_types.items(), key=lambda item: item[1], reverse=True
        )[:ENV_FILE_TYPES_LIMIT]
    ]

    top_repos_raw = sorted(all_repos, key=lambda repo: repo["size"], reverse=True)[:TOP_REPOS_LIMIT]

    def _decorate(repo: dict[str, Any]) -> dict[str, Any]:
        return {
            **repo,
            "type_label": REPO_TYPE_LABELS.get(repo["type"], repo["type"]),
            "size_text": format_size(repo["size"]),
            "modified_text": format_time(repo["last_modified"]),
        }

    # 按类型统计(跨所有目录合并)
    type_counts: dict[str, dict[str, Any]] = {}
    for repo in all_repos:
        tp = repo["type"]
        if tp not in type_counts:
            type_counts[tp] = {"count": 0, "size": 0, "label": REPO_TYPE_LABELS.get(tp, tp)}
        type_counts[tp]["count"] += 1
        type_counts[tp]["size"] += repo["size"]
    type_counts_list = [
        {
            "type": tp,
            "label": info["label"],
            "count": info["count"],
            "size": info["size"],
            "size_text": format_size(info["size"]),
        }
        for tp, info in sorted(type_counts.items(), key=lambda kv: kv[1]["size"], reverse=True)
    ]

    return {
        "cache_dir": str(cache_dirs[0]) if cache_dirs else "",
        "cache_exists": any(d.exists() for d in cache_dirs),
        "repo_count": len(all_repos),
        "repo_total_size": total_repo_size,
        "repo_total_size_text": format_size(total_repo_size),
        "repos": [_decorate(r) for r in all_repos],
        "dir_infos": dir_infos,
        "file_types": top_file_types,
        "top_repos": [_decorate(r) for r in top_repos_raw],
        "type_counts": type_counts_list,
        "env_vars": get_hf_env_vars(),
        "system_info": get_system_info(),
    }


def delete_repo(path_str: str) -> dict[str, Any]:
    """删除一个仓库缓存目录.破坏性操作,调用方必须二次确认."""
    p = Path(path_str)
    if not p.exists():
        return {"ok": False, "error": f"路径不存在: {path_str}"}
    if not p.is_dir():
        return {"ok": False, "error": f"不是目录: {path_str}"}
    try:
        shutil.rmtree(p)
        return {"ok": True, "deleted": path_str}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"删除失败: {exc}"}


def find_incomplete_files() -> dict[str, Any]:
    """扫描所有缓存目录,列出所有 .incomplete 残留文件."""
    cache_dirs = get_hf_cache_dirs()
    files: list[dict[str, Any]] = []
    total_size = 0
    for cache_dir in cache_dirs:
        hub = cache_dir / "hub"
        if not hub.exists():
            continue
        for dirpath, _, filenames in os.walk(hub):
            for fn in filenames:
                if fn.endswith(".incomplete"):
                    fp = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(fp)
                    except OSError:
                        sz = 0
                    rel = os.path.relpath(fp, cache_dir)
                    files.append(
                        {
                            "path": fp,
                            "size": sz,
                            "size_text": format_size(sz),
                            "rel_path": rel,
                        }
                    )
                    total_size += sz
    return {
        "files": files,
        "count": len(files),
        "total_size": total_size,
        "total_size_text": format_size(total_size),
    }


def clean_incomplete_files() -> dict[str, Any]:
    """删除所有缓存目录中的 .incomplete 残留文件."""
    cache_dirs = get_hf_cache_dirs()
    deleted = 0
    freed_size = 0
    errors: list[str] = []
    for cache_dir in cache_dirs:
        hub = cache_dir / "hub"
        if not hub.exists():
            continue
        for dirpath, _, filenames in os.walk(hub):
            for fn in filenames:
                if fn.endswith(".incomplete"):
                    fp = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(fp)
                        os.remove(fp)
                        deleted += 1
                        freed_size += sz
                    except OSError as exc:
                        errors.append(f"{fp}: {exc}")
    return {
        "ok": True,
        "deleted": deleted,
        "freed_size": freed_size,
        "freed_size_text": format_size(freed_size),
        "errors": errors,
    }


def download_model(repo_id: str) -> dict[str, Any]:
    """从 Hugging Face 下载模型到默认缓存目录."""
    from huggingface_hub import snapshot_download

    cache_dirs = get_hf_cache_dirs()
    target = cache_dirs[0]

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "cache_dir": str(target),
        "resume_download": True,
    }
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if endpoint:
        kwargs["endpoint"] = endpoint

    local_path = snapshot_download(**kwargs)
    return {
        "ok": True,
        "repo_id": repo_id,
        "local_path": local_path,
        "cache_dir": str(target),
    }


# ============================================================
# 长任务管理(参考 gpt_image_editor.py 的 TASKS 模式)
# ============================================================

TASKS: dict[str, dict[str, Any]] = {}


def start_task(target, *args, **kwargs) -> str:
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "已启动",
        "result": None,
        "error": None,
    }

    def _runner() -> None:
        try:
            TASKS[task_id]["progress"] = 0.3
            TASKS[task_id]["message"] = "执行中..."
            result = target(*args, **kwargs)
            TASKS[task_id]["status"] = "done"
            TASKS[task_id]["progress"] = 1.0
            TASKS[task_id]["message"] = "完成"
            TASKS[task_id]["result"] = result
        except Exception as exc:  # noqa: BLE001
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["progress"] = 0.0
            TASKS[task_id]["message"] = "失败"
            TASKS[task_id]["error"] = str(exc)

    threading.Thread(target=_runner, daemon=True).start()
    return task_id


# ============================================================
# FastAPI 路由
# ============================================================

app = FastAPI(title=TITLE)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.get("/api/summary")
async def api_summary() -> dict[str, Any]:
    return await asyncio.to_thread(collect_summary)


@app.post("/api/delete")
async def api_delete(payload: dict[str, Any]) -> dict[str, Any]:
    path = (payload or {}).get("path", "").strip()
    if not path:
        return {"ok": False, "error": "缺少 path 字段"}
    task_id = start_task(delete_repo, path)
    return {"ok": True, "task_id": task_id}


@app.get("/api/task/{task_id}")
def api_task_status(task_id: str) -> dict[str, Any]:
    info = TASKS.get(task_id)
    if not info:
        return {"status": "unknown", "progress": 0, "message": "任务不存在"}
    return info


@app.post("/api/env-instructions")
async def api_env_instructions(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload or {}
    settings: dict[str, str] = {}
    for key in ("HF_HOME", "HF_ENDPOINT", "HF_TOKEN"):
        v = (payload.get(key) or "").strip()
        if v:
            settings[key] = v
    if not settings:
        return {"ok": False, "error": "请至少填写一项配置"}
    return {"ok": True, "instructions": build_env_instructions(settings)}


@app.get("/api/incomplete-files")
async def api_incomplete_files() -> dict[str, Any]:
    return await asyncio.to_thread(find_incomplete_files)


@app.post("/api/clean-incomplete")
async def api_clean_incomplete() -> dict[str, Any]:
    task_id = start_task(clean_incomplete_files)
    return {"ok": True, "task_id": task_id}


@app.post("/api/download")
async def api_download(payload: dict[str, Any]) -> dict[str, Any]:
    repo_id = (payload or {}).get("repo_id", "").strip()
    if not repo_id:
        return {"ok": False, "error": "缺少 repo_id 字段"}
    if "/" not in repo_id:
        return {"ok": False, "error": "仓库格式无效,应为 org/repo_name"}
    task_id = start_task(download_model, repo_id)
    return {"ok": True, "task_id": task_id}


# ============================================================
# HTML(f-string 内嵌)
# ============================================================

HTML = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>
  :root {{
    --bg: #1e1e1e;
    --bg-2: #252526;
    --bg-3: #2d2d30;
    --line: #3c3c3c;
    --text: #d4d4d4;
    --text-dim: #9d9d9d;
    --primary: #007acc;
    --primary-dim: #1f8ad2;
    --negative: #ff5252;
    --warning: #f2c037;
    --positive: #21ba45;
    --bar-bg: #1a1a1a;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px; height: 100%; }}
  body {{ display: flex; flex-direction: column; }}
  header {{ background: var(--bg-2); border-bottom: 1px solid var(--line);
    padding: 8px 16px; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
  header .title {{ font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
  header .spacer {{ flex: 1; }}
  header input.search {{ background: var(--bg-3); border: 1px solid var(--line);
    color: var(--text); padding: 5px 10px; border-radius: 3px; width: 240px; outline: none; }}
  header input.search:focus {{ border-color: var(--primary); }}
  button {{ background: var(--bg-3); color: var(--text); border: 1px solid var(--line);
    padding: 5px 12px; border-radius: 3px; cursor: pointer; font-size: 12px;
    display: inline-flex; align-items: center; gap: 4px; }}
  button:hover {{ background: #3a3a3d; }}
  button.primary {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
  button.primary:hover {{ background: var(--primary-dim); }}
  button.danger {{ color: var(--negative); border-color: var(--negative); }}
  button.danger:hover {{ background: rgba(255, 82, 82, 0.1); }}
  button.flat {{ background: transparent; border: 1px solid transparent;
    padding: 3px 8px; font-size: 11px; }}
  button.flat:hover {{ background: var(--bg-3); border-color: var(--line); }}
  main {{ display: flex; gap: 12px; padding: 12px; flex: 1; min-height: 0; }}
  .col-left {{ width: 420px; flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; min-height: 0; overflow-y: auto; }}
  .col-right {{ flex: 1; min-width: 0; display: flex; flex-direction: column; min-height: 0; }}
  .card {{ background: var(--bg-2); border: 1px solid var(--line);
    border-radius: 4px; padding: 12px; }}
  .card h3 {{ margin: 0 0 8px 0; font-size: 13px; font-weight: 600; color: var(--text); }}
  .kv {{ display: flex; align-items: center; gap: 8px; padding: 2px 0; font-size: 12px; }}
  .kv .k {{ font-family: ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--text-dim); width: 100px; flex-shrink: 0; }}
  .kv .v {{ flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .kv .v.path {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; }}
  .kv button {{ flex-shrink: 0; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; padding: 2px 0; font-size: 12px; }}
  .bar-row .label {{ width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-row .bar-bg {{ flex: 1; height: 8px; background: var(--bar-bg); border-radius: 2px; overflow: hidden; }}
  .bar-row .bar-fill {{ height: 100%; background: var(--primary); }}
  .bar-row .size {{ width: 70px; text-align: right; color: var(--text-dim); font-size: 11px; flex-shrink: 0; }}
  .bar-row .rank {{ width: 20px; color: var(--text-dim); flex-shrink: 0; }}
  .bar-row .type {{ width: 56px; color: var(--text-dim); font-size: 11px; flex-shrink: 0; }}
  .bar-row .name {{ flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-row .name-wide {{ flex: 2; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .table-wrap {{ flex: 1; min-height: 0; overflow: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead th {{ position: sticky; top: 0; background: var(--bg-3);
    border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left;
    font-weight: 600; color: var(--text-dim); cursor: pointer; user-select: none; }}
  thead th:hover {{ color: var(--text); }}
  thead th.sorted-asc::after {{ content: " ▲"; font-size: 9px; color: var(--primary); }}
  thead th.sorted-desc::after {{ content: " ▼"; font-size: 9px; color: var(--primary); }}
  tbody tr {{ border-bottom: 1px solid var(--bg-3); cursor: pointer; }}
  tbody tr:hover {{ background: var(--bg-3); }}
  tbody tr.selected {{ background: rgba(0, 122, 204, 0.2); }}
  td {{ padding: 5px 8px; vertical-align: middle; }}
  td.right {{ text-align: right; }}
  .chip {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; }}
  .chip.model {{ background: rgba(0, 122, 204, 0.25); color: #6cb8ff; }}
  .chip.dataset {{ background: rgba(38, 166, 154, 0.25); color: #4dd0c1; }}
  .chip.space {{ background: rgba(156, 39, 176, 0.25); color: #ce93d8; }}
  .pagination {{ display: flex; align-items: center; gap: 8px; padding: 8px 0; font-size: 12px; color: var(--text-dim); }}
  .pagination button {{ padding: 3px 10px; }}
  .pagination .info {{ margin-left: auto; }}
  footer {{ background: var(--bg-2); border-top: 1px solid var(--line);
    padding: 6px 16px; display: flex; align-items: center; gap: 16px;
    font-size: 11px; color: var(--text-dim); flex-shrink: 0; }}
  footer .status {{ flex: 1; }}
  /* 模态框 */
  .modal-mask {{ position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    display: none; align-items: center; justify-content: center; z-index: 100; }}
  .modal-mask.open {{ display: flex; }}
  .modal {{ background: var(--bg-2); border: 1px solid var(--line);
    border-radius: 6px; padding: 20px; min-width: 320px; max-width: 90vw;
    max-height: 90vh; overflow: auto; }}
  .modal h2 {{ margin: 0 0 12px 0; font-size: 15px; font-weight: 600; }}
  .modal .row {{ margin-bottom: 10px; }}
  .modal .row label {{ display: block; font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }}
  .modal input, .modal textarea {{ width: 100%; background: var(--bg-3);
    border: 1px solid var(--line); color: var(--text); padding: 6px 8px;
    border-radius: 3px; font-family: inherit; font-size: 12px; outline: none; }}
  .modal input:focus, .modal textarea:focus {{ border-color: var(--primary); }}
  .modal textarea {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; resize: vertical; min-height: 100px; }}
  .modal .actions {{ display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px; }}
  .modal pre {{ background: var(--bg-3); padding: 10px; border-radius: 3px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px;
    overflow: auto; max-height: 240px; white-space: pre-wrap; word-break: break-all; }}
  .toast {{ position: fixed; bottom: 24px; right: 24px;
    background: var(--bg-3); border: 1px solid var(--line);
    border-left: 3px solid var(--primary);
    padding: 10px 16px; border-radius: 3px; font-size: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 200;
    transform: translateX(120%); transition: transform 0.2s; max-width: 360px; }}
  .toast.show {{ transform: translateX(0); }}
  .toast.success {{ border-left-color: var(--positive); }}
  .toast.error {{ border-left-color: var(--negative); }}
  .toast.warning {{ border-left-color: var(--warning); }}
  .empty {{ color: var(--text-dim); font-size: 12px; padding: 16px; text-align: center; }}
  .progress {{ display: none; align-items: center; gap: 6px; }}
  .progress.show {{ display: inline-flex; }}
  .progress .dot {{ width: 6px; height: 6px; background: var(--primary); border-radius: 50%;
    animation: pulse 1s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} }}
  /* 类型筛选 */
  .filter-bar {{ display: flex; gap: 6px; padding-bottom: 8px; flex-wrap: wrap; }}
  .filter-chip {{ padding: 3px 12px; border-radius: 12px; border: 1px solid var(--line);
    background: var(--bg-3); color: var(--text-dim); cursor: pointer;
    font-size: 11px; user-select: none; transition: all 0.15s; }}
  .filter-chip:hover {{ color: var(--text); border-color: var(--primary-dim); }}
  .filter-chip.active {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
  .filter-chip .count {{ opacity: 0.7; margin-left: 2px; font-size: 10px; }}
  /* 仓库列表卡片 */
  .repo-list {{ max-height: 240px; overflow-y: auto; }}
  .repo-item {{ display: flex; align-items: center; gap: 8px; padding: 3px 0;
    font-size: 12px; border-bottom: 1px solid var(--bg-3); cursor: pointer; }}
  .repo-item:hover {{ background: var(--bg-3); }}
  .repo-item .repo-type {{ flex-shrink: 0; font-size: 10px; }}
  .repo-item .repo-name {{ flex: 1; min-width: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }}
  .repo-item .repo-size {{ flex-shrink: 0; color: var(--text-dim); font-size: 11px;
    width: 70px; text-align: right; }}
  .repo-root {{ color: var(--text-dim); font-size: 10px; }}
</style>
</head>
<body>

<header>
  <div class="title">☁ Hugging Face Manager</div>
  <input id="search" class="search" type="text" placeholder="搜索模型或数据集...">
  <button id="btn-refresh">刷新</button>
  <button id="btn-env">环境配置</button>
  <button id="btn-copy-env" class="flat">复制环境变量</button>
  <button id="btn-copy-path" class="flat">复制路径</button>
  <button id="btn-detail">查看详情</button>
  <button id="btn-delete" class="danger">删除缓存</button>
  <div class="spacer"></div>
  <button id="btn-clean-incomplete" class="flat" style="color: var(--warning);">清理未完成</button>
  <button id="btn-download" class="primary">下载模型</button>
  <div id="progress" class="progress"><span class="dot"></span><span id="progress-msg">扫描中...</span></div>
</header>

<main>
  <div class="col-left">
    <div class="card">
      <h3>缓存概览</h3>
      <div id="overview"></div>
    </div>
    <div class="card">
      <h3>环境变量</h3>
      <div id="env-vars"></div>
    </div>
    <div class="card">
      <h3>文件类型分布</h3>
      <div id="file-types"></div>
    </div>
    <div class="card">
      <h3>已缓存仓库 <span style="color: var(--text-dim); font-weight: 400; font-size: 11px;">- 单击选中,表格联动</span></h3>
      <div class="repo-list" id="repo-list"></div>
    </div>
  </div>
  <div class="col-right">
    <div class="card" style="flex: 1; display: flex; flex-direction: column; min-height: 0;">
      <h3>缓存列表 <span style="color: var(--text-dim); font-weight: 400; font-size: 11px;">- 单击选中,双击查看详情,点击表头排序</span></h3>
      <div class="filter-bar" id="filter-bar">
        <span class="filter-chip active" data-type="">全部 <span class="count" id="filter-count-all">0</span></span>
        <span class="filter-chip" data-type="model">模型 <span class="count" id="filter-count-model">0</span></span>
        <span class="filter-chip" data-type="dataset">数据集 <span class="count" id="filter-count-dataset">0</span></span>
        <span class="filter-chip" data-type="space">空间 <span class="count" id="filter-count-space">0</span></span>
      </div>
      <div class="table-wrap">
        <table id="repos-table">
          <thead>
            <tr>
              <th data-key="type_label">类型</th>
              <th data-key="name">名称</th>
              <th data-key="size" class="right">大小</th>
              <th data-key="files" class="right">文件数</th>
              <th data-key="last_modified">修改时间</th>
            </tr>
          </thead>
          <tbody id="repos-body"></tbody>
        </table>
      </div>
      <div class="pagination">
        <button id="page-prev">上一页</button>
        <button id="page-next">下一页</button>
        <span class="info" id="page-info">第 1 / 1 页</span>
      </div>
    </div>
  </div>
</main>

<footer>
  <div class="status" id="status">就绪</div>
  <div id="system-info"></div>
</footer>

<!-- 详情模态框 -->
<div class="modal-mask" id="modal-detail">
  <div class="modal" style="min-width: 480px;">
    <h2>缓存详情</h2>
    <div id="detail-body"></div>
    <div class="actions">
      <button onclick="closeModal('modal-detail')">关闭</button>
    </div>
  </div>
</div>

<!-- 环境配置模态框 -->
<div class="modal-mask" id="modal-env">
  <div class="modal" style="min-width: 480px;">
    <h2>环境配置</h2>
    <div class="row">
      <label>HF_HOME</label>
      <input id="env-hf-home" type="text" placeholder="例如: /data/hf">
    </div>
    <div class="row">
      <label>HF_ENDPOINT</label>
      <input id="env-hf-endpoint" type="text" placeholder="例如: https://hf-mirror.com">
    </div>
    <div class="row">
      <label>HF_TOKEN</label>
      <input id="env-hf-token" type="password" placeholder="hf_xxx">
    </div>
    <div class="actions">
      <button onclick="closeModal('modal-env')">取消</button>
      <button class="primary" id="btn-env-generate">生成配置</button>
    </div>
  </div>
</div>

<!-- 配置结果模态框 -->
<div class="modal-mask" id="modal-env-result">
  <div class="modal" style="min-width: 560px;">
    <h2>配置说明</h2>
    <div class="row">
      <label id="env-shell-label">Unix Shell</label>
      <pre id="env-unix"></pre>
      <div style="text-align: right;"><button class="flat" id="btn-copy-unix">复制 Unix</button></div>
    </div>
    <div class="row">
      <label>Windows</label>
      <pre id="env-windows"></pre>
      <div style="text-align: right;"><button class="flat" id="btn-copy-windows">复制 Windows</button></div>
    </div>
    <div class="actions">
      <button onclick="closeModal('modal-env-result')">关闭</button>
    </div>
  </div>
</div>

<!-- 下载模型模态框 -->
<div class="modal-mask" id="modal-download">
  <div class="modal" style="min-width: 440px;">
    <h2>下载模型</h2>
    <div class="row">
      <label>Hugging Face 仓库 ID</label>
      <input id="download-repo-id" type="text" placeholder="例如: openbmb/MiniCPM5-1B">
    </div>
    <div id="download-hint" style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">
      模型将下载到默认缓存目录,使用镜像请在环境变量中设置 HF_ENDPOINT.
    </div>
    <div class="actions">
      <button onclick="closeModal('modal-download')">取消</button>
      <button class="primary" id="btn-start-download">开始下载</button>
    </div>
  </div>
</div>

<!-- 清理未完成模态框 -->
<div class="modal-mask" id="modal-clean-incomplete">
  <div class="modal" style="min-width: 440px;">
    <h2 style="color: var(--warning);">清理未完成下载</h2>
    <div id="clean-incomplete-body">
      <div class="empty">正在扫描...</div>
    </div>
    <div class="actions">
      <button onclick="closeModal('modal-clean-incomplete')">取消</button>
      <button class="danger" id="btn-confirm-clean" style="display: none;">确认清理</button>
      <button id="btn-rescan-incomplete" style="display: none;" onclick="scanIncomplete()">重新扫描</button>
    </div>
  </div>
</div>

<!-- 删除确认模态框 -->
<div class="modal-mask" id="modal-delete">
  <div class="modal" style="min-width: 360px;">
    <h2 style="color: var(--warning);">确认删除</h2>
    <div id="delete-body"></div>
    <div class="actions">
      <button onclick="closeModal('modal-delete')">取消</button>
      <button class="danger" id="btn-confirm-delete">确认删除</button>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
const state = {{
  summary: null,
  repos: [],
  search: "",
  sortKey: "size",
  sortDir: "desc",
  page: 0,
  typeFilter: "",
  selectedPath: null,
  pendingDelete: null,
  pollInterval: null,
}};

const PAGE_SIZE = {PAGE_SIZE};

const $ = (id) => document.getElementById(id);

function showToast(msg, type) {{
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show " + (type || "");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => t.classList.remove("show"), 2400);
}}

function setStatus(msg) {{ $("status").textContent = msg; }}

function setProgress(show, msg) {{
  $("progress").classList.toggle("show", show);
  if (msg) $("progress-msg").textContent = msg;
}}

function openModal(id) {{ $(id).classList.add("open"); }}
function closeModal(id) {{ $(id).classList.remove("open"); }}

document.querySelectorAll(".modal-mask").forEach(m => {{
  m.addEventListener("click", (e) => {{ if (e.target === m) m.classList.remove("open"); }});
}});

function fmtSize(bytes) {{
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  for (const u of units) {{
    if (v < 1024) return v.toFixed(2) + " " + u;
    v /= 1024;
  }}
  return v.toFixed(2) + " PB";
}}

function fmtTime(ts) {{
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${{d.getFullYear()}}-${{pad(d.getMonth()+1)}}-${{pad(d.getDate())}} ${{pad(d.getHours())}}:${{pad(d.getMinutes())}}`;
}}

async function copyToClipboard(text) {{
  try {{
    await navigator.clipboard.writeText(text);
    return true;
  }} catch (e) {{
    // 兜底:选中文本让用户 Ctrl+C
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {{ ok = document.execCommand("copy"); }} catch (_) {{ ok = false; }}
    document.body.removeChild(ta);
    return ok;
  }}
}}

function renderOverview() {{
  const el = $("overview");
  if (!state.summary) {{ el.innerHTML = '<div class="empty">尚未扫描</div>'; return; }}
  const s = state.summary;
  let html = "";
  html += kvRow("缓存项", String(s.repo_count) + " 个仓库");
  html += kvRow("模型数据", s.repo_total_size_text);
  // 类型统计
  const tc = s.type_counts || [];
  if (tc.length) {{
    for (const t of tc) {{
      html += `<div class="kv">
        <span class="k">${{escHtml(t.label)}}</span>
        <span class="v">${{t.count}} 个 · ${{escHtml(t.size_text)}}</span>
      </div>`;
    }}
  }}
  // 缓存目录详情
  const dirs = s.dir_infos || [];
  if (dirs.length) {{
    html += '<div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--line);">';
    html += '<div style="font-size: 11px; color: var(--text-dim); margin-bottom: 4px;">磁盘占用</div>';
    for (const d of dirs) {{
      const label = shortPath(d.path);
      html += `<div class="kv" title="${{escAttr(d.path)}}">
        <span class="k" style="width: 140px;">${{escHtml(label)}}</span>
        <span class="v">${{d.exists ? d.repo_count + " 项 · " + escHtml(d.repo_size_text) : "不存在"}}</span>
        <button class="flat" onclick="copyPath('${{escAttr(d.path)}}')" title="复制路径">⧉</button>
      </div>`;
      html += `<div class="kv" style="padding-left: 12px; opacity: 0.8;">
        <span class="k" style="width: 120px;">└ hub/ 物理占用</span>
        <span class="v">${{escHtml(d.raw_hub_size_text)}}</span>
      </div>`;
      if (d.incomplete_count > 0) {{
        html += `<div class="kv" style="padding-left: 12px; opacity: 0.8;">
          <span class="k" style="width: 120px; color: var(--negative);">└ 未完成下载</span>
          <span class="v" style="color: var(--negative);">${{d.incomplete_count}} 个 · ${{escHtml(d.incomplete_size_text)}}</span>
        </div>`;
      }}
    }}
    html += '<div style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">snapshots/ 为符号链接,不占额外磁盘空间</div>';
    html += '</div>';
  }}
  el.innerHTML = html;
}}

function renderEnv() {{
  const el = $("env-vars");
  const vars = (state.summary && state.summary.env_vars) || {{}};
  const keys = Object.keys(vars);
  if (!keys.length) {{ el.innerHTML = '<div class="empty">当前没有检测到 Hugging Face 环境变量</div>'; return; }}
  let html = "";
  for (const k of keys) {{
    const v = vars[k];
    html += `<div class="kv">
      <span class="k">${{k}}</span>
      <span class="v" title="${{v}}">${{v}}</span>
      <button class="flat" onclick="copyEnvVar('${{k}}','${{escAttr(v)}}')" title="复制">⧉</button>
    </div>`;
  }}
  el.innerHTML = html;
}}

function renderFileTypes() {{
  const el = $("file-types");
  const items = (state.summary && state.summary.file_types) || [];
  if (!items.length) {{ el.innerHTML = '<div class="empty">没有可展示的数据</div>'; return; }}
  const max = items[0].size || 1;
  let html = "";
  for (const it of items) {{
    const pct = ((it.size / max) * 100).toFixed(1);
    html += `<div class="bar-row">
      <span class="label" title="${{escAttr(it.label)}}">${{escHtml(it.label)}}</span>
      <div class="bar-bg"><div class="bar-fill" style="width: ${{pct}}%"></div></div>
      <span class="size">${{escHtml(it.size_text)}}</span>
    </div>`;
  }}
  el.innerHTML = html;
}}

function renderRepoList() {{
  const el = $("repo-list");
  const repos = getFilteredRepos();
  if (!repos.length) {{ el.innerHTML = '<div class="empty">没有匹配的数据</div>'; return; }}
  const multiDir = (state.summary && state.summary.dir_infos && state.summary.dir_infos.length > 1);
  let html = "";
  repos.forEach((r) => {{
    const sel = r.path === state.selectedPath ? ' style="background: rgba(0, 122, 204, 0.25);"' : "";
    const rootLabel = multiDir && r.cache_root ? ' [' + shortPath(r.cache_root) + ']' : '';
    const fullTitle = r.cache_root ? `${{r.name}}\\n来源: ${{r.cache_root}}` : r.name;
    html += `<div class="repo-item"${{sel}} data-path="${{escAttr(r.path)}}" title="${{escAttr(fullTitle)}}">
      <span class="repo-type"><span class="chip ${{r.type}}">${{escHtml(r.type_label)}}</span></span>
      <span class="repo-name">${{escHtml(r.name)}}<span class="repo-root">${{escHtml(rootLabel)}}</span></span>
      <span class="repo-size">${{escHtml(r.size_text)}}</span>
    </div>`;
  }});
  el.innerHTML = html;
}}

function getFilteredRepos() {{
  let repos = (state.summary && state.summary.repos) || [];
  if (state.search) {{
    const q = state.search.toLowerCase();
    repos = repos.filter(r => r.name.toLowerCase().includes(q) || r.type.toLowerCase().includes(q));
  }}
  if (state.typeFilter) {{
    repos = repos.filter(r => r.type === state.typeFilter);
  }}
  repos = repos.slice();
  repos.sort((a, b) => {{
    const av = a[state.sortKey], bv = b[state.sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return state.sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
    return state.sortDir === "asc" ? av - bv : bv - av;
  }});
  return repos;
}}

function renderTable() {{
  const repos = getFilteredRepos();
  const total = repos.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (state.page >= totalPages) state.page = totalPages - 1;
  if (state.page < 0) state.page = 0;
  const start = state.page * PAGE_SIZE;
  const pageRepos = repos.slice(start, start + PAGE_SIZE);
  $("page-info").textContent = `第 ${{state.page + 1}} / ${{totalPages}} 页 · 共 ${{total}} 项`;

  const body = $("repos-body");
  if (!pageRepos.length) {{
    body.innerHTML = `<tr><td colspan="5" class="empty">${{total ? "当前页无数据" : "没有匹配的数据"}}</td></tr>`;
  }} else {{
    body.innerHTML = pageRepos.map(r => {{
      const sel = r.path === state.selectedPath ? " selected" : "";
      const multiDir = (state.summary && state.summary.dir_infos && state.summary.dir_infos.length > 1);
      const rootLabel = multiDir && r.cache_root ? ' [' + shortPath(r.cache_root) + ']' : '';
      const nameTitle = r.cache_root ? `${{escAttr(r.name)}}\\n来源: ${{escAttr(r.cache_root)}}` : escAttr(r.name);
      const repoNameHtml = `${{escHtml(r.name)}}<span class="repo-root">${{escHtml(rootLabel)}}</span>`;
      return `<tr class="${{sel.trim()}}" data-path="${{escAttr(r.path)}}">
        <td><span class="chip ${{r.type}}">${{escHtml(r.type_label)}}</span></td>
        <td title="${{nameTitle}}">${{repoNameHtml}}</td>
        <td class="right">${{escHtml(r.size_text)}}</td>
        <td class="right">${{r.files}}</td>
        <td>${{escHtml(r.modified_text)}}</td>
      </tr>`;
    }}).join("");
  }}

  document.querySelectorAll("thead th").forEach(th => {{
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.key === state.sortKey) th.classList.add(state.sortDir === "asc" ? "sorted-asc" : "sorted-desc");
  }});

  $("page-prev").disabled = state.page === 0;
  $("page-next").disabled = state.page >= totalPages - 1;
}}

function renderSystemInfo() {{
  if (!state.summary) return;
  const info = state.summary.system_info || {{}};
  $("system-info").textContent = `Python ${{info.python || "?"}} | huggingface_hub ${{info.huggingface_hub || "?"}} | ${{info.system || "?"}} ${{info.release || ""}}`;
}}

function renderFilterChips() {{
  const types = (state.summary && state.summary.type_counts) || [];
  let total = 0;
  const counts = {{ model: 0, dataset: 0, space: 0 }};
  for (const t of types) {{
    counts[t.type] = t.count;
    total += t.count;
  }}
  $("filter-count-all").textContent = total;
  $("filter-count-model").textContent = counts.model;
  $("filter-count-dataset").textContent = counts.dataset;
  $("filter-count-space").textContent = counts.space;
  document.querySelectorAll("#filter-bar .filter-chip").forEach(chip => {{
    chip.classList.toggle("active", chip.dataset.type === state.typeFilter);
  }});
}}

function renderAll() {{
  renderOverview();
  renderEnv();
  renderFileTypes();
  renderFilterChips();
  renderRepoList();
  renderTable();
  renderSystemInfo();
}}

function kvRow(k, v, isPath) {{
  return `<div class="kv"><span class="k">${{k}}</span><span class="v ${{isPath ? "path" : ""}}" title="${{escAttr(v)}}">${{escHtml(v)}}</span></div>`;
}}

function escHtml(s) {{
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[c]));
}}
function escAttr(s) {{
  return String(s == null ? "" : s).replace(/"/g, "&quot;");
}}

function shortPath(p) {{
  // 提取路径最后两段,便于紧凑展示
  if (!p) return "";
  const parts = p.replace(/[\\\\/]+$/, '').split(/[\\\\/]/);
  if (parts.length <= 2) return p;
  return ".../" + parts.slice(-2).join("/");
}}

async function refresh() {{
  setStatus("正在扫描缓存...");
  setProgress(true, "扫描中...");
  $("btn-refresh").disabled = true;
  try {{
    const resp = await fetch("/api/summary");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    state.summary = data;
    // 选中失效
    if (state.selectedPath && !data.repos.find(r => r.path === state.selectedPath)) {{
      state.selectedPath = null;
    }}
    renderAll();
    const note = data.cache_exists ? "" : "(缓存目录不存在)";
    setStatus(`扫描完成,共 ${{data.repo_count}} 个缓存项,总大小 ${{data.repo_total_size_text}}${{note}}`);
    if (data.repo_count) showToast(`扫描完成,共 ${{data.repo_count}} 个缓存项`, "success");
  }} catch (e) {{
    setStatus("扫描失败: " + e.message);
    showToast("扫描失败: " + e.message, "error");
  }} finally {{
    setProgress(false);
    $("btn-refresh").disabled = false;
  }}
}}

function showDetail(repo) {{
  const el = $("detail-body");
  let rows = `
    ${{kvRow("类型", repo.type_label)}}
    ${{kvRow("名称", repo.name)}}
    ${{kvRow("大小", repo.size_text)}}
    ${{kvRow("文件数", String(repo.files))}}
    ${{kvRow("修改时间", repo.modified_text)}}
  `;
  if (repo.cache_root) {{
    rows += `<div class="kv">
      <span class="k">来源</span>
      <span class="v path" title="${{escAttr(repo.cache_root)}}">${{escHtml(repo.cache_root)}}</span>
      <button class="flat" onclick="copyPath('${{escAttr(repo.cache_root)}}')">⧉</button>
    </div>`;
  }}
  rows += `<div class="kv">
    <span class="k">路径</span>
    <span class="v path" style="word-break: break-all; white-space: normal;">${{escHtml(repo.path)}}</span>
    <button class="flat" onclick="copyPath('${{escAttr(repo.path)}}')">⧉</button>
  </div>`;
  el.innerHTML = rows;
  openModal("modal-detail");
}}

function showSelectedDetail() {{
  if (!state.selectedPath) {{ showToast("请先选择一个缓存项", "warning"); return; }}
  const repo = state.summary.repos.find(r => r.path === state.selectedPath);
  if (!repo) {{ showToast("找不到选中的缓存项", "warning"); return; }}
  showDetail(repo);
}}

function confirmDelete() {{
  if (!state.selectedPath) {{ showToast("请先选择一个缓存项", "warning"); return; }}
  const repo = state.summary.repos.find(r => r.path === state.selectedPath);
  if (!repo) {{ showToast("找不到选中的缓存项", "warning"); return; }}
  state.pendingDelete = repo;
  $("delete-body").innerHTML = `
    <p style="color: var(--warning);">确定要删除以下缓存吗?此操作不可撤销.</p>
    <div class="kv"><span class="k">名称</span><span class="v">${{escHtml(repo.name)}}</span></div>
    <div class="kv"><span class="k">大小</span><span class="v">${{escHtml(repo.size_text)}}</span></div>
    <div class="kv"><span class="k">路径</span><span class="v path" style="word-break: break-all; white-space: normal;">${{escHtml(repo.path)}}</span></div>
  `;
  openModal("modal-delete");
}}

async function doDelete() {{
  if (!state.pendingDelete) return;
  const path = state.pendingDelete.path;
  closeModal("modal-delete");
  setStatus(`正在删除 ${{path}}...`);
  setProgress(true, "删除中...");
  try {{
    const resp = await fetch("/api/delete", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ path }}),
    }});
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "请求失败");
    // 轮询进度
    await pollTask(data.task_id);
  }} catch (e) {{
    setStatus("删除失败: " + e.message);
    showToast("删除失败: " + e.message, "error");
    setProgress(false);
  }}
}}

async function pollTask(taskId) {{
  return new Promise((resolve) => {{
    const tick = async () => {{
      try {{
        const resp = await fetch(`/api/task/${{taskId}}`);
        const info = await resp.json();
        if (info.status === "done") {{
          const r = info.result || {{}};
          if (r.ok) {{
            showToast("已删除缓存", "success");
            setStatus("已删除 " + r.deleted);
            state.selectedPath = null;
            state.pendingDelete = null;
            setProgress(false);
            await refresh();
          }} else {{
            showToast("删除失败: " + (r.error || "未知错误"), "error");
            setStatus("删除失败: " + (r.error || "未知错误"));
            setProgress(false);
          }}
          clearInterval(state.pollInterval);
          state.pollInterval = null;
          resolve();
        }} else if (info.status === "error") {{
          showToast("删除失败: " + info.error, "error");
          setStatus("删除失败: " + info.error);
          setProgress(false);
          clearInterval(state.pollInterval);
          state.pollInterval = null;
          resolve();
        }} else {{
          setProgress(true, info.message || "执行中...");
        }}
      }} catch (e) {{
        clearInterval(state.pollInterval);
        state.pollInterval = null;
        setProgress(false);
        resolve();
      }}
    }};
    state.pollInterval = setInterval(tick, 300);
    tick();
  }});
}}

async function openEnvDialog() {{
  if (state.summary) $("env-hf-home").value = state.summary.cache_dir;
  openModal("modal-env");
}}

async function generateEnvConfig() {{
  const settings = {{
    HF_HOME: $("env-hf-home").value.trim(),
    HF_ENDPOINT: $("env-hf-endpoint").value.trim(),
    HF_TOKEN: $("env-hf-token").value.trim(),
  }};
  const filtered = Object.fromEntries(Object.entries(settings).filter(([_, v]) => v));
  if (!Object.keys(filtered).length) {{
    showToast("请至少填写一项配置", "warning");
    return;
  }}
  try {{
    const resp = await fetch("/api/env-instructions", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(filtered),
    }});
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "生成失败");
    const ins = data.instructions;
    $("env-shell-label").textContent = `Unix Shell (${{ins.rc_file}}) · 当前 shell: ${{ins.shell}}`;
    $("env-unix").textContent = ins.unix_script;
    $("env-windows").textContent = ins.windows_commands.join("\\n");
    $("btn-copy-unix").onclick = async () => {{
      const ok = await copyToClipboard(ins.unix_script);
      showToast(ok ? "已复制 Unix 配置" : "复制失败", ok ? "success" : "error");
    }};
    $("btn-copy-windows").onclick = async () => {{
      const ok = await copyToClipboard(ins.windows_commands.join("\\n"));
      showToast(ok ? "已复制 Windows 配置" : "复制失败", ok ? "success" : "error");
    }};
    closeModal("modal-env");
    openModal("modal-env-result");
  }} catch (e) {{
    showToast("生成失败: " + e.message, "error");
  }}
}}

async function copyEnvVars() {{
  const vars = (state.summary && state.summary.env_vars) || {{}};
  const keys = Object.keys(vars);
  if (!keys.length) {{ showToast("当前没有可复制的环境变量", "warning"); return; }}
  const text = keys.map(k => `${{k}}=${{vars[k]}}`).join("\\n");
  const ok = await copyToClipboard(text);
  showToast(ok ? "已复制当前环境变量" : "复制失败", ok ? "success" : "error");
}}

async function copySelectedPath() {{
  if (!state.selectedPath) {{ showToast("请先选择一个缓存项", "warning"); return; }}
  const ok = await copyToClipboard(state.selectedPath);
  showToast(ok ? "已复制缓存路径" : "复制失败", ok ? "success" : "error");
}}

async function copyEnvVar(k, v) {{
  const ok = await copyToClipboard(`${{k}}=${{v}}`);
  showToast(ok ? `已复制 ${{k}}` : "复制失败", ok ? "success" : "error");
}}

async function copyPath(p) {{
  const ok = await copyToClipboard(p);
  showToast(ok ? "已复制缓存路径" : "复制失败", ok ? "success" : "error");
}}

// ---- 清理未完成下载 ----

async function scanIncomplete() {{
  const body = $("clean-incomplete-body");
  body.innerHTML = '<div class="empty">正在扫描...</div>';
  $("btn-confirm-clean").style.display = "none";
  $("btn-rescan-incomplete").style.display = "none";
  try {{
    const resp = await fetch("/api/incomplete-files");
    const data = await resp.json();
    if (!data.files.length) {{
      body.innerHTML = '<div class="empty" style="color: var(--positive);">✓ 没有未完成的下载,缓存很干净!</div>';
    }} else {{
      let html = `<p style="margin: 0 0 8px 0;">发现 <b>${{data.count}}</b> 个未完成下载,共 <b>${{escHtml(data.total_size_text)}}</b>,可以安全清理:</p>`;
      html += '<div style="max-height: 200px; overflow: auto; font-size: 11px; font-family: ui-monospace, monospace; background: var(--bg-3); padding: 8px; border-radius: 3px;">';
      for (const f of data.files) {{
        html += `<div style="padding: 2px 0; border-bottom: 1px solid var(--line);">${{escHtml(f.rel_path)}} <span style="color: var(--warning); float: right;">${{escHtml(f.size_text)}}</span></div>`;
      }}
      html += '</div>';
      body.innerHTML = html;
      $("btn-confirm-clean").style.display = "";
      $("btn-confirm-clean").textContent = "确认清理 · " + data.total_size_text;
    }}
  }} catch (e) {{
    body.innerHTML = '<div class="empty" style="color: var(--negative);">扫描失败: ' + escHtml(e.message) + '</div>';
    $("btn-rescan-incomplete").style.display = "";
  }}
}}

async function doCleanIncomplete() {{
  $("btn-confirm-clean").disabled = true;
  $("btn-confirm-clean").textContent = "清理中...";
  try {{
    const resp = await fetch("/api/clean-incomplete", {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: "{{}}" }});
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "请求失败");
    setProgress(true, "清理中...");
    await pollTask(data.task_id);
    closeModal("modal-clean-incomplete");
  }} catch (e) {{
    showToast("清理失败: " + e.message, "error");
    $("btn-confirm-clean").disabled = false;
    $("btn-confirm-clean").textContent = "确认清理";
  }}
}}

function openCleanIncompleteDialog() {{
  openModal("modal-clean-incomplete");
  scanIncomplete();
}}

// ---- 下载模型 ----

function openDownloadDialog() {{
  $("download-repo-id").value = "";
  openModal("modal-download");
  setTimeout(() => $("download-repo-id").focus(), 150);
}}

async function doDownload() {{
  const repoId = $("download-repo-id").value.trim();
  if (!repoId) {{ showToast("请输入仓库 ID", "warning"); return; }}
  if (repoId.indexOf("/") === -1) {{ showToast('仓库格式无效,应为 org/repo_name', "warning"); return; }}
  closeModal("modal-download");
  setStatus(`正在下载 ${{repoId}}...`);
  setProgress(true, "下载 " + repoId + " ...");
  $("btn-download").disabled = true;
  try {{
    const resp = await fetch("/api/download", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ repo_id: repoId }}),
    }});
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || "请求失败");
    await pollTask(data.task_id);
  }} catch (e) {{
    showToast("下载失败: " + e.message, "error");
    setStatus("下载失败: " + e.message);
    setProgress(false);
  }} finally {{
    $("btn-download").disabled = false;
  }}
}}

// 事件绑定
$("btn-refresh").onclick = refresh;
$("btn-env").onclick = openEnvDialog;
$("btn-env-generate").onclick = generateEnvConfig;
$("btn-detail").onclick = showSelectedDetail;
$("btn-delete").onclick = confirmDelete;
$("btn-confirm-delete").onclick = doDelete;
$("btn-copy-env").onclick = copyEnvVars;
$("btn-copy-path").onclick = copySelectedPath;
$("btn-clean-incomplete").onclick = openCleanIncompleteDialog;
$("btn-confirm-clean").onclick = doCleanIncomplete;
$("btn-download").onclick = openDownloadDialog;
$("btn-start-download").onclick = doDownload;

$("search").addEventListener("input", (e) => {{
  state.search = e.target.value;
  state.page = 0;
  renderRepoList();
  renderTable();
}});

// 类型筛选
$("filter-bar").addEventListener("click", (e) => {{
  const chip = e.target.closest(".filter-chip");
  if (!chip) return;
  state.typeFilter = chip.dataset.type;
  state.page = 0;
  renderFilterChips();
  renderRepoList();
  renderTable();
}});

// 仓库列表选中 / 双击
$("repo-list").addEventListener("click", (e) => {{
  const item = e.target.closest(".repo-item");
  if (!item || !item.dataset.path) return;
  state.selectedPath = item.dataset.path;
  renderRepoList();
  renderTable();
}});

$("repo-list").addEventListener("dblclick", (e) => {{
  const item = e.target.closest(".repo-item");
  if (!item || !item.dataset.path) return;
  const repo = state.summary && state.summary.repos.find(r => r.path === item.dataset.path);
  if (repo) showDetail(repo);
}});

document.querySelectorAll("thead th").forEach(th => {{
  th.addEventListener("click", () => {{
    const key = th.dataset.key;
    if (state.sortKey === key) {{
      state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    }} else {{
      state.sortKey = key;
      state.sortDir = key === "name" || key === "type_label" || key === "last_modified" ? "asc" : "desc";
    }}
    renderRepoList();
    renderTable();
  }});
}});

$("repos-body").addEventListener("click", (e) => {{
  const tr = e.target.closest("tr");
  if (!tr || !tr.dataset.path) return;
  state.selectedPath = tr.dataset.path;
  renderRepoList();
  document.querySelectorAll("#repos-body tr").forEach(r => r.classList.remove("selected"));
  tr.classList.add("selected");
}});

$("repos-body").addEventListener("dblclick", (e) => {{
  const tr = e.target.closest("tr");
  if (!tr || !tr.dataset.path) return;
  const repo = state.summary && state.summary.repos.find(r => r.path === tr.dataset.path);
  if (repo) showDetail(repo);
}});

$("page-prev").onclick = () => {{ state.page--; renderTable(); }};
$("page-next").onclick = () => {{ state.page++; renderTable(); }};

document.addEventListener("keydown", (e) => {{
  if (e.key === "Enter" && e.target.id === "download-repo-id") {{
    e.preventDefault();
    doDownload();
    return;
  }}
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft" && !$("page-prev").disabled) {{ state.page--; renderTable(); }}
  else if (e.key === "ArrowRight" && !$("page-next").disabled) {{ state.page++; renderTable(); }}
  else if (e.key === "Escape") {{
    document.querySelectorAll(".modal-mask.open").forEach(m => m.classList.remove("open"));
  }}
}});

// 启动时立即扫描
refresh();
</script>
</body>
</html>
"""


# ============================================================
# pywebview API class
# ============================================================


class API:
    """暴露给前端的 pywebview API(仅放系统能力,剪贴板走 JS 即可)"""

    def close_window(self) -> None:
        webview.windows[0].destroy()


# ============================================================
# 入口
# ============================================================


def main() -> None:
    url = f"http://{HOST}:{PORT}"

    # 1) 启动 FastAPI(后台线程)
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001
            print(f"FastAPI 启动失败: {exc}", file=sys.stderr)

    threading.Thread(target=_run_server, daemon=True).start()

    # 2) 等服务器起来
    time.sleep(1.0)

    # 3) 打开 pywebview 窗口
    print(f"Hugging Face Manager 启动: {url}")
    print("按 Ctrl+C 或关闭窗口停止")
    window = webview.create_window(
        title=TITLE,
        url=url,
        width=1280,
        height=800,
        min_size=(960, 600),
        js_api=API(),
        text_select=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
