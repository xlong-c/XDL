#!/usr/bin/env python3
"""Hugging Face 管理工具 — FastAPI + HTML 版本。

用法:
  python tools/setup/hf_manager_GUI.py

环境变量:
  XDL_WEB_HOST=127.0.0.1
  XDL_HF_MANAGER_PORT=8012
  XDL_NO_BROWSER=1
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import time
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Hugging Face 管理工具")


def format_size(size_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_time(timestamp: float) -> str:
    if timestamp <= 0:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
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
    for key in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser()
    if platform.system() == "Windows":
        return Path.home() / ".cache" / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def get_hf_env_vars() -> dict[str, str]:
    keys = [
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
    ]
    result: dict[str, str] = {}
    for key in keys:
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
    extensions: dict[str, int] = defaultdict(int)
    try:
        for dirpath, _, filenames in os.walk(cache_dir):
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
    if not cache_dir.exists():
        return []
    try:
        from huggingface_hub import scan_cache_dir as hf_scan

        result = hf_scan(cache_dir)
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
        return repos
    except Exception:
        return scan_cache_manual(cache_dir)


def repo_type_label(repo_type: str) -> str:
    return {"model": "Model", "dataset": "Dataset", "space": "Space"}.get(repo_type, repo_type)


def file_type_label(ext: str) -> str:
    mapping = {
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
        ".incomplete": "Incomplete Download",
        ".so": "Shared Library",
        "(noext)": "No Extension",
    }
    return mapping.get(ext, ext)


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
    cache_dir = get_hf_cache_dir().resolve()
    repos = get_repo_list(cache_dir)
    total_repo_size = sum(repo["size"] for repo in repos)
    subdirs = []
    for name in ("hub", "blobs", "assets", "refs", "snapshots", "modules"):
        path = cache_dir / name
        if path.exists():
            size = get_dir_size(path)
            subdirs.append({"name": name, "size": size, "size_text": format_size(size)})

    file_types = scan_cache_file_types(cache_dir) if cache_dir.exists() else {}
    top_file_types = [
        {
            "suffix": suffix,
            "label": file_type_label(suffix),
            "size": size,
            "size_text": format_size(size),
        }
        for suffix, size in sorted(file_types.items(), key=lambda item: item[1], reverse=True)[:8]
    ]
    top_repos = sorted(repos, key=lambda repo: repo["size"], reverse=True)[:5]
    env_vars = get_hf_env_vars()
    sys_info = get_system_info()

    return {
        "cache_dir": str(cache_dir),
        "cache_exists": cache_dir.exists(),
        "repo_count": len(repos),
        "repo_total_size": total_repo_size,
        "repo_total_size_text": format_size(total_repo_size),
        "repos": [
            {
                **repo,
                "type_label": repo_type_label(repo["type"]),
                "size_text": format_size(repo["size"]),
                "modified_text": format_time(repo["last_modified"]),
            }
            for repo in repos
        ],
        "subdirs": subdirs,
        "file_types": top_file_types,
        "top_repos": [
            {
                **repo,
                "type_label": repo_type_label(repo["type"]),
                "size_text": format_size(repo["size"]),
                "modified_text": format_time(repo["last_modified"]),
            }
            for repo in top_repos
        ],
        "env_vars": env_vars,
        "system_info": sys_info,
    }


def resolve_repo_path(path_text: str) -> Path:
    path = Path((path_text or "").strip()).expanduser().resolve()
    return path


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    try:
        return int(value)
    except ValueError:
        return default


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_PAGE


@app.get("/api/summary")
async def api_summary() -> dict[str, Any]:
    return collect_summary()


@app.post("/api/delete-repo", response_model=None)
async def api_delete_repo(data: dict[str, Any]):
    repo_path = resolve_repo_path(str(data.get("path", "")))
    if not repo_path.exists():
        return JSONResponse({"error": f"路径不存在: {repo_path}"}, status_code=404)
    try:
        shutil.rmtree(repo_path)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"deleted": str(repo_path)}


@app.post("/api/env-instructions", response_model=None)
async def api_env_instructions(data: dict[str, Any]):
    settings: dict[str, str] = {}
    for key in ("HF_HOME", "HF_ENDPOINT", "HF_TOKEN"):
        value = str(data.get(key, "")).strip()
        if value:
            settings[key] = value
    if not settings:
        return JSONResponse({"error": "没有可生成的配置项"}, status_code=400)
    return build_env_instructions(settings)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hugging Face 管理工具</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
    background: #1e1e1e;
    color: #d0d0d0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
#toolbar {
    height: 52px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 14px;
    background: #252525;
    border-bottom: 1px solid #343434;
}
#title {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
}
.spacer { flex: 1; }
.input, .btn, textarea {
    border: 1px solid #4b4b4b;
    border-radius: 6px;
    background: #303030;
    color: #e0e0e0;
    font-size: 13px;
}
.input {
    min-height: 34px;
    padding: 0 10px;
}
.btn {
    min-height: 34px;
    padding: 0 14px;
    cursor: pointer;
}
.btn:hover { background: #404040; }
.btn.primary {
    background: #007acc;
    border-color: #007acc;
    color: #fff;
}
#search-input {
    width: 240px;
}
#layout {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(320px, 0.95fr) minmax(520px, 1.45fr);
    gap: 12px;
    padding: 12px;
}
.column {
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.panel {
    background: #252525;
    border: 1px solid #343434;
    border-radius: 8px;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.panel-header {
    padding: 12px 14px;
    border-bottom: 1px solid #343434;
    font-size: 13px;
    color: #f0f0f0;
    font-weight: 600;
}
.panel-body {
    padding: 12px 14px;
    overflow: auto;
    min-height: 0;
}
.kv-list {
    display: grid;
    gap: 8px;
}
.kv {
    display: grid;
    grid-template-columns: 96px 1fr auto;
    gap: 8px;
    align-items: center;
    font-size: 13px;
}
.kv .key { color: #8c8c8c; font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; }
.kv .value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.muted { color: #888888; }
#repo-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
#repo-table th, #repo-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #303030;
    text-align: left;
    white-space: nowrap;
}
#repo-table th {
    position: sticky;
    top: 0;
    background: #252525;
    color: #a9a9a9;
    cursor: pointer;
    z-index: 1;
}
#repo-table tbody tr:hover { background: #2d2d2d; }
#repo-table tbody tr.selected { background: rgba(0, 122, 204, 0.22); }
.repo-name {
    max-width: 380px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tag {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    background: #303030;
    color: #d0d0d0;
    font-size: 12px;
}
.stat-block {
    display: grid;
    gap: 8px;
}
.stat-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
}
.bar {
    height: 8px;
    border-radius: 999px;
    background: #1a1a1a;
    overflow: hidden;
    flex: 1;
}
.bar > span {
    display: block;
    height: 100%;
    background: #007acc;
}
#statusbar {
    padding: 8px 14px;
    border-top: 1px solid #343434;
    background: #252525;
    color: #8a8a8a;
    font-size: 12px;
    display: flex;
    gap: 12px;
    align-items: center;
}
#status-text { color: #88c0ff; }
#dialog-mask {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.6);
}
#dialog {
    width: min(720px, calc(100vw - 40px));
    max-height: calc(100vh - 60px);
    overflow: auto;
    background: #252525;
    border: 1px solid #343434;
    border-radius: 8px;
}
#dialog .panel-body textarea {
    width: 100%;
    min-height: 160px;
    padding: 10px;
    resize: vertical;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
}
#dialog .field {
    display: grid;
    gap: 6px;
    margin-bottom: 10px;
}
#dialog .field label {
    color: #9a9a9a;
    font-size: 12px;
}
@media (max-width: 1180px) {
    #layout {
        grid-template-columns: 1fr;
    }
}
</style>
</head>
<body>
<div id="toolbar">
    <div id="title">Hugging Face Manager</div>
    <div class="spacer"></div>
    <input id="search-input" class="input" placeholder="搜索模型或数据集">
    <button id="refresh-btn" class="btn primary">刷新</button>
    <button id="config-btn" class="btn">环境配置</button>
    <button id="copy-env-btn" class="btn">复制环境变量</button>
    <button id="copy-path-btn" class="btn">复制路径</button>
    <button id="delete-btn" class="btn">删除缓存</button>
</div>

<div id="layout">
    <div class="column">
        <div class="panel">
            <div class="panel-header">缓存概览</div>
            <div class="panel-body">
                <div class="kv-list" id="overview-list"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">环境变量</div>
            <div class="panel-body">
                <div class="kv-list" id="env-list"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">文件类型分布</div>
            <div class="panel-body">
                <div class="stat-block" id="file-types"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">Top 5 占用</div>
            <div class="panel-body">
                <div class="stat-block" id="top-repos"></div>
            </div>
        </div>
    </div>
    <div class="column">
        <div class="panel" style="flex:1;">
            <div class="panel-header">缓存列表</div>
            <div class="panel-body" style="padding:0;">
                <table id="repo-table">
                    <thead>
                        <tr>
                            <th data-sort="type">类型</th>
                            <th data-sort="name">名称</th>
                            <th data-sort="size">大小</th>
                            <th data-sort="files">文件数</th>
                            <th data-sort="last_modified">修改时间</th>
                        </tr>
                    </thead>
                    <tbody id="repo-table-body"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<div id="statusbar">
    <div id="status-text">就绪</div>
    <div id="system-text"></div>
</div>

<div id="dialog-mask">
    <div id="dialog">
        <div class="panel-header" id="dialog-title">环境配置</div>
        <div class="panel-body" id="dialog-body"></div>
    </div>
</div>

<script>
const state = {
    summary: null,
    selectedRepo: null,
    search: '',
    sortKey: 'size',
    sortDesc: true,
};

const statusTextEl = document.getElementById('status-text');
const systemTextEl = document.getElementById('system-text');
const dialogMaskEl = document.getElementById('dialog-mask');
const dialogTitleEl = document.getElementById('dialog-title');
const dialogBodyEl = document.getElementById('dialog-body');

function setStatus(text, isError = false) {
    statusTextEl.textContent = text;
    statusTextEl.style.color = isError ? '#ff7b72' : '#88c0ff';
}

function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || `请求失败: ${response.status}`);
    }
    return data;
}

async function refreshSummary() {
    setStatus('正在扫描缓存...');
    try {
        state.summary = await fetchJSON('/api/summary');
        renderAll();
        setStatus(`扫描完成，共 ${state.summary.repo_count} 个缓存项，总大小 ${state.summary.repo_total_size_text}`);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function renderAll() {
    renderOverview();
    renderEnv();
    renderFileTypes();
    renderTopRepos();
    renderRepoTable();
    renderSystemInfo();
}

function renderOverview() {
    const root = document.getElementById('overview-list');
    const summary = state.summary;
    if (!summary) {
        root.innerHTML = '';
        return;
    }
    const rows = [
        {key: '路径', value: summary.cache_dir},
        {key: '缓存目录', value: summary.cache_exists ? '存在' : '不存在'},
        {key: '缓存项', value: `${summary.repo_count}`},
        {key: '总大小', value: summary.repo_total_size_text},
    ];
    for (const subdir of summary.subdirs) {
        rows.push({key: subdir.name, value: subdir.size_text});
    }
    root.innerHTML = rows.map((row) => `
        <div class="kv">
            <div class="key">${escapeHtml(row.key)}</div>
            <div class="value" title="${escapeHtml(row.value)}">${escapeHtml(row.value)}</div>
            <div></div>
        </div>
    `).join('');
}

function renderEnv() {
    const root = document.getElementById('env-list');
    const envVars = state.summary ? state.summary.env_vars : {};
    const entries = Object.entries(envVars || {});
    if (!entries.length) {
        root.innerHTML = '<div class="muted">当前没有检测到 Hugging Face 环境变量。</div>';
        return;
    }
    root.innerHTML = entries.map(([key, value]) => `
        <div class="kv">
            <div class="key">${escapeHtml(key)}</div>
            <div class="value" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
            <button class="btn" data-copy="${escapeHtml(`${key}=${value}`)}">复制</button>
        </div>
    `).join('');

    root.querySelectorAll('[data-copy]').forEach((button) => {
        button.addEventListener('click', async () => {
            const text = button.getAttribute('data-copy') || '';
            await navigator.clipboard.writeText(text);
            setStatus(`已复制 ${text.split('=')[0]}`);
        });
    });
}

function renderFileTypes() {
    const root = document.getElementById('file-types');
    const items = state.summary ? state.summary.file_types : [];
    if (!items.length) {
        root.innerHTML = '<div class="muted">没有可展示的数据。</div>';
        return;
    }
    const maxSize = Math.max(...items.map((item) => item.size), 1);
    root.innerHTML = items.map((item) => `
        <div class="stat-row">
            <div style="width:150px">${escapeHtml(item.label)}</div>
            <div class="bar"><span style="width:${(item.size / maxSize) * 100}%"></span></div>
            <div style="width:92px;text-align:right">${escapeHtml(item.size_text)}</div>
        </div>
    `).join('');
}

function renderTopRepos() {
    const root = document.getElementById('top-repos');
    const items = state.summary ? state.summary.top_repos : [];
    if (!items.length) {
        root.innerHTML = '<div class="muted">没有可展示的数据。</div>';
        return;
    }
    const maxSize = Math.max(...items.map((item) => item.size), 1);
    root.innerHTML = items.map((item, index) => `
        <div class="stat-row">
            <div style="width:24px">${index + 1}.</div>
            <div style="width:90px"><span class="tag">${escapeHtml(item.type_label)}</span></div>
            <div style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
            <div class="bar"><span style="width:${(item.size / maxSize) * 100}%"></span></div>
            <div style="width:92px;text-align:right">${escapeHtml(item.size_text)}</div>
        </div>
    `).join('');
}

function getSortedRepos() {
    if (!state.summary) {
        return [];
    }
    const query = state.search.trim().toLowerCase();
    const repos = state.summary.repos.filter((repo) => {
        if (!query) {
            return true;
        }
        return repo.name.toLowerCase().includes(query) || repo.type.toLowerCase().includes(query);
    });
    const sorted = [...repos].sort((left, right) => {
        const key = state.sortKey;
        const leftValue = left[key];
        const rightValue = right[key];
        if (typeof leftValue === 'number' && typeof rightValue === 'number') {
            return leftValue - rightValue;
        }
        return String(leftValue).localeCompare(String(rightValue), 'zh-CN');
    });
    if (state.sortDesc) {
        sorted.reverse();
    }
    return sorted;
}

function renderRepoTable() {
    const tbody = document.getElementById('repo-table-body');
    const repos = getSortedRepos();
    if (!repos.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="muted">没有匹配的数据。</td></tr>';
        return;
    }
    tbody.innerHTML = repos.map((repo) => `
        <tr data-path="${escapeHtml(repo.path)}" class="${state.selectedRepo && state.selectedRepo.path === repo.path ? 'selected' : ''}">
            <td><span class="tag">${escapeHtml(repo.type_label)}</span></td>
            <td class="repo-name" title="${escapeHtml(repo.name)}">${escapeHtml(repo.name)}</td>
            <td>${escapeHtml(repo.size_text)}</td>
            <td>${escapeHtml(String(repo.files))}</td>
            <td>${escapeHtml(repo.modified_text)}</td>
        </tr>
    `).join('');

    tbody.querySelectorAll('tr[data-path]').forEach((row) => {
        row.addEventListener('click', () => {
            const path = row.getAttribute('data-path');
            state.selectedRepo = repos.find((repo) => repo.path === path) || null;
            renderRepoTable();
            if (state.selectedRepo) {
                setStatus(`已选中: ${state.selectedRepo.name}`);
            }
        });
        row.addEventListener('dblclick', () => {
            const path = row.getAttribute('data-path');
            const repo = repos.find((item) => item.path === path);
            if (repo) {
                showRepoDetail(repo);
            }
        });
    });
}

function renderSystemInfo() {
    if (!state.summary) {
        systemTextEl.textContent = '';
        return;
    }
    const info = state.summary.system_info;
    systemTextEl.textContent = `Python ${info.python} | huggingface_hub ${info.huggingface_hub} | ${info.system} ${info.release}`;
}

function closeDialog() {
    dialogMaskEl.style.display = 'none';
    dialogTitleEl.textContent = '';
    dialogBodyEl.innerHTML = '';
}

function openDialog(title, bodyHtml) {
    dialogTitleEl.textContent = title;
    dialogBodyEl.innerHTML = bodyHtml;
    dialogMaskEl.style.display = 'flex';
}

function showRepoDetail(repo) {
    openDialog('缓存详情', `
        <div class="kv-list">
            <div class="kv"><div class="key">类型</div><div class="value">${escapeHtml(repo.type_label)}</div><div></div></div>
            <div class="kv"><div class="key">名称</div><div class="value" title="${escapeHtml(repo.name)}">${escapeHtml(repo.name)}</div><div></div></div>
            <div class="kv"><div class="key">大小</div><div class="value">${escapeHtml(repo.size_text)}</div><div></div></div>
            <div class="kv"><div class="key">文件数</div><div class="value">${escapeHtml(String(repo.files))}</div><div></div></div>
            <div class="kv"><div class="key">修改时间</div><div class="value">${escapeHtml(repo.modified_text)}</div><div></div></div>
            <div class="kv"><div class="key">路径</div><div class="value" title="${escapeHtml(repo.path)}">${escapeHtml(repo.path)}</div><button class="btn" id="dialog-copy-path">复制</button></div></div>
        </div>
        <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
            <button class="btn" id="dialog-close">关闭</button>
        </div>
    `);
    document.getElementById('dialog-close').addEventListener('click', closeDialog);
    document.getElementById('dialog-copy-path').addEventListener('click', async () => {
        await navigator.clipboard.writeText(repo.path);
        setStatus('已复制缓存路径');
    });
}

function showEnvConfigDialog() {
    openDialog('环境配置', `
        <div class="field">
            <label for="cfg-hf-home">HF_HOME</label>
            <input id="cfg-hf-home" class="input" value="${escapeHtml(state.summary ? state.summary.cache_dir : '')}">
        </div>
        <div class="field">
            <label for="cfg-hf-endpoint">HF_ENDPOINT</label>
            <input id="cfg-hf-endpoint" class="input" placeholder="https://hf-mirror.com">
        </div>
        <div class="field">
            <label for="cfg-hf-token">HF_TOKEN</label>
            <input id="cfg-hf-token" class="input" type="password" placeholder="hf_xxx">
        </div>
        <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
            <button class="btn" id="dialog-cancel">取消</button>
            <button class="btn primary" id="dialog-build">生成配置</button>
        </div>
    `);

    document.getElementById('dialog-cancel').addEventListener('click', closeDialog);
    document.getElementById('dialog-build').addEventListener('click', async () => {
        try {
            const payload = {
                HF_HOME: document.getElementById('cfg-hf-home').value.trim(),
                HF_ENDPOINT: document.getElementById('cfg-hf-endpoint').value.trim(),
                HF_TOKEN: document.getElementById('cfg-hf-token').value.trim(),
            };
            const result = await fetchJSON('/api/env-instructions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            openDialog('配置说明', `
                <div class="field">
                    <label>Unix Shell (${escapeHtml(result.rc_file)})</label>
                    <textarea readonly>${escapeHtml(result.unix_script)}</textarea>
                </div>
                <div class="field">
                    <label>Windows</label>
                    <textarea readonly>${escapeHtml(result.windows_commands.join('\n'))}</textarea>
                </div>
                <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
                    <button class="btn" id="dialog-copy-unix">复制 Unix</button>
                    <button class="btn" id="dialog-copy-win">复制 Windows</button>
                    <button class="btn primary" id="dialog-close2">关闭</button>
                </div>
            `);
            document.getElementById('dialog-copy-unix').addEventListener('click', async () => {
                await navigator.clipboard.writeText(result.unix_script);
                setStatus('已复制 Unix 配置');
            });
            document.getElementById('dialog-copy-win').addEventListener('click', async () => {
                await navigator.clipboard.writeText(result.windows_commands.join('\n'));
                setStatus('已复制 Windows 配置');
            });
            document.getElementById('dialog-close2').addEventListener('click', closeDialog);
        } catch (error) {
            setStatus(error.message, true);
        }
    });
}

async function copyCurrentEnv() {
    if (!state.summary) {
        return;
    }
    const lines = Object.entries(state.summary.env_vars).map(([key, value]) => `${key}=${value}`);
    if (!lines.length) {
        setStatus('当前没有可复制的环境变量', true);
        return;
    }
    await navigator.clipboard.writeText(lines.join('\n'));
    setStatus('已复制当前环境变量');
}

async function copySelectedPath() {
    if (!state.selectedRepo) {
        setStatus('请先选择一个缓存项', true);
        return;
    }
    await navigator.clipboard.writeText(state.selectedRepo.path);
    setStatus(`已复制路径: ${state.selectedRepo.path}`);
}

async function deleteSelectedRepo() {
    if (!state.selectedRepo) {
        setStatus('请先选择一个缓存项', true);
        return;
    }
    const repo = state.selectedRepo;
    const ok = window.confirm(`确定删除以下缓存吗？\n\n${repo.name}\n${repo.size_text}\n${repo.path}\n\n此操作不可撤销。`);
    if (!ok) {
        return;
    }
    try {
        await fetchJSON('/api/delete-repo', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: repo.path}),
        });
        state.selectedRepo = null;
        await refreshSummary();
        setStatus(`已删除缓存: ${repo.name}`);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function bindEvents() {
    document.getElementById('refresh-btn').addEventListener('click', refreshSummary);
    document.getElementById('config-btn').addEventListener('click', showEnvConfigDialog);
    document.getElementById('copy-env-btn').addEventListener('click', copyCurrentEnv);
    document.getElementById('copy-path-btn').addEventListener('click', copySelectedPath);
    document.getElementById('delete-btn').addEventListener('click', deleteSelectedRepo);
    document.getElementById('search-input').addEventListener('input', (event) => {
        state.search = event.target.value;
        renderRepoTable();
    });
    document.querySelectorAll('#repo-table th[data-sort]').forEach((th) => {
        th.addEventListener('click', () => {
            const key = th.getAttribute('data-sort');
            if (state.sortKey === key) {
                state.sortDesc = !state.sortDesc;
            } else {
                state.sortKey = key;
                state.sortDesc = ['size', 'files', 'last_modified'].includes(key);
            }
            renderRepoTable();
        });
    });
    dialogMaskEl.addEventListener('click', (event) => {
        if (event.target === dialogMaskEl) {
            closeDialog();
        }
    });
}

bindEvents();
refreshSummary();
</script>
</body>
</html>
"""


def main() -> None:
    host = os.environ.get("XDL_WEB_HOST", "127.0.0.1")
    port = env_int("XDL_HF_MANAGER_PORT", 8012)
    no_browser = os.environ.get("XDL_NO_BROWSER", "").strip() == "1"
    url = f"http://{host}:{port}"

    def open_browser() -> None:
        import time as _time

        _time.sleep(1.2)
        webbrowser.open(url)

    if not no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Hugging Face 管理工具启动: {url}")
    print("按 Ctrl+C 停止")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
