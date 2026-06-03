#!/usr/bin/env python3
"""批量图片查看器 — FastAPI + HTML 版本。

用法:
  python tools/image/batch_viewer_GUI.py

环境变量:
  XDL_WEB_HOST=127.0.0.1
  XDL_BATCH_VIEWER_PORT=8011
  XDL_NO_BROWSER=1

功能:
  - 按目录批量浏览图片
  - 支持切片拼接预览
  - 支持选择并删除图片
  - 支持分页、缩放、跳转
"""

from __future__ import annotations

import io
import os
import re
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image

app = FastAPI(title="批量图片查看器")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def natural_sort_key(path: str) -> list[Any]:
    filename = Path(path).name.lower()
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", filename)]


def resolve_directory(path_text: str) -> Path:
    raw = (path_text or ".").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def resolve_file(path_text: str) -> Path:
    raw = unquote((path_text or "").strip())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def list_images(directory: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for child in sorted(directory.iterdir(), key=lambda item: natural_sort_key(str(item))):
        if child.is_file() and child.suffix.lower() in VALID_EXTENSIONS:
            entries.append({"name": child.name, "path": str(child.resolve())})
    return entries


def parse_slice_indices(slice_text: str, slice_parts: int) -> list[int]:
    if slice_parts < 1:
        return []
    indices: list[int] = []
    for part in (slice_text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if 1 <= value <= slice_parts:
            indices.append(value)
    return sorted(set(indices))


def normalize_hex_color(color: str) -> str:
    value = (color or "#000000").strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
            return value.lower()
        except ValueError:
            pass
    return "#000000"


def parse_hex_color(color: str) -> tuple[int, int, int]:
    value = normalize_hex_color(color)
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))


def compose_preview(
    image: Image.Image,
    slice_enabled: bool,
    slice_indices: list[int],
    slice_parts: int,
) -> Image.Image:
    preview = image.convert("RGB")
    if not slice_enabled or not slice_indices or slice_parts < 1:
        return preview

    width, height = preview.size
    if width <= 0 or height <= 0:
        return preview

    boundaries = [round(width * idx / slice_parts) for idx in range(slice_parts + 1)]
    pieces: list[Image.Image] = []
    for index in slice_indices:
        start = boundaries[index - 1]
        end = boundaries[index]
        if end <= start:
            continue
        pieces.append(preview.crop((start, 0, end, height)))

    if not pieces:
        return preview

    total_width = sum(piece.width for piece in pieces)
    max_height = max(piece.height for piece in pieces)
    merged = Image.new("RGB", (total_width, max_height), (0, 0, 0))
    offset_x = 0
    for piece in pieces:
        merged.paste(piece, (offset_x, 0))
        offset_x += piece.width
    return merged


def render_thumbnail(
    image_path: Path,
    width: int,
    height: int,
    slice_enabled: bool,
    slice_indices: list[int],
    slice_parts: int,
    bg_color: str,
) -> io.BytesIO:
    target_width = max(1, width)
    target_height = max(1, height)
    with Image.open(image_path) as image:
        preview = compose_preview(image, slice_enabled, slice_indices, slice_parts)
        preview.thumbnail((target_width, target_height), RESAMPLE)
        canvas = Image.new("RGB", (target_width, target_height), parse_hex_color(bg_color))
        offset_x = max(0, (target_width - preview.width) // 2)
        offset_y = max(0, (target_height - preview.height) // 2)
        canvas.paste(preview, (offset_x, offset_y))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    try:
        return int(value)
    except ValueError:
        return default


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_PAGE


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "default_dir": str(Path.cwd()),
        "default_cols": 2,
        "default_slice_enabled": True,
        "default_slice_indices": "2,3",
        "default_slice_parts": 4,
        "default_pad_x": 0,
        "default_pad_y": 0,
        "default_ratio": "4/1",
        "default_bg_color": "#000000",
    }


@app.post("/api/list-dir")
async def api_list_dir(data: dict[str, Any]) -> dict[str, Any]:
    directory = resolve_directory(str(data.get("path", ".")))
    if not directory.exists():
        return JSONResponse({"error": f"目录不存在: {directory}"}, status_code=404)
    if not directory.is_dir():
        return JSONResponse({"error": f"不是目录: {directory}"}, status_code=400)

    images = list_images(directory)
    return {
        "directory": str(directory),
        "images": images,
        "count": len(images),
    }


@app.get("/api/thumb", response_model=None)
async def api_thumb(
    path: str = Query(...),
    width: int = Query(480, ge=1, le=4096),
    height: int = Query(160, ge=1, le=4096),
    slice_enabled: bool = Query(True),
    slice_indices: str = Query("2,3"),
    slice_parts: int = Query(4, ge=1, le=128),
    bg_color: str = Query("#000000"),
):
    image_path = resolve_file(path)
    if not image_path.exists() or not image_path.is_file():
        return JSONResponse({"error": "image not found"}, status_code=404)

    indices = parse_slice_indices(slice_indices, slice_parts)
    try:
        buf = render_thumbnail(
            image_path=image_path,
            width=width,
            height=height,
            slice_enabled=slice_enabled,
            slice_indices=indices,
            slice_parts=slice_parts,
            bg_color=bg_color,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/image", response_model=None)
async def api_image(path: str = Query(...)):
    image_path = resolve_file(path)
    if not image_path.exists() or not image_path.is_file():
        return JSONResponse({"error": "image not found"}, status_code=404)

    try:
        with Image.open(image_path) as image:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            buf.seek(0)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/delete", response_model=None)
async def api_delete(data: dict[str, Any]):
    raw_paths = data.get("paths")
    if not isinstance(raw_paths, list):
        return JSONResponse({"error": "paths must be a list"}, status_code=400)

    deleted_paths: list[str] = []
    failed: list[dict[str, str]] = []
    for raw_path in raw_paths:
        try:
            path = resolve_file(str(raw_path))
            path.unlink()
            deleted_paths.append(str(path))
        except Exception as exc:
            failed.append({"path": str(raw_path), "error": str(exc)})

    return {
        "deleted_count": len(deleted_paths),
        "deleted_paths": deleted_paths,
        "failed": failed,
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>批量图片查看器</title>
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
    background: #2a2a2a;
    border-bottom: 1px solid #3a3a3a;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: minmax(260px, 1.6fr) repeat(9, minmax(72px, auto)) minmax(90px, 0.8fr);
    gap: 8px;
    align-items: center;
}
.input, .btn, .check {
    min-height: 34px;
    border-radius: 6px;
    border: 1px solid #4a4a4a;
    background: #343434;
    color: #e0e0e0;
    font-size: 13px;
}
.input {
    padding: 0 10px;
    width: 100%;
}
.btn {
    padding: 0 12px;
    cursor: pointer;
    transition: background 0.15s ease;
}
.btn:hover { background: #444444; }
.btn.primary { background: #007acc; border-color: #007acc; color: #fff; }
.btn.primary:hover { background: #1390ea; }
.check {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 10px;
    white-space: nowrap;
}
.check input { accent-color: #007acc; }
.field {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}
.field label {
    color: #8e8e8e;
    font-size: 12px;
}
#content {
    min-height: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
}
#statusbar {
    background: #252525;
    border-bottom: 1px solid #343434;
    padding: 8px 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    white-space: nowrap;
    overflow: hidden;
}
#status { color: #88c0ff; font-size: 13px; }
#meta {
    margin-left: auto;
    color: #8a8a8a;
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
}
#grid-wrap {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 10px;
    background: #111111;
}
#grid {
    display: grid;
    align-content: start;
}
.tile {
    position: relative;
    border: 1px solid #2f2f2f;
    background: #000;
    overflow: hidden;
    cursor: pointer;
    user-select: none;
}
.tile:hover { border-color: #4d4d4d; }
.tile.selected { border-color: #d33f3f; box-shadow: inset 0 0 0 1px #d33f3f; }
.tile img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
}
.tile .name {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 6px 8px;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.85));
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tile .badge {
    position: absolute;
    top: 8px;
    left: 8px;
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.72);
    font-size: 11px;
    color: #fff;
}
.tile .cross {
    position: absolute;
    inset: 0;
    pointer-events: none;
    display: none;
}
.tile.selected .cross { display: block; }
.tile .cross::before,
.tile .cross::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 150%;
    height: 3px;
    background: rgba(255, 64, 64, 0.9);
    transform-origin: center center;
}
.tile .cross::before { transform: translate(-50%, -50%) rotate(30deg); }
.tile .cross::after { transform: translate(-50%, -50%) rotate(-30deg); }
#empty {
    display: none;
    height: 100%;
    align-items: center;
    justify-content: center;
    color: #7a7a7a;
    font-size: 14px;
}
@media (max-width: 1280px) {
    #toolbar {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
}
</style>
</head>
<body>
<div id="toolbar">
    <input id="dir-input" class="input" placeholder="输入图片目录，例如 ./ 或 /data/images">
    <button id="open-btn" class="btn primary">打开目录</button>
    <button id="prev-btn" class="btn">上一页</button>
    <button id="next-btn" class="btn">下一页</button>
    <input id="jump-input" class="input" placeholder="跳到索引" inputmode="numeric">
    <button id="jump-btn" class="btn">跳转</button>
    <button id="zoom-in-btn" class="btn">放大</button>
    <button id="zoom-out-btn" class="btn">缩小</button>
    <button id="delete-btn" class="btn">删除选中</button>
    <label class="check"><input type="checkbox" id="slice-enabled" checked>切片预览</label>
    <div class="field"><label for="slice-input">切片</label><input id="slice-input" class="input" value="2,3"></div>
    <div class="field"><label for="parts-input">份数</label><input id="parts-input" class="input" value="4" inputmode="numeric"></div>
    <div class="field"><label for="ratio-input">比例</label><input id="ratio-input" class="input" value="4/1"></div>
    <div class="field"><label for="bg-input">背景</label><input id="bg-input" class="input" value="#000000"></div>
</div>
<div id="content">
    <div id="statusbar">
        <div id="status">就绪</div>
        <div id="meta"></div>
    </div>
    <div id="grid-wrap">
        <div id="empty">当前目录没有可显示的图片。</div>
        <div id="grid"></div>
    </div>
</div>

<script>
const state = {
    folder: '.',
    images: [],
    selected: new Set(),
    pageStart: 0,
    cols: 2,
    sliceEnabled: true,
    sliceIndices: '2,3',
    sliceParts: 4,
    ratioText: '4/1',
    bgColor: '#000000',
    padX: 0,
    padY: 0,
};

const gridWrapEl = document.getElementById('grid-wrap');
const gridEl = document.getElementById('grid');
const emptyEl = document.getElementById('empty');
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#ff7b72' : '#88c0ff';
}

function parseRatio(value) {
    const raw = String(value || '').trim();
    if (raw.includes('/')) {
        const parts = raw.split('/');
        const left = Number(parts[0]);
        const right = Number(parts[1]);
        if (left > 0 && right > 0) {
            return left / right;
        }
    }
    const number = Number(raw);
    return number > 0 ? number : 4;
}

function normalizeSliceIndices() {
    const sliceParts = Math.max(1, Number(document.getElementById('parts-input').value) || 4);
    const raw = document.getElementById('slice-input').value;
    const values = raw.split(',').map((item) => Number(item.trim())).filter((item) => Number.isInteger(item) && item >= 1 && item <= sliceParts);
    const unique = [...new Set(values)];
    if (!unique.length) {
        unique.push(1);
    }
    state.sliceParts = sliceParts;
    state.sliceIndices = unique.join(',');
    document.getElementById('slice-input').value = state.sliceIndices;
    document.getElementById('parts-input').value = String(state.sliceParts);
    return unique;
}

function getEffectiveCols() {
    const indices = normalizeSliceIndices();
    if (!state.sliceEnabled) {
        return Math.max(1, state.cols);
    }
    return Math.max(1, Math.floor((state.cols * state.sliceParts) / Math.max(1, indices.length)));
}

function estimateRows() {
    const effectiveCols = getEffectiveCols();
    const ratio = parseRatio(state.ratioText);
    const availableWidth = Math.max(320, gridWrapEl.clientWidth - 20);
    const availableHeight = Math.max(220, gridWrapEl.clientHeight - 20);
    const cellWidth = Math.max(60, (availableWidth - (effectiveCols - 1) * state.padX) / effectiveCols);
    const cellHeight = Math.max(36, cellWidth / ratio);
    return Math.max(1, Math.floor((availableHeight + state.padY) / (cellHeight + state.padY)));
}

function getPageSize() {
    return getEffectiveCols() * estimateRows();
}

function syncStateFromInputs() {
    state.folder = document.getElementById('dir-input').value.trim() || '.';
    state.sliceEnabled = document.getElementById('slice-enabled').checked;
    state.ratioText = document.getElementById('ratio-input').value.trim() || '4/1';
    state.bgColor = document.getElementById('bg-input').value.trim() || '#000000';
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || `请求失败: ${response.status}`);
    }
    return data;
}

async function openDirectory() {
    syncStateFromInputs();
    setStatus(`正在读取目录: ${state.folder}`);
    try {
        const data = await postJSON('/api/list-dir', {path: state.folder});
        state.folder = data.directory;
        state.images = data.images;
        state.selected.clear();
        state.pageStart = 0;
        document.getElementById('dir-input').value = data.directory;
        render();
        if (state.images.length) {
            setStatus(`已加载 ${state.images.length} 张图片`);
        } else {
            setStatus('目录中没有找到图片');
        }
    } catch (error) {
        setStatus(error.message, true);
    }
}

function render() {
    syncStateFromInputs();
    normalizeSliceIndices();

    const ratio = parseRatio(state.ratioText);
    const effectiveCols = getEffectiveCols();
    const rows = estimateRows();
    const pageSize = effectiveCols * rows;
    const total = state.images.length;
    if (state.pageStart >= total && total > 0) {
        state.pageStart = Math.max(0, Math.floor((total - 1) / pageSize) * pageSize);
    }
    const pageImages = state.images.slice(state.pageStart, state.pageStart + pageSize);
    emptyEl.style.display = pageImages.length ? 'none' : 'flex';
    gridEl.innerHTML = '';
    gridEl.style.gridTemplateColumns = `repeat(${effectiveCols}, minmax(0, 1fr))`;
    gridEl.style.gap = `${state.padY}px ${state.padX}px`;

    const availableWidth = Math.max(320, gridWrapEl.clientWidth - 20);
    const cellWidth = Math.max(60, Math.floor((availableWidth - (effectiveCols - 1) * state.padX) / effectiveCols));
    const cellHeight = Math.max(36, Math.floor(cellWidth / ratio));

    for (let index = 0; index < pageImages.length; index += 1) {
        const image = pageImages[index];
        const tile = document.createElement('div');
        tile.className = 'tile';
        if (state.selected.has(image.path)) {
            tile.classList.add('selected');
        }
        tile.style.aspectRatio = String(ratio);
        tile.title = image.path;

        const badge = document.createElement('div');
        badge.className = 'badge';
        badge.textContent = `#${state.pageStart + index + 1}`;

        const img = document.createElement('img');
        const params = new URLSearchParams({
            path: image.path,
            width: String(cellWidth),
            height: String(cellHeight),
            slice_enabled: String(state.sliceEnabled),
            slice_indices: state.sliceIndices,
            slice_parts: String(state.sliceParts),
            bg_color: state.bgColor,
        });
        img.src = `/api/thumb?${params.toString()}`;
        img.loading = 'lazy';
        img.alt = image.name;

        const cross = document.createElement('div');
        cross.className = 'cross';

        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = image.name;

        tile.appendChild(img);
        tile.appendChild(badge);
        tile.appendChild(cross);
        tile.appendChild(name);

        tile.addEventListener('click', () => {
            if (state.selected.has(image.path)) {
                state.selected.delete(image.path);
            } else {
                state.selected.add(image.path);
            }
            tile.classList.toggle('selected');
            updateMeta(effectiveCols, rows);
        });

        tile.addEventListener('dblclick', () => {
            const url = `/api/image?path=${encodeURIComponent(image.path)}`;
            window.open(url, '_blank');
        });

        gridEl.appendChild(tile);
    }

    updateMeta(effectiveCols, rows);
}

function updateMeta(effectiveCols, rows) {
    const total = state.images.length;
    const pageSize = Math.max(1, effectiveCols * rows);
    const start = total ? state.pageStart + 1 : 0;
    const end = Math.min(total, state.pageStart + pageSize);
    const sliceText = state.sliceEnabled ? `切片 ${state.sliceIndices}/${state.sliceParts}` : '原图';
    metaEl.textContent = `${start}-${end} / ${total} | 列 ${effectiveCols} | 行 ${rows} | ${sliceText} | 已选 ${state.selected.size}`;
}

function prevPage() {
    const step = getPageSize();
    state.pageStart = Math.max(0, state.pageStart - step);
    render();
}

function nextPage() {
    const step = getPageSize();
    if (state.pageStart + step < state.images.length) {
        state.pageStart += step;
        render();
    }
}

function jumpToIndex() {
    const value = Number(document.getElementById('jump-input').value);
    if (!Number.isInteger(value) || value < 1 || value > state.images.length) {
        setStatus(`请输入 1 到 ${state.images.length || 1} 之间的索引`, true);
        return;
    }
    const pageSize = getPageSize();
    state.pageStart = Math.floor((value - 1) / pageSize) * pageSize;
    render();
}

function zoomIn() {
    state.cols = Math.max(1, state.cols - 1);
    render();
}

function zoomOut() {
    state.cols = Math.min(20, state.cols + 1);
    render();
}

async function deleteSelected() {
    if (!state.selected.size) {
        setStatus('还没有选中任何图片', true);
        return;
    }
    const count = state.selected.size;
    if (!window.confirm(`确定删除选中的 ${count} 张图片吗？此操作不可撤销。`)) {
        return;
    }
    try {
        const result = await postJSON('/api/delete', {paths: [...state.selected]});
        const deleted = new Set(result.deleted_paths);
        state.images = state.images.filter((item) => !deleted.has(item.path));
        state.selected.clear();
        const failedCount = result.failed.length;
        render();
        if (failedCount) {
            setStatus(`已删除 ${result.deleted_count} 张，失败 ${failedCount} 张`, true);
        } else {
            setStatus(`已删除 ${result.deleted_count} 张图片`);
        }
    } catch (error) {
        setStatus(error.message, true);
    }
}

function bindInputEvents() {
    document.getElementById('open-btn').addEventListener('click', openDirectory);
    document.getElementById('prev-btn').addEventListener('click', prevPage);
    document.getElementById('next-btn').addEventListener('click', nextPage);
    document.getElementById('jump-btn').addEventListener('click', jumpToIndex);
    document.getElementById('zoom-in-btn').addEventListener('click', zoomIn);
    document.getElementById('zoom-out-btn').addEventListener('click', zoomOut);
    document.getElementById('delete-btn').addEventListener('click', deleteSelected);

    document.getElementById('dir-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            openDirectory();
        }
    });
    document.getElementById('jump-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            jumpToIndex();
        }
    });

    ['slice-enabled', 'slice-input', 'parts-input', 'ratio-input', 'bg-input'].forEach((id) => {
        document.getElementById(id).addEventListener('change', render);
    });

    window.addEventListener('keydown', (event) => {
        const activeTag = document.activeElement ? document.activeElement.tagName : '';
        if (activeTag === 'INPUT' || activeTag === 'TEXTAREA') {
            return;
        }
        if (event.key === 'ArrowLeft') {
            prevPage();
        } else if (event.key === 'ArrowRight') {
            nextPage();
        } else if (event.key === 'ArrowUp') {
            zoomIn();
        } else if (event.key === 'ArrowDown') {
            zoomOut();
        } else if (event.key === 'Delete') {
            deleteSelected();
        }
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => render(), 80);
    });
}

async function init() {
    bindInputEvents();
    const config = await fetch('/api/config').then((response) => response.json());
    document.getElementById('dir-input').value = config.default_dir;
    document.getElementById('slice-enabled').checked = config.default_slice_enabled;
    document.getElementById('slice-input').value = config.default_slice_indices;
    document.getElementById('parts-input').value = String(config.default_slice_parts);
    document.getElementById('ratio-input').value = config.default_ratio;
    document.getElementById('bg-input').value = config.default_bg_color;
    state.folder = config.default_dir;
    state.cols = config.default_cols;
    await openDirectory();
}

init();
</script>
</body>
</html>
"""


def main() -> None:
    host = os.environ.get("XDL_WEB_HOST", "127.0.0.1")
    port = env_int("XDL_BATCH_VIEWER_PORT", 8011)
    no_browser = os.environ.get("XDL_NO_BROWSER", "").strip() == "1"
    url = f"http://{host}:{port}"

    def open_browser() -> None:
        import time

        time.sleep(1.2)
        webbrowser.open(url)

    if not no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"批量图片查看器启动: {url}")
    print("按 Ctrl+C 停止")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
