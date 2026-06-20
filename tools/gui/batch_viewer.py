#!/usr/bin/env python3
"""批量图片查看器 - pywebview + FastAPI 单文件版.

用法:
    python tools/gui/batch_viewer.py

环境变量:
    XDL_BATCH_VIEWER_PORT=8768   端口(默认 8768)
    XDL_BATCH_VIEWER_HOST=127.0.0.1  监听地址(默认 127.0.0.1,不绑 0.0.0.0)

功能:
    - 按目录批量浏览图片(自然排序: a1 < a2 < a10)
    - 支持切片拼接预览(如横向 4 段, 取第 2,3 段拼成缩略图)
    - 支持选择并删除图片(破坏性操作, 前端有 confirm)
    - 分页,缩放,跳转,键盘快捷键
    - 缩略图懒加载, 避免一次拉满内存

与原 FastAPI+HTML 版本的差异:
    - 启动方式从 uvicorn.run 改为 pywebview 原生窗口(无浏览器 tab)
    - 目录输入旁加 "📁 浏览..." 按钮, 走 pywebview.js_api 调系统目录选择器
    - PIL 缩略图/读图改用 asyncio.to_thread, 避免阻塞事件循环
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import uvicorn
import webview
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

HOST = os.environ.get("XDL_BATCH_VIEWER_HOST", "127.0.0.1")
PORT = int(os.environ.get("XDL_BATCH_VIEWER_PORT", "8768") or "8768")
TITLE = "批量图片查看器"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
RESAMPLE = Image.LANCZOS  # pyright: ignore[reportAttributeAccessIssue]  # Pillow stub 移除了旧名
THUMB_CACHE_SIZE = 512


# ============================================================
# 业务函数(纯 Python, 可单测)
# ============================================================


def natural_sort_key(path: str) -> list[Any]:
    filename = Path(path).name.lower()
    return [
        int(chunk) if chunk.isdigit() else chunk
        for chunk in re.split(r"(\d+)", filename)
    ]


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
    for child in sorted(
        directory.iterdir(), key=lambda item: natural_sort_key(str(item))
    ):
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
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


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


def resize_to_fit(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if resized_width == width and resized_height == height:
        return image
    return image.resize((resized_width, resized_height), RESAMPLE)


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
        preview = resize_to_fit(preview, target_width, target_height)
        canvas = Image.new(
            "RGB", (target_width, target_height), parse_hex_color(bg_color)
        )
        offset_x = max(0, (target_width - preview.width) // 2)
        offset_y = max(0, (target_height - preview.height) // 2)
        canvas.paste(preview, (offset_x, offset_y))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


@lru_cache(maxsize=THUMB_CACHE_SIZE)
def render_thumbnail_cached(
    path_text: str,
    mtime_ns: int,
    file_size: int,
    width: int,
    height: int,
    slice_enabled: bool,
    slice_indices_text: str,
    slice_parts: int,
    bg_color: str,
) -> bytes:
    image_path = Path(path_text)
    slice_indices = parse_slice_indices(slice_indices_text, slice_parts)
    return render_thumbnail(
        image_path=image_path,
        width=width,
        height=height,
        slice_enabled=slice_enabled,
        slice_indices=slice_indices,
        slice_parts=slice_parts,
        bg_color=bg_color,
    ).getvalue()


def read_image_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def delete_images(paths: list[str]) -> dict[str, Any]:
    deleted_paths: list[str] = []
    failed: list[dict[str, str]] = []
    for raw_path in paths:
        try:
            path = resolve_file(str(raw_path))
            path.unlink()
            deleted_paths.append(str(path))
        except Exception as exc:  # noqa: BLE001
            failed.append({"path": str(raw_path), "error": str(exc)})
    return {
        "deleted_count": len(deleted_paths),
        "deleted_paths": deleted_paths,
        "failed": failed,
    }


def save_painted_image(image_path: str, base64_data: str) -> dict[str, Any]:
    """将 base64 PNG 数据解码后覆盖写回原文件."""
    import base64

    path = resolve_file(image_path)
    if not path.exists():
        return {"ok": False, "error": f"文件不存在: {path}"}

    # 去掉可能的 data URL 前缀
    payload = base64_data
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        return {"ok": False, "error": f"Base64 解码失败: {exc}"}

    try:
        path.write_bytes(raw)
        return {"ok": True, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "error": f"写入文件失败: {exc}"}


# ============================================================
# FastAPI app + 路由
# ============================================================

app = FastAPI(title=TITLE)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
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
async def api_list_dir(data: dict[str, Any]):
    directory = resolve_directory(str(data.get("path", ".")))
    if not directory.exists():
        return JSONResponse({"error": f"目录不存在: {directory}"}, status_code=404)
    if not directory.is_dir():
        return JSONResponse({"error": f"不是目录: {directory}"}, status_code=400)

    # list_images 可能很慢(千张图目录的 stat 调用), 放线程里不阻塞事件循环
    images = await asyncio.to_thread(list_images, directory)
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

    image_stat = await asyncio.to_thread(image_path.stat)
    normalized_indices = ",".join(
        str(index) for index in parse_slice_indices(slice_indices, slice_parts)
    )
    normalized_bg_color = normalize_hex_color(bg_color)
    try:
        thumbnail_bytes = await asyncio.to_thread(
            render_thumbnail_cached,
            path_text=str(image_path),
            mtime_ns=image_stat.st_mtime_ns,
            file_size=image_stat.st_size,
            width=width,
            height=height,
            slice_enabled=slice_enabled,
            slice_indices_text=normalized_indices,
            slice_parts=slice_parts,
            bg_color=normalized_bg_color,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return Response(content=thumbnail_bytes, media_type="image/png")


@app.get("/api/image", response_model=None)
async def api_image(path: str = Query(...)):
    image_path = resolve_file(path)
    if not image_path.exists() or not image_path.is_file():
        return JSONResponse({"error": "image not found"}, status_code=404)

    try:
        image_bytes = await asyncio.to_thread(read_image_bytes, image_path)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return Response(content=image_bytes, media_type="image/png")


@app.post("/api/delete", response_model=None)
async def api_delete(data: dict[str, Any]):
    raw_paths = data.get("paths")
    if not isinstance(raw_paths, list):
        return JSONResponse({"error": "paths must be a list"}, status_code=400)
    # 删除通常很快, 但批量千张时仍可能阻塞
    result = await asyncio.to_thread(delete_images, [str(p) for p in raw_paths])
    return result


@app.post("/api/save-painted", response_model=None)
async def api_save_painted(data: dict[str, Any]):
    image_path = str(data.get("path", ""))
    base64_data = str(data.get("image", ""))
    if not image_path or not base64_data:
        return JSONResponse({"error": "path 和 image(data URL/base64) 都是必填项"}, status_code=400)
    result = await asyncio.to_thread(save_painted_image, image_path, base64_data)
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error", "unknown")}, status_code=400)
    return result


# ============================================================
# HTML(f-string 内嵌, CSS/JS 主体沿用原版, 加了 📁 浏览按钮)
# ============================================================

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{TITLE}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; }}
body {{
    background: #1e1e1e;
    color: #d0d0d0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
#toolbar {{
    background: #2a2a2a;
    border-bottom: 1px solid #3a3a3a;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}}
.toolbar-sep {{
    width: 1px;
    height: 24px;
    background: #4a4a4a;
    flex-shrink: 0;
}}
.input, .btn, .check {{
    min-height: 34px;
    border-radius: 6px;
    border: 1px solid #4a4a4a;
    background: #343434;
    color: #e0e0e0;
    font-size: 13px;
}}
.input {{
    padding: 0 10px;
    width: 100%;
}}
.btn {{
    padding: 0 12px;
    cursor: pointer;
    transition: background 0.15s ease;
}}
.btn:hover {{ background: #444444; }}
.btn.primary {{ background: #007acc; border-color: #007acc; color: #fff; }}
.btn.primary:hover {{ background: #1390ea; }}
.btn.icon {{ padding: 0 10px; min-width: 34px; justify-content: center; }}
.check {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 10px;
    white-space: nowrap;
}}
.check input {{ accent-color: #007acc; }}
.field {{
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}}
.field label {{
    color: #8e8e8e;
    font-size: 12px;
}}
#content {{
    min-height: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
}}
#statusbar {{
    background: #252525;
    border-bottom: 1px solid #343434;
    padding: 8px 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    white-space: nowrap;
    overflow: hidden;
}}
#status {{ color: #88c0ff; font-size: 13px; }}
#meta {{
    margin-left: auto;
    color: #8a8a8a;
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
}}
#grid-wrap {{
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 10px;
    background: #111111;
}}
#grid {{
    display: grid;
    align-content: start;
}}
.tile {{
    position: relative;
    border: 1px solid #2f2f2f;
    background: #000;
    overflow: hidden;
    cursor: pointer;
    user-select: none;
}}
.tile:hover {{ border-color: #4d4d4d; }}
.tile.selected {{ border-color: #d33f3f; box-shadow: inset 0 0 0 1px #d33f3f; }}
.tile img,
.tile canvas {{
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
}}
.tile .name {{
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
}}
.tile .badge {{
    position: absolute;
    top: 8px;
    left: 8px;
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.72);
    font-size: 11px;
    color: #fff;
}}
.tile .cross {{
    position: absolute;
    inset: 0;
    pointer-events: none;
    display: none;
}}
.tile.selected .cross {{ display: block; }}
.tile .cross::before,
.tile .cross::after {{
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 150%;
    height: 3px;
    background: rgba(255, 64, 64, 0.9);
    transform-origin: center center;
}}
.tile .cross::before {{ transform: translate(-50%, -50%) rotate(30deg); }}
.tile .cross::after {{ transform: translate(-50%, -50%) rotate(-30deg); }}
#empty {{
    display: none;
    height: 100%;
    align-items: center;
    justify-content: center;
    color: #7a7a7a;
    font-size: 14px;
}}
/* ---------- 大图模态框 ---------- */
#viewer-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    flex-direction: column;
}}
#viewer-overlay.open {{ display: flex; }}
#viewer-toolbar {{
    background: #2a2a2a;
    border-bottom: 1px solid #3a3a3a;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}}
#viewer-toolbar .btn.active {{
    background: #007acc;
    border-color: #007acc;
}}
#viewer-canvas-wrap {{
    flex: 1;
    min-height: 0;
    overflow: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}}
#viewer-layer {{
    position: relative;
    display: inline-block;
    line-height: 0;
}}
#viewer-layer img {{
    display: block;
    max-width: 95vw;
    max-height: calc(100vh - 100px);
    object-fit: contain;
}}
#viewer-paint-canvas {{
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
}}
#viewer-paint-canvas.brush-active {{
    pointer-events: auto;
    cursor: none;
}}
#viewer-statusbar {{
    background: #252525;
    border-top: 1px solid #343434;
    padding: 6px 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 12px;
    color: #aaa;
    flex-shrink: 0;
}}
#viewer-color-swatch {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #555;
    display: inline-block;
    vertical-align: middle;
}}
#viewer-brush-indicator {{
    position: fixed;
    pointer-events: none;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.15);
    transform: translate(-50%, -50%);
    display: none;
    z-index: 1001;
}}
</style>
</head>
<body>
<div id="toolbar">
    <input id="dir-input" class="input" placeholder="输入图片目录,例如 ./ 或 /data/images" style="flex:1;min-width:180px;max-width:360px;">
    <button id="browse-btn" class="btn icon" title="系统目录选择">📂</button>
    <button id="open-btn" class="btn primary">打开</button>
    <span class="toolbar-sep"></span>
    <button id="prev-btn" class="btn">◀</button>
    <button id="next-btn" class="btn">▶</button>
    <input id="jump-input" class="input" placeholder="#索引" inputmode="numeric" style="width:58px;">
    <button id="jump-btn" class="btn">跳转</button>
    <span class="toolbar-sep"></span>
    <button id="zoom-in-btn" class="btn">🔍</button>
    <button id="zoom-out-btn" class="btn">🔎</button>
    <span class="toolbar-sep"></span>
    <button id="delete-btn" class="btn" style="color:#ff7b72;">🗑️删除</button>
    <button id="picker-btn" class="btn" style="color:#88c0ff;">🎨取色</button>
    <button id="brush-btn" class="btn" style="color:#88c0ff;">🖌️涂抹</button>
    <button id="fullscreen-btn" class="btn" style="color:#88c0ff;">⛶</button>
    <span class="toolbar-sep"></span>
    <label class="check"><input type="checkbox" id="slice-enabled" checked>切片</label>
    <div class="field"><input id="slice-input" class="input" value="2,3" style="width:52px;"></div>
    <div class="field"><input id="parts-input" class="input" value="4" inputmode="numeric" style="width:40px;"></div>
    <div class="field"><input id="ratio-input" class="input" value="4/1" style="width:48px;"></div>
    <div class="field"><input id="bg-input" class="input" value="#000000" style="width:68px;"></div>
</div>
<div id="content">
    <div id="statusbar">
        <div id="status">就绪</div>
        <div id="meta"></div>
    </div>
    <div id="grid-wrap">
        <div id="empty">当前目录没有可显示的图片.</div>
        <div id="grid"></div>
    </div>
</div>

<div id="viewer-overlay">
    <div id="viewer-toolbar">
        <button id="viewer-picker-btn" class="btn active" style="color:#88c0ff;">🎨取色</button>
        <button id="viewer-brush-btn" class="btn" style="color:#88c0ff;">🖌️涂抹</button>
        <span id="viewer-brush-size-wrap" class="field" style="display:none;">
            <label for="viewer-brush-size">笔刷</label>
            <input id="viewer-brush-size" class="input" value="20" inputmode="numeric" style="width:52px;">px
        </span>
        <span class="toolbar-sep"></span>
        <button id="viewer-undo-btn" class="btn">↩撤销</button>
        <button id="viewer-save-btn" class="btn primary">💾保存</button>
        <span style="flex:1;"></span>
        <button id="viewer-close-btn" class="btn">✕关闭</button>
    </div>
    <div id="viewer-canvas-wrap">
        <div id="viewer-layer">
            <img id="viewer-image" src="">
            <canvas id="viewer-paint-canvas"></canvas>
        </div>
    </div>
    <div id="viewer-statusbar">
        <span>模式: <span id="viewer-mode-text">🎨取色</span></span>
        <span>当前色: <span id="viewer-color-swatch"></span> <span id="viewer-color-text">-</span></span>
        <span>笔刷: ● <span id="viewer-brush-text">-</span> px</span>
        <span style="margin-left:auto;" id="viewer-filename"></span>
    </div>
</div>
<div id="viewer-brush-indicator"></div>

<script>
const state = {{
    folder: '.',
    images: [],
    selected: new Set(),
    prefetchedThumbs: new Map(),
    thumbImageCache: new Map(),
    thumbLoadPromises: new Map(),
    pageStart: 0,
    renderToken: 0,
    cols: 2,
    sliceEnabled: true,
    sliceIndices: '2,3',
    sliceParts: 4,
    ratioText: '4/1',
    bgColor: '#000000',
    padX: 0,
    padY: 0,
}};

const gridWrapEl = document.getElementById('grid-wrap');
const gridEl = document.getElementById('grid');
const emptyEl = document.getElementById('empty');
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');
const PREFETCH_LIMIT = 256;

const viewerState = {{
    isOpen: false,
    imagePath: '',
    mode: 'picker',
    brushSize: 20,
    currentColor: '#ff0000',
    isDrawing: false,
    lastX: 0,
    lastY: 0,
}};

const viewerOverlayEl = document.getElementById('viewer-overlay');
const viewerImageEl = document.getElementById('viewer-image');
const viewerPaintCanvas = document.getElementById('viewer-paint-canvas');
const viewerBrushIndicator = document.getElementById('viewer-brush-indicator');
const viewerPickerBtn = document.getElementById('viewer-picker-btn');
const viewerBrushBtn = document.getElementById('viewer-brush-btn');
const viewerBrushSizeWrap = document.getElementById('viewer-brush-size-wrap');
const viewerBrushSizeInput = document.getElementById('viewer-brush-size');
const viewerModeText = document.getElementById('viewer-mode-text');
const viewerColorSwatch = document.getElementById('viewer-color-swatch');
const viewerColorText = document.getElementById('viewer-color-text');
const viewerBrushText = document.getElementById('viewer-brush-text');
const viewerFilename = document.getElementById('viewer-filename');

function updateViewerStatus() {{
    viewerColorSwatch.style.backgroundColor = viewerState.currentColor;
    viewerColorText.textContent = viewerState.currentColor;
    viewerBrushText.textContent = viewerState.mode === 'brush' ? String(viewerState.brushSize) : '-';
}}

function syncPaintCanvasSize() {{
    const displayW = viewerImageEl.clientWidth;
    const displayH = viewerImageEl.clientHeight;
    if (displayW <= 0 || displayH <= 0) {{ return; }}
    viewerPaintCanvas.width = displayW;
    viewerPaintCanvas.height = displayH;
    viewerPaintCanvas.style.width = displayW + 'px';
    viewerPaintCanvas.style.height = displayH + 'px';
}}

function clearPaintCanvas() {{
    const ctx = viewerPaintCanvas.getContext('2d');
    if (ctx) {{ ctx.clearRect(0, 0, viewerPaintCanvas.width, viewerPaintCanvas.height); }}
}}

function updateBrushIndicator() {{
    viewerBrushIndicator.style.width = viewerState.brushSize + 'px';
    viewerBrushIndicator.style.height = viewerState.brushSize + 'px';
}}

function setViewerMode(mode) {{
    viewerState.mode = mode;
    const isBrush = mode === 'brush';
    viewerPickerBtn.classList.toggle('active', !isBrush);
    viewerBrushBtn.classList.toggle('active', isBrush);
    viewerBrushSizeWrap.style.display = isBrush ? '' : 'none';
    viewerModeText.textContent = isBrush ? '🖌️涂抹' : '🎨取色';
    if (isBrush) {{
        viewerImageEl.style.cursor = 'none';
        viewerPaintCanvas.style.cursor = 'none';
        syncPaintCanvasSize();
        viewerBrushIndicator.style.display = '';
        updateBrushIndicator();
    }} else {{
        viewerImageEl.style.cursor = 'crosshair';
        viewerBrushIndicator.style.display = 'none';
    }}
    updateViewerStatus();
}}

async function openViewer(path) {{
    viewerState.isOpen = true;
    viewerState.imagePath = path;
    viewerFilename.textContent = path.split(/[\\\\/]/).pop() || path;
    const url = `/api/image?path=${{encodeURIComponent(path)}}`;
    viewerImageEl.src = url;
    viewerOverlayEl.classList.add('open');
    await new Promise((resolve) => {{
        if (viewerImageEl.complete) {{ resolve(); return; }}
        viewerImageEl.onload = resolve;
        viewerImageEl.onerror = resolve;
    }});
    // 等待浏览器完成 layout 计算, 否则 clientWidth/clientHeight 可能为 0
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    syncPaintCanvasSize();
    clearPaintCanvas();
    setViewerMode('picker');
}}

function closeViewer() {{
    viewerState.isOpen = false;
    viewerOverlayEl.classList.remove('open');
    viewerBrushIndicator.style.display = 'none';
    viewerImageEl.src = '';
    clearPaintCanvas();
}}

function setStatus(text, isError = false) {{
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#ff7b72' : '#88c0ff';
}}

function parseRatio(value) {{
    const raw = String(value || '').trim();
    if (raw.includes('/')) {{
        const parts = raw.split('/');
        const left = Number(parts[0]);
        const right = Number(parts[1]);
        if (left > 0 && right > 0) {{
            return left / right;
        }}
    }}
    const number = Number(raw);
    return number > 0 ? number : 4;
}}

function normalizeSliceIndices() {{
    const sliceParts = Math.max(1, Number(document.getElementById('parts-input').value) || 4);
    const raw = document.getElementById('slice-input').value;
    const values = raw.split(',').map((item) => Number(item.trim())).filter((item) => Number.isInteger(item) && item >= 1 && item <= sliceParts);
    const unique = [...new Set(values)];
    if (!unique.length) {{
        unique.push(1);
    }}
    state.sliceParts = sliceParts;
    state.sliceIndices = unique.join(',');
    document.getElementById('slice-input').value = state.sliceIndices;
    document.getElementById('parts-input').value = String(state.sliceParts);
    return unique;
}}

function quantizeThumbSize(value) {{
    return Math.max(1, Math.round(value / 32) * 32);
}}

function buildThumbUrl(imagePath, width, height) {{
    const params = new URLSearchParams({{
        path: imagePath,
        width: String(width),
        height: String(height),
        slice_enabled: String(state.sliceEnabled),
        slice_indices: state.sliceIndices,
        slice_parts: String(state.sliceParts),
        bg_color: state.bgColor,
    }});
    return `/api/thumb?${{params.toString()}}`;
}}

function rememberPrefetchedThumb(url) {{
    if (state.prefetchedThumbs.has(url)) {{
        state.prefetchedThumbs.delete(url);
    }}
    state.prefetchedThumbs.set(url, Date.now());
    while (state.prefetchedThumbs.size > PREFETCH_LIMIT) {{
        const oldestKey = state.prefetchedThumbs.keys().next().value;
        if (!oldestKey) {{
            break;
        }}
        state.prefetchedThumbs.delete(oldestKey);
        state.thumbImageCache.delete(oldestKey);
        state.thumbLoadPromises.delete(oldestKey);
    }}
}}

function ensureThumbReady(url) {{
    if (state.thumbLoadPromises.has(url)) {{
        return state.thumbLoadPromises.get(url);
    }}
    const promise = new Promise((resolve) => {{
        const img = new Image();
        let settled = false;
        img.decoding = 'async';
        img.loading = 'eager';
        const settle = (value) => {{
            if (settled) {{
                return;
            }}
            settled = true;
            resolve(value);
        }};
        const finish = () => {{
            if (typeof img.decode === 'function') {{
                img.decode().catch(() => null).finally(() => {{
                    state.thumbImageCache.set(url, img);
                    settle(img);
                }});
                return;
            }}
            state.thumbImageCache.set(url, img);
            settle(img);
        }};
        img.onload = finish;
        img.onerror = () => settle(null);
        img.src = url;
        if (img.complete && img.naturalWidth > 0) {{
            finish();
        }} else if (img.complete) {{
            settle(null);
        }}
    }});
    state.thumbLoadPromises.set(url, promise);
    rememberPrefetchedThumb(url);
    return promise;
}}

function prefetchThumb(url) {{
    void ensureThumbReady(url);
}}

function prefetchPage(pageStart, pageSize, thumbWidth, thumbHeight) {{
    if (pageStart < 0 || pageStart >= state.images.length || pageSize <= 0) {{
        return;
    }}
    const pageImages = state.images.slice(pageStart, pageStart + pageSize);
    for (const image of pageImages) {{
        prefetchThumb(buildThumbUrl(image.path, thumbWidth, thumbHeight));
    }}
}}

function getEffectiveCols() {{
    const indices = normalizeSliceIndices();
    if (!state.sliceEnabled) {{
        return Math.max(1, state.cols);
    }}
    return Math.max(1, Math.floor((state.cols * state.sliceParts) / Math.max(1, indices.length)));
}}

function estimateRows() {{
    const effectiveCols = getEffectiveCols();
    const ratio = parseRatio(state.ratioText);
    const availableWidth = Math.max(320, gridWrapEl.clientWidth - 20);
    const availableHeight = Math.max(220, gridWrapEl.clientHeight - 20);
    const cellWidth = Math.max(60, (availableWidth - (effectiveCols - 1) * state.padX) / effectiveCols);
    const cellHeight = Math.max(36, cellWidth / ratio);
    return Math.max(1, Math.floor((availableHeight + state.padY) / (cellHeight + state.padY)));
}}

function getPageSize() {{
    return getEffectiveCols() * estimateRows();
}}

function syncStateFromInputs() {{
    state.folder = document.getElementById('dir-input').value.trim() || '.';
    state.sliceEnabled = document.getElementById('slice-enabled').checked;
    state.ratioText = document.getElementById('ratio-input').value.trim() || '4/1';
    state.bgColor = document.getElementById('bg-input').value.trim() || '#000000';
}}

async function postJSON(url, payload) {{
    const response = await fetch(url, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload),
    }});
    const data = await response.json();
    if (!response.ok) {{
        throw new Error(data.error || `请求失败: ${{response.status}}`);
    }}
    return data;
}}

async function openDirectory() {{
    syncStateFromInputs();
    setStatus(`正在读取目录: ${{state.folder}}`);
    try {{
        const data = await postJSON('/api/list-dir', {{path: state.folder}});
        state.folder = data.directory;
        state.images = data.images;
        state.selected.clear();
        state.pageStart = 0;
        document.getElementById('dir-input').value = data.directory;
        render();
        if (state.images.length) {{
            setStatus(`已加载 ${{state.images.length}} 张图片`);
        }} else {{
            setStatus('目录中没有找到图片');
        }}
    }} catch (error) {{
        setStatus(error.message, true);
    }}
}}

async function pickDirectory() {{
    // pywebview 6.x: window.create_file_dialog 返回 tuple; 不存在时直接降级手输
    if (typeof pywebview === 'undefined' || !pywebview.api || !pywebview.api.pick_directory) {{
        setStatus('系统目录选择不可用, 请直接输入路径', true);
        return;
    }}
    try {{
        const path = await pywebview.api.pick_directory();
        if (path) {{
            document.getElementById('dir-input').value = path;
            await openDirectory();
        }}
    }} catch (error) {{
        setStatus('选择目录失败: ' + error.message, true);
    }}
}}

async function render() {{
    const renderToken = state.renderToken + 1;
    state.renderToken = renderToken;
    syncStateFromInputs();
    normalizeSliceIndices();

    const ratio = parseRatio(state.ratioText);
    const effectiveCols = getEffectiveCols();
    const rows = estimateRows();
    const pageSize = effectiveCols * rows;
    const total = state.images.length;
    if (state.pageStart >= total && total > 0) {{
        state.pageStart = Math.max(0, Math.floor((total - 1) / pageSize) * pageSize);
    }}
    const pageImages = state.images.slice(state.pageStart, state.pageStart + pageSize);
    const availableWidth = Math.max(320, gridWrapEl.clientWidth - 20);
    const cellWidth = Math.max(60, Math.floor((availableWidth - (effectiveCols - 1) * state.padX) / effectiveCols));
    const cellHeight = Math.max(36, Math.floor(cellWidth / ratio));
    const thumbWidth = quantizeThumbSize(cellWidth);
    const thumbHeight = quantizeThumbSize(cellHeight);
    const pageThumbs = pageImages.map((image) => ({{
        image,
        url: buildThumbUrl(image.path, thumbWidth, thumbHeight),
    }}));

    if (!pageThumbs.length) {{
        emptyEl.style.display = 'flex';
        gridEl.innerHTML = '';
        updateMeta(effectiveCols, rows);
        return;
    }}

    const loadedImages = await Promise.all(
        pageThumbs.map((item) => ensureThumbReady(item.url))
    );
    if (renderToken !== state.renderToken) {{
        return;
    }}

    emptyEl.style.display = 'none';
    gridEl.innerHTML = '';
    gridEl.style.gridTemplateColumns = `repeat(${{effectiveCols}}, minmax(0, 1fr))`;
    gridEl.style.gap = `${{state.padY}}px ${{state.padX}}px`;
    const fragment = document.createDocumentFragment();

    for (let index = 0; index < pageThumbs.length; index += 1) {{
        const {{ image, url }} = pageThumbs[index];
        const tile = document.createElement('div');
        tile.className = 'tile';
        if (state.selected.has(image.path)) {{
            tile.classList.add('selected');
        }}
        tile.style.aspectRatio = String(ratio);
        tile.title = image.path;

        const badge = document.createElement('div');
        badge.className = 'badge';
        badge.textContent = `#${{state.pageStart + index + 1}}`;

        const canvas = document.createElement('canvas');
        canvas.width = thumbWidth;
        canvas.height = thumbHeight;
        canvas.title = image.name;
        const context = canvas.getContext('2d');
        const loadedImage = loadedImages[index] || state.thumbImageCache.get(url);
        if (context && loadedImage) {{
            context.drawImage(loadedImage, 0, 0, thumbWidth, thumbHeight);
        }}

        const cross = document.createElement('div');
        cross.className = 'cross';

        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = image.name;

        tile.appendChild(canvas);
        tile.appendChild(badge);
        tile.appendChild(cross);
        tile.appendChild(name);

        tile.addEventListener('click', () => {{
            if (state.selected.has(image.path)) {{
                state.selected.delete(image.path);
            }} else {{
                state.selected.add(image.path);
            }}
            tile.classList.toggle('selected');
            updateMeta(effectiveCols, rows);
        }});

        tile.addEventListener('dblclick', () => {{
            openViewer(image.path).then(() => setViewerMode('picker'));
        }});

        fragment.appendChild(tile);
    }}

    gridEl.appendChild(fragment);
    prefetchPage(state.pageStart + pageSize, pageSize, thumbWidth, thumbHeight);
    prefetchPage(state.pageStart - pageSize, pageSize, thumbWidth, thumbHeight);
    updateMeta(effectiveCols, rows);
}}

function updateMeta(effectiveCols, rows) {{
    const total = state.images.length;
    const pageSize = Math.max(1, effectiveCols * rows);
    const start = total ? state.pageStart + 1 : 0;
    const end = Math.min(total, state.pageStart + pageSize);
    const sliceText = state.sliceEnabled ? `切片 ${{state.sliceIndices}}/${{state.sliceParts}}` : '原图';
    metaEl.textContent = `${{start}}-${{end}} / ${{total}} | 列 ${{effectiveCols}} | 行 ${{rows}} | ${{sliceText}} | 已选 ${{state.selected.size}}`;
}}

function prevPage() {{
    const step = getPageSize();
    state.pageStart = Math.max(0, state.pageStart - step);
    render();
}}

function nextPage() {{
    const step = getPageSize();
    if (state.pageStart + step < state.images.length) {{
        state.pageStart += step;
        render();
    }}
}}

function jumpToIndex() {{
    const value = Number(document.getElementById('jump-input').value);
    if (!Number.isInteger(value) || value < 1 || value > state.images.length) {{
        setStatus(`请输入 1 到 ${{state.images.length || 1}} 之间的索引`, true);
        return;
    }}
    const pageSize = getPageSize();
    state.pageStart = Math.floor((value - 1) / pageSize) * pageSize;
    render();
}}

function zoomIn() {{
    state.cols = Math.max(1, state.cols - 1);
    render();
}}

function zoomOut() {{
    state.cols = Math.min(20, state.cols + 1);
    render();
}}

async function deleteSelected() {{
    if (!state.selected.size) {{
        setStatus('还没有选中任何图片', true);
        return;
    }}
    const count = state.selected.size;
    if (!window.confirm(`确定删除选中的 ${{count}} 张图片吗?此操作不可撤销.`)) {{
        return;
    }}
    try {{
        const result = await postJSON('/api/delete', {{paths: [...state.selected]}});
        const deleted = new Set(result.deleted_paths);
        state.images = state.images.filter((item) => !deleted.has(item.path));
        state.selected.clear();
        const failedCount = result.failed.length;
        render();
        if (failedCount) {{
            setStatus(`已删除 ${{result.deleted_count}} 张,失败 ${{failedCount}} 张`, true);
        }} else {{
            setStatus(`已删除 ${{result.deleted_count}} 张图片`);
        }}
    }} catch (error) {{
        setStatus(error.message, true);
    }}
}}

function pickColor(clientX, clientY) {{
    const imgRect = viewerImageEl.getBoundingClientRect();
    const naturalW = viewerImageEl.naturalWidth;
    const naturalH = viewerImageEl.naturalHeight;
    if (!naturalW || !naturalH) {{ return; }}
    const scaleX = naturalW / imgRect.width;
    const scaleY = naturalH / imgRect.height;
    const px = Math.round((clientX - imgRect.left) * scaleX);
    const py = Math.round((clientY - imgRect.top) * scaleY);
    if (px < 0 || py < 0 || px >= naturalW || py >= naturalH) {{ return; }}
    const offscreen = document.createElement('canvas');
    offscreen.width = naturalW;
    offscreen.height = naturalH;
    const ctx = offscreen.getContext('2d');
    ctx.drawImage(viewerImageEl, 0, 0);
    const [r, g, b] = ctx.getImageData(px, py, 1, 1).data;
    viewerState.currentColor = '#' + r.toString(16).padStart(2, '0') + g.toString(16).padStart(2, '0') + b.toString(16).padStart(2, '0');
    updateViewerStatus();
}}

function drawDot(canvasX, canvasY) {{
    const ctx = viewerPaintCanvas.getContext('2d');
    const r = viewerState.brushSize / 2;
    ctx.fillStyle = viewerState.currentColor;
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, r, 0, Math.PI * 2);
    ctx.fill();
}}

function drawLine(fromX, fromY, toX, toY) {{
    const ctx = viewerPaintCanvas.getContext('2d');
    ctx.strokeStyle = viewerState.currentColor;
    ctx.lineWidth = viewerState.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
}}

async function savePainted() {{
    const mergeCanvas = document.createElement('canvas');
    mergeCanvas.width = viewerImageEl.naturalWidth;
    mergeCanvas.height = viewerImageEl.naturalHeight;
    const ctx = mergeCanvas.getContext('2d');
    ctx.drawImage(viewerImageEl, 0, 0);
    const paintW = viewerPaintCanvas.width;
    const paintH = viewerPaintCanvas.height;
    if (paintW > 0 && paintH > 0) {{
        const scaleX = mergeCanvas.width / paintW;
        const scaleY = mergeCanvas.height / paintH;
        ctx.save();
        ctx.scale(scaleX, scaleY);
        ctx.drawImage(viewerPaintCanvas, 0, 0);
        ctx.restore();
    }}
    const base64 = mergeCanvas.toDataURL('image/png');
    try {{
        const result = await postJSON('/api/save-painted', {{ path: viewerState.imagePath, image: base64 }});
        if (result.ok) {{
            setStatus('已保存: ' + viewerFilename.textContent);
            closeViewer();
            state.thumbImageCache.clear();
            state.thumbLoadPromises.clear();
            render();
        }} else {{
            setStatus('保存失败: ' + (result.error || 'unknown'), true);
        }}
    }} catch (error) {{
        setStatus('保存失败: ' + error.message, true);
    }}
}}

function bindInputEvents() {{
    document.getElementById('open-btn').addEventListener('click', openDirectory);
    document.getElementById('browse-btn').addEventListener('click', pickDirectory);
    document.getElementById('prev-btn').addEventListener('click', prevPage);
    document.getElementById('next-btn').addEventListener('click', nextPage);
    document.getElementById('jump-btn').addEventListener('click', jumpToIndex);
    document.getElementById('zoom-in-btn').addEventListener('click', zoomIn);
    document.getElementById('zoom-out-btn').addEventListener('click', zoomOut);
    document.getElementById('delete-btn').addEventListener('click', deleteSelected);

    document.getElementById('dir-input').addEventListener('keydown', (event) => {{
        if (event.key === 'Enter') {{
            openDirectory();
        }}
    }});
    document.getElementById('jump-input').addEventListener('keydown', (event) => {{
        if (event.key === 'Enter') {{
            jumpToIndex();
        }}
    }});

    ['slice-enabled', 'slice-input', 'parts-input', 'ratio-input', 'bg-input'].forEach((id) => {{
        document.getElementById(id).addEventListener('change', render);
    }});

    // ---- viewer modal buttons ----
    viewerPickerBtn.addEventListener('click', () => setViewerMode('picker'));
    viewerBrushBtn.addEventListener('click', () => setViewerMode('brush'));
    document.getElementById('viewer-close-btn').addEventListener('click', closeViewer);
    document.getElementById('viewer-undo-btn').addEventListener('click', clearPaintCanvas);
    document.getElementById('viewer-save-btn').addEventListener('click', savePainted);

    viewerOverlayEl.addEventListener('click', (event) => {{
        if (event.target === viewerOverlayEl) {{ closeViewer(); }}
    }});
    document.addEventListener('keydown', (event) => {{
        if (event.key === 'Escape' && viewerState.isOpen) {{
            closeViewer();
        }}
    }});

    viewerBrushSizeInput.addEventListener('input', () => {{
        const v = Math.max(2, Math.min(200, Number(viewerBrushSizeInput.value) || 20));
        viewerState.brushSize = v;
        updateViewerStatus();
        updateBrushIndicator();
    }});

    // ---- 取色: click on viewer image ----
    viewerImageEl.addEventListener('click', (event) => {{
        if (viewerState.mode !== 'picker') {{ return; }}
        pickColor(event.clientX, event.clientY);
    }});

    // ---- 涂抹 mouse events (绑定在 overlay 上, 不依赖 paint canvas 的 pointer-events) ----
    viewerOverlayEl.addEventListener('mousedown', (event) => {{
        if (viewerState.mode !== 'brush') {{ return; }}
        // 不在 toolbar/statusbar 上触发绘制
        if (event.target.closest('#viewer-toolbar') || event.target.closest('#viewer-statusbar')) {{ return; }}
        // 防御: 若 canvas 尺寸仍为 0 则重新同步
        if (viewerPaintCanvas.width <= 0 || viewerPaintCanvas.height <= 0) {{
            syncPaintCanvasSize();
        }}
        if (viewerPaintCanvas.width <= 0 || viewerPaintCanvas.height <= 0) {{ return; }}
        viewerState.isDrawing = true;
        const rect = viewerPaintCanvas.getBoundingClientRect();
        viewerState.lastX = event.clientX - rect.left;
        viewerState.lastY = event.clientY - rect.top;
        drawDot(viewerState.lastX, viewerState.lastY);
    }});

    viewerOverlayEl.addEventListener('mousemove', (event) => {{
        if (viewerState.mode === 'brush') {{
            viewerBrushIndicator.style.left = event.clientX + 'px';
            viewerBrushIndicator.style.top = event.clientY + 'px';
        }}
        if (!viewerState.isDrawing) {{ return; }}
        const rect = viewerPaintCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        drawLine(viewerState.lastX, viewerState.lastY, x, y);
        viewerState.lastX = x;
        viewerState.lastY = y;
    }});

    // mouseup 绑定在 document 上, 确保拖拽到 overlay 外也能结束绘制
    document.addEventListener('mouseup', () => {{ viewerState.isDrawing = false; }});

    // ---- 滚轮调笔刷大小 ----
    viewerOverlayEl.addEventListener('wheel', (event) => {{
        if (viewerState.mode !== 'brush') {{ return; }}
        event.preventDefault();
        const step = event.deltaY > 0 ? -2 : 2;
        viewerState.brushSize = Math.max(2, Math.min(200, viewerState.brushSize + step));
        viewerBrushSizeInput.value = String(viewerState.brushSize);
        updateViewerStatus();
        updateBrushIndicator();
    }}, {{ passive: false }});

    // ---- 全屏 ----
    document.getElementById('fullscreen-btn').addEventListener('click', () => {{
        if (document.fullscreenElement) {{
            document.exitFullscreen();
        }} else {{
            document.documentElement.requestFullscreen();
        }}
    }});

    // ---- 工具栏取色/涂抹按钮: 打开最近图片(有选中则第一个选中, 否则第一张) ----
    document.getElementById('picker-btn').addEventListener('click', () => {{
        if (!state.images.length) {{ setStatus('没有图片可查看', true); return; }}
        const idx = state.selected.size ? [...state.selected][0] : state.images[0].path;
        openViewer(idx).then(() => setViewerMode('picker'));
    }});

    document.getElementById('brush-btn').addEventListener('click', () => {{
        if (!state.images.length) {{ setStatus('没有图片可查看', true); return; }}
        const idx = state.selected.size ? [...state.selected][0] : state.images[0].path;
        openViewer(idx).then(() => setViewerMode('brush'));
    }});

    window.addEventListener('keydown', (event) => {{
        const activeTag = document.activeElement ? document.activeElement.tagName : '';
        if (activeTag === 'INPUT' || activeTag === 'TEXTAREA') {{
            return;
        }}
        if (event.key === 'ArrowLeft') {{
            prevPage();
        }} else if (event.key === 'ArrowRight') {{
            nextPage();
        }} else if (event.key === 'ArrowUp') {{
            zoomIn();
        }} else if (event.key === 'ArrowDown') {{
            zoomOut();
        }} else if (event.key === 'Delete') {{
            deleteSelected();
        }}
    }});

    let resizeTimer = null;
    window.addEventListener('resize', () => {{
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => render(), 80);
    }});
}}

async function init() {{
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
}}

init();
</script>
</body>
</html>
"""


# ============================================================
# pywebview API class
# ============================================================


class API:
    """暴露给前端的 pywebview API(系统级操作)"""

    def __init__(self) -> None:
        self._window: webview.Window | None = None

    def bind_window(self, window: webview.Window) -> None:
        self._window = window

    def pick_directory(self) -> str | None:
        """系统目录选择对话框, 走 pywebview.create_file_dialog(FOLDER_DIALOG).
        pywebview 6.x 返回 tuple[str, ...], 取第一个; 用户取消时为空 tuple.
        """
        if self._window is None:
            return None
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)  # pyright: ignore[reportArgumentType]
        if not result:
            return None
        return result[0] if isinstance(result, (tuple, list)) else str(result)

    def close_window(self) -> None:
        if self._window is not None:
            self._window.destroy()


# ============================================================
# 入口
# ============================================================


def main() -> None:
    url = f"http://{HOST}:{PORT}"

    # 1) 启动 FastAPI(后台线程)
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    threading.Thread(target=_run_server, daemon=True).start()

    # 2) 等服务器起来
    time.sleep(1.0)

    # 3) 打开 pywebview 窗口
    print(f"{TITLE} 启动: {url}")
    print("按 Ctrl+C 或关闭窗口停止")
    api = API()
    window = webview.create_window(
        title=TITLE,
        url=url,
        width=1400,
        height=900,
        min_size=(960, 600),
        js_api=api,
        text_select=True,
    )
    assert window is not None  # stub 标注可能为 None
    api.bind_window(window)
    webview.start()


if __name__ == "__main__":
    main()
