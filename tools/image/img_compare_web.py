"""图片像素级对比器 — FastAPI + HTML 版本。

用法:
  python img_compare_web.py
  python img_compare_web.py --port 8080

功能:
  - 双图对比模式：加载两张图，拖拽分界线对比
  - 拼合图拆分模式：加载拼合图，切分为 N 块，选择任意两块对比
  - 所有操作通过键盘快捷键或鼠标完成

操作:
  鼠标拖拽分界线   — 移动分界线
  滚轮             — 微调分界线位置
  ← ↑ → ↓ 方向键  — 微调分界线 (1px)
  Shift+方向键     — 快速移动分界线 (10px)
  H                — 切换横线 / 竖线
  F                — 切换适应窗口 / 原始大小
  S                — 保存当前对比截图
  R                — 交换左右/上下图片
  O / P            — 打开图片1 / 图片2 (双图模式)
  M                — 切换双图模式 / 拼合图模式
  1~9 数字键       — 拼合图模式: 快速选择对比块
  Esc / Q          — 退出
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import webbrowser
from typing import Any

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="图片对比器")

# 内存存储
images: dict[str, dict[str, Any]] = {}
slices: dict[str, list[dict[str, Any]]] = {}
image_counter = 0


# ════════════════════════════════════════════════════════════════
#  API 端点
# ════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片，返回图片 ID 和信息"""
    global image_counter
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    image_counter += 1
    img_id = f"img_{image_counter}"
    images[img_id] = {
        "id": img_id,
        "name": file.filename or "unknown",
        "width": img.width,
        "height": img.height,
        "image": img,
    }
    return {"id": img_id, "name": file.filename, "width": img.width, "height": img.height}


@app.get("/api/image/{img_id}")
async def get_image(img_id: str):
    """获取图片 bytes"""
    if img_id not in images:
        return JSONResponse({"error": "not found"}, status_code=404)
    img = images[img_id]["image"]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/split")
async def split_image(data: dict):
    """切分拼合图为 N 块"""
    img_id = data.get("image_id")
    cols = data.get("cols", 5)
    if img_id not in images:
        return JSONResponse({"error": "image not found"}, status_code=404)

    img = images[img_id]["image"]
    cols = max(2, int(cols))
    slice_w = img.width // cols

    slice_list = []
    for i in range(cols):
        x0 = i * slice_w
        x1 = x0 + slice_w if i < cols - 1 else img.width
        cropped = img.crop((x0, 0, x1, img.height))

        slice_id = f"{img_id}_slice_{i}"
        images[slice_id] = {
            "id": slice_id,
            "name": f"{images[img_id]['name']}#{i + 1}",
            "width": cropped.width,
            "height": cropped.height,
            "image": cropped,
        }
        slice_list.append({
            "id": slice_id,
            "index": i,
            "width": cropped.width,
            "height": cropped.height,
        })

    slices[img_id] = slice_list
    return {"slices": slice_list, "cols": cols}


@app.post("/api/screenshot")
async def save_screenshot(data: dict):
    """生成对比截图"""
    left_id = data.get("left_id")
    right_id = data.get("right_id")
    divider_x = data.get("divider_x", 0.5)
    horizontal = data.get("horizontal", False)

    img_left = images.get(left_id, {}).get("image") if left_id else None
    img_right = images.get(right_id, {}).get("image") if right_id else None

    if img_left is None and img_right is None:
        return JSONResponse({"error": "no images"}, status_code=400)

    # 使用右边图的尺寸作为基准
    ref = img_right or img_left
    target_w, target_h = ref.width, ref.height

    if img_right:
        result = img_right.copy()
    else:
        result = Image.new("RGB", (target_w, target_h), (0, 0, 0))

    draw = __import__("PIL").ImageDraw.Draw(result)

    if horizontal:
        div = int(target_h * divider_x)
        if img_left and div > 0:
            left_resized = _resize(img_left, target_w, target_h)
            result.paste(left_resized.crop((0, 0, target_w, div)), (0, 0))
        draw.line([(0, div), (target_w, div)], fill="#ffffff", width=1)
    else:
        div = int(target_w * divider_x)
        if img_left and div > 0:
            left_resized = _resize(img_left, target_w, target_h)
            result.paste(left_resized.crop((0, 0, div, target_h)), (0, 0))
        draw.line([(div, 0), (div, target_h)], fill="#ffffff", width=1)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=screenshot.png"},
    )


def _resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    ratio = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ════════════════════════════════════════════════════════════════
#  HTML / CSS / JS (内联)
# ════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图片对比器</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #1e1e1e;
    color: #cccccc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ── 工具栏 ── */
#toolbar {
    background: #2d2d2d;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 1px solid #3d3d3d;
    flex-shrink: 0;
}
.btn {
    background: #3d3d3d;
    color: #cccccc;
    border: none;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.15s;
}
.btn:hover { background: #505050; color: #fff; }
.btn.active { background: #007acc; color: #fff; }

#grid-controls {
    display: none;
    align-items: center;
    gap: 4px;
    margin-right: 8px;
}
#grid-controls.visible { display: flex; }
#grid-controls label { color: #888; font-size: 12px; }
#cols-input {
    width: 40px;
    background: #3d3d3d;
    color: #cccccc;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 13px;
    text-align: center;
}

#info {
    margin-left: auto;
    color: #888;
    font-size: 12px;
    white-space: nowrap;
}

/* ── 画布区域 ── */
#canvas-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
}
canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: sb-h-double-arrow;
}

/* ── 底部块选择面板 ── */
#block-panel {
    display: none;
    background: #252525;
    padding: 6px 12px;
    border-top: 1px solid #3d3d3d;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
}
#block-panel.visible { display: flex; }
#block-panel label { color: #888; font-size: 12px; margin-right: 6px; }
.block-btn {
    background: #3d3d3d;
    color: #cccccc;
    border: none;
    padding: 4px 12px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 13px;
    min-width: 32px;
}
.block-btn:hover { background: #505050; }
.block-btn.left { background: #007acc; color: #fff; }
.block-btn.right { background: #cc5500; color: #fff; }

/* ── 文件输入隐藏 ── */
#file-input { display: none; }

/* ── 提示遮罩 ── */
#hint {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #666;
    font-size: 18px;
    font-weight: bold;
    pointer-events: none;
    text-align: center;
    line-height: 1.6;
}
</style>
</head>
<body>

<!-- 工具栏 -->
<div id="toolbar">
    <button class="btn" id="mode-btn" onclick="toggleMode()">拼合图模式 (M)</button>
    <div id="grid-controls">
        <label>切分:</label>
        <input type="number" id="cols-input" value="5" min="2" max="100">
        <button class="btn" onclick="applyGridSplit()">切分</button>
    </div>
    <button class="btn" onclick="openImage(1)">打开图1 (O)</button>
    <button class="btn" onclick="openImage(2)">打开图2 (P)</button>
    <button class="btn" onclick="toggleFit()">适应/原始 (F)</button>
    <button class="btn" onclick="swapImages()">交换 (R)</button>
    <button class="btn" onclick="saveScreenshot()">截图保存 (S)</button>
    <button class="btn" id="orient-btn" onclick="toggleOrientation()">竖线 ▯ (H)</button>
    <span id="info"></span>
</div>

<!-- 画布 -->
<div id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="hint">按 O / P 打开图片</div>
</div>

<!-- 底部块选择面板 -->
<div id="block-panel">
    <label>点击选择对比块:</label>
    <div id="block-buttons"></div>
</div>

<!-- 隐藏文件输入 -->
<input type="file" id="file-input" accept="image/*">

<script>
// ════════════════════════════════════════════════════════════════
//  状态
// ════════════════════════════════════════════════════════════════
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const hint = document.getElementById('hint');
const infoEl = document.getElementById('info');
const blockPanel = document.getElementById('block-panel');
const blockBtns = document.getElementById('block-buttons');
const gridControls = document.getElementById('grid-controls');
const modeBtn = document.getElementById('mode-btn');
const orientBtn = document.getElementById('orient-btn');
const colsInput = document.getElementById('cols-input');
const fileInput = document.getElementById('file-input');

let mode = 'single';  // 'single' | 'grid'
let fitToWindow = true;
let horizontal = false;
let dividerX = 0.5;
let dragging = false;

// 双图模式
let img1 = null;  // {id, name, width, height}
let img2 = null;

// 拼合图模式
let sourceImg = null;
let currentSlices = [];  // [{id, index, width, height}]
let idxLeft = 0;
let idxRight = 1;

// ════════════════════════════════════════════════════════════════
//  初始化
// ════════════════════════════════════════════════════════════════
function init() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    setupEvents();
    updateInfo();
}

function resizeCanvas() {
    const wrap = document.getElementById('canvas-wrap');
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    draw();
}

// ════════════════════════════════════════════════════════════════
//  事件绑定
// ════════════════════════════════════════════════════════════════
function setupEvents() {
    // 鼠标拖拽
    canvas.addEventListener('mousedown', e => { dragging = true; });
    canvas.addEventListener('mousemove', e => {
        if (!dragging) return;
        const pos = horizontal ? e.offsetY : e.offsetX;
        const dim = horizontal ? canvas.height : canvas.width;
        dividerX = Math.max(0, Math.min(1, pos / dim));
        draw();
    });
    canvas.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('mouseleave', () => { dragging = false; });

    // 滚轮
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        nudgeDivider(e.deltaY > 0 ? -1 : 1);
    });

    // 键盘
    document.addEventListener('keydown', handleKey);

    // 文件选择
    fileInput.addEventListener('change', handleFileSelect);
}

let pendingSide = 1;
function openImage(side) {
    pendingSide = side;
    fileInput.click();
}

async function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    fileInput.value = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) { alert(data.error); return; }

        if (mode === 'grid') {
            sourceImg = data;
            await applyGridSplit();
        } else {
            if (pendingSide === 1) img1 = data;
            else img2 = data;
        }
        hint.style.display = 'none';
        draw();
        updateInfo();
    } catch (err) {
        alert('上传失败: ' + err.message);
    }
}

// ════════════════════════════════════════════════════════════════
//  键盘处理
// ════════════════════════════════════════════════════════════════
function handleKey(e) {
    const key = e.key.toLowerCase();

    if (key === 'escape' || key === 'q') { window.close(); return; }
    if (key === 'o') { openImage(1); return; }
    if (key === 'p') { openImage(2); return; }
    if (key === 'f') { toggleFit(); return; }
    if (key === 's') { saveScreenshot(); return; }
    if (key === 'r') { swapImages(); return; }
    if (key === 'h') { toggleOrientation(); return; }
    if (key === 'm') { toggleMode(); return; }

    // 数字键
    if (/^[0-9]$/.test(e.key)) {
        quickSelect(parseInt(e.key));
        return;
    }

    // 方向键
    if (['arrowleft', 'arrowright', 'arrowup', 'arrowdown'].includes(key)) {
        e.preventDefault();
        const isH = key === 'arrowup' || key === 'arrowdown';
        const dir = (key === 'arrowright' || key === 'arrowdown') ? 1 : -1;
        const step = e.shiftKey ? 10 : 1;
        if (isH === horizontal) nudgeDivider(dir * step);
    }
}

function nudgeDivider(px) {
    const dim = horizontal ? canvas.height : canvas.width;
    if (dim > 0) {
        dividerX += px / dim;
        dividerX = Math.max(0, Math.min(1, dividerX));
        draw();
    }
}

// ════════════════════════════════════════════════════════════════
//  功能
// ════════════════════════════════════════════════════════════════
function toggleMode() {
    mode = mode === 'single' ? 'grid' : 'single';
    modeBtn.textContent = mode === 'single' ? '拼合图模式 (M)' : '双图模式 (M)';
    gridControls.classList.toggle('visible', mode === 'grid');
    blockPanel.classList.toggle('visible', mode === 'grid' && currentSlices.length > 0);
    draw();
    updateInfo();
}

function toggleOrientation() {
    horizontal = !horizontal;
    orientBtn.textContent = horizontal ? '横线 ▬ (H)' : '竖线 ▯ (H)';
    canvas.style.cursor = horizontal ? 'sb-v-double-arrow' : 'sb-h-double-arrow';
    draw();
}

function toggleFit() {
    fitToWindow = !fitToWindow;
    draw();
    updateInfo();
}

function swapImages() {
    if (mode === 'grid') {
        [idxLeft, idxRight] = [idxRight, idxLeft];
        updateBlockButtons();
    } else {
        [img1, img2] = [img2, img1];
    }
    dividerX = 1.0 - dividerX;
    draw();
    updateInfo();
}

async function applyGridSplit() {
    if (!sourceImg) {
        alert('请先打开拼合图');
        return;
    }
    const cols = parseInt(colsInput.value) || 5;
    if (cols < 2) { alert('切分数量必须 >= 2'); return; }

    try {
        const resp = await fetch('/api/split', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: sourceImg.id, cols }),
        });
        const data = await resp.json();
        if (data.error) { alert(data.error); return; }

        currentSlices = data.slices;
        idxLeft = 0;
        idxRight = Math.min(1, currentSlices.length - 1);

        buildBlockPanel();
        draw();
        updateInfo();
    } catch (err) {
        alert('切分失败: ' + err.message);
    }
}

function buildBlockPanel() {
    blockBtns.innerHTML = '';
    currentSlices.forEach((s, i) => {
        const btn = document.createElement('button');
        btn.className = 'block-btn';
        btn.textContent = i + 1;
        btn.onclick = () => onBlockClick(i);
        blockBtns.appendChild(btn);
    });
    updateBlockButtons();
}

function updateBlockButtons() {
    const btns = blockBtns.querySelectorAll('.block-btn');
    btns.forEach((btn, i) => {
        btn.classList.remove('left', 'right');
        if (i === idxLeft) btn.classList.add('left');
        else if (i === idxRight) btn.classList.add('right');
    });
}

function onBlockClick(idx) {
    if (idx === idxLeft) {
        idxLeft = idxRight;
        idxRight = idx;
    } else if (idx !== idxRight) {
        idxLeft = idxRight;
        idxRight = idx;
    }
    updateBlockButtons();
    draw();
    updateInfo();
}

function quickSelect(n) {
    if (mode !== 'grid' || currentSlices.length === 0) return;
    if (n === 0) n = 10;
    n -= 1;
    if (n >= 0 && n < currentSlices.length) onBlockClick(n);
}

async function saveScreenshot() {
    const leftId = getLeftId();
    const rightId = getRightId();
    if (!leftId && !rightId) return;

    try {
        const resp = await fetch('/api/screenshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                left_id: leftId,
                right_id: rightId,
                divider_x: dividerX,
                horizontal,
            }),
        });
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'screenshot.png';
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert('保存失败: ' + err.message);
    }
}

// ════════════════════════════════════════════════════════════════
//  绘制
// ════════════════════════════════════════════════════════════════
function getLeft() {
    if (mode === 'grid' && currentSlices[idxLeft]) return currentSlices[idxLeft];
    return img1;
}
function getRight() {
    if (mode === 'grid' && currentSlices[idxRight]) return currentSlices[idxRight];
    return img2;
}
function getLeftId() { const l = getLeft(); return l ? l.id : null; }
function getRightId() { const r = getRight(); return r ? r.id : null; }

let imgLeftEl = new Image();
let imgRightEl = new Image();
let loadedLeft = false;
let loadedRight = false;

function draw() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#2d2d2d';
    ctx.fillRect(0, 0, w, h);

    const left = getLeft();
    const right = getRight();

    if (!left && !right) {
        hint.style.display = 'block';
        hint.textContent = mode === 'single'
            ? '按 O / P 打开图片'
            : '按 O 打开拼合图，输入切分数后回车';
        return;
    }
    hint.style.display = 'none';

    const displayW = fitToWindow ? w : (right || left).width;
    const displayH = fitToWindow ? h : (right || left).height;

    // 加载并绘制右边图
    if (right && right.id) {
        const newSrc = `/api/image/${right.id}`;
        if (imgRightEl.src !== location.origin + newSrc) {
            loadedRight = false;
            imgRightEl.onload = () => { loadedRight = true; draw(); };
            imgRightEl.src = newSrc;
            return;
        }
        if (loadedRight) {
            const ratio = Math.min(displayW / imgRightEl.width, displayH / imgRightEl.height);
            const rw = imgRightEl.width * ratio;
            const rh = imgRightEl.height * ratio;
            ctx.drawImage(imgRightEl, 0, 0, rw, rh);
        }
    }

    // 加载并绘制左边图（裁剪到分界线）
    if (left && left.id) {
        const newSrc = `/api/image/${left.id}`;
        if (imgLeftEl.src !== location.origin + newSrc) {
            loadedLeft = false;
            imgLeftEl.onload = () => { loadedLeft = true; draw(); };
            imgLeftEl.src = newSrc;
            return;
        }
        if (loadedLeft) {
            const ratio = Math.min(displayW / imgLeftEl.width, displayH / imgLeftEl.height);
            const lw = imgLeftEl.width * ratio;
            const lh = imgLeftEl.height * ratio;

            ctx.save();
            if (horizontal) {
                const div = Math.floor(displayH * dividerX);
                ctx.beginPath();
                ctx.rect(0, 0, lw, div);
                ctx.clip();
                ctx.drawImage(imgLeftEl, 0, 0, lw, lh);
            } else {
                const div = Math.floor(displayW * dividerX);
                ctx.beginPath();
                ctx.rect(0, 0, div, lh);
                ctx.clip();
                ctx.drawImage(imgLeftEl, 0, 0, lw, lh);
            }
            ctx.restore();
        }
    }

    // 画分界线
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    if (horizontal) {
        const div = Math.floor(displayH * dividerX);
        ctx.beginPath();
        ctx.moveTo(0, div);
        ctx.lineTo(displayW, div);
        ctx.stroke();
        // 中心圆点
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(displayW / 2, div, 5, 0, Math.PI * 2);
        ctx.strokeStyle = '#aaaaaa';
        ctx.stroke();
    } else {
        const div = Math.floor(displayW * dividerX);
        ctx.beginPath();
        ctx.moveTo(div, 0);
        ctx.lineTo(div, displayH);
        ctx.stroke();
        // 中心圆点
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(div, displayH / 2, 5, 0, Math.PI * 2);
        ctx.strokeStyle = '#aaaaaa';
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // 拼合图模式: 块编号标签
    if (mode === 'grid') {
        ctx.font = 'bold 16px monospace';
        if (left) {
            ctx.fillStyle = '#007acc';
            ctx.textAlign = 'left';
            ctx.fillText(`#${idxLeft + 1}`, 16, 28);
        }
        if (right) {
            ctx.fillStyle = '#cc5500';
            ctx.textAlign = 'right';
            ctx.fillText(`#${idxRight + 1}`, displayW - 16, 28);
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  信息栏
// ════════════════════════════════════════════════════════════════
function updateInfo() {
    const parts = [];
    if (mode === 'grid') {
        if (sourceImg) parts.push(`拼合: ${sourceImg.name} (${sourceImg.width}x${sourceImg.height})`);
        if (currentSlices.length) {
            parts.push(`共 ${currentSlices.length} 块`);
            parts.push(`对比: #${idxLeft + 1} vs #${idxRight + 1}`);
        }
    } else {
        if (img1) parts.push(`图1: ${img1.name} (${img1.width}x${img1.height})`);
        if (img2) parts.push(`图2: ${img2.name} (${img2.width}x${img2.height})`);
    }
    parts.push(fitToWindow ? '适应' : '原始');
    infoEl.textContent = parts.join('  |  ');
}

// ════════════════════════════════════════════════════════════════
//  启动
// ════════════════════════════════════════════════════════════════
init();
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="图片对比器 (FastAPI 版)")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(url)

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"图片对比器启动: {url}")
    print("按 Ctrl+C 停止")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
