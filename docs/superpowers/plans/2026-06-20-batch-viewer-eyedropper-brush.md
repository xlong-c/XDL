# batch_viewer 取色器+涂抹+全屏 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 batch_viewer.py 添加大图模态框(取色器/涂抹/笔刷滚轮调大小),工具栏单行 flexbox 布局,全屏功能.

**Architecture:** 单文件修改 `tools/gui/batch_viewer.py`.后端新增 `POST /api/save-painted` 端点(PIL 解码 base64 → 覆盖写回原文件); 前端新增大图模态框(原图 `<img>` + 透明 `<canvas>` 叠加), 取色模式点 img 读像素, 涂抹模式在 canvas 上用圆点绘制, 滚轮调笔刷半径, 保存时 canvas.toBlob() → base64 → POST 到后端覆盖.

**Tech Stack:** Python 3.12+, FastAPI, pywebview, Pillow, vanilla JS (no framework)

---

### Task 1: 后端 - 新增 save_painted 业务函数 + API 端点

**Files:**
- Modify: `tools/gui/batch_viewer.py` (add function in 业务函数区, add route in FastAPI 区)

- [ ] **Step 1: 添加纯业务函数 `save_painted_image`**

在 `delete_images` 函数后面 (line ~252) 添加:

```python
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
```

- [ ] **Step 2: 添加 `POST /api/save-painted` 路由**

在 `api_delete` 路由后面 (line ~355) 添加:

```python
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
```

- [ ] **Step 3: 验证后端可启动**

```bash
timeout 3 python tools/gui/batch_viewer.py 2>&1 || true
```
预期: 无 import/syntax 错误.

---

### Task 2: 前端 CSS - 工具栏 flexbox + 模态框 + 绘画层样式

**Files:**
- Modify: `tools/gui/batch_viewer.py` CSS 段 (lines ~377-533)

- [ ] **Step 1: 替换 `#toolbar` CSS**

```css
#toolbar {
    background: #2a2a2a;
    border-bottom: 1px solid #3a3a3a;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.toolbar-sep {
    width: 1px;
    height: 24px;
    background: #4a4a4a;
    flex-shrink: 0;
}
```

- [ ] **Step 2: 新增模态框 + 绘画相关 CSS**

在 `</style>` 前添加:

```css
/* ---------- 大图模态框 ---------- */
#viewer-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    flex-direction: column;
}
#viewer-overlay.open { display: flex; }
#viewer-toolbar {
    background: #2a2a2a;
    border-bottom: 1px solid #3a3a3a;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}
#viewer-toolbar .btn.active {
    background: #007acc;
    border-color: #007acc;
}
#viewer-canvas-wrap {
    flex: 1;
    min-height: 0;
    overflow: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
#viewer-layer {
    position: relative;
    display: inline-block;
    line-height: 0;
}
#viewer-layer img {
    display: block;
    max-width: 95vw;
    max-height: calc(100vh - 100px);
    object-fit: contain;
}
#viewer-paint-canvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
}
#viewer-paint-canvas.brush-active {
    pointer-events: auto;
    cursor: none;
}
#viewer-statusbar {
    background: #252525;
    border-top: 1px solid #343434;
    padding: 6px 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 12px;
    color: #aaa;
    flex-shrink: 0;
}
#viewer-color-swatch {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #555;
    display: inline-block;
    vertical-align: middle;
}
#viewer-brush-indicator {
    position: fixed;
    pointer-events: none;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.15);
    transform: translate(-50%, -50%);
    display: none;
    z-index: 1001;
}
```

- [ ] **Step 3: 删除旧的 `@media (max-width: 1280px)` toolbar 规则**

旧的 grid 媒体查询已不适用.

---

### Task 3: 前端 HTML - 工具栏改为 flexbox 单行

**Files:**
- Modify: `tools/gui/batch_viewer.py` HTML body 段 (lines ~537-553)

- [ ] **Step 1: 替换 `#toolbar` HTML**

```html
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
```

---

### Task 4: 前端 HTML - 大图模态框结构

**Files:**
- Modify: `tools/gui/batch_viewer.py` HTML body 段, 在 `#content` 之后, `<script>` 之前插入

- [ ] **Step 1: 插入模态框 HTML**

在 `</div>` (content div 结束) 后面,`<script>` 前面添加:

```html
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
```

---

### Task 5: 前端 JS - 模态框状态 + 打开/关闭逻辑

**Files:**
- Modify: `tools/gui/batch_viewer.py` JS 段 - 在 `state` 对象后添加 viewer state, 在 `init()` 前添加 viewer 函数

- [ ] **Step 1: 添加 viewer state**

在 `state` 对象定义后 (`};`) 添加:

```javascript
const viewerState = {
    isOpen: false,
    imagePath: '',
    mode: 'picker',       // 'picker' | 'brush'
    brushSize: 20,
    currentColor: '#ff0000',
    isDrawing: false,
    lastX: 0,
    lastY: 0,
};
```

- [ ] **Step 2: 添加 DOM 引用**

在现有 DOM 引用区添加:

```javascript
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
```

- [ ] **Step 3: 添加 `setViewerMode()` 函数**

```javascript
function setViewerMode(mode) {
    viewerState.mode = mode;
    const isBrush = mode === 'brush';
    viewerPickerBtn.classList.toggle('active', !isBrush);
    viewerBrushBtn.classList.toggle('active', isBrush);
    viewerBrushSizeWrap.style.display = isBrush ? '' : 'none';
    viewerModeText.textContent = isBrush ? '🖌️涂抹' : '🎨取色';
    viewerPaintCanvas.classList.toggle('brush-active', isBrush);
    if (isBrush) {
        viewerImageEl.style.cursor = 'none';
        viewerPaintCanvas.style.cursor = 'none';
        syncPaintCanvasSize();
        viewerBrushIndicator.style.display = '';
    } else {
        viewerImageEl.style.cursor = 'crosshair';
        viewerBrushIndicator.style.display = 'none';
    }
    updateViewerStatus();
}
```

- [ ] **Step 4: 添加 `updateViewerStatus()` 函数**

```javascript
function updateViewerStatus() {
    viewerColorSwatch.style.backgroundColor = viewerState.currentColor;
    viewerColorText.textContent = viewerState.currentColor;
    viewerBrushText.textContent = viewerState.mode === 'brush' ? String(viewerState.brushSize) : '-';
}
```

- [ ] **Step 5: 添加 `openViewer(path)` 函数**

```javascript
async function openViewer(path) {
    viewerState.isOpen = true;
    viewerState.imagePath = path;
    viewerFilename.textContent = path.split(/[\\/]/).pop() || path;
    
    const url = `/api/image?path=${encodeURIComponent(path)}`;
    viewerImageEl.src = url;
    viewerOverlayEl.classList.add('open');
    
    // 等图片加载完再同步 canvas 尺寸
    await new Promise((resolve) => {
        if (viewerImageEl.complete) { resolve(); return; }
        viewerImageEl.onload = resolve;
        viewerImageEl.onerror = resolve;
    });
    syncPaintCanvasSize();
    clearPaintCanvas();
    setViewerMode('picker');
}
```

- [ ] **Step 6: 添加 `closeViewer()` 函数**

```javascript
function closeViewer() {
    viewerState.isOpen = false;
    viewerOverlayEl.classList.remove('open');
    viewerBrushIndicator.style.display = 'none';
    viewerImageEl.src = '';
    clearPaintCanvas();
}
```

- [ ] **Step 7: 修改双击事件**

将 line 885-888 的:
```javascript
tile.addEventListener('dblclick', () => {
    const url = `/api/image?path=${encodeURIComponent(image.path)}`;
    window.open(url, '_blank');
});
```
替换为:
```javascript
tile.addEventListener('dblclick', () => {
    openViewer(image.path);
});
```

- [ ] **Step 8: 绑定模态框按钮事件**

在 `bindInputEvents()` 中添加:

```javascript
viewerPickerBtn.addEventListener('click', () => setViewerMode('picker'));
viewerBrushBtn.addEventListener('click', () => setViewerMode('brush'));
document.getElementById('viewer-close-btn').addEventListener('click', closeViewer);

// 按 Esc 关闭
viewerOverlayEl.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { closeViewer(); }
});
// 点击 overlay 背景关闭
viewerOverlayEl.addEventListener('click', (event) => {
    if (event.target === viewerOverlayEl) { closeViewer(); }
});

// 笔刷大小输入
viewerBrushSizeInput.addEventListener('input', () => {
    const v = Math.max(2, Math.min(200, Number(viewerBrushSizeInput.value) || 20));
    viewerState.brushSize = v;
    updateViewerStatus();
    updateBrushIndicator();
});
```

---

### Task 6: 前端 JS - 取色器逻辑

**Files:**
- Modify: `tools/gui/batch_viewer.py` JS 段

- [ ] **Step 1: 添加取色函数**

```javascript
function pickColor(clientX, clientY) {
    const imgRect = viewerImageEl.getBoundingClientRect();
    const naturalW = viewerImageEl.naturalWidth;
    const naturalH = viewerImageEl.naturalHeight;
    if (!naturalW || !naturalH) { return; }
    
    // 计算图片在容器中的实际显示区域
    const scaleX = naturalW / imgRect.width;
    const scaleY = naturalH / imgRect.height;
    const px = Math.round((clientX - imgRect.left) * scaleX);
    const py = Math.round((clientY - imgRect.top) * scaleY);
    if (px < 0 || py < 0 || px >= naturalW || py >= naturalH) { return; }
    
    // 用隐藏 canvas 读像素
    const offscreen = document.createElement('canvas');
    offscreen.width = naturalW;
    offscreen.height = naturalH;
    const ctx = offscreen.getContext('2d');
    ctx.drawImage(viewerImageEl, 0, 0);
    const [r, g, b] = ctx.getImageData(px, py, 1, 1).data;
    viewerState.currentColor = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    updateViewerStatus();
}
```

- [ ] **Step 2: 绑定取色点击**

在模态框初始化中添加:

```javascript
viewerImageEl.addEventListener('click', (event) => {
    if (viewerState.mode !== 'picker') { return; }
    pickColor(event.clientX, event.clientY);
});
```

---

### Task 7: 前端 JS - 涂抹绘画逻辑

**Files:**
- Modify: `tools/gui/batch_viewer.py` JS 段

- [ ] **Step 1: 添加 painting 辅助函数**

```javascript
function syncPaintCanvasSize() {
    const displayW = viewerImageEl.clientWidth;
    const displayH = viewerImageEl.clientHeight;
    if (displayW <= 0 || displayH <= 0) { return; }
    viewerPaintCanvas.width = displayW;
    viewerPaintCanvas.height = displayH;
    viewerPaintCanvas.style.width = displayW + 'px';
    viewerPaintCanvas.style.height = displayH + 'px';
}

function clearPaintCanvas() {
    const ctx = viewerPaintCanvas.getContext('2d');
    if (ctx) { ctx.clearRect(0, 0, viewerPaintCanvas.width, viewerPaintCanvas.height); }
}

function updateBrushIndicator() {
    viewerBrushIndicator.style.width = viewerState.brushSize + 'px';
    viewerBrushIndicator.style.height = viewerState.brushSize + 'px';
}

function drawDot(canvasX, canvasY) {
    const ctx = viewerPaintCanvas.getContext('2d');
    const r = viewerState.brushSize / 2;
    ctx.fillStyle = viewerState.currentColor;
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, r, 0, Math.PI * 2);
    ctx.fill();
}

function drawLine(fromX, fromY, toX, toY) {
    const ctx = viewerPaintCanvas.getContext('2d');
    const r = viewerState.brushSize / 2;
    ctx.strokeStyle = viewerState.currentColor;
    ctx.lineWidth = viewerState.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
}
```

- [ ] **Step 2: 绑定绘画鼠标事件**

```javascript
viewerPaintCanvas.addEventListener('mousedown', (event) => {
    if (viewerState.mode !== 'brush') { return; }
    viewerState.isDrawing = true;
    const rect = viewerPaintCanvas.getBoundingClientRect();
    viewerState.lastX = event.clientX - rect.left;
    viewerState.lastY = event.clientY - rect.top;
    drawDot(viewerState.lastX, viewerState.lastY);
});

viewerPaintCanvas.addEventListener('mousemove', (event) => {
    const rect = viewerPaintCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // 更新笔刷指示器位置
    if (viewerState.mode === 'brush') {
        viewerBrushIndicator.style.left = event.clientX + 'px';
        viewerBrushIndicator.style.top = event.clientY + 'px';
    }
    
    if (!viewerState.isDrawing) { return; }
    drawLine(viewerState.lastX, viewerState.lastY, x, y);
    viewerState.lastX = x;
    viewerState.lastY = y;
});

viewerPaintCanvas.addEventListener('mouseup', () => {
    viewerState.isDrawing = false;
});

viewerPaintCanvas.addEventListener('mouseleave', () => {
    viewerState.isDrawing = false;
});
```

- [ ] **Step 3: 绑定滚轮调笔刷大小**

```javascript
viewerOverlayEl.addEventListener('wheel', (event) => {
    if (viewerState.mode !== 'brush') { return; }
    event.preventDefault();
    const step = event.deltaY > 0 ? -2 : 2;
    viewerState.brushSize = Math.max(2, Math.min(200, viewerState.brushSize + step));
    viewerBrushSizeInput.value = String(viewerState.brushSize);
    updateViewerStatus();
    updateBrushIndicator();
}, { passive: false });
```

---

### Task 8: 前端 JS - 撤销 + 保存 + 全屏

**Files:**
- Modify: `tools/gui/batch_viewer.py` JS 段

- [ ] **Step 1: 撤销功能**

```javascript
document.getElementById('viewer-undo-btn').addEventListener('click', () => {
    clearPaintCanvas();
});
```

- [ ] **Step 2: 保存功能(合成 → POST)**

```javascript
document.getElementById('viewer-save-btn').addEventListener('click', async () => {
    // 合成绘画层到原图上
    const mergeCanvas = document.createElement('canvas');
    mergeCanvas.width = viewerImageEl.naturalWidth;
    mergeCanvas.height = viewerImageEl.naturalHeight;
    const ctx = mergeCanvas.getContext('2d');
    ctx.drawImage(viewerImageEl, 0, 0);
    
    // 缩放绘画层到原图尺寸
    const paintW = viewerPaintCanvas.width;
    const paintH = viewerPaintCanvas.height;
    const scaleX = mergeCanvas.width / paintW;
    const scaleY = mergeCanvas.height / paintH;
    ctx.save();
    ctx.scale(scaleX, scaleY);
    ctx.drawImage(viewerPaintCanvas, 0, 0);
    ctx.restore();
    
    const base64 = mergeCanvas.toDataURL('image/png');
    
    try {
        const result = await postJSON('/api/save-painted', {
            path: viewerState.imagePath,
            image: base64,
        });
        if (result.ok) {
            setStatus(`已保存: ${viewerFilename.textContent}`);
            closeViewer();
            // 刷新缩略图缓存
            state.thumbImageCache.clear();
            state.thumbLoadPromises.clear();
            render();
        } else {
            setStatus(`保存失败: ${result.error}`, true);
        }
    } catch (error) {
        setStatus(`保存失败: ${error.message}`, true);
    }
});
```

- [ ] **Step 3: 全屏功能**

```javascript
document.getElementById('fullscreen-btn').addEventListener('click', () => {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        document.documentElement.requestFullscreen();
    }
});

// 工具栏取色/涂抹按钮也打开最近的图片
document.getElementById('picker-btn').addEventListener('click', () => {
    if (!state.images.length) { setStatus('没有图片可查看', true); return; }
    const idx = state.selected.size ? [...state.selected][0] : state.images[0].path;
    openViewer(idx);
    // 确保进入取色模式
    setTimeout(() => setViewerMode('picker'), 300);
});

document.getElementById('brush-btn').addEventListener('click', () => {
    if (!state.images.length) { setStatus('没有图片可查看', true); return; }
    const idx = state.selected.size ? [...state.selected][0] : state.images[0].path;
    openViewer(idx);
    setTimeout(() => setViewerMode('brush'), 300);
});
```

- [ ] **Step 4: 视图初始化时更新笔刷指示器**

在 `setViewerMode` 中 brush 分支添加: `updateBrushIndicator();`

---

### Task 9: 验证 - 启动测试

**Files:**
- Test: 手动启动验证

- [ ] **Step 1: 语法检查**

```bash
python -c "import ast; ast.parse(open('tools/gui/batch_viewer.py').read()); print('OK')"
```

- [ ] **Step 2: import 检查**

```bash
python -c "import tools.gui.batch_viewer; print('import OK')"
```
(如果 `gi` / `webview` 不可用, 至少保证纯 Python 部分无 import 错误)

- [ ] **Step 3: 运行 ruff 检查**

```bash
ruff check tools/gui/batch_viewer.py
```

---

## 修改位置汇总

| 区域 | 行号范围 | 改动类型 |
|---|---|---|
| 业务函数区 | ~252 后 | 新增 `save_painted_image` |
| API 路由区 | ~355 后 | 新增 `POST /api/save-painted` |
| CSS `#toolbar` | ~379-387 | 替换为 flexbox |
| CSS `@media` | ~529-533 | 删除旧 grid 媒体查询 |
| CSS `</style>` 前 | ~534 前 | 新增模态框+绘画 CSS |
| HTML toolbar | ~537-553 | 替换为 flexbox 单行 |
| HTML content 后 | ~563 后 | 新增模态框 HTML |
| JS dblclick | ~885-888 | 改为 `openViewer(path)` |
| JS state 后 | ~583 后 | 添加 `viewerState` |
| JS DOM 引用 | ~585-590 后 | 添加 viewer DOM refs |
| JS bindInputEvents | ~969-1017 | 添加 viewer 按钮绑定 |
| JS init 前 | ~1019 前 | 添加所有 viewer 函数 |
