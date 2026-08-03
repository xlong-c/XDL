#!/usr/bin/env python3
"""图片对比器 - pywebview + FastAPI 单文件版.

按 tools/gui/AGENTS.md 规范组织.功能与原 Flet 版一致:
  - 双图对比: 加载两张图, 拖拽分界线对比
  - 拼合图拆分: 加载拼合图, 切分 N 块, 选任意两块对比
  - 鼠标 / 滚轮 / 方向键 / 快捷键 全部支持

启动:
    python tools/gui/img_compare.py

依赖:
    pip install pywebview>=5.0 fastapi uvicorn pillow
    Linux: sudo apt install python3-gi gir1.2-webkit2-4.0

快捷键:
    O / P          打开图1 / 图2
    H              切换横线 / 竖线
    F              适应窗口 / 原始大小
    S              保存当前对比截图
    R              交换
    M              切换双图 / 拼合图模式
    1~9 / 0        拼合图模式选块 (0 = 10)
    ←↑→↓           微调分界线 (1px)
    Shift+方向键    快速移动 (10px)
    Esc / Q        退出
"""
from __future__ import annotations

import base64
import io
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image

# ════════════════════════════════════════════════════════════════
#  1. CONFIG(不引入命令行参数解析库)
# ════════════════════════════════════════════════════════════════

CONFIG: dict[str, Any] = {
    "title": "图片对比器",
    "width": 1280,
    "height": 820,
    "host": "127.0.0.1",
    "port": 8765,
    "mode": "single",        # "single" 双图对比 | "grid" 拼合图拆分
    "cols": 5,                # 拼合图横向切分数
}

# ════════════════════════════════════════════════════════════════
#  2. 业务函数(无 UI 依赖,可独立单测)
# ════════════════════════════════════════════════════════════════


def load_pil(path: str) -> Image.Image:
    """打开图片为 RGB."""
    return Image.open(path).convert("RGB")


def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image → base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def split_image_horizontal(img: Image.Image, cols: int) -> list[Image.Image]:
    """横向等分切分图片为 N 块, 最后一块吃到剩余像素."""
    cols = max(2, int(cols))
    src_w, src_h = img.size
    slice_w = src_w // cols
    out: list[Image.Image] = []
    for i in range(cols):
        x0 = i * slice_w
        x1 = x0 + slice_w if i < cols - 1 else src_w
        out.append(img.crop((x0, 0, x1, src_h)))
    return out


# ════════════════════════════════════════════════════════════════
#  3. STATE(Python 端的"业务真相")
#
#  显示状态(分界线比例,画布尺寸,拖拽标志)放前端 JS,
#  业务数据(图片路径,模式,块索引)放这里.
# ════════════════════════════════════════════════════════════════


class State:
    """集中管理业务数据, 路由都从这里读/写."""

    def __init__(self) -> None:
        # 双图模式
        self.img1: Image.Image | None = None
        self.img1_path: str | None = None
        self.img1_uri: str | None = None
        self.img2: Image.Image | None = None
        self.img2_path: str | None = None
        self.img2_uri: str | None = None
        # 拼合图模式
        self.source_img: Image.Image | None = None
        self.source_path: str | None = None
        self.slices: list[Image.Image] = []
        self.slices_uri: list[str] = []
        self.idx_left: int = 0
        self.idx_right: int = 1
        # 模式 / 切分数
        self.mode: str = CONFIG["mode"]
        self.cols: int = max(2, int(CONFIG["cols"]))

    # ── 双图加载 ──
    def load_image(self, side: int, path: str) -> dict[str, Any]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        img = load_pil(path)
        uri = pil_to_data_uri(img)
        if side == 1:
            self.img1, self.img1_path, self.img1_uri = img, path, uri
        else:
            self.img2, self.img2_path, self.img2_uri = img, path, uri
        return self.snapshot()

    # ── 拼合图加载 ──
    def load_grid(self, path: str) -> dict[str, Any]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        self.source_img = load_pil(path)
        self.source_path = path
        self._do_split()
        return self.snapshot()

    def _do_split(self) -> None:
        if self.source_img is None:
            self.slices, self.slices_uri = [], []
            return
        self.slices = split_image_horizontal(self.source_img, self.cols)
        self.slices_uri = [pil_to_data_uri(s) for s in self.slices]
        n = len(self.slices)
        if n:
            self.idx_left = max(0, min(self.idx_left, n - 1))
            self.idx_right = max(0, min(self.idx_right, n - 1))

    def set_cols(self, cols: int) -> dict[str, Any]:
        if cols < 2:
            raise ValueError("切分数量必须 ≥ 2")
        self.cols = cols
        self._do_split()
        return self.snapshot()

    def select_block(self, idx: int) -> dict[str, Any]:
        """点击 / 数字键: 复刻 Flet 版 _on_block_click 行为."""
        if not self.slices:
            return self.snapshot()
        if idx == self.idx_left:
            self.idx_left, self.idx_right = self.idx_right, self.idx_left
        elif idx == self.idx_right:
            pass
        else:
            self.idx_left, self.idx_right = self.idx_right, idx
        return self.snapshot()

    # ── 模式 / 交换 ──
    def toggle_mode(self) -> dict[str, Any]:
        self.mode = "grid" if self.mode == "single" else "single"
        return self.snapshot()

    def swap(self) -> dict[str, Any]:
        if self.mode == "grid":
            self.idx_left, self.idx_right = self.idx_right, self.idx_left
        else:
            self.img1, self.img2 = self.img2, self.img1
            self.img1_path, self.img2_path = self.img2_path, self.img1_path
            self.img1_uri, self.img2_uri = self.img2_uri, self.img1_uri
        return self.snapshot()

    # ── 给前端的状态 ──
    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cols": self.cols,
            "img1": self._img_info(self.img1, self.img1_path),
            "img1_uri": self.img1_uri,
            "img2": self._img_info(self.img2, self.img2_path),
            "img2_uri": self.img2_uri,
            "source": self._img_info(self.source_img, self.source_path),
            "slices": [self._img_info(s) for s in self.slices],
            "slices_uri": self.slices_uri,
            "idx_left": self.idx_left,
            "idx_right": self.idx_right,
        }

    @staticmethod
    def _img_info(img: Image.Image | None, path: str | None = None) -> dict[str, Any] | None:
        if img is None:
            return None
        return {"w": img.size[0], "h": img.size[1], "path": path}


STATE = State()

# ════════════════════════════════════════════════════════════════
#  4. FastAPI app + 路由
# ════════════════════════════════════════════════════════════════

app = FastAPI(title=CONFIG["title"])


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    """前端启动时 / 模式切换后拉状态."""
    return STATE.snapshot()


@app.get("/api/load_image")
async def load_image(side: int, path: str) -> dict[str, Any]:
    """双图模式加载图片.side=1 / 2"""
    return STATE.load_image(side, path)


@app.get("/api/load_grid")
async def load_grid(path: str) -> dict[str, Any]:
    """拼合图模式加载图片."""
    return STATE.load_grid(path)


@app.post("/api/set_cols")
async def set_cols(cols: int) -> dict[str, Any]:
    return STATE.set_cols(cols)


@app.post("/api/select_block")
async def select_block(idx: int) -> dict[str, Any]:
    return STATE.select_block(idx)


@app.post("/api/toggle_mode")
async def toggle_mode() -> dict[str, Any]:
    return STATE.toggle_mode()


@app.post("/api/swap")
async def swap() -> dict[str, Any]:
    return STATE.swap()


# ════════════════════════════════════════════════════════════════
#  5. HTML 模板
# ════════════════════════════════════════════════════════════════

HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ background:#1e1e1e; color:#e5e5e5; margin:0; height:100vh;
         font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         display:flex; flex-direction:column; }}
  .panel {{ background:#2d2d2d; }}
  .btn {{ background:#3d3d3d; color:#ccc; padding:6px 12px;
         border-radius:4px; cursor:pointer; border:1px solid #525252;
         font-size:13px; }}
  .btn:hover {{ background:#505050; }}
  .btn.active {{ background:#525252; color:#fff; }}
  .input {{ background:#3d3d3d; color:#fff; padding:6px 10px;
           border-radius:4px; border:1px solid #525252; width:60px;
           text-align:center; }}
  #toolbar {{ padding:8px 12px; }}
  #canvas-wrap {{ flex:1; position:relative; background:#000; overflow:hidden; }}
  #canvas {{ position:absolute; inset:0; width:100%; height:100%; cursor:ew-resize; }}
  #canvas.h {{ cursor:ns-resize; }}
  #info {{ padding:6px 12px; font-size:12px; color:#888;
         background:#252525; border-top:1px solid #3d3d3d; }}
  #hint {{ position:absolute; inset:0; display:flex; align-items:center;
          justify-content:center; color:#666; font-size:18px;
          font-weight:bold; pointer-events:none; }}
  .block-btn {{ width:30px; height:30px; padding:0; }}
  .block-btn.left {{ background:#007acc; color:white; border-color:#007acc; }}
  .block-btn.right {{ background:#cc5500; color:white; border-color:#cc5500; }}
</style>
</head>
<body>

  <!-- 工具栏 -->
  <div id="toolbar" class="panel flex flex-wrap items-center gap-2">
    <button class="btn" onclick="toggleMode()">模式 <span id="mode-text">(M)</span></button>
    <div id="grid-row" class="flex items-center gap-1" style="display:none">
      <span style="color:#888;font-size:13px">切分:</span>
      <input id="cols" class="input" type="number" value="5" min="2">
      <button class="btn" onclick="applyCols()">应用</button>
    </div>
    <button class="btn" onclick="openImage(1)">打开图1 (O)</button>
    <button class="btn" onclick="openImage(2)">打开图2 (P)</button>
    <button class="btn" onclick="toggleFit()">适应/原始 (F)</button>
    <button class="btn" onclick="swap()">交换 (R)</button>
    <button class="btn" onclick="saveSnapshot()">截图保存 (S)</button>
    <button class="btn" onclick="toggleOrient()">分界 <span id="orient-text">竖线</span> (H)</button>
    <div class="flex-1"></div>
    <span id="status" style="color:#888;font-size:12px"></span>
  </div>

  <!-- 画布 -->
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="hint">按 O / P 打开图片, 或按 M 切到拼合图模式</div>
  </div>

  <!-- 块选择面板 (拼合图模式) -->
  <div id="block-panel" class="panel" style="display:none;padding:6px 12px">
    <span style="color:#888;font-size:12px;margin-right:6px">点击选择对比块:</span>
    <div id="block-buttons" class="inline-flex gap-1"></div>
  </div>

  <!-- 信息栏 -->
  <div id="info"></div>

  <script>
    // ── 前端"显示状态"(业务数据从后端 STATE 拉) ──
    const view = {{
      divider_x: 0.5,
      horizontal: false,
      fit_to_window: true,
      dragging: false,
      canvas_w: 0,
      canvas_h: 0,
    }};

    // 业务数据缓存(从后端拉)
    let data = {{
      mode: 'single', cols: 5,
      img1: null, img2: null, source: null,
      slices: [], slices_uri: [], idx_left: 0, idx_right: 1,
    }};

    // 图片对象缓存(用于 canvas drawImage, 异步加载)
    const imgCache = {{}};

    // ── 工具函数 ──
    async function callApi(path, params = {{}}, method = 'GET') {{
      const qs = new URLSearchParams(params).toString();
      const r = await fetch(path + (qs ? '?' + qs : ''), {{ method }});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    }}

    function loadImg(uri) {{
      if (imgCache[uri]) return Promise.resolve(imgCache[uri]);
      return new Promise((resolve, reject) => {{
        const im = new Image();
        im.onload = () => {{ imgCache[uri] = im; resolve(im); }};
        im.onerror = reject;
        im.src = uri;
      }});
    }}

    function setHint(text) {{
      document.getElementById('hint').textContent = text;
      document.getElementById('hint').style.display = text ? 'flex' : 'none';
    }}

    // ── 启动: 拉状态, 装事件 ──
    window.addEventListener('pywebviewready', async () => {{
      data = await callApi('/api/state');
      applyModeVisibility();
      render();
      updateInfo();
      setupCanvas();
      setupKeyboard();
    }});

    function applyModeVisibility() {{
      document.getElementById('grid-row').style.display = data.mode === 'grid' ? 'flex' : 'none';
      document.getElementById('block-panel').style.display = (data.mode === 'grid' && data.slices.length) ? 'block' : 'none';
      document.getElementById('mode-text').textContent = data.mode === 'grid' ? '拼合图' : '双图';
      if (data.mode === 'grid' && data.slices.length) buildBlockButtons();
    }}

    function buildBlockButtons() {{
      const wrap = document.getElementById('block-buttons');
      wrap.innerHTML = '';
      data.slices.forEach((s, i) => {{
        const b = document.createElement('button');
        b.className = 'btn block-btn';
        b.textContent = i === 9 ? '10' : String(i + 1);
        b.onclick = () => selectBlock(i);
        if (i === data.idx_left) b.classList.add('left');
        else if (i === data.idx_right) b.classList.add('right');
        wrap.appendChild(b);
      }});
    }}

    // ── 画布 ──
    function setupCanvas() {{
      const wrap = document.getElementById('canvas-wrap');
      const cvs = document.getElementById('canvas');
      const ctx = cvs.getContext('2d');
      const resize = () => {{
        const rect = wrap.getBoundingClientRect();
        view.canvas_w = rect.width;
        view.canvas_h = rect.height;
        cvs.width = rect.width * devicePixelRatio;
        cvs.height = rect.height * devicePixelRatio;
        cvs.style.width = rect.width + 'px';
        cvs.style.height = rect.height + 'px';
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        render();
      }};
      new ResizeObserver(resize).observe(wrap);
      resize();

      cvs.addEventListener('mousedown', (e) => {{
        view.dragging = true;
        setDividerFromEvent(e);
      }});
      window.addEventListener('mousemove', (e) => {{
        if (view.dragging) setDividerFromEvent(e);
      }});
      window.addEventListener('mouseup', () => {{ view.dragging = false; }});
      cvs.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const delta = view.horizontal ? e.deltaX : e.deltaY;
        const px = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        nudgeDivider(Math.round(-px));
      }}, {{ passive: false }});
    }}

    function setDividerFromEvent(e) {{
      const cvs = document.getElementById('canvas');
      const rect = cvs.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const pos = view.horizontal ? y : x;
      const dim = view.horizontal ? view.canvas_h : view.canvas_w;
      if (dim <= 0) return;
      view.divider_x = Math.max(0, Math.min(1, pos / dim));
      render();
    }}

    function nudgeDivider(px) {{
      const dim = view.horizontal ? view.canvas_h : view.canvas_w;
      if (dim <= 0) return;
      view.divider_x = Math.max(0, Math.min(1, view.divider_x + px / dim));
      render();
    }}

    // ── 渲染 ──
    function calcDisplayGeometry() {{
      const left = (data.mode === 'grid' ? data.slices[data.idx_left] : data.img1);
      const right = (data.mode === 'grid' ? data.slices[data.idx_right] : data.img2);
      if (!left && !right) return null;
      const ref = right || left;
      if (view.canvas_w < 1 || view.canvas_h < 1) return null;
      const rw = ref.w, rh = ref.h;
      const ratio = view.fit_to_window
        ? Math.min(view.canvas_w / rw, view.canvas_h / rh)
        : Math.min(1, view.canvas_w / rw, view.canvas_h / rh);
      const draw_w = Math.max(1, rw * ratio);
      const draw_h = Math.max(1, rh * ratio);
      const draw_x = (view.canvas_w - draw_w) / 2;
      const draw_y = (view.canvas_h - draw_h) / 2;
      return {{ draw_x, draw_y, draw_w, draw_h, ref_w: rw, ref_h: rh }};
    }}

    async function render() {{
      const cvs = document.getElementById('canvas');
      const ctx = cvs.getContext('2d');
      ctx.fillStyle = '#1e1e1e';
      ctx.fillRect(0, 0, view.canvas_w, view.canvas_h);

      const geom = calcDisplayGeometry();
      if (!geom) {{
        const msg = data.mode === 'grid'
          ? '按 O 打开拼合图, 输入切分数后回车'
          : '按 O / P 打开图片';
        setHint(msg);
        return;
      }}
      setHint(null);

      // 取当前左右 URI
      const rightUri = data.mode === 'grid' ? data.slices_uri[data.idx_right] : data.img2_uri;
      const leftUri  = data.mode === 'grid' ? data.slices_uri[data.idx_left]  : data.img1_uri;

      const divPos = view.horizontal
        ? Math.round(view.canvas_h * view.divider_x)
        : Math.round(view.canvas_w * view.divider_x);

      // 1. 画完整右图
      if (rightUri) {{
        const im = await loadImg(rightUri);
        ctx.drawImage(im, geom.draw_x, geom.draw_y, geom.draw_w, geom.draw_h);
      }}
      // 2. 画左图(裁到分界线)
      if (leftUri) {{
        const im = await loadImg(leftUri);
        ctx.save();
        if (view.horizontal) {{
          const h = Math.max(0, Math.min(divPos - geom.draw_y, geom.draw_h));
          if (h > 0) {{
            ctx.beginPath();
            ctx.rect(geom.draw_x, geom.draw_y, geom.draw_w, h);
            ctx.clip();
            ctx.drawImage(im, geom.draw_x, geom.draw_y, geom.draw_w, geom.draw_h);
          }}
        }} else {{
          const w = Math.max(0, Math.min(divPos - geom.draw_x, geom.draw_w));
          if (w > 0) {{
            ctx.beginPath();
            ctx.rect(geom.draw_x, geom.draw_y, w, geom.draw_h);
            ctx.clip();
            ctx.drawImage(im, geom.draw_x, geom.draw_y, geom.draw_w, geom.draw_h);
          }}
        }}
        ctx.restore();
      }}
      // 3. 分界线
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      if (view.horizontal) {{
        ctx.moveTo(0, divPos); ctx.lineTo(view.canvas_w, divPos);
      }} else {{
        ctx.moveTo(divPos, 0); ctx.lineTo(divPos, view.canvas_h);
      }}
      ctx.stroke();
      ctx.setLineDash([]);
      // 4. 中心把手
      ctx.strokeStyle = '#aaaaaa';
      ctx.beginPath();
      ctx.arc(view.horizontal ? view.canvas_w / 2 : divPos,
              view.horizontal ? divPos : view.canvas_h / 2,
              5, 0, Math.PI * 2);
      ctx.stroke();
      // 5. 拼合图模式: 块编号
      if (data.mode === 'grid') {{
        ctx.font = 'bold 16px system-ui';
        ctx.textBaseline = 'top';
        if (leftUri) {{
          ctx.fillStyle = '#007acc';
          ctx.fillText('#' + (data.idx_left + 1), geom.draw_x + 16, geom.draw_y + 8);
        }}
        if (rightUri) {{
          ctx.fillStyle = '#cc5500';
          const t = '#' + (data.idx_right + 1);
          const tw = ctx.measureText(t).width;
          ctx.fillText(t, geom.draw_x + geom.draw_w - tw - 16, geom.draw_y + 8);
        }}
      }}
    }}

    // ── 状态刷新 ──
    async function refreshState() {{
      data = await callApi('/api/state');
      applyModeVisibility();
      render();
      updateInfo();
    }}

    function updateInfo() {{
      const parts = [];
      if (data.mode === 'grid') {{
        if (data.source) parts.push('拼合: ' + (data.source.path?.split(/[\\\\/]/).pop() || '...') + ' (' + data.source.w + 'x' + data.source.h + ')');
        if (data.slices.length) {{
          parts.push('共 ' + data.slices.length + ' 块');
          parts.push('对比: #' + (data.idx_left + 1) + ' vs #' + (data.idx_right + 1));
        }}
      }} else {{
        if (data.img1) parts.push('图1: ' + (data.img1.path?.split(/[\\\\/]/).pop() || '...') + ' (' + data.img1.w + 'x' + data.img1.h + ')');
        if (data.img2) parts.push('图2: ' + (data.img2.path?.split(/[\\\\/]/).pop() || '...') + ' (' + data.img2.w + 'x' + data.img2.h + ')');
      }}
      parts.push(view.fit_to_window ? '适应' : '原始');
      document.getElementById('info').textContent = parts.join('  |  ') || '就绪';
    }}

    // ── 业务操作 ──
    async function openImage(side) {{
      try {{
        const path = await pywebview.api.pick_image();
        if (!path) return;
        if (data.mode === 'grid') {{
          await callApi('/api/load_grid', {{ path }});
        }} else {{
          await callApi('/api/load_image', {{ side, path }});
        }}
        await refreshState();
        toast('已加载: ' + path.split(/[\\\\/]/).pop());
      }} catch (e) {{ toast('加载失败: ' + e.message, true); }}
    }}

    async function applyCols() {{
      const cols = parseInt(document.getElementById('cols').value);
      try {{
        await callApi('/api/set_cols', {{ cols }}, 'POST');
        await refreshState();
      }} catch (e) {{ toast(e.message, true); }}
    }}

    async function selectBlock(idx) {{
      await callApi('/api/select_block', {{ idx }}, 'POST');
      await refreshState();
    }}

    async function toggleMode() {{
      await callApi('/api/toggle_mode', {{}}, 'POST');
      view.divider_x = 0.5;
      await refreshState();
    }}

    async function swap() {{
      await callApi('/api/swap', {{}}, 'POST');
      view.divider_x = 1 - view.divider_x;
      await refreshState();
    }}

    function toggleFit() {{
      view.fit_to_window = !view.fit_to_window;
      render();
      updateInfo();
    }}

    function toggleOrient() {{
      view.horizontal = !view.horizontal;
      document.getElementById('orient-text').textContent = view.horizontal ? '横线' : '竖线';
      document.getElementById('canvas').classList.toggle('h', view.horizontal);
      render();
    }}

    async function saveSnapshot() {{
      const cvs = document.getElementById('canvas');
      const dataUrl = cvs.toDataURL('image/png');
      const base64 = dataUrl.split(',')[1];
      try {{
        const path = await pywebview.api.save_snapshot(base64);
        if (path) toast('已保存: ' + path);
      }} catch (e) {{ toast('保存失败: ' + e.message, true); }}
    }}

    function toast(msg, error = false) {{
      const el = document.getElementById('status');
      el.textContent = msg;
      el.style.color = error ? '#ff7755' : '#88cc88';
      setTimeout(() => {{ if (el.textContent === msg) {{ el.textContent = ''; }} }}, 3000);
    }}

    // ── 键盘 ──
    function setupKeyboard() {{
      document.addEventListener('keydown', async (e) => {{
        // 在 input 框里不拦截
        if (e.target.tagName === 'INPUT') return;
        const k = e.key.toLowerCase();
        const shift = e.shiftKey;
        if (k === 'escape' || k === 'q') {{ pywebview.api.close_window(); return; }}
        if (k === 'o') {{ e.preventDefault(); openImage(1); return; }}
        if (k === 'p') {{ e.preventDefault(); openImage(2); return; }}
        if (k === 'f') {{ toggleFit(); return; }}
        if (k === 's') {{ e.preventDefault(); saveSnapshot(); return; }}
        if (k === 'r') {{ e.preventDefault(); swap(); return; }}
        if (k === 'h') {{ toggleOrient(); return; }}
        if (k === 'm') {{ e.preventDefault(); toggleMode(); return; }}
        if (/^[0-9]$/.test(k)) {{
          // 拼合图模式: 0=10, 1-9=对应块
          if (data.mode === 'grid' && data.slices.length) {{
            const idx = k === '0' ? 9 : parseInt(k) - 1;
            if (idx < data.slices.length) selectBlock(idx);
          }}
          return;
        }}
        if (k.startsWith('arrow')) {{
          const vertical = k === 'arrowup' || k === 'arrowdown';
          const positive = k === 'arrowright' || k === 'arrowdown';
          if (vertical === view.horizontal) {{
            const step = shift ? 10 : 1;
            nudgeDivider(positive ? step : -step);
            e.preventDefault();
          }}
        }}
      }});
    }}
  </script>

</body>
</html>
""".format(title=CONFIG["title"])


# ════════════════════════════════════════════════════════════════
#  6. pywebview js_api(系统级操作)
# ════════════════════════════════════════════════════════════════


class API:
    """前端通过 pywebview.api.xxx() 调这里, 只放系统集成."""

    def __init__(self) -> None:
        self._window: webview.Window | None = None

    def bind_window(self, window: webview.Window) -> None:
        self._window = window

    def pick_image(self) -> str | None:
        """弹原生文件选择对话框, 返回选中路径或 None."""
        assert self._window is not None
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=("图片 (*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tiff)", "全部 (*.*)"),
        )
        return result[0] if result else None

    def save_snapshot(self, base64_data: str) -> str | None:
        """弹保存对话框, 写盘, 返回写入路径或 None."""
        assert self._window is not None
        result = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=("PNG 图片 (*.png)",),
            save_filename="screenshot.png",
        )
        if not result:
            return None
        path = result[0]  # FileDialog.SAVE 返回 tuple, 单元素
        # 用户没写扩展名, 补 .png
        if not Path(path).suffix:
            path = path + ".png"
        try:
            data = base64.b64decode(base64_data)
            Path(path).write_bytes(data)
        except Exception as exc:
            raise RuntimeError(f"写入失败: {exc}") from exc
        return path

    def close_window(self) -> None:
        """Esc / Q 退出."""
        if self._window is not None:
            self._window.destroy()


# ════════════════════════════════════════════════════════════════
#  7. main: 起 uvicorn 线程 → 探活 → 开窗口
# ════════════════════════════════════════════════════════════════


def _run_server() -> None:
    config = uvicorn.Config(
        app,
        host=CONFIG["host"],
        port=CONFIG["port"],
        log_level="warning",
    )
    uvicorn.Server(config).run()


def _wait_server_ready(timeout: float = 10.0) -> None:
    url = f"http://{CONFIG['host']}:{CONFIG['port']}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"FastAPI 未在 {timeout}s 内启动: {url}")


def main() -> None:
    threading.Thread(target=_run_server, daemon=True).start()
    _wait_server_ready()
    api = API()
    window = webview.create_window(
        CONFIG["title"],
        url=f"http://{CONFIG['host']}:{CONFIG['port']}",
        js_api=api,
        width=CONFIG["width"],
        height=CONFIG["height"],
    )
    assert window is not None  # stub 标注可能为 None
    api.bind_window(window)
    webview.start()


if __name__ == "__main__":
    main()
