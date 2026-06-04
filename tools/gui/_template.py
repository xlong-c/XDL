#!/usr/bin/env python3
"""pywebview + FastAPI 单文件 GUI 模板 — 双击即跑。

按 tools/gui/AGENTS.md 规范组织。新建 GUI 工具时，复制本文件再改业务。

运行:
    python tools/gui/_template.py

依赖:
    pip install pywebview>=5.0 fastapi uvicorn
    Linux 额外: sudo apt install python3-gi gir1.2-webkit2-4.0

启动顺序:
    1. uvicorn 在后台线程起 FastAPI
    2. 主线程等 / 探活
    3. pywebview 开原生窗口, 加载 http://127.0.0.1:PORT
    4. 窗口关闭 → webview.start() 返回 → 进程退出
"""
from __future__ import annotations

import threading
import time
import urllib.request
import uuid
from typing import Any

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ════════════════════════════════════════════════════════════════
#  1. CONFIG（不引入命令行参数解析库，直接改这里）
# ════════════════════════════════════════════════════════════════

CONFIG: dict[str, Any] = {
    "title": "pywebview + FastAPI 模板",
    "width": 960,
    "height": 640,
    "host": "127.0.0.1",
    "port": 8765,
}

# ════════════════════════════════════════════════════════════════
#  2. 业务函数（无 UI 依赖，可独立单测）
# ════════════════════════════════════════════════════════════════


def do_thing(name: str) -> str:
    """示例: 纯业务逻辑, 返回字符串。"""
    return f"已处理: {name}"


# ════════════════════════════════════════════════════════════════
#  3. FastAPI app + 路由（业务数据走这里）
# ════════════════════════════════════════════════════════════════

app = FastAPI(title=CONFIG["title"])


# ── 页面 ──
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


# ── 业务 API ──
@app.get("/api/run")
async def run(name: str = "world") -> dict[str, str]:
    """同步业务调用: 前端 fetch, 后端直接返回结果。"""
    return {"ok": True, "result": do_thing(name)}


# ── 长任务: 内存任务表 + 轮询 ──
TASKS: dict[str, dict[str, Any]] = {}


@app.post("/api/long/start")
async def long_start() -> dict[str, str]:
    """启动长任务, 返回 task_id, 前端轮询 /api/long/status。"""
    tid = uuid.uuid4().hex

    def run() -> None:
        for i in range(20):
            TASKS[tid] = {"done": False, "progress": (i + 1) / 20,
                          "msg": f"步骤 {i + 1}/20"}
            time.sleep(0.1)
        TASKS[tid] = {"done": True, "progress": 1.0, "msg": "完成"}

    TASKS[tid] = {"done": False, "progress": 0.0, "msg": "开始"}
    threading.Thread(target=run, daemon=True).start()
    return {"task_id": tid}


@app.get("/api/long/status")
async def long_status(task_id: str) -> dict[str, Any]:
    """前端轮询长任务状态。"""
    return TASKS.get(task_id, {"done": True, "error": "unknown task"})


# ════════════════════════════════════════════════════════════════
#  4. HTML 模板（f-string, 可引用 CONFIG / Python 变量）
# ════════════════════════════════════════════════════════════════

HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ background:#1e1e1e; color:#e5e5e5;
         font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }}
  .panel {{ background:#2d2d2d; }}
  .btn {{ background:#2563eb; color:white; padding:8px 16px;
         border-radius:6px; cursor:pointer; border:none; }}
  .btn:hover {{ background:#1d4ed8; }}
  .btn-2 {{ background:#3d3d3d; }}
  .btn-2:hover {{ background:#505050; }}
  input.txt {{ background:#3d3d3d; color:white; padding:8px 12px;
              border-radius:6px; border:1px solid #525252; }}
  pre {{ background:#111; padding:12px; border-radius:6px;
        max-height:240px; overflow:auto; font-size:13px; }}
  #bar {{ background:#3d3d3d; height:8px; border-radius:4px; overflow:hidden; }}
  #bar > div {{ background:#2563eb; height:100%; width:0%; transition:width 0.2s; }}
</style>
</head>
<body class="p-6">

  <h1 class="text-2xl mb-4">🛠️ {title}</h1>

  <!-- 控制区 -->
  <div class="panel p-4 rounded-lg mb-4">
    <div class="flex gap-2 mb-3 flex-wrap items-center">
      <input id="name" class="txt" placeholder="输入名字" value="world">
      <button class="btn"   onclick="runSync()">同步执行</button>
      <button class="btn btn-2" onclick="pickFile()">选文件</button>
      <button class="btn btn-2" onclick="startLong()">长任务</button>
    </div>
    <div id="bar"><div></div></div>
  </div>

  <!-- 输出区 -->
  <div class="panel p-4 rounded-lg">
    <div class="text-sm text-zinc-400 mb-2">输出</div>
    <pre id="out">就绪</pre>
  </div>

  <script>
    // ── 全局 helper ──
    window.show = (msg) => {{
      document.getElementById('out').textContent = msg;
    }};
    window.log = (msg) => {{
      const out = document.getElementById('out');
      out.textContent += (out.textContent === '就绪' ? '' : '\\n') + msg;
      out.scrollTop = out.scrollHeight;
    }};
    window.setProgress = (p) => {{
      document.querySelector('#bar > div').style.width = (p * 100) + '%';
    }};

    // ── 业务 API 统一封装 ──
    async function callApi(path, params = {{}}, method = 'GET') {{
      const qs = new URLSearchParams(params).toString();
      const url = path + (qs ? '?' + qs : '');
      const r = await fetch(url, {{ method }});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    }}

    // ── 同步业务: 直接 fetch /api/run ──
    async function runSync() {{
      const name = document.getElementById('name').value;
      try {{
        const r = await callApi('/api/run', {{ name }});
        show(r.ok ? r.result : '失败: ' + JSON.stringify(r));
      }} catch (e) {{
        show('错误: ' + e.message);
      }}
    }}

    // ── 系统对话框: pywebview.js_api → 拿到路径再 fetch ──
    async function pickFile() {{
      const path = await pywebview.api.pick_file();
      if (path) show('已选: ' + path);
    }}

    // ── 长任务: 轮询 ──
    async function startLong() {{
      try {{
        const {{ task_id }} = await callApi('/api/long/start', {{}}, 'POST');
        const timer = setInterval(async () => {{
          const s = await callApi('/api/long/status', {{ task_id }});
          if (s.error) {{ clearInterval(timer); show('错误: ' + s.error); return; }}
          setProgress(s.progress);
          log(s.msg);
          if (s.done) clearInterval(timer);
        }}, 200);
      }} catch (e) {{
        show('启动失败: ' + e.message);
      }}
    }}
  </script>

</body>
</html>
""".format(title=CONFIG["title"])


# ════════════════════════════════════════════════════════════════
#  5. pywebview js_api class（仅系统级操作）
#
#  业务数据走 FastAPI HTTP, 这里只放系统集成（对话框、消息框等）。
# ════════════════════════════════════════════════════════════════


class API:
    """前端通过 pywebview.api.xxx() 调这里。"""

    def __init__(self) -> None:
        self._window: webview.Window | None = None

    def bind_window(self, window: webview.Window) -> None:
        self._window = window

    def pick_file(self) -> str | None:
        """弹原生文件选择对话框, 返回选中路径或 None。"""
        assert self._window is not None
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("图片 (*.png;*.jpg;*.jpeg)", "全部 (*.*)"),
        )
        return result[0] if result else None


# ════════════════════════════════════════════════════════════════
#  6. main: 起 uvicorn 线程 → 探活 → 开窗口
# ════════════════════════════════════════════════════════════════


def _run_server() -> None:
    """后台线程跑 uvicorn。"""
    config = uvicorn.Config(
        app,
        host=CONFIG["host"],
        port=CONFIG["port"],
        log_level="warning",
    )
    uvicorn.Server(config).run()


def _wait_server_ready(timeout: float = 10.0) -> None:
    """轮询 / 直到 FastAPI 起来, 或超时。"""
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
    # 1. 后台起服务
    threading.Thread(target=_run_server, daemon=True).start()
    # 2. 等就绪
    _wait_server_ready()
    # 3. 开窗口
    api = API()
    window = webview.create_window(
        CONFIG["title"],
        url=f"http://{CONFIG['host']}:{CONFIG['port']}",
        js_api=api,
        width=CONFIG["width"],
        height=CONFIG["height"],
    )
    api.bind_window(window)
    webview.start()


if __name__ == "__main__":
    main()
