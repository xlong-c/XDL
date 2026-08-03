# tools/gui - pywebview + FastAPI 单文件 GUI 工具规范

## 目录职责

- 收容"一个 `.py` 文件 = 一个 GUI 工具"的脚本,统一使用 **pywebview(窗口)+ FastAPI(本地后端)** 实现
- 沉淀单文件模板,HTML 风格,跨平台约束,供所有 GUI 工具复用

## 架构总览

```
┌─────────────────────┐         ┌──────────────────────┐
│   pywebview 窗口     │  HTTP   │   FastAPI (uvicorn)   │
│   (系统原生 WebView) │ ──────> │   本地 127.0.0.1:port  │
│                     │ <────── │   业务 API + 静态/HTML  │
│   js_api: 系统对话框 │  JS桥   │                       │
└─────────────────────┘         └──────────────────────┘
       前端 HTML/JS                  Python 业务逻辑
```

**两种通信通道,按用途分工**:

| 通道 | 用途 |
|---|---|
| **HTTP / `fetch`** | 业务数据交互(CRUD,计算,调外部 API,读文件) |
| **pywebview `js_api`** | 系统级操作(文件选择对话框,消息框,窗口控制) |

业务调用走 HTTP 是为了享受 FastAPI 的路由,参数校验,`/docs` 调试页面;系统对话框走 js_api 是因为这些是桌面集成,不属于"业务 HTTP".

## 依赖与安装

```bash
pip install pywebview>=5.0 fastapi uvicorn
```

Linux 额外装 GTK WebKit 后端(Windows / macOS 零依赖):

```bash
# Ubuntu 24.04+ / Debian 12+ (webkit 4.1)
sudo apt install -y python3-gi gir1.2-webkit2-4.1 libwebkit2gtk-4.1-0

# Ubuntu 22.04 / Debian 11 (webkit 4.0)
sudo apt install -y python3-gi gir1.2-webkit2-4.0
```

> **conda 用户额外一步**:系统装的 `gi` 装在 `/usr/lib/python3/dist-packages/gi`,conda 环境的 Python 找不到.软链过去:
>
> ```bash
> ln -sf /usr/lib/python3/dist-packages/gi \
>        /root/miniconda3/envs/<env>/lib/python3.12/site-packages/gi
> ```
>
> 改 `<env>` 和 `python3.12` 为你实际的 conda 环境和 Python 版本.**不要用 `PYTHONPATH=/usr/lib/python3/dist-packages`**--会把系统老版本 `typing_extensions` 排到前面,pydantic 立即崩.
>
> **容器 / SSH / 无完整 D-Bus session 环境**:启动时会刷 `(process): dconf-WARNING **: failed to commit changes to dconf: Could not connect`.GTK 想存窗口几何,session dbus-daemon 没在跑.用 `dbus-run-session` 包一层:
>
> ```bash
> dbus-run-session python tools/gui/img_compare.py
> ```
>
> 或者强制 X11 backend 绕开 dconf(窗口几何不持久化):
>
> ```bash
> GDK_BACKEND=x11 python tools/gui/img_compare.py
> ```

工具级按需安装,不下沉到 `xdl` 核心包.

## 文件组织

### 强约束

1. **一个工具 = 一个 `.py` 文件**.HTML 字符串内嵌在 Python 中,不允许分离 `.html` / `.css` / `.js` 文件
2. **文件名禁止带 UI 框架后缀**:
   - ❌ `img_compare_GUI.py`,`img_compare_web.py`,`img_compare_v2.py`
   - ✅ `img_compare.py`
3. **同一业务工具只允许存在一个 GUI 实现**,用 git 找回历史,不再造 `*_v3.py`
4. **本目录只放"无业务归属"的通用 GUI 工具**(模板,示例,跨业务的可视化工具).有明确业务归属的工具应放在 `tools/image/`,`tools/setup/` 等子目录,仍然遵守本规范

### 目录结构

```
tools/gui/
├── AGENTS.md         # 本文档
├── _template.py      # 可复用模板(双击即跑)
└── README.md         # (可选)本目录工具列表
```

模板以 `_` 前缀命名,排在目录最前,不会被误当成业务工具.

## 单文件模板结构

每个 GUI 工具必须严格按以下顺序组织代码块:

```python
#!/usr/bin/env python3
"""<工具名> - 一句话说明."""
from __future__ import annotations

import threading
import time
from typing import Any

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ═══ 1. CONFIG(不引入命令行参数解析库)═══
CONFIG: dict[str, Any] = {
    "title": "工具名",
    "width": 1024,
    "height": 720,
    "host": "127.0.0.1",
    "port": 8765,
}

# ═══ 2. 业务函数(无 UI 依赖,可独立单测)═══
def do_thing(x: str) -> str:
    ...

# ═══ 3. FastAPI app + 路由 ═══
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML

@app.get("/api/run")
async def run(name: str = "world") -> dict[str, str]:
    return {"result": do_thing(name)}

# ═══ 4. HTML 模板(f-string)═══
HTML = """<!DOCTYPE html>..."""

# ═══ 5. pywebview js_api class(仅系统级操作)═══
class API:
    def pick_file(self): ...
    def alert(self, msg: str): ...

# ═══ 6. main():起 uvicorn 线程 → 等就绪 → 开窗口 ═══
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
    webview.start()
```

完整可跑骨架见 [`tools/gui/_template.py`](_template.py).

## 运行工具

不同环境的启动方式不一样:

| 环境 | 命令 |
|---|---|
| **Windows** | `python tools\gui\<tool>.py` |
| **macOS** | `python tools/gui/<tool>.py` |
| **Linux 完整桌面**(GNOME / KDE,有 D-Bus / portal / PipeWire) | `python tools/gui/<tool>.py` |
| **Linux 容器 / SSH / 无完整桌面 portal** | `./tools/gui/quiet.sh tools/gui/<tool>.py` |

`quiet.sh` 用 `dbus-run-session` 启动 session bus 消 dconf 警告,再统一 grep 过滤 xdg-desktop-portal / PipeWire / RealtimeKit / MESA ZINK / dbus-daemon 激活日志 / SpiRegistry 这些已知无害输出.**真 traceback 不受影响**--Python 错误信息不会匹配过滤模式.

只推荐"个人用 / 自己机".服务器 / 分发场景不要静默,让用户看到所有问题.


## HTML 风格约定

所有工具共享一致的视觉风格.

- **CDN 引入 Tailwind CSS**(不打包,不编译):
  ```html
  <script src="https://cdn.tailwindcss.com"></script>
  ```
  离线环境改为内置精简 CSS,但**默认走 CDN**
- **深色主题**:背景 `#1e1e1e`,面板 `#2d2d2d`,主色蓝 `#2563eb`,强调橙 `#cc5500`
- **业务调用统一用 `fetch`**:
  ```javascript
  async function callApi(path, params = {}) {
    const qs = new URLSearchParams(params).toString();
    const r = await fetch(`${path}${qs ? '?' + qs : ''}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  }
  ```
  后端约定返回 JSON,业务异常走 `r.ok` 检查
- **进度 / 日志用 `fetch` 轮询** 或 **WebSocket / SSE**(按需选择;简单工具用轮询即可)
- **系统对话框统一调 `pywebview.api.xxx`**:
  ```javascript
  const path = await pywebview.api.pick_file();
  ```
- **图标统一用 emoji 或 inline SVG**,不引入图标库
- **不要写外部 `.css` / `.js` 文件**

## 前后端通信

### 业务数据:HTTP `fetch`

```python
# 后端
@app.get("/api/process")
async def process(path: str, mode: str = "default") -> dict[str, Any]:
    return {"ok": True, "data": do_thing(path, mode)}
```

```javascript
// 前端
const r = await callApi("/api/process", { path: "...", mode: "fast" });
if (r.ok) show(r.data);
```

### 系统操作:pywebview `js_api`

```python
# 后端(不是 FastAPI 路由,是 webview 暴露给前端的对象)
class API:
    def pick_file(self) -> str | None:
        result = self._window.create_file_dialog(webview.OPEN_DIALOG)
        return result[0] if result else None

    def alert(self, msg: str) -> None:
        self._window.evaluate_js(f"show({msg!r})")
```

```javascript
// 前端
const path = await pywebview.api.pick_file();
if (path) {
  const r = await callApi("/api/load", { path });
  show(r.data);
}
```

### 图片 / 缩略图性能约定

图片浏览,批量查看,对比器这类工具,性能瓶颈通常不在"按钮逻辑",而在 **解码,缩放,重绘,翻页时机**.WebView 下如果处理得粗糙,常见症状是:

- 点下一页时先出来半页,另一半慢半拍
- 同一张图出现"上半部分先显示,下半部分后补"的分段感
- 缩放时格子尺寸变了,但图片内容没同步放大
- 每次翻页 / 缩放都重新 PIL 解码,导致明显卡顿

推荐约定:

1. **后端缩略图生成做内存缓存**
   - 同一张图,同一组参数(路径,mtime,文件大小,目标宽高,切片参数,背景色)应命中缓存
   - Python 侧优先用 `functools.lru_cache`
   - 缩略图 key 里要带 `mtime` / `size`,避免原图更新后继续吃旧缓存

2. **缩略图接口优先返回完整 bytes,不要默认走流式分块显示**
   - 对小到中等 PNG/JPEG 缩略图,优先 `Response(content=..., media_type=...)`
   - `StreamingResponse` 更适合大文件下载,真正的流式场景;缩略图场景下可能出现 WebView 一边收一边画的分段感

3. **缩放不要只改容器尺寸,要让图像本体同步缩放**
   - Pillow 的 `thumbnail()` 只会缩小,不会放大
   - 需要放大预览时,自己算目标比例后走 `resize()` / 等比 fit 逻辑

4. **前端请求尺寸要量化**
   - 不要让 `381px`,`384px`,`387px` 这种细碎尺寸各打一套缓存
   - 对缩略图宽高做 bucket 量化(例如按 16 / 32 像素取整),显著提高缓存复用率

5. **翻页时先预取,再切页**
   - 当前页渲染后,后台预取前后页缩略图
   - 翻页时优先命中浏览器缓存或前端内存缓存,避免"第一页刚清空,第二页才开始请求"

6. **可见页尽量整页 ready 后再替换,不要边到边补图**
   - 先把下一页缩略图 preload 完,再一次性挂到 DOM
   - 需要避免竞态时,前端维护 `renderToken` / version,新的渲染请求发出后,旧请求完成也不能回写页面

7. **对 WebView 明显出现分段显示的场景,优先用 `canvas` 一次性绘制**
   - `<img>` 在某些 WebView / 平台组合下会出现边解码边显示
   - 先 preload / decode 图片,再画到 `<canvas>`,通常比直接把 `<img>` 挂进页面更稳

8. **预加载不只等 `onload`,有条件时等 `img.decode()`**
   - `onload` 只表示资源到达,未必代表解码和首帧绘制成本已经消化完
   - 若运行环境支持 `HTMLImageElement.decode()`,应优先等待 decode 完成后再显示

9. **批量插入节点时用 `DocumentFragment`**
   - 翻页或重排时,不要循环里逐个 `appendChild` 到 live DOM
   - 先拼 fragment,最后一次性挂载,减少 layout / paint 抖动

10. **把性能相关逻辑收敛成通用 helper**
   - 例如:`build_thumb_url()`,`ensure_thumb_ready()`,`prefetch_page()`,`resize_to_fit()`
   - 不要把缓存 key 拼接,预取,decode 等细节散落在事件回调里

经验上,图片型 GUI 工具的优化顺序建议是:

1. 先修正确性:缩放逻辑是否真的放大了图像本体
2. 再压后端:缩略图缓存,线程化 PIL 处理
3. 再压前端:尺寸量化,fragment,预取前后页
4. 最后压显示链路:整页 ready 后切换,必要时改 `canvas`

### 长任务:HTTP 流式 / 轮询

pywebview 的 js_api 是同步阻塞的,长任务**禁止**放 js_api.两条路:

**路线 A:后端异步 + 前端轮询**(简单,推荐)

```python
# 任务状态存到内存 dict
TASKS: dict[str, dict] = {}

@app.post("/api/long/start")
async def long_start() -> dict[str, str]:
    tid = str(uuid.uuid4())
    TASKS[tid] = {"progress": 0.0, "done": False}
    threading.Thread(target=_run_long, args=(tid,), daemon=True).start()
    return {"task_id": tid}

@app.get("/api/long/status")
async def long_status(task_id: str) -> dict:
    return TASKS.get(task_id, {"done": True, "error": "unknown"})
```

```javascript
// 前端轮询
const { task_id } = await callApi("/api/long/start");
const timer = setInterval(async () => {
  const s = await callApi("/api/long/status", { task_id });
  setProgress(s.progress);
  if (s.done) { clearInterval(timer); show("完成"); }
}, 200);
```

**路线 B:SSE 流式**(适合进度密集,轮询太频繁)

```python
from sse_starlette.sse import EventSourceResponse

@app.get("/api/long/stream")
async def long_stream():
    async def gen():
        for i in range(100):
            yield {"event": "progress", "data": json.dumps({"p": i / 100})}
            await asyncio.sleep(0.1)
    return EventSourceResponse(gen())
```

```javascript
const es = new EventSource("/api/long/stream");
es.addEventListener("progress", (e) => setProgress(JSON.parse(e.data).p));
```

简单工具用 A,复杂工具再考虑 B.

## 跨平台行为差异

| 平台 | 后端 | 备注 |
|---|---|---|
| Windows | Edge WebView2 | Win10 1803+ / Win11 自带 |
| macOS | WKWebView | 系统自带 |
| Linux | GTK WebKit | 24.04+ 装 `gir1.2-webkit2-4.1`;22.04 装 `gir1.2-webkit2-4.0`;conda 用户还需把系统 `gi` 软链到 env site-packages |

文件路径统一用 `pathlib.Path`.HTTP 监听绑定 `127.0.0.1`(不要 `0.0.0.0`,避免暴露到局域网).

## 反例(不要做的事)

1. ❌ 写 `xxx_GUI.py` / `xxx_web.py` / `xxx_v2.py` 后缀
2. ❌ 拆出独立的 `.html` / `.css` / `.js` 文件
3. ❌ 用命令行参数解析库解析参数(违反 `tools/AGENTS.md`)
4. ❌ 把业务逻辑直接写在 `@app.get(...)` 路由里--业务函数应独立,纯 Python,可单测
5. ❌ 在 pywebview `js_api` 方法里跑耗时同步循环--会卡死前端
6. ❌ uvicorn 绑定 `0.0.0.0`--本地工具不要对外暴露
7. ❌ FastAPI 路由和 js_api 混用职责--业务走 HTTP,系统对话框走 js_api
8. ❌ 提交用户运行时配置(`*.json` / `.cache`)到 git

## 迁移指南

### 把 Tkinter / Flet 改写成 pywebview + FastAPI

1. 抽出**业务函数**(去掉 `tk.Button(...).pack()` 之类的 UI 代码)
2. 业务函数放进"业务函数"区
3. Tkinter 窗口布局改成 HTML + Tailwind 类名
4. `tk.Button(command=on_click).pack()` → `<button onclick="callApi('/api/on_click')">`
5. `tk.filedialog.askopenfilename()` → 前端 `pywebview.api.pick_file()`(业务侧再 `fetch('/api/load?path=...')`)
6. 启动方式:删 `tk.Tk().mainloop()` / Flet 的 `ft.app(target=main)`,改用模板的 `main()`(uvicorn 线程 + webview 窗口)

迁移后用 `git rm` 删除旧文件,**不留备份**--git 历史里能找回来.

## 关联资源

- [pywebview 官方文档](https://pywebview.flowrl.com/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- 模板:[`tools/gui/_template.py`](_template.py)
- 历史停用方案下的工具脚本(待迁移):`tools/image/img_compare_*.py`,`tools/setup/gpt_image_editor_*.py`,`tools/setup/hf_manager_*.py`
- 上级规范:[`tools/AGENTS.md`](../AGENTS.md)
