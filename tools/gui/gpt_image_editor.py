#!/usr/bin/env python3
"""GPT 图像编辑器 - pywebview + FastAPI 单文件版.

按 tools/gui/AGENTS.md 规范组织.功能与原 NiceGUI 版一致:
  - 配置 OpenAI 兼容图像 API (providers, base_url, api_key, model, size)
  - 6 个内置预设 (换发/换装/背景替换/风格迁移/写实照片/动漫)
  - 文生图 (无图片) / 图像编辑 (1-2 张图)
  - 结果保存到本地 + 40 条历史记录
  - 持久化配置到 ~/.config/xdl-gpt-image-editor.json

启动:
    python tools/gui/gpt_image_editor.py

依赖:
    pip install pywebview>=5.0 fastapi uvicorn pillow requests
    Linux: sudo apt install python3-gi gir1.2-webkit2-4.1 libwebkit2gtk-4.1-0
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image

# ════════════════════════════════════════════════════════════════
#  1. CONFIG
# ════════════════════════════════════════════════════════════════

CONFIG: dict[str, Any] = {
    "title": "GPT 图像编辑器",
    "width": 1400,
    "height": 900,
    "host": "127.0.0.1",
    "port": 8766,
    "config_file": Path.home() / ".config" / "xdl-gpt-image-editor.json",
}

DEFAULT_PROVIDERS: list[dict[str, str]] = [
    {"name": "custom", "base_url": "", "api_key": "", "model": "gpt-image-2"},
    {"name": "openai", "base_url": "https://api.openai.com/v1",
     "api_key": "", "model": "gpt-image-1"},
]

DEFAULT_PRESETS: list[dict[str, str]] = [
    {"name": "换发", "prefix": "hairswap",
     "prompt": ("Take the person in the first image. Replace their hair with the "
                "hairstyle and hair color of the person in the second image. Keep "
                "everything else exactly the same: face, facial features, body, "
                "clothing, and background must remain unchanged.")},
    {"name": "换装", "prefix": "clothswap",
     "prompt": ("Take the person in the first image. Replace their clothing with "
                "the clothing worn by the person in the second image. Keep the "
                "person's face, body shape, pose, hairstyle, and background "
                "exactly the same. Only change the clothes.")},
    {"name": "背景替换", "prefix": "bgswap",
     "prompt": ("Take the person in the first image and place them into the "
                "background shown in the second image. Keep the person exactly as "
                "they are. Only replace the background with the scene from the "
                "second image. Make it look natural.")},
    {"name": "风格迁移", "prefix": "styletransfer",
     "prompt": ("Apply the artistic style, color palette, lighting, and visual "
                "aesthetic of the second image to the first image. Preserve "
                "content, subjects, and composition of the first image exactly.")},
    {"name": "写实照片", "prefix": "realistic",
     "prompt": ("A photorealistic, high-quality photograph. Natural lighting, "
                "sharp focus, professional composition, 8K resolution, highly "
                "detailed.")},
    {"name": "动漫风格", "prefix": "anime",
     "prompt": ("Anime and manga style illustration. Clean linework, vibrant "
                "colors, cel-shaded, high quality Japanese animation art style.")},
]

SIZE_OPTIONS: list[str] = [
    "1024x1024", "1024x1536", "1536x1024", "1792x1024", "1024x1792",
]

# ════════════════════════════════════════════════════════════════
#  2. 业务函数(无 UI 依赖,可独立单测)
# ════════════════════════════════════════════════════════════════


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        w, h = size_text.lower().split("x", 1)
        return max(1, int(w)), max(1, int(h))
    except Exception as exc:
        raise ValueError(f"非法尺寸: {size_text}") from exc


def ensure_dir(path_text: str) -> Path | None:
    raw = path_text.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def file_stem(name: str | None, default: str) -> str:
    if not name:
        return default
    stem = Path(name).stem.strip()
    return stem or default


def safe_write_png(image_bytes: bytes, path: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.convert("RGBA" if image.mode in ("RGBA", "LA") else "RGB").save(
            path, format="PNG"
        )


def image_to_png_bytes(file_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(file_bytes)) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def png_bytes_to_data_url(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def fetch_binary(url: str, timeout: int = 120) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def call_generation_api(
    *, base_url: str, api_key: str, model: str, size: str, prompt: str
) -> bytes:
    response = requests.post(
        base_url.rstrip("/") + "/images/generations",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model, "prompt": prompt, "size": size, "n": 1},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return _extract_image_bytes(payload)


def call_edit_api(
    *, base_url: str, api_key: str, model: str, size: str, prompt: str,
    image_payloads: list[dict[str, Any]],
) -> bytes:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for i, payload in enumerate(image_payloads, start=1):
        files.append(("image", (f"image{i}.png", payload["png_bytes"], "image/png")))
    response = requests.post(
        base_url.rstrip("/") + "/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data={"model": model, "prompt": prompt, "size": size, "n": "1"},
        files=files,
        timeout=180,
    )
    response.raise_for_status()
    return _extract_image_bytes(response.json())


def _extract_image_bytes(payload: dict) -> bytes:
    data = payload.get("data")
    entry = data[0] if isinstance(data, list) and data else data
    if not isinstance(entry, dict):
        raise ValueError("API 返回缺少 data")
    if entry.get("b64_json"):
        return base64.b64decode(entry["b64_json"])
    if entry.get("url"):
        return fetch_binary(entry["url"])
    raise ValueError("API 返回中没有 b64_json 或 url")


def save_outputs(
    *, output_dir: Path | None, result_bytes: bytes,
    image1_bytes: bytes | None, image2_bytes: bytes | None,
    preset_prefix: str, image1_name: str, image2_name: str, is_edit: bool,
) -> tuple[list[str], str]:
    if output_dir is None:
        return [], ""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    prefix = (preset_prefix or "result").strip() or "result"
    saved_paths: list[str] = []

    if is_edit and image1_bytes is not None:
        p = output_dir / f"{prefix}_{file_stem(image1_name, 'img1')}_img1_{timestamp}_{unique}.png"
        safe_write_png(image1_bytes, p)
        saved_paths.append(str(p))
    if is_edit and image2_bytes is not None:
        p = output_dir / f"{prefix}_{file_stem(image2_name, 'img2')}_img2_{timestamp}_{unique}.png"
        safe_write_png(image2_bytes, p)
        saved_paths.append(str(p))

    if is_edit:
        result_name = (f"{prefix}_{file_stem(image1_name, 'img')}_with_"
                       f"{file_stem(image2_name, 'ref')}_{timestamp}_{unique}.png")
    else:
        result_name = f"{prefix}_{timestamp}_{unique}.png"
    rp = output_dir / result_name
    safe_write_png(result_bytes, rp)
    saved_paths.append(str(rp))
    return saved_paths, str(output_dir)


def stitch_preview(parts: list[bytes | None], size: str) -> bytes:
    width, height = parse_size(size)
    valid = [p for p in parts if p is not None]
    canvas = Image.new(
        "RGB", (max(1, len(valid)) * width, height), color=(243, 244, 246)
    )
    for i, part in enumerate(valid):
        with Image.open(io.BytesIO(part)) as image:
            frame = image.convert("RGB").resize((width, height))
            canvas.paste(frame, (i * width, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════
#  3. STATE + 持久化(替换 NiceGUI 的 app.storage.user)
# ════════════════════════════════════════════════════════════════


def default_config() -> dict[str, Any]:
    return {
        "providers": [dict(p) for p in DEFAULT_PROVIDERS],
        "presets": [dict(p) for p in DEFAULT_PRESETS],
        "selected_provider": "custom",
        "selected_preset": 0,
        "base_url": "",
        "api_key": "",
        "model": "gpt-image-2",
        "img_size": "1024x1024",
        "output_dir": "./saveimages",
        "history": [],
        "max_history": 40,
    }


def load_config() -> dict[str, Any]:
    cfg_file: Path = CONFIG["config_file"]
    cfg = default_config()
    if cfg_file.is_file():
        try:
            saved = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for k, v in saved.items():
                    cfg[k] = v
        except Exception:
            pass
    # 兜底: providers / presets 必须是 list
    if not isinstance(cfg.get("providers"), list) or not cfg["providers"]:
        cfg["providers"] = [dict(p) for p in DEFAULT_PROVIDERS]
    if not isinstance(cfg.get("presets"), list) or not cfg["presets"]:
        cfg["presets"] = [dict(p) for p in DEFAULT_PRESETS]
    if not isinstance(cfg.get("history"), list):
        cfg["history"] = []
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    cfg_file: Path = CONFIG["config_file"]
    try:
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    except Exception as exc:
        print(f"[warn] 持久化配置失败: {exc}")


@dataclass
class ImagePayload:
    name: str
    png_bytes: bytes
    preview_data_url: str


class State:
    """集中管理运行期状态.config 持久化, 业务数据走内存."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = load_config()
        self.images: dict[int, ImagePayload | None] = {1: None, 2: None}
        self.current_result: dict[str, Any] | None = None  # 最近一次结果

    # ── config 持久化 ──
    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        self.config.update(patch)
        save_config(self.config)
        return self.config

    # ── provider / preset CRUD ──
    def add_provider(self, data: dict[str, str]) -> dict[str, Any]:
        name = data.get("name", "").strip()
        if not name:
            raise ValueError("Provider 名称不能为空")
        providers = [p for p in self.config["providers"] if p.get("name") != name]
        providers.append({"name": name, "base_url": data.get("base_url", "").strip(),
                          "api_key": data.get("api_key", "").strip(),
                          "model": data.get("model", "gpt-image-2").strip()})
        self.config["providers"] = providers
        self.config["selected_provider"] = name
        save_config(self.config)
        return self.config

    def edit_provider(self, original_name: str, data: dict[str, str]) -> dict[str, Any]:
        name = data.get("name", "").strip()
        if not name:
            raise ValueError("Provider 名称不能为空")
        new_prov = {"name": name, "base_url": data.get("base_url", "").strip(),
                    "api_key": data.get("api_key", "").strip(),
                    "model": data.get("model", "gpt-image-2").strip()}
        providers = []
        for p in self.config["providers"]:
            if p.get("name") == original_name:
                providers.append(new_prov)
            elif p.get("name") != name:
                providers.append(p)
        self.config["providers"] = providers
        self.config["selected_provider"] = name
        save_config(self.config)
        return self.config

    def delete_provider(self, name: str) -> dict[str, Any]:
        if len(self.config["providers"]) <= 1:
            raise ValueError("至少保留一个 Provider")
        self.config["providers"] = [p for p in self.config["providers"]
                                    if p.get("name") != name]
        if self.config.get("selected_provider") == name:
            self.config["selected_provider"] = self.config["providers"][0]["name"]
        save_config(self.config)
        return self.config

    def add_preset(self, data: dict[str, str]) -> dict[str, Any]:
        name = data.get("name", "").strip()
        if not name:
            raise ValueError("预设名称不能为空")
        self.config["presets"].append({
            "name": name,
            "prefix": data.get("prefix", "result").strip() or "result",
            "prompt": data.get("prompt", "").strip(),
        })
        self.config["selected_preset"] = len(self.config["presets"]) - 1
        save_config(self.config)
        return self.config

    def edit_preset(self, index: int, data: dict[str, str]) -> dict[str, Any]:
        name = data.get("name", "").strip()
        if not name:
            raise ValueError("预设名称不能为空")
        if not 0 <= index < len(self.config["presets"]):
            raise ValueError("预设索引越界")
        self.config["presets"][index] = {
            "name": name,
            "prefix": data.get("prefix", "result").strip() or "result",
            "prompt": data.get("prompt", "").strip(),
        }
        save_config(self.config)
        return self.config

    def delete_preset(self, index: int) -> dict[str, Any]:
        if len(self.config["presets"]) <= 1:
            raise ValueError("至少保留一个预设")
        if not 0 <= index < len(self.config["presets"]):
            raise ValueError("预设索引越界")
        del self.config["presets"][index]
        self.config["selected_preset"] = max(0, min(
            self.config.get("selected_preset", 0), len(self.config["presets"]) - 1
        ))
        save_config(self.config)
        return self.config

    # ── 图片 ──
    def set_image(self, side: int, path: str) -> dict[str, Any]:
        if side not in (1, 2):
            raise ValueError("side 必须 1 或 2")
        if not Path(path).is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, "rb") as f:
            raw = f.read()
        png_bytes = image_to_png_bytes(raw)
        payload = ImagePayload(
            name=Path(path).name,
            png_bytes=png_bytes,
            preview_data_url=png_bytes_to_data_url(png_bytes),
        )
        self.images[side] = payload
        return self._image_view(side)

    def clear_image(self, side: int) -> dict[str, Any]:
        self.images[side] = None
        return {"side": side, "cleared": True}

    def _image_view(self, side: int) -> dict[str, Any]:
        p = self.images.get(side)
        return {
            "side": side,
            "loaded": p is not None,
            "name": p.name if p else None,
            "preview": p.preview_data_url if p else None,
        }

    # ── 历史 ──
    def push_history(self, entry: dict[str, Any]) -> dict[str, Any]:
        history = self.config.setdefault("history", [])
        history.insert(0, entry)
        max_n = int(self.config.get("max_history", 40))
        self.config["history"] = history[:max_n]
        save_config(self.config)
        return self.config

    def clear_history(self) -> dict[str, Any]:
        self.config["history"] = []
        save_config(self.config)
        return self.config

    # ── 给前端的全量状态 ──
    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "image1": self._image_view(1),
            "image2": self._image_view(2),
            "current_result": self.current_result,
            "size_options": SIZE_OPTIONS,
        }


STATE = State()
TASKS: dict[str, dict[str, Any]] = {}  # task_id -> {status, progress, result, error}

# ════════════════════════════════════════════════════════════════
#  4. FastAPI app + 路由
# ════════════════════════════════════════════════════════════════

app = FastAPI(title=CONFIG["title"])


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return STATE.snapshot()


@app.post("/api/config")
async def update_config(patch: dict[str, Any]) -> dict[str, Any]:
    return STATE.update_config(patch)


@app.post("/api/provider/add")
async def add_provider(data: dict[str, str]) -> dict[str, Any]:
    return STATE.add_provider(data)


@app.post("/api/provider/edit")
async def edit_provider(original_name: str, data: dict[str, str]) -> dict[str, Any]:
    return STATE.edit_provider(original_name, data)


@app.post("/api/provider/delete")
async def delete_provider(name: str) -> dict[str, Any]:
    return STATE.delete_provider(name)


@app.post("/api/preset/add")
async def add_preset(data: dict[str, str]) -> dict[str, Any]:
    return STATE.add_preset(data)


@app.post("/api/preset/edit")
async def edit_preset(index: int, data: dict[str, str]) -> dict[str, Any]:
    return STATE.edit_preset(index, data)


@app.post("/api/preset/delete")
async def delete_preset(index: int) -> dict[str, Any]:
    return STATE.delete_preset(index)


@app.post("/api/image/set")
async def set_image(side: int, path: str) -> dict[str, Any]:
    return STATE.set_image(side, path)


@app.post("/api/image/clear")
async def clear_image(side: int) -> dict[str, Any]:
    return STATE.clear_image(side)


@app.post("/api/history/clear")
async def clear_history() -> dict[str, Any]:
    return STATE.clear_history()


# ── 长任务: 启动 + 轮询 ──
def _run_generate(task_id: str, params: dict[str, Any]) -> None:
    try:
        TASKS[task_id] = {"status": "running", "progress": 0.1,
                           "msg": "正在调用远端图像接口..."}

        # 决定 is_edit
        imgs = []
        if params.get("image1_path"):
            try:
                with open(params["image1_path"], "rb") as f:
                    raw = f.read()
                p1_bytes = image_to_png_bytes(raw)
                imgs.append({"name": Path(params["image1_path"]).name,
                             "png_bytes": p1_bytes})
            except Exception as exc:
                raise RuntimeError(f"读取图片1失败: {exc}") from exc
        if params.get("image2_path"):
            try:
                with open(params["image2_path"], "rb") as f:
                    raw = f.read()
                p2_bytes = image_to_png_bytes(raw)
                imgs.append({"name": Path(params["image2_path"]).name,
                             "png_bytes": p2_bytes})
            except Exception as exc:
                raise RuntimeError(f"读取图片2失败: {exc}") from exc

        is_edit = bool(imgs)
        if is_edit and not imgs[0].get("png_bytes"):
            raise ValueError("编辑模式至少需要图片1")

        # 调 API
        TASKS[task_id] = {"status": "running", "progress": 0.4,
                           "msg": "等待 API 响应 (可能 30-180s)..."}
        if is_edit:
            result_bytes = call_edit_api(
                base_url=params["base_url"], api_key=params["api_key"],
                model=params["model"], size=params["size"],
                prompt=params["prompt"], image_payloads=imgs,
            )
        else:
            result_bytes = call_generation_api(
                base_url=params["base_url"], api_key=params["api_key"],
                model=params["model"], size=params["size"],
                prompt=params["prompt"],
            )

        TASKS[task_id] = {"status": "running", "progress": 0.8,
                           "msg": "保存输出..."}
        out_dir = ensure_dir(params.get("output_dir", ""))
        preset_prefix = params.get("preset_prefix", "result")
        saved_files, saved_dir = save_outputs(
            output_dir=out_dir, result_bytes=result_bytes,
            image1_bytes=imgs[0]["png_bytes"] if imgs else None,
            image2_bytes=imgs[1]["png_bytes"] if len(imgs) > 1 else None,
            preset_prefix=preset_prefix,
            image1_name=imgs[0]["name"] if imgs else "img1.png",
            image2_name=imgs[1]["name"] if len(imgs) > 1 else "img2.png",
            is_edit=is_edit,
        )

        # 拼预览
        if is_edit:
            stitch_parts: list[bytes | None] = [
                imgs[0]["png_bytes"],
                imgs[1]["png_bytes"] if len(imgs) > 1 else None,
                result_bytes,
            ]
        else:
            stitch_parts: list[bytes | None] = [result_bytes]
        stitched = stitch_preview(stitch_parts, params["size"])

        bundle = {
            "stitched_b64": base64.b64encode(stitched).decode("ascii"),
            "result_b64": base64.b64encode(result_bytes).decode("ascii"),
            "image1_b64": base64.b64encode(imgs[0]["png_bytes"]).decode("ascii") if imgs else None,
            "image2_b64": base64.b64encode(imgs[1]["png_bytes"]).decode("ascii") if len(imgs) > 1 else None,
            "is_edit": is_edit,
            "saved_files": saved_files,
            "saved_dir": saved_dir,
            "title": f"[{params.get('preset_name', '未命名')}]",
            "status_message": (f"完成: 已保存 {len(saved_files)} 个文件到 {saved_dir}"
                                if saved_files else "完成: 未落盘,仅保留页面结果"),
        }
        STATE.current_result = bundle

        # 入历史
        STATE.push_history({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": bundle["title"],
            "is_edit": is_edit,
            "saved_message": bundle["status_message"].replace("完成: ", "", 1),
            "stitched_b64": bundle["stitched_b64"],
            "result_b64": bundle["result_b64"],
            "image1_b64": bundle["image1_b64"],
            "image2_b64": bundle["image2_b64"],
        })

        TASKS[task_id] = {"status": "done", "progress": 1.0, "result": bundle}
    except Exception as exc:
        TASKS[task_id] = {"status": "error", "error": str(exc)}


@app.post("/api/generate")
async def start_generate(params: dict[str, Any]) -> dict[str, str]:
    if not params.get("base_url"):
        raise ValueError("请输入 Base URL")
    if not params.get("api_key"):
        raise ValueError("请输入 API Key")
    if not params.get("prompt"):
        raise ValueError("请输入提示词")
    parse_size(params.get("size", "1024x1024"))
    STATE.update_config({
        "selected_provider": params.get("provider_name"),
        "selected_preset": params.get("preset_index"),
        "base_url": params["base_url"],
        "api_key": params["api_key"],
        "model": params["model"],
        "img_size": params["size"],
        "output_dir": params.get("output_dir", ""),
    })
    tid = uuid.uuid4().hex
    TASKS[tid] = {"status": "queued", "progress": 0.0, "msg": "排队中..."}
    threading.Thread(target=_run_generate, args=(tid, params), daemon=True).start()
    return {"task_id": tid}


@app.get("/api/task/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    return TASKS.get(task_id, {"status": "unknown"})


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
  body {{ background:#101216; color:#e6e9ef;
         font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin:0; min-height:100vh; }}
  .card {{ background:#222730; border:1px solid #303744; border-radius:8px; padding:14px; }}
  .panel-title {{ font-size:12px; font-weight:700; color:#98a2b3;
                 text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px; }}
  .muted {{ color:#98a2b3; font-size:12px; }}
  .btn {{ background:#3b82f6; color:white; padding:6px 12px;
         border-radius:6px; border:none; cursor:pointer; font-size:13px; }}
  .btn:hover {{ background:#2563eb; }}
  .btn-ghost {{ background:#3d3d3d; color:#e6e9ef; }}
  .btn-ghost:hover {{ background:#505050; }}
  .btn-danger {{ background:#ef4444; color:white; }}
  .btn-danger:hover {{ background:#dc2626; }}
  .btn-icon {{ padding:4px 8px; font-size:12px; }}
  .input, .select, textarea {{ background:#191d24; color:#e6e9ef;
         border:1px solid #303744; border-radius:6px; padding:6px 10px;
         font-size:13px; width:100%; }}
  textarea {{ min-height:120px; font-family:inherit; resize:vertical; }}
  .preview {{ width:100%; height:172px; border:1px dashed #4b5563;
             border-radius:8px; background:#141922; overflow:hidden;
             display:flex; align-items:center; justify-content:center;
             position:relative; }}
  .preview img {{ max-width:100%; max-height:100%; object-fit:contain; }}
  .preview .empty {{ color:#4b5563; font-size:13px; }}
  .result-stage {{ background:#0b0d11; border:1px solid #303744;
                  border-radius:8px; min-height:480px;
                  display:flex; align-items:center; justify-content:center;
                  padding:20px; }}
  .result-stage img {{ max-width:100%; max-height:600px; object-fit:contain; }}
  .placeholder {{ color:#98a2b3; text-align:center; white-space:pre-line; line-height:1.8; }}
  .tab {{ padding:10px 16px; cursor:pointer; color:#98a2b3;
         border-bottom:2px solid transparent; font-size:13px; }}
  .tab.active {{ color:#e6e9ef; border-bottom-color:#3b82f6; }}
  .history-card {{ background:#222730; border:1px solid #303744; border-radius:8px;
                  padding:10px; cursor:pointer; }}
  .history-card:hover {{ border-color:#3b82f6; }}
  .history-card img {{ width:100%; height:120px; object-fit:cover;
                       border-radius:4px; margin-bottom:6px; }}
  .modal {{ position:fixed; inset:0; background:rgba(0,0,0,0.6);
           display:flex; align-items:center; justify-content:center; z-index:50; }}
  .modal .box {{ background:#222730; border:1px solid #303744;
                border-radius:8px; padding:20px; max-width:560px; width:calc(100% - 32px); }}
  .toast {{ position:fixed; top:16px; right:16px; padding:10px 16px;
           border-radius:6px; font-size:13px; z-index:60; }}
  .toast.success {{ background:#16a34a; color:white; }}
  .toast.error {{ background:#ef4444; color:white; }}
  .toast.info {{ background:#3b82f6; color:white; }}
  .progress {{ background:#303744; height:6px; border-radius:3px; overflow:hidden; }}
  .progress > div {{ background:#3b82f6; height:100%; transition:width 0.3s; }}
  .row {{ display:flex; gap:8px; align-items:center; }}
</style>
</head>
<body>

<div class="flex p-4 gap-4" style="min-height:100vh; align-items:flex-start;">
  <!-- 左侧栏 -->
  <div class="w-[420px] min-w-[360px] flex flex-col gap-3" id="sidebar">

    <!-- API 配置 -->
    <div class="card">
      <div class="panel-title">API 配置</div>
      <div class="row">
        <select id="provider-select" class="select" style="flex:1" onchange="onProviderChange()"></select>
        <button class="btn btn-ghost btn-icon" onclick="openProviderDialog()">+ 添加</button>
        <button class="btn btn-ghost btn-icon" onclick="editCurrentProvider()">编辑</button>
        <button class="btn btn-danger btn-icon" onclick="deleteCurrentProvider()">删</button>
      </div>
      <div class="mt-2"><input id="base-url" class="input" placeholder="https://api.example.com/v1" oninput="saveFormConfig()"></div>
      <div class="mt-2"><input id="api-key" class="input" type="password" placeholder="sk-..." oninput="saveFormConfig()"></div>
      <div class="mt-2 row">
        <input id="model" class="input" placeholder="gpt-image-2" oninput="saveFormConfig()">
        <select id="img-size" class="select" style="width:140px" onchange="saveFormConfig()"></select>
      </div>
    </div>

    <!-- 图片 1 -->
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div class="panel-title" style="margin:0">图片 1 (主图)</div>
        <button id="clear-1-btn" class="btn btn-ghost btn-icon" style="display:none" onclick="clearImage(1)">清空</button>
      </div>
      <div class="preview mt-2" onclick="pickAndSetImage(1)">
        <div id="preview-1-empty" class="empty">点击上传主图</div>
        <img id="preview-1" style="display:none">
      </div>
      <div class="muted mt-1" id="image-1-name"></div>
    </div>

    <!-- 图片 2 -->
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div class="panel-title" style="margin:0">图片 2 (参考图, 可选)</div>
        <button id="clear-2-btn" class="btn btn-ghost btn-icon" style="display:none" onclick="clearImage(2)">清空</button>
      </div>
      <div class="preview mt-2" onclick="pickAndSetImage(2)">
        <div id="preview-2-empty" class="empty">点击上传参考图</div>
        <img id="preview-2" style="display:none">
      </div>
      <div class="muted mt-1" id="image-2-name"></div>
    </div>

    <!-- 提示词 -->
    <div class="card">
      <div class="panel-title">提示词</div>
      <div class="row">
        <select id="preset-select" class="select" style="flex:1" onchange="onPresetChange()"></select>
        <button class="btn btn-ghost btn-icon" onclick="openPresetDialog()">+ 添加</button>
        <button class="btn btn-ghost btn-icon" onclick="editCurrentPreset()">编辑</button>
        <button class="btn btn-danger btn-icon" onclick="deleteCurrentPreset()">删</button>
      </div>
      <textarea id="prompt" class="mt-2" oninput="saveFormConfig()"></textarea>
      <div class="row mt-2" style="justify-content:space-between">
        <button class="btn btn-ghost btn-icon" onclick="restorePreset()">恢复预设</button>
        <span class="muted">有图走编辑, 无图走文生图</span>
      </div>
    </div>

    <!-- 输出 -->
    <div class="card">
      <div class="panel-title">输出</div>
      <input id="output-dir" class="input" placeholder="./saveimages" oninput="saveFormConfig()">
      <div class="muted mt-1">留空则不落盘, 只保留页面下载</div>
    </div>

    <!-- 生成 -->
    <button id="generate-btn" class="btn" style="height:48px; font-size:15px; font-weight:700" onclick="generate()">生成</button>
    <div id="status" class="muted">就绪</div>
    <div class="progress"><div id="progress-bar" style="width:0%"></div></div>
  </div>

  <!-- 右侧主区 -->
  <div class="flex-1 min-w-0 flex flex-col gap-3">
    <div class="row" style="border-bottom:1px solid #303744">
      <div id="tab-result" class="tab active" onclick="switchTab('result')">结果</div>
      <div id="tab-history" class="tab" onclick="switchTab('history')">历史</div>
      <div class="flex-1"></div>
      <button id="download-stitched" class="btn btn-ghost" style="display:none" onclick="downloadKind('stitched')">保存拼接图</button>
      <button id="download-img1" class="btn btn-ghost" style="display:none" onclick="downloadKind('img1')">图1</button>
      <button id="download-img2" class="btn btn-ghost" style="display:none" onclick="downloadKind('img2')">图2</button>
      <button id="download-result" class="btn btn-ghost" style="display:none" onclick="downloadKind('result')">结果图</button>
    </div>

    <div id="pane-result" class="result-stage">
      <div class="placeholder">生成结果会显示在这里

编辑模式会拼接显示: 图1 / 图2 / 结果</div>
    </div>

    <div id="pane-history" style="display:none">
      <div class="row" style="margin-bottom:10px">
        <span class="muted" id="history-info"></span>
        <div class="flex-1"></div>
        <button class="btn btn-danger" onclick="clearHistory()">清空历史</button>
      </div>
      <div id="history-list" class="grid" style="display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px"></div>
    </div>
  </div>
</div>

<!-- 通用对话框 -->
<div id="modal" class="modal" style="display:none" onclick="if(event.target===this)closeModal()">
  <div class="box" id="modal-body"></div>
</div>

<div id="toast"></div>

<script>
  let data = {{
    config: null, image1: null, image2: null, current_result: null,
    size_options: [],
  }};
  let currentTab = 'result';

  async function callApi(path, params = {{}}, method = 'GET') {{
    const qs = new URLSearchParams(params).toString();
    const url = path + (qs ? '?' + qs : '');
    const r = await fetch(url, {{ method, headers: {{ 'Content-Type': 'application/json' }},
      body: method === 'GET' ? undefined : JSON.stringify(params) }});
    if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + (await r.text().catch(() => '')));
    return await r.json();
  }}

  function toast(msg, type = 'info') {{
    const t = document.getElementById('toast');
    t.className = 'toast ' + type;
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 3500);
  }}

  function setStatus(msg) {{ document.getElementById('status').textContent = msg; }}

  function setProgress(p) {{
    document.getElementById('progress-bar').style.width = (p * 100) + '%';
  }}

  // ── 启动: 拉状态, 装表单 ──
  window.addEventListener('pywebviewready', async () => {{
    data = await callApi('/api/state');
    rebuildForm();
  }});

  function rebuildForm() {{
    const cfg = data.config;
    // provider select
    const ps = document.getElementById('provider-select');
    ps.innerHTML = '';
    cfg.providers.forEach(p => {{
      const o = document.createElement('option');
      o.value = p.name; o.textContent = p.name;
      ps.appendChild(o);
    }});
    ps.value = cfg.selected_provider || cfg.providers[0].name;
    // base_url / api_key / model: 跟随 provider, 优先用 cfg 当前值(用户可能手动改过)
    document.getElementById('base-url').value = cfg.base_url || '';
    document.getElementById('api-key').value = cfg.api_key || '';
    document.getElementById('model').value = cfg.model || 'gpt-image-2';
    // size
    const ss = document.getElementById('img-size');
    ss.innerHTML = '';
    data.size_options.forEach(s => {{
      const o = document.createElement('option'); o.value = s; o.textContent = s;
      ss.appendChild(o);
    }});
    ss.value = cfg.img_size || '1024x1024';
    // preset
    const prs = document.getElementById('preset-select');
    prs.innerHTML = '';
    cfg.presets.forEach((p, i) => {{
      const o = document.createElement('option'); o.value = i; o.textContent = p.name;
      prs.appendChild(o);
    }});
    prs.value = cfg.selected_preset || 0;
    document.getElementById('prompt').value = (cfg.presets[prs.value] || {{}}).prompt || '';
    // output dir
    document.getElementById('output-dir').value = cfg.output_dir || '';
    // images
    applyImageView(1, data.image1);
    applyImageView(2, data.image2);
    // history
    renderHistory();
    // current result
    if (data.current_result) applyResultView(data.current_result);
  }}

  function saveFormConfig() {{
    callApi('/api/config', {{
      selected_provider: document.getElementById('provider-select').value,
      base_url: document.getElementById('base-url').value,
      api_key: document.getElementById('api-key').value,
      model: document.getElementById('model').value,
      img_size: document.getElementById('img-size').value,
      selected_preset: parseInt(document.getElementById('preset-select').value),
      output_dir: document.getElementById('output-dir').value,
    }}).catch(() => {{}});
  }}

  function onProviderChange() {{
    const name = document.getElementById('provider-select').value;
    const p = data.config.providers.find(x => x.name === name);
    if (p) {{
      document.getElementById('base-url').value = p.base_url || '';
      document.getElementById('api-key').value = p.api_key || '';
      document.getElementById('model').value = p.model || 'gpt-image-2';
    }}
    saveFormConfig();
  }}

  function onPresetChange() {{
    const i = parseInt(document.getElementById('preset-select').value);
    const p = data.config.presets[i];
    if (p) document.getElementById('prompt').value = p.prompt || '';
    saveFormConfig();
  }}

  function restorePreset() {{ onPresetChange(); }}

  // ── 图片 ──
  async function pickAndSetImage(side) {{
    const path = await pywebview.api.pick_image();
    if (!path) return;
    try {{
      const view = await callApi('/api/image/set', {{ side, path }}, 'POST');
      data['image' + side] = view;
      applyImageView(side, view);
    }} catch (e) {{ toast('加载失败: ' + e.message, 'error'); }}
  }}

  async function clearImage(side) {{
    try {{
      await callApi('/api/image/clear', {{ side }}, 'POST');
      data['image' + side] = {{ loaded: false }};
      applyImageView(side, data['image' + side]);
    }} catch (e) {{ toast('清空失败: ' + e.message, 'error'); }}
  }}

  function applyImageView(side, view) {{
    document.getElementById('preview-' + side).style.display = view.loaded ? 'block' : 'none';
    document.getElementById('preview-' + side + '-empty').style.display = view.loaded ? 'none' : 'flex';
    document.getElementById('clear-' + side + '-btn').style.display = view.loaded ? 'inline-block' : 'none';
    if (view.loaded) {{
      document.getElementById('preview-' + side).src = view.preview;
      document.getElementById('image-' + side + '-name').textContent = view.name;
    }} else {{
      document.getElementById('image-' + side + '-name').textContent = '';
    }}
  }}

  // ── 生成 ──
  async function generate() {{
    const ps = parseInt(document.getElementById('preset-select').value);
    const preset = data.config.presets[ps];
    const params = {{
      base_url: document.getElementById('base-url').value.trim(),
      api_key: document.getElementById('api-key').value.trim(),
      model: document.getElementById('model').value.trim() || 'gpt-image-2',
      size: document.getElementById('img-size').value,
      prompt: document.getElementById('prompt').value.trim(),
      output_dir: document.getElementById('output-dir').value.trim(),
      image1_path: data.image1.loaded ? (data.image1.path || '') : '',
      image2_path: data.image2.loaded ? (data.image2.path || '') : '',
      provider_name: document.getElementById('provider-select').value,
      preset_index: ps,
      preset_name: preset ? preset.name : '未命名',
      preset_prefix: preset ? preset.prefix : 'result',
    }};
    document.getElementById('generate-btn').disabled = true;
    setStatus('启动中...'); setProgress(0);
    try {{
      const {{ task_id }} = await callApi('/api/generate', params, 'POST');
      pollTask(task_id);
    }} catch (e) {{
      toast(e.message, 'error'); setStatus('错误: ' + e.message);
      document.getElementById('generate-btn').disabled = false;
    }}
  }}

  async function pollTask(tid) {{
    const timer = setInterval(async () => {{
      try {{
        const s = await callApi('/api/task/' + tid);
        if (s.progress !== undefined) setProgress(s.progress);
        if (s.msg) setStatus(s.msg);
        if (s.status === 'done') {{
          clearInterval(timer);
          document.getElementById('generate-btn').disabled = false;
          applyResultView(s.result);
          data = await callApi('/api/state');  // 刷新历史
          renderHistory();
          setStatus(s.result.status_message);
          setProgress(1.0);
          toast('生成完成', 'success');
        }} else if (s.status === 'error') {{
          clearInterval(timer);
          document.getElementById('generate-btn').disabled = false;
          setProgress(0);
          setStatus('错误: ' + s.error);
          toast(s.error, 'error');
        }}
      }} catch (e) {{
        clearInterval(timer); document.getElementById('generate-btn').disabled = false;
        setStatus('轮询失败: ' + e.message);
      }}
    }}, 500);
  }}

  function applyResultView(bundle) {{
    data.current_result = bundle;
    document.getElementById('pane-result').innerHTML =
      `<img src="data:image/png;base64,${{bundle.stitched_b64}}">`;
    document.getElementById('download-stitched').style.display = 'inline-block';
    document.getElementById('download-result').style.display = 'inline-block';
    document.getElementById('download-img1').style.display = bundle.image1_b64 ? 'inline-block' : 'none';
    document.getElementById('download-img2').style.display = bundle.image2_b64 ? 'inline-block' : 'none';
    switchTab('result');
  }}

  async function downloadKind(kind) {{
    if (!data.current_result) return;
    let b64, fname;
    if (kind === 'stitched') {{ b64 = data.current_result.stitched_b64; fname = 'stitched.png'; }}
    else if (kind === 'result') {{ b64 = data.current_result.result_b64; fname = 'result.png'; }}
    else if (kind === 'img1') {{ b64 = data.current_result.image1_b64; fname = 'img1.png'; }}
    else if (kind === 'img2') {{ b64 = data.current_result.image2_b64; fname = 'img2.png'; }}
    if (!b64) {{ toast('当前没有该图片', 'error'); return; }}
    try {{
      const path = await pywebview.api.save_image(b64, fname);
      if (path) toast('已保存: ' + path, 'success');
    }} catch (e) {{ toast('保存失败: ' + e.message, 'error'); }}
  }}

  // ── 标签页 ──
  function switchTab(name) {{
    currentTab = name;
    document.getElementById('tab-result').classList.toggle('active', name === 'result');
    document.getElementById('tab-history').classList.toggle('active', name === 'history');
    document.getElementById('pane-result').style.display = name === 'result' ? 'flex' : 'none';
    document.getElementById('pane-history').style.display = name === 'history' ? 'block' : 'none';
  }}

  function renderHistory() {{
    const history = (data.config && data.config.history) || [];
    document.getElementById('history-info').textContent = `共 ${{history.length}} 条历史`;
    const list = document.getElementById('history-list');
    list.innerHTML = '';
    if (!history.length) {{
      list.innerHTML = '<div class="muted" style="grid-column:1/-1; text-align:center; padding:40px">暂无历史记录</div>';
      return;
    }}
    history.forEach((h, i) => {{
      const card = document.createElement('div');
      card.className = 'history-card';
      card.innerHTML = `
        <img src="data:image/png;base64,${{h.stitched_b64}}">
        <div style="font-size:13px; font-weight:500">${{h.title || '未命名'}}</div>
        <div class="muted" style="font-size:11px">${{h.timestamp}} | ${{h.is_edit ? '编辑' : '生成'}}</div>
      `;
      card.onclick = () => applyResultView({{
        stitched_b64: h.stitched_b64, result_b64: h.result_b64,
        image1_b64: h.image1_b64, image2_b64: h.image2_b64,
        is_edit: h.is_edit, title: h.title,
        status_message: '已载入历史: ' + (h.title || ''),
      }});
      list.appendChild(card);
    }});
  }}

  async function clearHistory() {{
    if (!confirm('确定清空所有历史记录?')) return;
    await callApi('/api/history/clear', {{}}, 'POST');
    data = await callApi('/api/state');
    renderHistory();
    toast('已清空', 'success');
  }}

  // ── Provider / Preset 对话框 ──
  function openProviderDialog(initial = null) {{
    const isEdit = initial !== null;
    showModal(`
      <div class="panel-title">${{isEdit ? '编辑' : '添加'}} Provider</div>
      <div class="mt-2"><input id="m-name" class="input" placeholder="名称" value="${{initial ? initial.name : ''}}"></div>
      <div class="mt-2"><input id="m-url" class="input" placeholder="Base URL" value="${{initial ? initial.base_url : 'https://api.example.com/v1'}}"></div>
      <div class="mt-2"><input id="m-key" class="input" type="password" placeholder="API Key" value="${{initial ? initial.api_key : ''}}"></div>
      <div class="mt-2"><input id="m-model" class="input" placeholder="Model" value="${{initial ? initial.model : 'gpt-image-2'}}"></div>
      <div class="row mt-3" style="justify-content:flex-end">
        <button class="btn btn-ghost" onclick="closeModal()">取消</button>
        <button class="btn" onclick="submitProvider(${{isEdit ? JSON.stringify(initial.name) : 'null'}})">确定</button>
      </div>
    `);
  }}

  async function submitProvider(originalName) {{
    const data_in = {{
      name: document.getElementById('m-name').value,
      base_url: document.getElementById('m-url').value,
      api_key: document.getElementById('m-key').value,
      model: document.getElementById('m-model').value,
    }};
    try {{
      if (originalName) {{
        data.config = await callApi('/api/provider/edit?original_name=' + encodeURIComponent(originalName), data_in, 'POST');
      }} else {{
        data.config = await callApi('/api/provider/add', data_in, 'POST');
      }}
      closeModal();
      rebuildForm();
      toast('Provider 已保存', 'success');
    }} catch (e) {{ toast(e.message, 'error'); }}
  }}

  function editCurrentProvider() {{
    const name = document.getElementById('provider-select').value;
    const p = data.config.providers.find(x => x.name === name);
    if (p) openProviderDialog(p);
  }}

  async function deleteCurrentProvider() {{
    const name = document.getElementById('provider-select').value;
    if (!confirm('确定删除 Provider: ' + name + '?')) return;
    try {{
      data.config = await callApi('/api/provider/delete?name=' + encodeURIComponent(name), {{}}, 'POST');
      rebuildForm();
      onProviderChange();
      toast('已删除', 'success');
    }} catch (e) {{ toast(e.message, 'error'); }}
  }}

  function openPresetDialog(initial = null) {{
    const isEdit = initial !== null;
    showModal(`
      <div class="panel-title">${{isEdit ? '编辑' : '添加'}} 预设</div>
      <div class="mt-2"><input id="m-pname" class="input" placeholder="名称" value="${{initial ? initial.name : ''}}"></div>
      <div class="mt-2"><input id="m-pprefix" class="input" placeholder="文件前缀" value="${{initial ? initial.prefix : 'result'}}"></div>
      <div class="mt-2"><textarea id="m-pprompt" placeholder="提示词" style="min-height:160px">${{initial ? initial.prompt : ''}}</textarea></div>
      <div class="row mt-3" style="justify-content:flex-end">
        <button class="btn btn-ghost" onclick="closeModal()">取消</button>
        <button class="btn" onclick="submitPreset(${{isEdit ? initial.index : -1}})">确定</button>
      </div>
    `);
  }}

  async function submitPreset(editIndex) {{
    const data_in = {{
      name: document.getElementById('m-pname').value,
      prefix: document.getElementById('m-pprefix').value,
      prompt: document.getElementById('m-pprompt').value,
    }};
    try {{
      if (editIndex >= 0) {{
        data.config = await callApi('/api/preset/edit?index=' + editIndex, data_in, 'POST');
      }} else {{
        data.config = await callApi('/api/preset/add', data_in, 'POST');
      }}
      closeModal();
      rebuildForm();
      onPresetChange();
      toast('预设已保存', 'success');
    }} catch (e) {{ toast(e.message, 'error'); }}
  }}

  function editCurrentPreset() {{
    const i = parseInt(document.getElementById('preset-select').value);
    const p = data.config.presets[i];
    if (p) openPresetDialog({{ ...p, index: i }});
  }}

  async function deleteCurrentPreset() {{
    const i = parseInt(document.getElementById('preset-select').value);
    if (!confirm('确定删除预设: ' + (data.config.presets[i] || {{}}).name + '?')) return;
    try {{
      data.config = await callApi('/api/preset/delete?index=' + i, {{}}, 'POST');
      rebuildForm();
      onPresetChange();
      toast('已删除', 'success');
    }} catch (e) {{ toast(e.message, 'error'); }}
  }}

  function showModal(html) {{
    document.getElementById('modal-body').innerHTML = html;
    document.getElementById('modal').style.display = 'flex';
  }}
  function closeModal() {{
    document.getElementById('modal').style.display = 'none';
  }}
</script>

</body>
</html>
""".format(title=CONFIG["title"])


# ════════════════════════════════════════════════════════════════
#  6. pywebview js_api(系统级操作)
# ════════════════════════════════════════════════════════════════


class API:
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

    def save_image(self, base64_data: str, suggested_name: str) -> str | None:
        """弹保存对话框, 写盘, 返回写入路径或 None."""
        assert self._window is not None
        result = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=("PNG 图片 (*.png)", "JPEG 图片 (*.jpg)"),
            save_filename=suggested_name,
        )
        if not result:
            return None
        path = result[0]
        if not Path(path).suffix:
            path = path + ".png"
        try:
            Path(path).write_bytes(base64.b64decode(base64_data))
        except Exception as exc:
            raise RuntimeError(f"写入失败: {exc}") from exc
        return path

    def close_window(self) -> None:
        if self._window is not None:
            self._window.destroy()


# ════════════════════════════════════════════════════════════════
#  7. main
# ════════════════════════════════════════════════════════════════


def _run_server() -> None:
    config = uvicorn.Config(
        app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning",
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
