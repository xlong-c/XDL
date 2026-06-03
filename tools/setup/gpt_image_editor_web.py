#!/usr/bin/env python3
"""GPT 图像编辑器 — FastAPI + HTML 版本。

用法:
  python tools/setup/gpt_image_editor_web.py

环境变量:
  XDL_WEB_HOST=127.0.0.1
  XDL_GPT_IMAGE_EDITOR_PORT=8013
  XDL_NO_BROWSER=1

说明:
  - 前端页面位于同目录 `gpt_image_editor.html`
  - 后端负责代理远端图像接口并可选落盘
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

app = FastAPI(title="GPT 图像编辑器")

HTML_PATH = Path(__file__).with_name("gpt_image_editor.html")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    try:
        return int(value)
    except ValueError:
        return default


def should_auto_open_browser() -> bool:
    if os.environ.get("XDL_NO_BROWSER", "").strip() == "1":
        return False
    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid) and geteuid() == 0:
        return False
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = size_text.lower().split("x", 1)
        width = max(1, int(width_text))
        height = max(1, int(height_text))
        return width, height
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
        image.convert("RGBA" if image.mode in ("RGBA", "LA") else "RGB").save(path, format="PNG")


def image_to_png_bytes(file_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(file_bytes)) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def fetch_binary(url: str, timeout: int = 120) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def call_generation_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    size: str,
    prompt: str,
) -> bytes:
    url = base_url.rstrip("/") + "/images/generations"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    entry = data[0] if isinstance(data, list) and data else data
    if not isinstance(entry, dict):
        raise ValueError("API 返回缺少 data")
    if entry.get("b64_json"):
        return base64.b64decode(entry["b64_json"])
    if entry.get("url"):
        return fetch_binary(entry["url"])
    raise ValueError("API 返回中没有 b64_json 或 url")


def call_edit_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    size: str,
    prompt: str,
    image_payloads: list[tuple[str, bytes]],
) -> bytes:
    url = base_url.rstrip("/") + "/images/edits"
    files = []
    for index, (_, data) in enumerate(image_payloads, start=1):
        files.append(("image", (f"image{index}.png", data, "image/png")))
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        data={
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": "1",
        },
        files=files,
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
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
    *,
    output_dir: Path | None,
    result_bytes: bytes,
    image1_bytes: bytes | None,
    image2_bytes: bytes | None,
    preset_prefix: str,
    image1_name: str,
    image2_name: str,
    is_edit: bool,
) -> tuple[list[str], str]:
    if output_dir is None:
        return [], ""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    prefix = (preset_prefix or "result").strip() or "result"
    saved_paths: list[str] = []

    if is_edit and image1_bytes is not None:
        img1_path = output_dir / f"{prefix}_{file_stem(image1_name, 'img1')}_img1_{timestamp}_{unique}.png"
        safe_write_png(image1_bytes, img1_path)
        saved_paths.append(str(img1_path))
    if is_edit and image2_bytes is not None:
        img2_path = output_dir / f"{prefix}_{file_stem(image2_name, 'img2')}_img2_{timestamp}_{unique}.png"
        safe_write_png(image2_bytes, img2_path)
        saved_paths.append(str(img2_path))

    result_name = f"{prefix}_{timestamp}_{unique}.png"
    if is_edit:
        result_name = (
            f"{prefix}_{file_stem(image1_name, 'img')}_with_"
            f"{file_stem(image2_name, 'ref')}_{timestamp}_{unique}.png"
        )
    result_path = output_dir / result_name
    safe_write_png(result_bytes, result_path)
    saved_paths.append(str(result_path))
    return saved_paths, str(output_dir)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.post("/api/generate", response_model=None)
async def api_generate(
    base_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-image-2"),
    size: str = Form("1024x1024"),
    prompt: str = Form(...),
    output_dir: str = Form(""),
    preset_prefix: str = Form("result"),
    preset_name: str = Form(""),
    image1: UploadFile | None = File(default=None),
    image2: UploadFile | None = File(default=None),
):
    if not base_url.strip():
        return JSONResponse({"error": "base_url 不能为空"}, status_code=400)
    if not api_key.strip():
        return JSONResponse({"error": "api_key 不能为空"}, status_code=400)
    if not prompt.strip():
        return JSONResponse({"error": "prompt 不能为空"}, status_code=400)

    try:
        parse_size(size)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    image1_bytes: bytes | None = None
    image2_bytes: bytes | None = None
    image_payloads: list[tuple[str, bytes]] = []

    try:
        if image1 is not None:
            image1_raw = await image1.read()
            image1_bytes = image_to_png_bytes(image1_raw)
            image_payloads.append((image1.filename or "image1.png", image1_bytes))
        if image2 is not None:
            image2_raw = await image2.read()
            image2_bytes = image_to_png_bytes(image2_raw)
            image_payloads.append((image2.filename or "image2.png", image2_bytes))
    except Exception as exc:
        return JSONResponse({"error": f"读取图片失败: {exc}"}, status_code=400)

    is_edit = bool(image_payloads)
    if is_edit and image1_bytes is None:
        return JSONResponse({"error": "编辑模式至少需要 image1"}, status_code=400)

    try:
        if is_edit:
            result_bytes = call_edit_api(
                base_url=base_url,
                api_key=api_key,
                model=model,
                size=size,
                prompt=prompt,
                image_payloads=image_payloads,
            )
        else:
            result_bytes = call_generation_api(
                base_url=base_url,
                api_key=api_key,
                model=model,
                size=size,
                prompt=prompt,
            )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = ""
        try:
            detail = exc.response.text[:500] if exc.response is not None else ""
        except Exception:
            detail = ""
        return JSONResponse({"error": f"远端 API 错误 [{status_code}] {detail}"}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    try:
        out_dir = ensure_dir(output_dir)
        saved_files, saved_dir = save_outputs(
            output_dir=out_dir,
            result_bytes=result_bytes,
            image1_bytes=image1_bytes,
            image2_bytes=image2_bytes,
            preset_prefix=preset_prefix,
            image1_name=image1.filename if image1 else "img1.png",
            image2_name=image2.filename if image2 else "img2.png",
            is_edit=is_edit,
        )
    except Exception as exc:
        return JSONResponse({"error": f"保存结果失败: {exc}"}, status_code=500)

    return {
        "ok": True,
        "is_edit": is_edit,
        "preset_name": preset_name,
        "result_b64": base64.b64encode(result_bytes).decode("ascii"),
        "saved_files": saved_files,
        "saved_dir": saved_dir,
    }


def main() -> None:
    host = os.environ.get("XDL_WEB_HOST", "127.0.0.1")
    port = env_int("XDL_GPT_IMAGE_EDITOR_PORT", 8013)
    url = f"http://{host}:{port}"

    def open_browser() -> None:
        time.sleep(1.2)
        webbrowser.open(url)

    if should_auto_open_browser():
        threading.Thread(target=open_browser, daemon=True).start()
    else:
        print("已跳过自动打开浏览器，请手动访问上面的地址。")

    print(f"GPT 图像编辑器启动: {url}")
    print("按 Ctrl+C 停止")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
