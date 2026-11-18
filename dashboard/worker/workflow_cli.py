"""工作流编译器命令行入口.

协议: 从 stdin 读一个 JSON 请求, 向 stdout 写一个 JSON 响应, 不解析命令行参数.
Go dashboard 的语义预检和 worker 的编译步骤都通过这个入口复用同一套 Python 语义.

请求形状见 ``docs/md/architecture/dashboard-workflow-ir.md``.
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from typing import Any, Dict, Mapping, Optional

from .workflow_compiler import CompileOptions, compile_workflow, validate_only
from .workflow_model import COMPILER_VERSION, payload_for_exception

VALID_ACTIONS = ("validate", "compile")


def handle_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    """处理一个编译请求并返回响应字典."""
    action = request.get("action")
    if action not in VALID_ACTIONS:
        return {
            "ok": False,
            "action": str(action),
            "compiler_version": COMPILER_VERSION,
            "issues": [
                {
                    "code": "invalid_snapshot",
                    "level": "error",
                    "message": f"action 必须是 {' 或 '.join(VALID_ACTIONS)}",
                    "path": "action",
                }
            ],
            "warnings": [],
        }
    snapshot = request.get("snapshot")
    raw_options = request.get("options") or {}
    if not isinstance(raw_options, Mapping):
        return payload_for_exception(
            ValueError("options 必须是 JSON 对象"), action=str(action)
        )
    options = CompileOptions(
        data_root=_optional_text(raw_options.get("data_root")),
        project_root=_optional_text(raw_options.get("project_root")),
        output_path=_optional_text(raw_options.get("output_path")),
        system_overrides=_system_overrides(raw_options.get("system_overrides")),
        check_registry=bool(raw_options.get("check_registry", True)),
        catalog_path=_optional_text(raw_options.get("catalog_path")),
    )
    result = (
        validate_only(snapshot, options)
        if action == "validate"
        else compile_workflow(snapshot, options)
    )
    payload = result.as_dict()
    payload["action"] = str(action)
    if action == "validate":
        payload.pop("config", None)
        payload.pop("config_path", None)
        payload.pop("config_hash", None)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    """读取 stdin 请求并写出 JSON 响应."""
    del argv  # 不使用命令行参数, 请求只走 stdin
    raw = sys.stdin.buffer.read()
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _write({"ok": False, "action": "", "issues": [_protocol_issue(str(exc))]})
        return 2
    if not isinstance(request, Mapping):
        _write(
            {
                "ok": False,
                "action": "",
                "issues": [_protocol_issue("请求必须是 JSON 对象")],
            }
        )
        return 2
    # 第三方组件导入可能向 stdout 打印, 必须隔离, 保证 stdout 只有协议 JSON.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            response = handle_request(request)
        except Exception as exc:  # noqa: BLE001 - 统一转成协议错误
            traceback.print_exc(file=sys.stderr)
            response = payload_for_exception(exc, action=str(request.get("action", "")))
    _write(response)
    return 0


def _write(response: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _protocol_issue(message: str) -> Dict[str, str]:
    return {
        "code": "invalid_snapshot",
        "level": "error",
        "message": f"无法解析编译请求: {message}",
    }


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _system_overrides(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


if __name__ == "__main__":
    raise SystemExit(main())
