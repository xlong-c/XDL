"""
轻量 dataclass CLI 解析辅助。
"""

import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .errors import ConfigValidationError

T = TypeVar("T")

_HELP_FLAGS = {"-h", "--help"}
_BOOL_LITERALS = {
    "0": "false",
    "1": "true",
    "false": "false",
    "no": "false",
    "off": "false",
    "on": "true",
    "true": "true",
    "yes": "true",
}


def parse_dataclass_cli(config_type: Type[T], args: Optional[Sequence[str]] = None) -> T:
    """
    将 CLI 参数解析为 dataclass 实例。

    支持两种写法：
    - `--input image.png --tile-size 64 --fp16`
    - `input=image.png tile_size=64 fp16=true`
    """

    if not isinstance(config_type, type) or not is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass type")

    arg_list = list(sys.argv[1:] if args is None else args)
    if any(arg in _HELP_FLAGS for arg in arg_list):
        print(format_dataclass_cli_help(config_type, program=Path(sys.argv[0]).name))
        raise SystemExit(0)

    dotlist = _args_to_dotlist(config_type, arg_list)

    try:
        base_cfg = OmegaConf.structured(config_type)
        override_cfg = OmegaConf.from_dotlist(dotlist)
        merged_cfg = OmegaConf.merge(base_cfg, override_cfg)
        return cast(T, OmegaConf.to_object(merged_cfg))
    except OmegaConfBaseException as exc:
        raise ConfigValidationError(f"Failed to parse CLI arguments: {exc}") from exc


def format_dataclass_cli_help(config_type: Type[Any], program: str) -> str:
    """生成简洁的 CLI 帮助文本。"""

    if not isinstance(config_type, type) or not is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass type")

    base_cfg = OmegaConf.structured(config_type)
    doc = (config_type.__doc__ or "").strip()
    lines: List[str] = []

    if doc:
        lines.append(doc)
        lines.append("")

    lines.extend([
        "用法:",
        f"  python {program} [--field value | field=value ...]",
        "",
        "参数:",
    ])

    for path, annotation in _iter_field_paths(config_type):
        default_value = OmegaConf.select(base_cfg, path)
        default_text = "<required>" if default_value == "???" else repr(default_value)
        cli_name = path.replace("_", "-")
        lines.append(
            f"  --{cli_name:<24} {_format_annotation(annotation):<18} default={default_text}"
        )

    lines.extend([
        "",
        "布尔参数支持 `--flag` / `--no-flag`，也支持 `flag=true`。",
    ])
    return "\n".join(lines)


def _args_to_dotlist(config_type: Type[Any], args: Sequence[str]) -> List[str]:
    field_map = {path: annotation for path, annotation in _iter_field_paths(config_type)}
    dotlist: List[str] = []
    index = 0

    while index < len(args):
        token = args[index]

        if token in _HELP_FLAGS:
            index += 1
            continue

        if token.startswith("--"):
            index = _consume_flag_token(
                args=args,
                index=index,
                field_map=field_map,
                dotlist=dotlist,
            )
            continue

        if "=" in token:
            raw_key, raw_value = token.split("=", 1)
            key = _normalize_key_path(raw_key)
            _ensure_known_field(field_map, key)
            dotlist.append(f"{key}={raw_value}")
            index += 1
            continue

        raise ConfigValidationError(
            f"Unsupported positional argument: {token}. "
            "Use `--field value` or `field=value`."
        )

    return dotlist


def _consume_flag_token(
    *,
    args: Sequence[str],
    index: int,
    field_map: Dict[str, Any],
    dotlist: List[str],
) -> int:
    token = args[index]
    body = token[2:]

    if "=" in body:
        raw_key, raw_value = body.split("=", 1)
        key = _normalize_key_path(raw_key)
        _ensure_known_field(field_map, key)
        dotlist.append(f"{key}={raw_value}")
        return index + 1

    if body.startswith("no-"):
        key = _normalize_key_path(body[3:])
        _ensure_bool_field(field_map, key)
        dotlist.append(f"{key}=false")
        return index + 1

    key = _normalize_key_path(body)
    _ensure_known_field(field_map, key)

    if _is_bool_annotation(field_map[key]):
        next_index = index + 1
        if next_index < len(args):
            next_token = args[next_index]
            normalized_bool = _normalize_bool_literal(next_token)
            if normalized_bool is not None:
                dotlist.append(f"{key}={normalized_bool}")
                return next_index + 1
        dotlist.append(f"{key}=true")
        return index + 1

    value_index = index + 1
    if value_index >= len(args) or args[value_index].startswith("--"):
        raise ConfigValidationError(f"Missing value for CLI argument: --{body}")

    dotlist.append(f"{key}={args[value_index]}")
    return value_index + 1


def _iter_field_paths(config_type: Type[Any], prefix: str = "") -> List[Tuple[str, Any]]:
    annotations = get_type_hints(config_type)
    items: List[Tuple[str, Any]] = []

    for field in fields(config_type):
        annotation = annotations.get(field.name, field.type)
        path = f"{prefix}{field.name}"
        nested_type = _extract_dataclass_type(annotation)
        if nested_type is not None:
            items.extend(_iter_field_paths(nested_type, prefix=f"{path}."))
            continue
        items.append((path, annotation))

    return items


def _normalize_key_path(raw_key: str) -> str:
    return ".".join(segment.replace("-", "_") for segment in raw_key.split("."))


def _ensure_known_field(field_map: Dict[str, Any], key: str) -> None:
    if key not in field_map:
        raise ConfigValidationError(f"Unknown CLI field: {key}")


def _ensure_bool_field(field_map: Dict[str, Any], key: str) -> None:
    _ensure_known_field(field_map, key)
    if not _is_bool_annotation(field_map[key]):
        raise ConfigValidationError(f"`--no-{key}` can only be used with bool fields")


def _extract_dataclass_type(annotation: Any) -> Optional[Type[Any]]:
    resolved = _strip_optional(annotation)
    if isinstance(resolved, type) and is_dataclass(resolved):
        return resolved
    return None


def _is_bool_annotation(annotation: Any) -> bool:
    return _strip_optional(annotation) is bool


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _normalize_bool_literal(value: str) -> Optional[str]:
    return _BOOL_LITERALS.get(value.lower())


def _format_annotation(annotation: Any) -> str:
    resolved = _strip_optional(annotation)
    if isinstance(resolved, type):
        return resolved.__name__
    return str(resolved)


__all__ = ["format_dataclass_cli_help", "parse_dataclass_cli"]
