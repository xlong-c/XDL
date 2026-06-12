"""Internal helpers for dataset templates — paths, records, and tensor conversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Common type aliases
# ---------------------------------------------------------------------------

PathLike = Union[str, Path]
Record = Dict[str, Any]
Transform = Optional[Callable[[Any], Any]]

DEFAULT_TEXT_KEYS: Tuple[str, ...] = ("text", "prompt", "caption")
DEFAULT_TARGET_TEXT_KEYS: Tuple[str, ...] = (
    "target_text",
    "response",
    "completion",
    "answer",
)


def apply_optional(transform: Transform, value: Any) -> Any:
    """当 transform 不为 None 时应用它, 否则原样返回 value."""

    return transform(value) if transform is not None else value


def is_int_like(value: Any) -> bool:
    """判断值是否可无损解释为 int, bool 不视作 int-like label."""

    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return str(value).strip() == str(int(value))


def build_label_mapping(
    labels: Sequence[Any],
    class_to_idx: Optional[Mapping[str, int]],
) -> Optional[Dict[str, int]]:
    """从标签值构造字符串标签到整数 id 的映射; 纯 int-like 标签保持原值."""

    if class_to_idx is not None:
        return {str(key): int(value) for key, value in class_to_idx.items()}
    if all(is_int_like(label) for label in labels):
        return None
    return {label: idx for idx, label in enumerate(sorted({str(item) for item in labels}))}


def target_from_label(label: Any, label_mapping: Optional[Mapping[str, int]]) -> int:
    """按可选 label mapping 将 manifest label 转为整数 target."""

    if label_mapping is None:
        return int(label)
    key = str(label)
    if key not in label_mapping:
        raise KeyError(f"Unknown label '{label}'")
    return int(label_mapping[key])


def parse_sequence_field(
    value: Any,
    *,
    delimiter: str = ",",
) -> List[Any]:
    """把 JSON/list/逗号分隔字符串字段统一解析为 list."""

    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("("):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
                raise TypeError("Decoded sequence field must be a sequence")
            return list(decoded)
        return [item.strip() for item in stripped.split(delimiter) if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError("Expected a sequence-like field")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


def normalize_extension(extension: str) -> str:
    """将单个扩展名统一为小写 '.' 前缀格式, 例如 'JPG' → '.jpg', 'png' → '.png'.

    Normalize a single file extension to lower-case dot form.
    """

    return extension.lower() if extension.startswith(".") else f".{extension.lower()}"


def normalize_extensions(extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """将扩展名列表统一规范化; 未提供时默认使用常见图片格式集合.

    Normalize extension filters, defaulting to common image formats.
    """

    values = extensions or DEFAULT_IMAGE_EXTENSIONS
    return tuple(normalize_extension(str(item)) for item in values)


def collect_image_paths(
    root: Path,
    *,
    extensions: Sequence[str],
    recursive: bool,
) -> List[Path]:
    """按字典序收集目录下的图片文件; 支持递归与非递归模式.

    Collect image files under a directory in deterministic order.
    """

    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    return [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in extensions
    ]


def load_image(path: Path, image_mode: str) -> Image.Image:
    """打开图片并转换为指定的 PIL 色彩模式, 例如 'RGB' / 'L' / 'RGBA'.

    Open an image and convert it to the requested PIL mode.
    """

    return Image.open(path).convert(image_mode)


def path_sample_id(
    path: Path,
    *,
    root: Path,
    index: int,
    sample_id_from: str,
) -> str:
    """基于文件路径生成稳定的样本标识, 支持 stem / name / relative_path / index 四种策略.

    Build a stable sample id from a path-based dataset item.
    """

    if sample_id_from == "stem":
        return path.stem
    if sample_id_from == "name":
        return path.name
    if sample_id_from == "relative_path":
        return path.relative_to(root).as_posix()
    if sample_id_from == "index":
        return str(index)
    raise ValueError(
        "sample_id_from must be one of: 'stem', 'name', 'relative_path', 'index'"
    )


def sidecar_path_for_image(
    image_path: Path,
    *,
    image_root: Path,
    sidecar_root: Optional[Path],
    sidecar_extension: Optional[str] = None,
) -> Path:
    """根据 basename 对齐规则, 由图片路径推算出对应 sidecar 文件的路径.

    若 sidecar_root 不为 None, 则保持相对目录结构从 sidecar_root 下映射;
    若 sidecar_extension 不为 None, 则将后缀替换为目标扩展名.

    Resolve a basename-aligned sidecar path for an image.
    """

    if sidecar_root is None:
        sidecar_path = image_path
    else:
        sidecar_path = sidecar_root / image_path.relative_to(image_root)

    if sidecar_extension is None:
        return sidecar_path
    return sidecar_path.with_suffix(normalize_extension(sidecar_extension))


def collect_sidecar_samples(
    image_paths: Sequence[Path],
    *,
    image_root: Path,
    sidecar_root: Optional[Path],
    sidecar_extension: Optional[str],
    missing: str,
    sidecar_name: str,
) -> List[Tuple[Path, Path]]:
    """将图片与 basename 对齐的 sidecar 文件逐一配对.

    missing='error' 时任何缺失都抛异常; missing='skip' 时静默跳过缺失样本.

    Pair image files with basename-aligned sidecar files.
    """

    if missing not in {"error", "skip"}:
        raise ValueError(f"missing_{sidecar_name} must be one of: 'error', 'skip'")

    # Sidecar 模板只负责建立 image -> annotation 的稳定文件对, 具体读取逻辑由 Dataset 处理.
    samples: List[Tuple[Path, Path]] = []
    missing_paths: List[Path] = []
    for image_path in image_paths:
        sidecar_path = sidecar_path_for_image(
            image_path,
            image_root=image_root,
            sidecar_root=sidecar_root,
            sidecar_extension=sidecar_extension,
        )
        if sidecar_path.is_file():
            samples.append((image_path, sidecar_path))
            continue
        if missing == "skip":
            continue
        missing_paths.append(sidecar_path)

    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:3])
        raise FileNotFoundError(f"Missing sidecar {sidecar_name} files: {preview}")
    return samples


# ---------------------------------------------------------------------------
# Record / manifest helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Record]:
    """读取 JSONL manifest 文件, 每行为一条 dict 记录, 跳过空行."""

    records: List[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise TypeError(f"JSONL record must be a mapping: {path}")
        records.append(dict(item))
    return records


def _read_json(path: Path) -> List[Record]:
    """读取 JSON manifest 文件, 支持顶层 list / {'records': [...]} / 单条 dict 三种形态."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise TypeError(f"JSON manifest must be a list or mapping: {path}")

    if not all(isinstance(item, dict) for item in records):
        raise TypeError(f"JSON manifest records must be mappings: {path}")
    return [dict(item) for item in records]


def _read_csv(path: Path) -> List[Record]:
    """读取 CSV manifest 文件, 首行为列名, 后续每行为一条记录."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_manifest_records(path: PathLike) -> List[Record]:
    """按文件后缀自动分发为 list[dict]: .jsonl → JSONL, .json → JSON, .csv → CSV."""

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(manifest_path)
    if suffix == ".json":
        return _read_json(manifest_path)
    if suffix == ".csv":
        return _read_csv(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def load_manifest_context(
    manifest_path: PathLike,
    *,
    base_dir: Optional[PathLike] = None,
) -> Tuple[Path, Path, List[Record]]:
    """解析 manifest 路径、基准目录和非空记录列表, 供上层 Dataset 构造函数统一入口.

    返回 tuple: (manifest 绝对路径, 基准目录, 记录列表)
    base_dir 缺省时回退到 manifest 所在目录.

    Resolve manifest path, base dir, and non-empty records.
    """

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    records = load_manifest_records(resolved_manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {resolved_manifest_path}")

    resolved_base_dir = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else resolved_manifest_path.parent
    )
    return resolved_manifest_path, resolved_base_dir, records


def resolve_path(value: Any, base_dir: Path) -> Path:
    """将 record 中的相对路径字段拼接 base_dir 转为绝对路径; 已是绝对路径时原样返回."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def require_record_keys(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    """校验 record 是否包含全部必需字段.

    allow_empty=True 时仅检查字段存在, 不拒绝空值 (None / '');
    默认拒绝空值, 缺失或为空均抛出 KeyError.
    """

    missing = [
        key
        for key in keys
        if key not in record or (not allow_empty and record.get(key) in (None, ""))
    ]
    if missing:
        if len(missing) == 1:
            raise KeyError(f"Manifest record requires key '{missing[0]}'")
        raise KeyError(f"Manifest record requires keys {missing}")


def resolve_record_path(record: Mapping[str, Any], key: str, base_dir: Path) -> Path:
    """先校验 record 包含指定 key, 再将其值解析为绝对路径."""

    require_record_keys(record, (key,))
    return resolve_path(record[key], base_dir)


def first_present_value(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    """按优先级依次尝试 keys, 返回第一个非空值; 全部为空时返回 default."""

    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def build_sample_id(
    record: Mapping[str, Any],
    *,
    index: int,
    sample_id_key: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """从 record 的常见字段中构建稳定的样本标识字符串.

    优先级: sample_id_key → 'sample_id' → 'id' → fallback → index.
    """

    if sample_id_key and record.get(sample_id_key) not in (None, ""):
        return str(record[sample_id_key])
    for key in ("sample_id", "id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    if fallback not in (None, ""):
        return str(fallback)
    return str(index)


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------


def to_image_tensor(image: Image.Image, *, normalize: bool) -> torch.Tensor:
    """PIL 图片 → CHW float tensor.

    normalize=False 输出 [0, 1]; normalize=True 输出 [-1, 1].
    HWC → CHW 置换在 GPU 友好的 contiguous 拷贝中完成.
    灰度图 (HW) 会自动补一个单通道维度变为 HWC.
    """

    array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def to_mask_tensor_int(mask: Image.Image) -> torch.Tensor:
    """单通道 PIL mask → int64 HW tensor (类别 ID).

    输入为三通道时仅取第一个通道; 不做归一化, 保留原始整型 ID.
    """

    array = np.array(mask, dtype=np.int64, copy=True)
    if array.ndim == 3:
        array = array[:, :, 0]
    return torch.from_numpy(array).contiguous()


def to_mask_tensor_float(mask: Image.Image) -> torch.Tensor:
    """单通道 PIL mask → float 1HW tensor, 值域 [0, 1].

    输入为三通道时仅取第一个通道; 添加单 batch 维度后除以 255.
    """

    array = np.array(mask, dtype=np.float32, copy=True)
    if array.ndim == 3:
        array = array[:, :, 0]
    return torch.from_numpy(array[None, ...]).contiguous() / 255.0
