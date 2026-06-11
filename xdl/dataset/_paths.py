"""Shared file path helpers for dataset templates."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PIL import Image

PathLike = Union[str, Path]

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
    """Normalize a single file extension to lower-case dot form."""

    return extension.lower() if extension.startswith(".") else f".{extension.lower()}"


def normalize_extensions(extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Normalize extension filters, defaulting to common image formats."""

    values = extensions or DEFAULT_IMAGE_EXTENSIONS
    return tuple(normalize_extension(str(item)) for item in values)


def collect_image_paths(
    root: Path,
    *,
    extensions: Sequence[str],
    recursive: bool,
) -> List[Path]:
    """Collect image files under a directory in deterministic order."""

    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    return [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in extensions
    ]


def load_image(path: Path, image_mode: str) -> Image.Image:
    """Open an image and convert it to the requested PIL mode."""

    return Image.open(path).convert(image_mode)


def path_sample_id(
    path: Path,
    *,
    root: Path,
    index: int,
    sample_id_from: str,
) -> str:
    """Build a stable sample id from a path-based dataset item."""

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
    """Resolve a basename-aligned sidecar path for an image."""

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
    """Pair image files with basename-aligned sidecar files."""

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
