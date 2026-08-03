"""Basename-aligned sidecar datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    PathLike,
    Record,
    Transform,
    apply_optional,
    collect_image_paths,
    collect_sidecar_samples,
    load_image,
    normalize_extension,
    normalize_extensions,
    path_sample_id,
)


# ---------------------------------------------------------------------------
# Basename-aligned sidecar reader registry
# ---------------------------------------------------------------------------

def _read_text_sidecar(path: Path) -> str:
    return path.read_text("utf-8").strip()


def _read_json_sidecar(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _read_image_sidecar(path: Path) -> Image.Image:
    return Image.open(path)


_SIDECAR_READERS: dict[str, Callable[[Path], Any]] = {
    ".txt": _read_text_sidecar,
    ".text": _read_text_sidecar,
    ".json": _read_json_sidecar,
    ".png": _read_image_sidecar,
    ".jpg": _read_image_sidecar,
    ".jpeg": _read_image_sidecar,
    ".webp": _read_image_sidecar,
    ".bmp": _read_image_sidecar,
    ".tif": _read_image_sidecar,
    ".tiff": _read_image_sidecar,
}


# ---------------------------------------------------------------------------
# BasenameAlignedDataset
# ---------------------------------------------------------------------------


class BasenameAlignedDataset(Dataset[Record]):
    """Generic dataset for basename-aligned image and sidecar files.

    Replaces the need for a new class per sidecar file format.  Image files
    are discovered under *image_root* (or *root*), and each image is paired
    with a same-basename sidecar file whose extension equals
    *sidecar_extension*.  The sidecar content is read by *sidecar_reader*
    (or an auto-selected built-in reader) and stored under *sidecar_key*
    in the returned dict.

    Built-in readers:

    - ``.txt`` / ``.text`` → ``str``
    - ``.json`` → parsed ``dict`` / ``list``
    - ``.png`` / ``.jpg`` / … → ``PIL.Image.Image``

    Parameters
    ----------
    sidecar_extension:
        Extension of the sidecar file (e.g. ``".json"``).
    root:
        Convenience – same directory for both images and sidecars.
    image_root:
        Directory containing images.  Takes precedence over *root*.
    sidecar_root:
        Directory containing sidecar files.  Defaults to *image_root*.
    sidecar_key:
        Dict key for the sidecar value.  Defaults to *sidecar_extension*
        without the leading dot (e.g. ``".json"`` → ``"json"``).
    sidecar_reader:
        Callable ``(path: Path) -> Any``.  When ``None`` an appropriate
        built-in reader is selected from *sidecar_extension*.
    sidecar_transform:
        Optional transform applied to the sidecar content before returning.
    transform:
        Image transform (applied to the image only).
    extensions:
        Image file extensions to scan for (default: common image formats).
    image_mode:
        PIL image mode for loading images.
    recursive:
        Whether to scan sub-directories for images.
    include_paths:
        Whether to include ``"image_path"`` and ``"<sidecar_key>_path"``
        in the returned dict.
    missing_sidecar:
        ``"error"`` or ``"skip"`` – behaviour when a sidecar file is missing.
    sample_id_from:
        One of ``"stem"``, ``"name"``, ``"relative_path"``, ``"index"``.
    repeat:
        Number of logical repetitions over the underlying samples.
    """

    def __init__(
        self,
        sidecar_extension: str,
        *,
        root: Optional[PathLike] = None,
        image_root: Optional[PathLike] = None,
        sidecar_root: Optional[PathLike] = None,
        sidecar_key: Optional[str] = None,
        sidecar_reader: Optional[Callable[[Path], Any]] = None,
        sidecar_transform: Optional[Callable[[Any], Any]] = None,
        transform: Transform = None,
        extensions: Optional[Sequence[str]] = None,
        image_mode: str = "RGB",
        recursive: bool = True,
        include_paths: bool = True,
        missing_sidecar: str = "error",
        sample_id_from: str = "stem",
        repeat: int = 1,
    ) -> None:
        if root is None and image_root is None:
            raise ValueError("Either root or image_root must be provided")
        if root is not None and image_root is not None:
            raise ValueError("Use either root or image_root, not both")
        if missing_sidecar not in {"error", "skip"}:
            raise ValueError("missing_sidecar must be one of: 'error', 'skip'")

        resolved_image_root = image_root if image_root is not None else root
        if resolved_image_root is None:
            raise ValueError("Either root or image_root must be provided")

        self.image_root = Path(resolved_image_root).expanduser().resolve()
        self.sidecar_root = (
            Path(sidecar_root).expanduser().resolve()
            if sidecar_root is not None
            else None
        )
        self.sidecar_extension = normalize_extension(sidecar_extension)
        self.sidecar_key = (
            sidecar_key
            if sidecar_key is not None
            else self.sidecar_extension.lstrip(".")
        )

        if sidecar_reader is not None:
            self.sidecar_reader = sidecar_reader
        else:
            builtin = _SIDECAR_READERS.get(self.sidecar_extension)
            if builtin is not None:
                self.sidecar_reader = builtin
            else:
                # Unknown extension → fallback to text.
                self.sidecar_reader = _read_text_sidecar

        self.sidecar_transform = sidecar_transform
        self.transform = transform
        self.image_mode = image_mode
        self.extensions = normalize_extensions(extensions)
        self.recursive = bool(recursive)
        self.include_paths = bool(include_paths)
        self.missing_sidecar = missing_sidecar
        self.sample_id_from = sample_id_from
        self.repeat = max(1, int(repeat))

        image_paths = collect_image_paths(
            self.image_root,
            extensions=self.extensions,
            recursive=self.recursive,
        )
        self.samples = collect_sidecar_samples(
            image_paths,
            image_root=self.image_root,
            sidecar_root=self.sidecar_root,
            sidecar_extension=self.sidecar_extension,
            missing=self.missing_sidecar,
            sidecar_name=self.sidecar_key,
        )
        if not self.samples:
            raise ValueError(
                f"No image/{self.sidecar_key} sidecar samples found under: {self.image_root}"
            )

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> Record:
        base_index = index % len(self.samples)
        image_path, sidecar_path = self.samples[base_index]
        sidecar_value = self.sidecar_reader(sidecar_path)
        if self.sidecar_transform is not None:
            sidecar_value = self.sidecar_transform(sidecar_value)

        sample: Record = {
            "image": apply_optional(
                self.transform,
                load_image(image_path, self.image_mode),
            ),
            self.sidecar_key: sidecar_value,
            "sample_id": path_sample_id(
                image_path,
                root=self.image_root,
                index=base_index,
                sample_id_from=self.sample_id_from,
            ),
        }
        if self.include_paths:
            sample["image_path"] = str(image_path)
            sample[f"{self.sidecar_key}_path"] = str(sidecar_path)
        return sample


__all__ = ["BasenameAlignedDataset"]
