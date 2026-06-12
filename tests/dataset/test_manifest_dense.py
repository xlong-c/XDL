import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from xdl.config.builder import build_dataset
from xdl.dataset import (
    DetectionCollate,
    ImageBoxesTransform,
    ImageMaskSidecarDataset,
    ImageMaskTransform,
    RecordDetectionDataset,
    RecordSegmentationDataset,
)
from xdl.utils.registry import COLLATE_REGISTRY, DATASET_REGISTRY, TRANSFORM_REGISTRY


def _save_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _save_mask(path: Path, value: int, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def test_manifest_segmentation_dataset_loads_sample_and_paths(tmp_path) -> None:
    _save_rgb(tmp_path / "image.png", (255, 0, 0))
    _save_mask(tmp_path / "mask.png", 7)
    manifest_path = tmp_path / "seg.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image": "image.png",
                "mask": "mask.png",
                "sample_id": "seg-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = RecordSegmentationDataset(
        manifest_path,
        transform=ImageMaskTransform(height=4, width=6),
        include_paths=True,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 4, 6)
    assert sample["mask"].shape == (4, 6)
    assert sample["mask"].dtype == torch.int64
    assert sample["sample_id"] == "seg-1"
    assert sample["image_path"] == str((tmp_path / "image.png").resolve())
    assert sample["mask_path"] == str((tmp_path / "mask.png").resolve())


def test_image_mask_sidecar_dataset_loads_split_roots_and_collates(tmp_path) -> None:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    _save_rgb(image_root / "a.png", (255, 0, 0))
    _save_rgb(image_root / "nested" / "b.jpg", (0, 255, 0))
    _save_mask(mask_root / "a.png", 3)
    _save_mask(mask_root / "nested" / "b.jpg", 5)

    dataset = ImageMaskSidecarDataset(
        image_root=image_root,
        mask_root=mask_root,
        transform=ImageMaskTransform(height=4, width=5, normalize=False),
        sample_id_from="relative_path",
        repeat=2,
    )
    sample = dataset[1]

    assert len(dataset) == 4
    assert sample["image"].shape == (3, 4, 5)
    assert sample["mask"].shape == (4, 5)
    assert sample["sample_id"] == "nested/b.jpg"
    assert sample["image_path"] == str((image_root / "nested" / "b.jpg").resolve())
    assert sample["mask_path"] == str((mask_root / "nested" / "b.jpg").resolve())
    assert dataset[3]["sample_id"] == "nested/b.jpg"

    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 4, 5)
    assert batch["mask"].shape == (2, 4, 5)
    assert batch["sample_id"] == ["a.png", "nested/b.jpg"]


def test_image_mask_sidecar_dataset_supports_same_dir_different_extension(
    tmp_path,
) -> None:
    _save_rgb(tmp_path / "sample.jpg", (255, 0, 0))
    _save_mask(tmp_path / "sample.png", 7)

    dataset = ImageMaskSidecarDataset(
        root=tmp_path,
        mask_extension=".png",
        extensions=[".jpg"],
        transform=ImageMaskTransform(height=3, width=4),
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 3, 4)
    assert sample["mask"].shape == (3, 4)
    assert sample["sample_id"] == "sample"
    assert sample["mask_path"] == str((tmp_path / "sample.png").resolve())


def test_image_mask_sidecar_dataset_skips_or_errors_on_missing_masks(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    _save_rgb(tmp_path / "b.png", (0, 255, 0))
    _save_mask(tmp_path / "a.mask.png", 1)

    dataset = ImageMaskSidecarDataset(
        root=tmp_path,
        mask_extension=".mask.png",
        extensions=[".png"],
        missing_mask="skip",
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "a"

    with pytest.raises(FileNotFoundError, match="Missing sidecar mask files"):
        ImageMaskSidecarDataset(
            root=tmp_path,
            mask_extension=".missing.png",
            extensions=[".png"],
        )


def test_manifest_detection_dataset_parses_json_strings_and_collates(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    _save_rgb(data_dir / "a.png", (10, 20, 30), size=(10, 8))
    _save_rgb(data_dir / "b.png", (40, 50, 60), size=(10, 8))
    manifest_path = data_dir / "det.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image": "a.png",
                "boxes": "[[1, 1, 5, 6], [2, 0, 8, 7]]",
                "labels": "[3, 4]",
                "sample_id": "det-a",
            }
        )
        + "\n"
        + json.dumps(
            {
                "image": "b.png",
                "boxes": [[0, 0, 3, 4]],
                "labels": [1],
                "sample_id": "det-b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(other_cwd)

    dataset = RecordDetectionDataset(
        manifest_path,
        transform=ImageBoxesTransform(height=4, width=5),
        include_paths=True,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 4, 5)
    assert sample["boxes"].shape == (2, 4)
    assert sample["labels"].tolist() == [3, 4]
    assert sample["sample_id"] == "det-a"
    assert sample["image_path"] == str((data_dir / "a.png").resolve())

    loader = DataLoader(dataset, batch_size=2, collate_fn=DetectionCollate())
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 4, 5)
    assert len(batch["boxes"]) == 2
    assert len(batch["labels"]) == 2
    assert batch["sample_id"] == ["det-a", "det-b"]
    assert batch["image_path"] == [
        str((data_dir / "a.png").resolve()),
        str((data_dir / "b.png").resolve()),
    ]


def test_dense_dataset_components_are_registered_and_buildable(tmp_path) -> None:
    _save_rgb(tmp_path / "image.png", (0, 0, 255))
    _save_mask(tmp_path / "mask.png", 1)
    manifest_path = tmp_path / "seg.jsonl"
    manifest_path.write_text(
        json.dumps({"image": "image.png", "mask": "mask.png"}) + "\n",
        encoding="utf-8",
    )

    dataset = build_dataset(
        {
            "target": "registry:RecordSegmentationDataset",
            "params": {
                "manifest_path": str(manifest_path),
                "transform": {
                    "target": "registry:ImageMaskTransform",
                    "params": {"height": 4, "width": 4},
                },
            },
        }
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 4, 4)
    assert sample["mask"].shape == (4, 4)
    sidecar_dataset = build_dataset(
        {
            "target": "registry:ImageMaskSidecarDataset",
            "params": {
                "image_root": str(tmp_path),
                "mask_root": str(tmp_path),
                "extensions": [".png"],
                "transform": {
                    "target": "registry:ImageMaskTransform",
                    "params": {"height": 4, "width": 4},
                },
            },
        }
    )

    assert sidecar_dataset[0]["image"].shape == (3, 4, 4)
    assert DATASET_REGISTRY.get("RecordSegmentationDataset") is RecordSegmentationDataset
    assert DATASET_REGISTRY.get("RecordDetectionDataset") is RecordDetectionDataset
    assert DATASET_REGISTRY.get("ImageMaskSidecarDataset") is ImageMaskSidecarDataset
    assert TRANSFORM_REGISTRY.get("ImageMaskTransform") is ImageMaskTransform
    assert TRANSFORM_REGISTRY.get("ImageBoxesTransform") is ImageBoxesTransform
    assert COLLATE_REGISTRY.get("DetectionCollate") is DetectionCollate
