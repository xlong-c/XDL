import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from xdl.dataset import (
    ImageEditCollate,
    ImageEditDataset,
)
from xdl.utils.registry import COLLATE_REGISTRY, DATASET_REGISTRY, TRANSFORM_REGISTRY


def _save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 6), color=color).save(path)


def _save_mask(path: Path, value: int) -> None:
    Image.new("L", (8, 6), color=value).save(path)


def test_manifest_image_edit_dataset_loads_jsonl_and_collates(tmp_path) -> None:
    _save_rgb(tmp_path / "source.png", (255, 0, 0))
    _save_rgb(tmp_path / "target.png", (0, 255, 0))
    _save_rgb(tmp_path / "reference.png", (0, 0, 255))
    _save_mask(tmp_path / "mask.png", 255)
    manifest_path = tmp_path / "data.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_image": "source.png",
                "target_image": "target.png",
                "reference_image": "reference.png",
                "edit_mask": "mask.png",
                "prompt": "make hair blue",
                "sample_id": "sample-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ImageEditDataset(manifest_path, height=4, width=6)
    sample = dataset[0]

    assert sample["source_image"].shape == (3, 4, 6)
    assert sample["target_image"].shape == (3, 4, 6)
    assert sample["reference_image"].shape == (3, 4, 6)
    assert sample["source_image"].min() >= -1.0
    assert sample["source_image"].max() <= 1.0
    assert sample["edit_mask"].shape == (1, 4, 6)
    assert sample["has_mask"].item() is True
    assert sample["prompt"] == "make hair blue"
    assert sample["sample_id"] == "sample-a"

    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ImageEditCollate(),
    )
    batch = next(iter(loader))

    assert batch["source_image"].shape == (1, 3, 4, 6)
    assert batch["prompt"] == ["make hair blue"]
    assert batch["sample_id"] == ["sample-a"]


def test_manifest_image_edit_dataset_loads_csv_without_mask_from_manifest_dir(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    _save_rgb(data_dir / "source.png", (10, 20, 30))
    _save_rgb(data_dir / "target.png", (40, 50, 60))
    _save_rgb(data_dir / "reference.png", (70, 80, 90))
    manifest_path = data_dir / "data.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "target_image", "reference_image", "caption"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_image": "source.png",
                "target_image": "target.png",
                "reference_image": "reference.png",
                "caption": "caption text",
            }
        )
    monkeypatch.chdir(other_cwd)

    dataset = ImageEditDataset(manifest_path, height=4, width=4)
    sample = dataset[0]

    assert sample["has_mask"].item() is False
    assert torch.count_nonzero(sample["edit_mask"]).item() == 0
    assert sample["prompt"] == "caption text"


def test_manifest_image_edit_components_are_registered() -> None:
    assert DATASET_REGISTRY.get("ImageEditDataset") is ImageEditDataset
    assert COLLATE_REGISTRY.get("ImageEditCollate") is ImageEditCollate
    assert TRANSFORM_REGISTRY.get("PairedImageTransform") is not None
