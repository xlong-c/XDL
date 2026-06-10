import csv
import json
from pathlib import Path

import torch
from PIL import Image

from xdl.dataset import (
    ImageFolderClassificationDataset,
    ManifestClassificationDataset,
    ManifestImageTextDataset,
    ManifestPairDataset,
    ManifestRegressionDataset,
    ManifestRecordDataset,
)
from xdl.utils.registry import DATASET_REGISTRY


def _save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (5, 4), color=color).save(path)


def _to_marker_tensor(image: Image.Image) -> torch.Tensor:
    return torch.tensor([image.size[0], image.size[1]], dtype=torch.long)


def test_manifest_record_dataset_loads_jsonl(tmp_path) -> None:
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "a", "value": 1}) + "\n"
        + json.dumps({"id": "b", "value": 2}) + "\n",
        encoding="utf-8",
    )

    dataset = ManifestRecordDataset(manifest_path)

    assert len(dataset) == 2
    assert dataset[0] == {"id": "a", "value": 1}


def test_image_folder_classification_dataset_discovers_class_dirs(tmp_path) -> None:
    _save_rgb(tmp_path / "cat" / "a.png", (255, 0, 0))
    _save_rgb(tmp_path / "dog" / "b.jpg", (0, 255, 0))

    dataset = ImageFolderClassificationDataset(
        tmp_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert dataset.classes == ["cat", "dog"]
    assert len(dataset) == 2
    assert torch.equal(image, torch.tensor([5, 4]))
    assert target == 0


def test_manifest_classification_dataset_supports_string_labels_and_csv(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    _save_rgb(data_dir / "img_a.png", (255, 0, 0))
    _save_rgb(data_dir / "img_b.png", (0, 255, 0))
    manifest_path = data_dir / "labels.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerow({"image": "img_a.png", "label": "zebra"})
        writer.writerow({"image": "img_b.png", "label": "ant"})
    monkeypatch.chdir(other_cwd)

    dataset = ManifestClassificationDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert dataset.classes == ["ant", "zebra"]
    assert torch.equal(image, torch.tensor([5, 4]))
    assert target == 1


def test_manifest_image_text_dataset_returns_text_sample_and_path(tmp_path) -> None:
    _save_rgb(tmp_path / "image.png", (0, 0, 255))
    manifest_path = tmp_path / "pairs.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image": "image.png",
                "prompt": "a blue square",
                "sample_id": "sample-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ManifestImageTextDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    sample = dataset[0]

    assert torch.equal(sample["image"], torch.tensor([5, 4]))
    assert sample["text"] == "a blue square"
    assert sample["sample_id"] == "sample-1"
    assert sample["image_path"] == str((tmp_path / "image.png").resolve())


def test_manifest_regression_dataset_returns_numeric_target(tmp_path) -> None:
    _save_rgb(tmp_path / "sample.png", (12, 34, 56))
    manifest_path = tmp_path / "scores.jsonl"
    manifest_path.write_text(
        json.dumps({"image": "sample.png", "target": "3.5"}) + "\n",
        encoding="utf-8",
    )

    dataset = ManifestRegressionDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert torch.equal(image, torch.tensor([5, 4]))
    assert target == 3.5


def test_manifest_pair_dataset_returns_pair_sample(tmp_path) -> None:
    _save_rgb(tmp_path / "left.png", (1, 2, 3))
    _save_rgb(tmp_path / "right.png", (4, 5, 6))
    manifest_path = tmp_path / "pairs.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_a": "left.png",
                "image_b": "right.png",
                "label": 1,
                "text_a": "left text",
                "text_b": "right text",
                "sample_id": "pair-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ManifestPairDataset(
        manifest_path,
        transform=_to_marker_tensor,
        text_a_key="text_a",
        text_b_key="text_b",
        include_paths=True,
    )
    sample = dataset[0]

    assert torch.equal(sample["image_a"], torch.tensor([5, 4]))
    assert torch.equal(sample["image_b"], torch.tensor([5, 4]))
    assert sample["label"] == 1
    assert sample["text_a"] == "left text"
    assert sample["text_b"] == "right text"
    assert sample["sample_id"] == "pair-1"
    assert sample["image_a_path"] == str((tmp_path / "left.png").resolve())
    assert sample["image_b_path"] == str((tmp_path / "right.png").resolve())


def test_dataset_templates_are_registered() -> None:
    assert DATASET_REGISTRY.get("ManifestRecordDataset") is ManifestRecordDataset
    assert (
        DATASET_REGISTRY.get("ImageFolderClassificationDataset")
        is ImageFolderClassificationDataset
    )
    assert (
        DATASET_REGISTRY.get("ManifestClassificationDataset")
        is ManifestClassificationDataset
    )
    assert DATASET_REGISTRY.get("ManifestRegressionDataset") is ManifestRegressionDataset
    assert DATASET_REGISTRY.get("ManifestImageTextDataset") is ManifestImageTextDataset
    assert DATASET_REGISTRY.get("ManifestPairDataset") is ManifestPairDataset
