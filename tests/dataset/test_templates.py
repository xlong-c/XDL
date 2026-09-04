import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from xdl.config.builder import build_dataset
from xdl.dataset import (
    BasenameAlignedDataset,
    ImageFolderDataset,
    ImageFolderClassificationDataset,
    ImagePromptDataset,
    ImageTextSidecarDataset,
    RecordClassificationDataset,
    RecordDataset,
    RecordDatasetBase,
    RecordImageTextDataset,
    RecordMultiLabelClassificationDataset,
    RecordPairDataset,
    RecordRegressionDataset,
    RecordTextDataset,
    RecordTripletDataset,
    split_dataset,
    train_val_split,
)
from xdl.utils.registry import DATASET_REGISTRY


def _save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (5, 4), color=color).save(path)


def _to_marker_tensor(image: Image.Image) -> torch.Tensor:
    return torch.tensor([image.size[0], image.size[1]], dtype=torch.long)


def _prefix_text(value: str) -> str:
    return f"tok::{value}"


def test_manifest_record_dataset_loads_jsonl_and_repeat(tmp_path) -> None:
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "a", "value": 1}) + "\n"
        + json.dumps({"id": "b", "value": 2}) + "\n",
        encoding="utf-8",
    )

    dataset = RecordDataset(manifest_path, repeat=2)

    assert len(dataset) == 4
    assert dataset[0] == {"id": "a", "value": 1}
    assert dataset[2] == {"id": "a", "value": 1}


def test_manifest_dataset_base_supports_thin_custom_adapters(tmp_path) -> None:
    _save_rgb(tmp_path / "sample.png", (255, 0, 0))
    manifest_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        json.dumps({"image": "sample.png", "sample_id": "thin-1"}) + "\n",
        encoding="utf-8",
    )

    class TinyManifestAdapter(RecordDatasetBase):
        def __getitem__(self, index: int) -> dict[str, str]:
            base_index, record = self._record_at(index)
            image_path = self._resolve_record_path(record, "image")
            return {
                "sample_id": self._sample_id_from_path(record, image_path, base_index),
                "image_path": str(image_path),
            }

    dataset = TinyManifestAdapter(manifest_path)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample == {
        "sample_id": "thin-1",
        "image_path": str((tmp_path / "sample.png").resolve()),
    }


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


def test_image_folder_dataset_returns_image_sample_and_repeat(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    _save_rgb(tmp_path / "nested" / "b.jpg", (0, 255, 0))

    dataset = ImageFolderDataset(
        tmp_path,
        transform=_to_marker_tensor,
        recursive=True,
        sample_id_from="relative_path",
        repeat=2,
    )
    sample = dataset[0]

    assert len(dataset) == 4
    assert torch.equal(sample["image"], torch.tensor([5, 4]))
    assert sample["sample_id"] == "a.png"
    assert sample["image_path"] == str((tmp_path / "a.png").resolve())
    assert dataset[2]["sample_id"] == "a.png"


def test_image_folder_dataset_can_disable_recursive_scan(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    _save_rgb(tmp_path / "nested" / "b.jpg", (0, 255, 0))

    dataset = ImageFolderDataset(tmp_path, recursive=False)

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "a"


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

    dataset = RecordClassificationDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert dataset.classes == ["ant", "zebra"]
    assert torch.equal(image, torch.tensor([5, 4]))
    assert target == 1


def test_manifest_image_prompt_dataset_supports_prompt_aliases_and_repeat(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "prompts.jsonl"
    _save_rgb(image_path, (255, 0, 0))
    manifest_path.write_text(
        json.dumps({"image": image_path.name, "caption": "red object"}) + "\n",
        encoding="utf-8",
    )

    dataset = ImagePromptDataset(
        manifest_path,
        transform=lambda image: torch.tensor(image.size),
        text_keys=("prompt", "caption", "text"),
        repeat=2,
    )

    image, prompt = dataset[1]
    assert tuple(image.tolist()) == (5, 4)
    assert prompt == "red object"
    assert len(dataset) == 2


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

    dataset = RecordImageTextDataset(
        manifest_path,
        transform=_to_marker_tensor,
        text_transform=_prefix_text,
    )
    sample = dataset[0]

    assert torch.equal(sample["image"], torch.tensor([5, 4]))
    assert sample["text"] == "tok::a blue square"
    assert sample["sample_id"] == "sample-1"
    assert sample["image_path"] == str((tmp_path / "image.png").resolve())


def test_image_text_sidecar_dataset_reads_same_directory_txt(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "000.png", (0, 0, 255))
    (tmp_path / "000.txt").write_text("first prompt\nsecond prompt\n", encoding="utf-8")

    dataset = ImageTextSidecarDataset(
        root=tmp_path,
        transform=_to_marker_tensor,
        text_transform=_prefix_text,
        text_selection="first_line",
    )
    sample = dataset[0]

    assert torch.equal(sample["image"], torch.tensor([5, 4]))
    assert sample["text"] == "tok::first prompt"
    assert sample["sample_id"] == "000"
    assert sample["image_path"] == str((tmp_path / "000.png").resolve())
    assert sample["text_path"] == str((tmp_path / "000.txt").resolve())


def test_image_text_sidecar_dataset_reads_split_roots_and_skips_missing(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    text_root = tmp_path / "texts"
    _save_rgb(image_root / "000.png", (0, 0, 255))
    _save_rgb(image_root / "001.png", (255, 0, 0))
    text_root.mkdir()
    (text_root / "000.txt").write_text("caption text\n", encoding="utf-8")

    dataset = ImageTextSidecarDataset(
        image_root=image_root,
        text_root=text_root,
        missing_text="skip",
        text_selection="full",
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["text"] == "caption text"
    assert sample["sample_id"] == "000"
    assert sample["text_path"] == str((text_root / "000.txt").resolve())


def test_image_text_sidecar_dataset_errors_on_missing_text(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "000.png", (0, 0, 255))

    with pytest.raises(FileNotFoundError, match="Missing sidecar text files"):
        ImageTextSidecarDataset(root=tmp_path)


def test_manifest_regression_dataset_returns_numeric_target(tmp_path) -> None:
    _save_rgb(tmp_path / "sample.png", (12, 34, 56))
    manifest_path = tmp_path / "scores.jsonl"
    manifest_path.write_text(
        json.dumps({"image": "sample.png", "target": "3.5"}) + "\n",
        encoding="utf-8",
    )

    dataset = RecordRegressionDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert torch.equal(image, torch.tensor([5, 4]))
    assert target == 3.5


def test_manifest_multilabel_dataset_supports_csv_string_labels(tmp_path) -> None:
    _save_rgb(tmp_path / "sample_a.png", (1, 2, 3))
    _save_rgb(tmp_path / "sample_b.png", (4, 5, 6))
    manifest_path = tmp_path / "multilabel.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "labels"])
        writer.writeheader()
        writer.writerow({"image": "sample_a.png", "labels": "cat,dog"})
        writer.writerow({"image": "sample_b.png", "labels": "dog"})

    dataset = RecordMultiLabelClassificationDataset(
        manifest_path,
        transform=_to_marker_tensor,
    )
    image, target = dataset[0]

    assert dataset.classes == ["cat", "dog"]
    assert torch.equal(image, torch.tensor([5, 4]))
    assert target.tolist() == [1.0, 1.0]
    assert dataset[1][1].tolist() == [0.0, 1.0]


def test_manifest_text_dataset_returns_target_text_and_record(tmp_path) -> None:
    manifest_path = tmp_path / "text.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt": "describe image",
                "response": "blue hair",
                "sample_id": "txt-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = RecordTextDataset(
        manifest_path,
        text_transform=_prefix_text,
        target_text_transform=_prefix_text,
        include_record=True,
    )
    sample = dataset[0]

    assert sample["text"] == "tok::describe image"
    assert sample["target_text"] == "tok::blue hair"
    assert sample["sample_id"] == "txt-1"
    assert sample["record"]["response"] == "blue hair"


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

    dataset = RecordPairDataset(
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


def test_manifest_triplet_dataset_returns_triplet_sample(tmp_path) -> None:
    _save_rgb(tmp_path / "anchor.png", (10, 11, 12))
    _save_rgb(tmp_path / "positive.png", (13, 14, 15))
    _save_rgb(tmp_path / "negative.png", (16, 17, 18))
    manifest_path = tmp_path / "triplets.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "anchor_image": "anchor.png",
                "positive_image": "positive.png",
                "negative_image": "negative.png",
                "anchor_text": "a",
                "positive_text": "p",
                "negative_text": "n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = RecordTripletDataset(
        manifest_path,
        transform=_to_marker_tensor,
        text_transform=_prefix_text,
        anchor_text_key="anchor_text",
        positive_text_key="positive_text",
        negative_text_key="negative_text",
        include_paths=True,
    )
    sample = dataset[0]

    assert torch.equal(sample["anchor_image"], torch.tensor([5, 4]))
    assert torch.equal(sample["positive_image"], torch.tensor([5, 4]))
    assert torch.equal(sample["negative_image"], torch.tensor([5, 4]))
    assert sample["anchor_text"] == "tok::a"
    assert sample["positive_text"] == "tok::p"
    assert sample["negative_text"] == "tok::n"
    assert sample["anchor_image_path"] == str((tmp_path / "anchor.png").resolve())
    assert sample["positive_image_path"] == str((tmp_path / "positive.png").resolve())
    assert sample["negative_image_path"] == str((tmp_path / "negative.png").resolve())


def test_builder_supports_manifest_text_dataset_text_transforms(tmp_path) -> None:
    manifest_path = tmp_path / "text.jsonl"
    manifest_path.write_text(
        json.dumps({"text": "hello", "target_text": "world"}) + "\n",
        encoding="utf-8",
    )

    dataset = build_dataset(
        {
            "target": "registry:RecordTextDataset",
            "params": {
                "manifest_path": str(manifest_path),
                "text_transform": {
                    "target": "torch.nn:Identity",
                    "params": {},
                },
                "target_text_transform": {
                    "target": "torch.nn:Identity",
                    "params": {},
                },
            },
        }
    )
    sample = dataset[0]

    assert sample["text"] == "hello"
    assert sample["target_text"] == "world"


def test_dataset_templates_are_registered() -> None:
    assert DATASET_REGISTRY.get("RecordDataset") is RecordDataset
    assert DATASET_REGISTRY.get("ImageFolderDataset") is ImageFolderDataset
    assert (
        DATASET_REGISTRY.get("ImageFolderClassificationDataset")
        is ImageFolderClassificationDataset
    )
    assert DATASET_REGISTRY.get("ImageTextSidecarDataset") is ImageTextSidecarDataset
    assert DATASET_REGISTRY.get("ImagePromptDataset") is ImagePromptDataset
    assert (
        DATASET_REGISTRY.get("RecordClassificationDataset")
        is RecordClassificationDataset
    )
    assert DATASET_REGISTRY.get("RecordRegressionDataset") is RecordRegressionDataset
    assert (
        DATASET_REGISTRY.get("RecordMultiLabelClassificationDataset")
        is RecordMultiLabelClassificationDataset
    )
    assert DATASET_REGISTRY.get("RecordImageTextDataset") is RecordImageTextDataset
    assert DATASET_REGISTRY.get("RecordTextDataset") is RecordTextDataset
    assert DATASET_REGISTRY.get("RecordPairDataset") is RecordPairDataset
    assert DATASET_REGISTRY.get("RecordTripletDataset") is RecordTripletDataset


# ---------------------------------------------------------------------------
# BasenameAlignedDataset
# ---------------------------------------------------------------------------


def test_basename_aligned_txt_sidecar_reads_file_and_returns_key(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    (tmp_path / "a.txt").write_text("hello world\n", encoding="utf-8")

    dataset = BasenameAlignedDataset(
        ".txt",
        root=tmp_path,
        sidecar_key="caption",
        sidecar_transform=lambda s: s.upper(),
    )
    sample = dataset[0]

    assert sample["caption"] == "HELLO WORLD"
    assert sample["sample_id"] == "a"
    assert sample["image_path"] == str((tmp_path / "a.png").resolve())
    assert sample["caption_path"] == str((tmp_path / "a.txt").resolve())


def test_basename_aligned_json_sidecar_parses_and_returns_key(tmp_path) -> None:
    _save_rgb(tmp_path / "img.png", (0, 255, 0))
    (tmp_path / "img.json").write_text(
        json.dumps({"tags": ["cat", "outdoor"], "score": 0.9}),
        encoding="utf-8",
    )

    dataset = BasenameAlignedDataset(".json", root=tmp_path)
    sample = dataset[0]

    assert sample["json"] == {"tags": ["cat", "outdoor"], "score": 0.9}
    assert sample["image_path"] == str((tmp_path / "img.png").resolve())
    assert sample["json_path"] == str((tmp_path / "img.json").resolve())


def test_basename_aligned_json_sidecar_with_custom_reader(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (0, 0, 0))
    (tmp_path / "a.meta").write_text("key=value\n", encoding="utf-8")

    dataset = BasenameAlignedDataset(
        ".meta",
        root=tmp_path,
        sidecar_key="meta",
        sidecar_reader=lambda p: dict(
            line.split("=", 1) for line in p.read_text("utf-8").strip().splitlines()
        ),
    )
    sample = dataset[0]

    assert sample["meta"] == {"key": "value"}


def test_basename_aligned_txt_split_roots_and_skip_missing(tmp_path) -> None:
    image_root = tmp_path / "images"
    text_root = tmp_path / "texts"
    _save_rgb(image_root / "001.png", (255, 0, 0))
    _save_rgb(image_root / "002.png", (0, 255, 0))
    text_root.mkdir()
    (text_root / "001.txt").write_text("present\n", encoding="utf-8")

    dataset = BasenameAlignedDataset(
        ".txt",
        image_root=image_root,
        sidecar_root=text_root,
        missing_sidecar="skip",
    )
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "001"


def test_basename_aligned_missing_sidecar_errors(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))

    with pytest.raises(FileNotFoundError, match="Missing sidecar"):
        BasenameAlignedDataset(".txt", root=tmp_path)


def test_basename_aligned_image_sidecar_reads_as_pil(tmp_path) -> None:
    # Use .jpg for the "sidecar" so it pairs correctly with .png main images.
    _save_rgb(tmp_path / "input.png", (100, 150, 200))
    Image.new("RGB", (5, 5), color=(50, 50, 50)).save(tmp_path / "input.jpg")

    dataset = BasenameAlignedDataset(
        ".jpg",
        root=tmp_path,
        sidecar_key="target",
        extensions=[".png"],
        image_mode="RGB",
    )
    sample = dataset[0]

    from PIL.Image import Image as PILImage

    assert isinstance(sample["image"], PILImage)
    assert isinstance(sample["target"], PILImage)
    assert sample["sample_id"] == "input"
    assert sample["target_path"] == str((tmp_path / "input.jpg").resolve())


def test_basename_aligned_can_apply_transform_to_image_only(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    (tmp_path / "a.txt").write_text("text\n", encoding="utf-8")

    dataset = BasenameAlignedDataset(
        ".txt",
        root=tmp_path,
        transform=_to_marker_tensor,
    )
    sample = dataset[0]

    assert torch.equal(sample["image"], torch.tensor([5, 4]))
    assert sample["txt"] == "text"


def test_basename_aligned_repeat(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    (tmp_path / "a.txt").write_text("ok\n", encoding="utf-8")

    dataset = BasenameAlignedDataset(".txt", root=tmp_path, repeat=3)
    assert len(dataset) == 3
    assert dataset[0]["sample_id"] == "a"
    assert dataset[1]["sample_id"] == "a"


def test_basename_aligned_is_registered() -> None:
    assert DATASET_REGISTRY.get("BasenameAlignedDataset") is BasenameAlignedDataset


# ---------------------------------------------------------------------------
# split_dataset / train_val_split
# ---------------------------------------------------------------------------


def test_split_dataset_produces_correct_subset_lengths(tmp_path) -> None:
    _save_rgb(tmp_path / "a.png", (255, 0, 0))
    _save_rgb(tmp_path / "b.png", (0, 255, 0))
    manifest_path = tmp_path / "rec.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "a", "value": 1}) + "\n"
        + json.dumps({"id": "b", "value": 2}) + "\n",
        encoding="utf-8",
    )
    dataset = RecordDataset(manifest_path)

    subsets = split_dataset(dataset, [1, 1], seed=42)
    assert len(subsets) == 2
    assert len(subsets[0]) == 1
    assert len(subsets[1]) == 1


def test_split_dataset_is_deterministic(tmp_path) -> None:
    manifest_path = tmp_path / "rec.jsonl"
    lines = "\n".join(
        json.dumps({"id": str(i), "value": i}) for i in range(10)
    )
    manifest_path.write_text(lines + "\n", encoding="utf-8")
    dataset = RecordDataset(manifest_path)

    a0 = [s.indices[:] for s in split_dataset(dataset, [5, 5], seed=99)]
    a1 = [s.indices[:] for s in split_dataset(dataset, [5, 5], seed=99)]
    assert a0 == a1

    # Different seed → different split.
    b = [s.indices[:] for s in split_dataset(dataset, [5, 5], seed=77)]
    assert a0 != b


def test_split_dataset_rejects_mismatched_lengths(tmp_path) -> None:
    manifest_path = tmp_path / "rec.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "x"}) + "\n" + json.dumps({"id": "y"}) + "\n",
        encoding="utf-8",
    )
    dataset = RecordDataset(manifest_path)

    with pytest.raises(ValueError, match="Sum of lengths"):
        split_dataset(dataset, [1, 2], seed=0)


def test_train_val_split_default_ratio(tmp_path) -> None:
    manifest_path = tmp_path / "rec.jsonl"
    lines = "\n".join(
        json.dumps({"id": str(i), "value": i}) for i in range(10)
    )
    manifest_path.write_text(lines + "\n", encoding="utf-8")
    dataset = RecordDataset(manifest_path)

    train, val = train_val_split(dataset, val_ratio=0.2, seed=42)
    assert len(train) == 8
    assert len(val) == 2


def test_train_val_split_zero_ratio(tmp_path) -> None:
    manifest_path = tmp_path / "rec.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n",
        encoding="utf-8",
    )
    dataset = RecordDataset(manifest_path)

    train, val = train_val_split(dataset, val_ratio=0.0)
    assert len(train) == 2
    assert len(val) == 0


def test_train_val_split_full_ratio(tmp_path) -> None:
    manifest_path = tmp_path / "rec.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n",
        encoding="utf-8",
    )
    dataset = RecordDataset(manifest_path)

    train, val = train_val_split(dataset, val_ratio=1.0)
    assert len(train) == 0
    assert len(val) == 2
