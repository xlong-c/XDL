"""hair 数据集契约回归测试."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("albumentations")
pytest.importorskip("cv2")


def test_grid_image_dataset_requires_explicit_base_dir() -> None:
    """base_dir 不得再带个人机器硬编码默认值."""

    from xdl.dataset.hair.hairdata import GridImageDataset

    init_signature = inspect.signature(GridImageDataset.__init__)
    assert init_signature.parameters["base_dir"].default is inspect.Parameter.empty

    paths_signature = inspect.signature(GridImageDataset.get_image_paths_from_csv)
    assert paths_signature.parameters["base_dir"].default is inspect.Parameter.empty


def test_hair_sources_have_no_personal_paths_or_demo_blocks() -> None:
    """框架 dataset 源码不保留个人路径和 __main__ 演示块."""

    import xdl.dataset.hair.hairdata as hairdata_module

    hair_dir = Path(hairdata_module.__file__).parent
    for path in sorted(hair_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' not in source, path
        assert "autodl" not in source, path
        assert "F:\\\\" not in source, path
