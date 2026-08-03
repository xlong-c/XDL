from pathlib import Path
import random

import pytest
import torch
import yaml

from xdl.utils import resolve_dtype, save_yaml, seed_everything


def test_resolve_dtype_aliases() -> None:
    assert resolve_dtype("fp32", "cpu") is torch.float32
    assert resolve_dtype("float16", "cuda") is torch.float16
    assert resolve_dtype("bf16", "cuda") is torch.bfloat16
    assert resolve_dtype("auto", "cpu") is torch.float32
    assert resolve_dtype("auto", "cuda") is torch.bfloat16


def test_resolve_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="不支持的 dtype"):
        resolve_dtype("int8", "cpu")


def test_seed_everything_seeds_python_and_torch_rng() -> None:
    seed_everything(123)
    first_random = random.random()
    first_tensor = torch.rand(3)

    seed_everything(123)
    assert random.random() == first_random
    assert torch.equal(torch.rand(3), first_tensor)


def test_save_yaml_converts_paths_and_tuples(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "config.yaml"
    save_yaml(
        {
            "path": tmp_path / "data",
            "tuple_value": ("a", tmp_path / "b"),
            "nested": {"items": [tmp_path / "c"]},
        },
        output_path,
    )

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "path": str(tmp_path / "data"),
        "tuple_value": ["a", str(tmp_path / "b")],
        "nested": {"items": [str(tmp_path / "c")]},
    }
