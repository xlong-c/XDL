from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "learn" / "gdn" / "gdn.py"
SPEC = importlib.util.spec_from_file_location("gdn", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
gdn = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gdn
SPEC.loader.exec_module(gdn)

SimpleGatedDeltaNet = gdn.SimpleGatedDeltaNet
gated_delta_rule_chunkwise = gdn.gated_delta_rule_chunkwise
gated_delta_rule_recurrent = gdn.gated_delta_rule_recurrent


def test_recurrent_rule_returns_expected_shapes() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 7, 3, 4)
    key = torch.randn(2, 7, 3, 4)
    value = torch.randn(2, 7, 3, 5)
    g = -torch.rand(2, 7, 3)
    beta = torch.rand(2, 7, 3)

    out, state = gated_delta_rule_recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        output_final_state=True,
    )

    assert out.shape == (2, 7, 3, 5)
    assert state is not None
    assert state.shape == (2, 3, 4, 5)


def test_chunkwise_matches_recurrent() -> None:
    torch.manual_seed(1)
    query = torch.randn(2, 9, 4, 6)
    key = torch.randn(2, 9, 4, 6)
    value = torch.randn(2, 9, 4, 5)
    g = -torch.rand(2, 9, 4)
    beta = torch.rand(2, 9, 4)

    recurrent_out, recurrent_state = gated_delta_rule_recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        output_final_state=True,
    )
    chunk_out, chunk_state = gated_delta_rule_chunkwise(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=4,
        output_final_state=True,
    )

    assert recurrent_state is not None
    assert chunk_state is not None
    assert torch.allclose(recurrent_out, chunk_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(recurrent_state, chunk_state, atol=1e-5, rtol=1e-5)


def test_simple_gdn_modes_match() -> None:
    torch.manual_seed(2)
    model = SimpleGatedDeltaNet(
        hidden_size=24,
        num_heads=3,
        head_dim=4,
        value_head_dim=6,
    )
    x = torch.randn(2, 10, 24)

    recurrent_out, recurrent_state = model(x, mode="recurrent", return_state=True)
    chunk_out, chunk_state = model(x, mode="chunkwise", chunk_size=5, return_state=True)

    assert recurrent_state is not None
    assert chunk_state is not None
    assert recurrent_out.shape == (2, 10, 24)
    assert torch.allclose(recurrent_out, chunk_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(recurrent_state, chunk_state, atol=1e-5, rtol=1e-5)


def test_chunkwise_rejects_invalid_chunk_size() -> None:
    query = torch.randn(1, 4, 2, 3)
    key = torch.randn(1, 4, 2, 3)
    value = torch.randn(1, 4, 2, 3)
    g = -torch.rand(1, 4, 2)
    beta = torch.rand(1, 4, 2)

    with pytest.raises(ValueError, match="chunk_size"):
        gated_delta_rule_chunkwise(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            chunk_size=0,
        )
