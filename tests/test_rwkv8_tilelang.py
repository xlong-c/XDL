from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "learn" / "rwkv" / "rwkv8" / "rwkv8_tilelang.py"
SPEC = importlib.util.spec_from_file_location("rwkv8_tilelang", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
rwkv8_tilelang = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rwkv8_tilelang
SPEC.loader.exec_module(rwkv8_tilelang)

RWKV8Rosa1Bit = rwkv8_tilelang.RWKV8Rosa1Bit
RWKV8Rosa4Bit = rwkv8_tilelang.RWKV8Rosa4Bit
TinyRWKV8RosaLM = rwkv8_tilelang.TinyRWKV8RosaLM
pack_bit_symbols = rwkv8_tilelang.pack_bit_symbols
rosa_suffix_match_ref = rwkv8_tilelang.rosa_suffix_match_ref
rosa_suffix_match_tilelang = rwkv8_tilelang.rosa_suffix_match_tilelang
unpack_bit_symbols = rwkv8_tilelang.unpack_bit_symbols


def test_rosa_suffix_match_ref_known_sequence() -> None:
    q = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    k = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1]], dtype=torch.uint8)
    v = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]], dtype=torch.uint8)

    out, hit = rosa_suffix_match_ref(q, k, v)

    assert out.tolist() == [[0, 0, 1, 0, 1, 1, 0, 1]]
    assert hit.tolist() == [[0, 0, 1, 1, 1, 1, 1, 1]]


def test_pack_and_unpack_bit_symbols_round_trip() -> None:
    x = torch.tensor(
        [
            [
                [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0],
                [1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0],
            ]
        ]
    )

    symbols = pack_bit_symbols(x, bits_per_symbol=4)
    restored = unpack_bit_symbols(symbols, bits_per_symbol=4, dtype=x.dtype)

    assert symbols.tolist() == [[[10, 5], [5, 10]]]
    assert torch.equal(restored, torch.sign(x))


def test_rwkv8_rosa_layers_return_expected_shapes() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)

    out1, hit1 = RWKV8Rosa1Bit(channels=8, backend="torch")(q, k, v, return_hit=True)
    out4, hit4 = RWKV8Rosa4Bit(channels=8, backend="torch")(q, k, v, return_hit=True)

    assert out1.shape == q.shape
    assert hit1.shape == q.shape
    assert out4.shape == q.shape
    assert hit4.shape == (2, 5, 2)


def test_tiny_rwkv8_rosa_lm_forward_cpu() -> None:
    model = TinyRWKV8RosaLM(
        vocab_size=32,
        channels=8,
        num_layers=1,
        rosa_bits=1,
        backend="torch",
    )
    tokens = torch.randint(0, 32, (2, 6))

    logits = model(tokens)

    assert logits.shape == (2, 6, 32)


def test_tilelang_backend_requires_cuda_tensors() -> None:
    q = torch.zeros(1, 4, dtype=torch.uint8)

    with pytest.raises(ValueError, match="CUDA tensors"):
        rosa_suffix_match_tilelang(q, q, q)
