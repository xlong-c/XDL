import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mhc import (
    DecoderBlockMHC,
    ManifoldHyperConnection,
    sinkhorn_normalize,
    split_mixes_with_sinkhorn,
)


def test_sinkhorn_normalize_preserves_device_and_dtype() -> None:
    logits = torch.randn(2, 3, 3, dtype=torch.float64)
    matrix = sinkhorn_normalize(logits, iters=4, eps=1e-8)

    assert matrix.dtype == logits.dtype
    assert matrix.device == logits.device


def test_pre_zero_input_is_finite() -> None:
    mhc = ManifoldHyperConnection(dim=4, rate=2, layer_id=0, sinkhorn_iters=4)
    residual = torch.zeros(1, 3, 2, 4)

    post_mix, comb_mix, layer_input = mhc.pre(residual)

    assert torch.isfinite(post_mix).all()
    assert torch.isfinite(comb_mix).all()
    assert torch.isfinite(layer_input).all()


def test_pre_matches_expected_weighted_sum() -> None:
    residual = torch.tensor([[[[1.0, 2.0], [10.0, 20.0]]]])
    pre_mix = torch.tensor([[[0.25, 0.75]]])
    layer_input = (pre_mix.unsqueeze(-1) * residual).sum(dim=-2)
    expected = torch.tensor([[[7.75, 15.5]]])
    assert torch.allclose(layer_input, expected)


def test_post_uses_transposed_combination() -> None:
    mhc = ManifoldHyperConnection(dim=2, rate=2, layer_id=0, sinkhorn_iters=2)

    residual = torch.tensor([[[[1.0, 2.0], [10.0, 20.0]]]])
    layer_output = torch.tensor([[[3.0, 4.0]]])
    post_mix = torch.tensor([[[[0.5], [1.5]]]])
    comb_mix = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])

    out = mhc.post(layer_output, residual, post_mix, comb_mix)
    expected = torch.tensor([[[[1.5, 2.0], [5.5, 8.0]]]])

    assert torch.allclose(out, expected)


def test_pre_post_path_has_gradient() -> None:
    mhc = ManifoldHyperConnection(dim=4, rate=2, layer_id=0, sinkhorn_iters=3)
    residual = torch.randn(2, 3, 2, 4, requires_grad=True)

    post_mix, comb_mix, layer_input = mhc.pre(residual)
    out = mhc.post(layer_input, residual, post_mix, comb_mix)
    loss = out.sum()
    loss.backward()

    assert mhc.mix_weight.grad is not None
    assert torch.isfinite(mhc.mix_weight.grad).all()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_split_mixes_sinkhorn_is_nearly_doubly_stochastic() -> None:
    mixes = torch.randn(2, 3, 8)
    scale = torch.ones(3) * 0.01
    base = torch.zeros(8)

    _, _, comb_mix = split_mixes_with_sinkhorn(
        mixes=mixes,
        scale=scale,
        base=base,
        hc_mult=2,
        sinkhorn_iters=8,
        eps=1e-6,
    )

    row_sums = comb_mix.sum(dim=-1)
    col_sums = comb_mix.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)


def test_decoder_block_mhc_shape_and_gradient() -> None:
    block = DecoderBlockMHC(dim=8, rate=2, layer_id=0, sinkhorn_iters=4, ffn_hidden_dim=16)
    residual = torch.randn(2, 3, 2, 8, requires_grad=True)

    out = block(residual)
    loss = out.sum()
    loss.backward()

    assert out.shape == residual.shape
    assert torch.isfinite(out).all()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
