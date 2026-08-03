import torch
from torch import nn

from xdl.analysis import attention_rollout, attention_rollout_for_model, capture_attention_maps
from xdl.model.vit import VisionTransformer


def test_attention_rollout_preserves_square_shape() -> None:
    attention_1 = torch.tensor(
        [[[[0.7, 0.3], [0.2, 0.8]]]],
        dtype=torch.float32,
    )
    attention_2 = torch.tensor(
        [[[[0.6, 0.4], [0.1, 0.9]]]],
        dtype=torch.float32,
    )

    rollout = attention_rollout([attention_1, attention_2])

    assert rollout.shape == (1, 2, 2)
    row_sums = rollout.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_capture_attention_maps_supports_xdl_vit() -> None:
    model = VisionTransformer(
        img_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=2,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
    )
    x = torch.randn(1, 3, 8, 8)

    attentions = capture_attention_maps(model, x)

    assert len(attentions) == 2
    assert attentions[0].shape == (1, 4, 5, 5)


class FakeHFOutput:
    def __init__(self, attentions: tuple[torch.Tensor, ...]) -> None:
        self.attentions = attentions


class FakeHFVisionModel(nn.Module):
    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> FakeHFOutput:
        del x
        if not output_attentions:
            return FakeHFOutput(())
        attention = torch.tensor(
            [[[[0.8, 0.2], [0.1, 0.9]]]],
            dtype=torch.float32,
        )
        return FakeHFOutput((attention,))


def test_attention_rollout_for_model_supports_hf_style_outputs() -> None:
    model = FakeHFVisionModel()
    rollout = attention_rollout_for_model(model, torch.randn(1, 3, 8, 8))

    assert rollout.shape == (1, 2, 2)
