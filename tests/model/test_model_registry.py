"""模型注册契约回归测试."""

from __future__ import annotations

from xdl.utils.registry import MODEL_REGISTRY

BUILDING_BLOCKS = (
    "BasicBlock",
    "Bottleneck",
    "PatchEmbedding",
    "MultiHeadAttention",
    "TransformerBlock",
)


def test_internal_building_blocks_are_not_registered() -> None:
    """内部构件不得再注册为 MODEL."""

    available = set(MODEL_REGISTRY.list_available())
    for name in BUILDING_BLOCKS:
        assert name not in available, name


def test_building_blocks_remain_importable() -> None:
    """取消注册不改变公共导入面."""

    from xdl.model import (
        BasicBlock,
        Bottleneck,
        MultiHeadAttention,
        PatchEmbedding,
        TransformerBlock,
    )

    for component in (
        BasicBlock,
        Bottleneck,
        MultiHeadAttention,
        PatchEmbedding,
        TransformerBlock,
    ):
        assert component is not None


def test_core_and_research_models_are_registered() -> None:
    """核心模型和低层 SR 研究模型都可从 registry 构建."""

    available = set(MODEL_REGISTRY.list_available())
    assert {"ResNet", "VGG", "VisionTransformer", "RealPLKSR"} <= available
