"""FSDP 配置构建测试."""

import pytest

from xdl.config import FSDPConfig, build_fsdp_plugin
from xdl.errors import TrainingError


def test_fsdp_int_shortcuts() -> None:
    p1 = build_fsdp_plugin(1)
    assert p1.fsdp_version != 2
    assert p1.use_orig_params is True
    assert "FULL_SHARD" in str(p1.reshard_after_forward)

    p2 = build_fsdp_plugin(2)
    assert p2.fsdp_version == 2
    assert p2.reshard_after_forward is True


def test_fsdp_dict_passthrough() -> None:
    plugin = build_fsdp_plugin(
        {
            "fsdp_version": 1,
            "auto_wrap_policy": "transformer_based_wrap",
            "transformer_cls_names_to_wrap": ["TransformerBlock"],
            "limit_all_gathers": True,
        }
    )
    assert plugin.fsdp_version == 1
    assert plugin.transformer_cls_names_to_wrap == ["TransformerBlock"]
    assert callable(plugin.auto_wrap_policy)


def test_fsdp_dict_sharding_strategy_maps_to_reshard_after_forward() -> None:
    plugin = build_fsdp_plugin(
        {
            "fsdp_version": 1,
            "sharding_strategy": "FULL_SHARD",
            "use_orig_params": True,
        }
    )
    assert "FULL_SHARD" in str(plugin.reshard_after_forward)
    assert "FULL_SHARD" in str(plugin.sharding_strategy)


def test_fsdp_structured_config() -> None:
    plugin = build_fsdp_plugin(
        FSDPConfig(fsdp_version=1, cpu_offload=False, activation_checkpointing=True)
    )
    assert plugin.fsdp_version == 1
    assert plugin.activation_checkpointing is True


def test_fsdp_invalid_values_raise() -> None:
    with pytest.raises(TrainingError):
        build_fsdp_plugin(3)
    with pytest.raises(TrainingError):
        build_fsdp_plugin({"not_a_plugin_key": 1})
