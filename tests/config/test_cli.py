from dataclasses import dataclass, field

import pytest

from xdl.config.cli import format_dataclass_cli_help, parse_dataclass_cli
from xdl.config.errors import ConfigValidationError


@dataclass
class NestedConfig:
    tile_size: int = 64
    tile: bool = True


@dataclass
class DemoConfig:
    input: str = "demo.png"
    fp16: bool = False
    nested: NestedConfig = field(default_factory=NestedConfig)


def test_parse_dataclass_cli_supports_flag_style() -> None:
    cfg = parse_dataclass_cli(
        DemoConfig,
        args=["--input", "x.png", "--fp16", "--nested.tile-size", "96", "--no-nested.tile"],
    )

    assert cfg.input == "x.png"
    assert cfg.fp16 is True
    assert cfg.nested.tile_size == 96
    assert cfg.nested.tile is False


def test_parse_dataclass_cli_supports_dotlist_style() -> None:
    cfg = parse_dataclass_cli(
        DemoConfig,
        args=["input=y.png", "fp16=true", "nested.tile_size=128"],
    )

    assert cfg.input == "y.png"
    assert cfg.fp16 is True
    assert cfg.nested.tile_size == 128
    assert cfg.nested.tile is True


def test_parse_dataclass_cli_rejects_unknown_field() -> None:
    with pytest.raises(ConfigValidationError, match="Unknown CLI field"):
        parse_dataclass_cli(DemoConfig, args=["--missing", "1"])


def test_parse_dataclass_cli_rejects_missing_value() -> None:
    with pytest.raises(ConfigValidationError, match="Missing value"):
        parse_dataclass_cli(DemoConfig, args=["--input"])


def test_format_dataclass_cli_help_lists_fields() -> None:
    help_text = format_dataclass_cli_help(DemoConfig, program="demo.py")

    assert "python demo.py" in help_text
    assert "--input" in help_text
    assert "--nested.tile-size" in help_text
