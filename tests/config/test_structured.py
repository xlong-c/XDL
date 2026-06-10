from dataclasses import dataclass, field
from typing import List

from xdl.config import load_structured_dataclass_config


@dataclass
class SmallConfig:
    output_dir: str = "./outputs"
    epochs: int = 1
    values: List[int] = field(default_factory=lambda: [1])


def test_load_structured_dataclass_config_merges_yaml_and_overrides(tmp_path) -> None:
    config_path = tmp_path / "small.yaml"
    config_path.write_text(
        """
output_dir: ${xdl.join_path:root,run}
values: [2, 3]
""",
        encoding="utf-8",
    )

    config = load_structured_dataclass_config(
        SmallConfig,
        config_path,
        overrides={"epochs": 4},
    )

    assert isinstance(config, SmallConfig)
    assert config.output_dir == "root/run"
    assert config.epochs == 4
    assert config.values == [2, 3]
