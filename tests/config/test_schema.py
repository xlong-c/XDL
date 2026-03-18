from xdl.config import (
    CONFIG_SCHEMA_VERSION,
    ConfigSchemaV1,
    create_structured_config,
    load_config_with_schema,
    to_plain_dict,
)
from xdl.config.errors import ConfigValidationError


def test_can_create_structured_schema() -> None:
    cfg = create_structured_config()

    assert cfg.config_version == CONFIG_SCHEMA_VERSION
    assert cfg.runtime.device == "cuda"
    assert isinstance(ConfigSchemaV1(), ConfigSchemaV1)


def test_can_merge_and_resolve_references() -> None:
    raw_config = {
        "runtime": {
            "experiment_name": "schema-smoke",
        },
        "trainer": {
            "max_epochs": 5,
        },
        "model": {
            "type": "vgg16_bn",
            "source": "registry",
            "params": {
                "num_classes": 100,
            },
        },
        "data": {
            "transforms": {
                "train": {
                    "type": "compose",
                    "items": [
                        {
                            "type": "Resize",
                            "source": "torchvision.transforms",
                            "params": {"size": [224, 224]},
                        }
                    ],
                }
            },
            "datasets": {
                "train": {
                    "type": "CIFAR100",
                    "source": "torchvision.datasets",
                    "params": {
                        "root": "./data",
                        "train": True,
                    },
                    "transform": "${data.transforms.train}",
                }
            },
            "dataloaders": {
                "train": {
                    "dataset": "${data.datasets.train}",
                    "params": {"batch_size": 8, "shuffle": True},
                }
            },
        },
        "optimization": {
            "optimizer": {
                "type": "SGD",
                "source": "torch.optim",
                "params": {"lr": 0.01},
            }
        },
        "loss": [
            {
                "type": "CrossEntropyLoss",
                "source": "torch.nn",
                "params": {},
            }
        ],
        "metrics": [
            {
                "type": "Accuracy",
                "source": "registry",
                "params": {"num_classes": 100},
            }
        ],
    }

    cfg = load_config_with_schema(raw_config)
    resolved = to_plain_dict(cfg, resolve=True)

    assert cfg.trainer.max_epochs == 5
    assert resolved["data"]["datasets"]["train"]["transform"]["type"] == "compose"
    assert resolved["data"]["dataloaders"]["train"]["dataset"]["type"] == "CIFAR100"
    assert resolved["model"]["type"] == "vgg16_bn"


def test_unknown_top_level_field_fails_validation() -> None:
    raw_config = {
        "runtime": {"device": "cpu"},
        "unknown_block": {"enabled": True},
    }

    try:
        load_config_with_schema(raw_config)
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError:
        pass
