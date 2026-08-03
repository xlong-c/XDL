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
    assert cfg.runtime.data_dir == "./data"
    assert isinstance(ConfigSchemaV1(), ConfigSchemaV1)


def test_can_merge_and_resolve_references() -> None:
    raw_config = {
        "runtime": {
            "experiment_name": "schema-smoke",
            "data_dir": "./dataset_root",
        },
        "trainer": {
            "max_epochs": 5,
            "batch_size": 8,
        },
        "model": {
            "target": "registry:vgg16_bn",
            "params": {
                "num_classes": 100,
            },
        },
        "dataloader_defaults": {
            "batch_size": "${trainer.batch_size}",
            "num_workers": 2,
            "pin_memory": False,
        },
        "train_transforms": {
            "target": "torchvision.transforms:Compose",
            "params": {
                "transforms": [
                    {
                        "target": "torchvision.transforms:Resize",
                        "params": {"size": [224, 224]},
                    }
                ]
            },
        },
        "train_dataset": {
            "target": "torchvision.datasets:CIFAR100",
            "params": {
                "root": "${runtime.data_dir}",
                "train": True,
                "transform": "${train_transforms}",
            },
        },
        "train_dataloader": {
            "dataset": "${train_dataset}",
            "params": {"shuffle": True},
        },
        "optimization": {
            "optimizer": {
                "target": "torch.optim:SGD",
                "params": {"lr": 0.01},
            }
        },
        "loss": [
            {
                "target": "torch.nn:CrossEntropyLoss",
                "params": {},
            }
        ],
        "metrics": [
            {
                "target": "registry:Accuracy",
                "params": {"num_classes": 100},
            }
        ],
    }

    cfg = load_config_with_schema(raw_config)
    resolved = to_plain_dict(cfg, resolve=True)

    assert cfg.trainer.max_epochs == 5
    assert resolved["runtime"]["data_dir"] == "./dataset_root"
    assert resolved["dataloader_defaults"]["batch_size"] == 8
    assert resolved["train_dataset"]["params"]["transform"]["target"] == "torchvision.transforms:Compose"
    assert resolved["train_dataset"]["params"]["root"] == "./dataset_root"
    assert resolved["train_dataloader"]["dataset"]["target"] == "torchvision.datasets:CIFAR100"
    assert resolved["model"]["target"] == "registry:vgg16_bn"


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


def test_unsupported_component_fields_fail_validation() -> None:
    raw_config = {
        "model": {
            "target": "torch.nn:Linear",
            "unexpected": True,
            "params": {
                "in_features": 4,
                "out_features": 2,
            },
        }
    }

    try:
        load_config_with_schema(raw_config)
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError:
        pass


def test_single_loss_object_is_normalized_to_list() -> None:
    cfg = load_config_with_schema(
        {
            "loss": {
                "target": "torch.nn:CrossEntropyLoss",
                "params": {},
            }
        }
    )
    resolved = to_plain_dict(cfg, resolve=True)

    assert isinstance(resolved["loss"], list)
    assert len(resolved["loss"]) == 1
    assert resolved["loss"][0]["target"] == "torch.nn:CrossEntropyLoss"
