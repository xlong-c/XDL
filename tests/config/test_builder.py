import torch

from xdl.config.builder import (
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transform,
)
from xdl.config.errors import ConfigValidationError


def test_builder_resolves_torch_import_paths() -> None:
    model = build_model(
        {
            "type": "Linear",
            "source": "torch.nn",
            "params": {"in_features": 4, "out_features": 2},
        }
    )
    optimizer = build_optimizer(
        model,
        {
            "type": "SGD",
            "source": "torch.optim",
            "params": {"lr": 0.01},
        },
    )
    scheduler = build_scheduler(
        optimizer,
        {
            "type": "StepLR",
            "source": "torch.optim.lr_scheduler",
            "params": {"step_size": 1, "gamma": 0.5},
        },
    )
    loss_fn = build_loss(
        {
            "type": "CrossEntropyLoss",
            "source": "torch.nn",
            "params": {},
        }
    )
    metrics = build_metrics(
        [
            {
                "type": "Accuracy",
                "source": "registry",
                "params": {"num_classes": 2},
            }
        ]
    )
    transform = build_transform(
        {
            "type": "compose",
            "items": [
                {
                    "type": "ToTensor",
                    "source": "torchvision.transforms",
                    "params": {},
                }
            ],
        }
    )

    assert isinstance(model, torch.nn.Linear)
    assert isinstance(optimizer, torch.optim.SGD)
    assert type(scheduler).__name__ == "StepLR"
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert len(metrics) == 1
    assert callable(transform)


def test_builder_keeps_legacy_torch_aliases_working() -> None:
    model = build_model(
        {
            "name": "Linear",
            "from_library": "torch",
            "params": {"in_features": 4, "out_features": 2},
        }
    )
    optimizer = build_optimizer(
        model,
        {
            "main_optimizer": {
                "name": "SGD",
                "from_library": "torch",
                "params": {"lr": 0.01},
            }
        },
    )

    assert isinstance(model, torch.nn.Linear)
    assert isinstance(optimizer, torch.optim.SGD)


def test_builder_can_build_local_dataset_from_registry() -> None:
    dataset = build_dataset(
        {
            "type": "SyntheticClassificationDataset",
            "source": "registry",
            "params": {
                "num_samples": 5,
                "input_shape": [4],
                "num_classes": 3,
            },
        }
    )

    feature, target = dataset[0]
    assert len(dataset) == 5
    assert tuple(feature.shape) == (4,)
    assert int(target) in {0, 1, 2}


def test_unknown_transform_pipeline_type_fails() -> None:
    try:
        build_transform({"type": "sequential", "items": []})
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError:
        pass
