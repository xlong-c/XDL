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
            "target": "torch.nn:Linear",
            "params": {"in_features": 4, "out_features": 2},
        }
    )
    optimizer = build_optimizer(
        model,
        {
            "target": "torch.optim:SGD",
            "params": {"lr": 0.01},
        },
    )
    scheduler = build_scheduler(
        optimizer,
        {
            "target": "torch.optim.lr_scheduler:StepLR",
            "params": {"step_size": 1, "gamma": 0.5},
        },
    )
    loss_fn = build_loss(
        {
            "target": "torch.nn:CrossEntropyLoss",
            "params": {},
        }
    )
    metrics = build_metrics(
        [
            {
                "target": "registry:Accuracy",
                "params": {"num_classes": 2},
            }
        ]
    )
    transform = build_transform(
        [
            "torchvision.transforms:ToTensor",
        ]
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
            "target": "registry:SyntheticClassificationDataset",
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


def test_builder_supports_dataset_param_transform_target() -> None:
    dataset = build_dataset(
        {
            "target": "torchvision.datasets:FakeData",
            "params": {
                "size": 2,
                "image_size": [1, 2, 2],
                "num_classes": 2,
                "transform": {
                    "target": "torchvision.transforms:Compose",
                    "transforms": [
                        "torchvision.transforms:ToTensor",
                    ],
                },
            },
        }
    )

    feature, target = dataset[0]
    assert tuple(feature.shape) == (1, 2, 2)
    assert int(target) in {0, 1}


def test_builder_supports_compact_transform_list_with_inline_params() -> None:
    transform = build_transform(
        [
            {
                "target": "torchvision.transforms:Resize",
                "size": [8, 8],
            },
            "torchvision.transforms:ToTensor",
        ]
    )

    assert callable(transform)
    assert type(transform).__name__ == "Compose"
