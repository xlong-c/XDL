import torch

from xdl.config.builder import (
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_task,
    build_optimizer,
    build_scheduler,
    build_transform,
)
from xdl.config.errors import ConfigValidationError
from xdl.model.generate import (
    IdentityRepresentor,
    RepresentationScatteringField,
    TBSMGenerator,
)
from xdl.trainer import TBSMCoreModel


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


def test_builder_materializes_nested_tbsm_task_components() -> None:
    task = build_task(
        {
            "target": "xdl.trainer:TBSMCoreModel",
            "params": {
                "generator": {
                    "target": "registry:TBSMGenerator",
                    "params": {
                        "backbone": {
                            "target": "torch.nn:Linear",
                            "params": {"in_features": 3, "out_features": 3},
                        }
                    },
                },
                "representation_fields": [
                    {
                        "target": "registry:RepresentationScatteringField",
                        "params": {
                            "representor": {
                                "target": "registry:IdentityRepresentor",
                                "params": {},
                            },
                            "lambda_weight": 0.0,
                            "rho": 0.0,
                            "num_classes": 3,
                        },
                    }
                ],
                "t_sampling": [1.0],
            },
        }
    )

    assert isinstance(task, TBSMCoreModel)
    assert isinstance(task.generator, TBSMGenerator)
    assert isinstance(task.representation_fields[0], RepresentationScatteringField)
    assert isinstance(task.representation_fields[0].representor, IdentityRepresentor)


def test_builder_rejects_unsupported_component_fields() -> None:
    try:
        build_model(
            {
                "target": "torch.nn:Linear",
                "unexpected": True,
                "params": {"in_features": 4, "out_features": 2},
            }
        )
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError as exc:
        assert "unsupported fields" in str(exc)


def test_builder_rejects_missing_target_wrapper_shape() -> None:
    model = build_model(
        {
            "target": "torch.nn:Linear",
            "params": {"in_features": 4, "out_features": 2},
        }
    )

    try:
        build_optimizer(
            model,
            {
                "optimizer": {
                    "target": "torch.optim:SGD",
                    "params": {"lr": 0.01},
                }
            },
        )
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError as exc:
        assert "requires 'target'" in str(exc)


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


def test_builder_rejects_transform_mapping_without_target() -> None:
    try:
        build_transform({"unexpected": []})
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError as exc:
        assert "requires 'target'" in str(exc)


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
