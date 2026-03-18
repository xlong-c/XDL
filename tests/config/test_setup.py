from textwrap import dedent

from xdl.config import load_config_with_schema, setup_from_yaml
from xdl.config.builder import build_model
from xdl.config.errors import ConfigValidationError


def test_setup_from_yaml_supports_new_schema(tmp_path) -> None:
    config_path = tmp_path / "new_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
              data_dir: ./tmp_data
              experiment_name: smoke_new
            trainer:
              max_epochs: 3
              batch_size: 2
            model:
              target: torch.nn:Linear
              params:
                in_features: 4
                out_features: 2
            dataloader_defaults:
              batch_size: ${trainer.batch_size}
              num_workers: 2
              pin_memory: false
            train_transforms:
              target: torchvision.transforms:Compose
              params:
                transforms:
                  - target: torchvision.transforms:ToTensor
                    params: {}
            train_dataset:
              target: torchvision.datasets:FakeData
              params:
                size: 4
                image_size: [1, 2, 2]
                num_classes: 2
                transform: ${train_transforms}
            train_dataloader:
              dataset: ${train_dataset}
              params:
                shuffle: false
            val_dataloader:
              dataset: ${train_dataset}
              params:
                batch_size: 4
                shuffle: false
            optimization:
              optimizer:
                target: torch.optim:SGD
                params:
                  lr: 0.01
              scheduler:
                target: torch.optim.lr_scheduler:StepLR
                params:
                  step_size: 1
                  gamma: 0.5
            loss:
              - target: torch.nn:CrossEntropyLoss
                params: {}
            metrics:
              - target: registry:Accuracy
                params:
                  num_classes: 2
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")

    assert type(setup.model).__name__ == "Linear"
    assert type(setup.optimizer).__name__ == "SGD"
    assert type(setup.scheduler).__name__ == "StepLR"
    assert type(setup.loss_fn).__name__ == "CrossEntropyLoss"
    assert setup.num_epochs == 3
    assert setup.batch_size == 2
    assert setup.device == "cpu"
    assert setup.train_loader is not None
    assert setup.train_loader.batch_size == 2
    assert setup.train_loader.num_workers == 2
    assert setup.train_loader.pin_memory is False
    assert setup.val_loader is not None
    assert setup.val_loader.batch_size == 4


def test_setup_from_yaml_rejects_unsupported_top_level_layout(tmp_path) -> None:
    config_path = tmp_path / "invalid_layout.yaml"
    config_path.write_text(
        dedent(
            """
            runtime:
              device: cpu
            trainer:
              max_epochs: 4
            data:
              transforms: {}
            """
        ),
        encoding="utf-8",
    )

    try:
        setup_from_yaml(config_path, device="cpu")
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError:
        pass


def test_official_unified_logger_example_builds() -> None:
    setup = setup_from_yaml("config/unified_logger_example.yaml", device="cpu")

    assert type(setup.model).__name__ == "SimpleMLP"
    assert type(setup.optimizer).__name__ == "Adam"
    assert type(setup.scheduler).__name__ == "StepLR"
    assert type(setup.loss_fn).__name__ == "CrossEntropyLoss"
    assert setup.train_loader is not None


def test_official_vgg_config_matches_schema_and_model_definition() -> None:
    cfg = load_config_with_schema("config/vgg_cifar100.yaml", resolve=True)

    assert cfg.model.target == "registry:vgg16_bn"
    assert cfg.trainer.max_epochs == 100
    assert cfg.train_dataset.target == "torchvision.datasets:CIFAR100"
    assert cfg.dataloader_defaults.batch_size == 128
    assert cfg.dataloader_defaults.num_workers == 4
    assert cfg.train_transforms.target == "torchvision.transforms:Compose"
    assert cfg.train_dataset.params.root == "./data"

    model = build_model(
        {
            "target": cfg.model.target,
            "params": dict(cfg.model.params),
        }
    )
    assert type(model).__name__ == "VGG"
