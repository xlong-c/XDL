from textwrap import dedent

from xdl.callbacks import Timer
from xdl.config import load_config_with_schema, setup_from_yaml
from xdl.config.builder import build_model
from xdl.config.errors import ConfigValidationError
from xdl.trainer import CoreModel
from xdl.trainer import Trainer


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
              precision: bf16
              gradient_accumulation_steps: 3
              grad_clip_max_norm: 0.5
              grad_clip_norm_type: 1.0
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
            logging:
              enable_tensorboard: false
              enable_console: false
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")

    assert type(setup.model).__name__ == "Linear"
    assert type(setup.optimizer).__name__ == "SGD"
    assert type(setup.scheduler).__name__ == "StepLR"
    assert type(setup.loss_fn).__name__ == "CrossEntropyLoss"
    assert setup.trainer.max_epochs == 3
    assert setup.trainer.batch_size == 2
    assert setup.runtime.device == "cpu"
    assert setup.trainer.precision == "bf16"
    assert setup.trainer.gradient_accumulation_steps == 3
    assert setup.trainer.grad_clip_max_norm == 0.5
    assert setup.trainer.grad_clip_norm_type == 1.0
    assert setup.train_loader is not None
    assert setup.train_loader.batch_size == 2
    assert setup.train_loader.num_workers == 2
    assert setup.train_loader.pin_memory is False
    assert setup.val_loader is not None
    assert setup.val_loader.batch_size == 4

    trainer = Trainer.from_setup(setup)
    assert trainer.precision == "bf16"
    assert trainer.gradient_accumulation_steps == 3
    assert trainer.grad_clip_max_norm == 0.5
    assert trainer.grad_clip_norm_type == 1.0


def test_setup_from_yaml_resolves_omegaconf_resolvers(tmp_path) -> None:
    config_path = tmp_path / "resolver_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
              output_dir: ${xdl.join_path:./tmp,outputs}
            trainer:
              max_epochs: 1
              batch_size: 1
            model:
              target: torch.nn:Linear
              params:
                in_features: 1
                out_features: 1
            train_dataset:
              target: registry:SyntheticClassificationDataset
              params:
                num_samples: 1
                input_shape: [1]
                num_classes: 1
                seed: 42
            train_dataloader:
              dataset: ${train_dataset}
              params:
                shuffle: false
            optimization:
              optimizer:
                target: torch.optim:SGD
                params:
                  lr: 0.01
            loss:
              - target: torch.nn:MSELoss
                params: {}
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")

    assert setup.full_config["runtime"]["output_dir"] == "tmp/outputs"


def test_setup_from_yaml_injects_config_relative_context(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    config_path = config_dir / "context_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
              output_dir: ${xdl.abspath:${xdl.config_dir},outputs}
            trainer:
              max_epochs: 1
              batch_size: 1
            model:
              target: torch.nn:Linear
              params:
                in_features: 1
                out_features: 1
            train_dataset:
              target: registry:SyntheticClassificationDataset
              params:
                num_samples: 1
                input_shape: [1]
                num_classes: 1
                seed: 42
            train_dataloader:
              dataset: ${train_dataset}
              params:
                shuffle: false
            optimization:
              optimizer:
                target: torch.optim:SGD
                params:
                  lr: 0.01
            loss:
              - target: torch.nn:MSELoss
                params: {}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(other_cwd)

    setup = setup_from_yaml(config_path, device="cpu")

    assert setup.full_config["xdl"]["config_dir"] == str(config_dir.resolve())
    assert setup.full_config["runtime"]["output_dir"] == str(
        (config_dir / "outputs").resolve()
    )


def test_setup_from_yaml_builds_callbacks_and_trainer_consumes_them(tmp_path) -> None:
    config_path = tmp_path / "callbacks_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
            trainer:
              max_epochs: 1
              batch_size: 1
            model:
              target: torch.nn:Linear
              params:
                in_features: 1
                out_features: 1
            train_dataset:
              target: registry:SyntheticClassificationDataset
              params:
                num_samples: 1
                input_shape: [1]
                num_classes: 1
                seed: 42
            train_dataloader:
              dataset: ${train_dataset}
              params:
                shuffle: false
            optimization:
              optimizer:
                target: torch.optim:SGD
                params:
                  lr: 0.01
            loss:
              - target: torch.nn:MSELoss
                params: {}
            callbacks:
              - target: xdl.callbacks:Timer
                params: {}
            logging:
              enable_tensorboard: false
              enable_console: false
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")
    trainer = Trainer.from_setup(setup)

    assert any(isinstance(callback, Timer) for callback in setup.callbacks)
    assert any(isinstance(callback, Timer) for callback in trainer.callbacks)


def test_setup_from_yaml_builds_coremodel_task_without_optimizer_or_loss(
    tmp_path,
    monkeypatch,
) -> None:
    module_path = tmp_path / "dummy_task_module.py"
    module_path.write_text(
        dedent(
            """
            from typing import Any

            from xdl.trainer import CoreModel


            class DummyTask(CoreModel):
                def __init__(self, scale: float = 1.0) -> None:
                    super().__init__()
                    self.scale = scale

                def configure_optimizers(self) -> None:
                    return None

                def training_step(self, batch: Any, batch_idx: int) -> None:
                    pass

                def validation_step(self, batch: Any, batch_idx: int) -> None:
                    pass
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "task_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
            trainer:
              max_epochs: 1
              batch_size: 1
            task:
              target: dummy_task_module:DummyTask
              params:
                scale: 2.5
            train_dataset:
              target: registry:SyntheticClassificationDataset
              params:
                num_samples: 1
                input_shape: [1]
                num_classes: 1
                seed: 42
            train_dataloader:
              dataset: ${train_dataset}
              params:
                shuffle: false
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")

    assert isinstance(setup.model, CoreModel)
    assert setup.model.scale == 2.5
    assert setup.optimizer is None
    assert setup.loss_fn is None
    assert setup.create_model() is setup.model


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


def test_official_manifest_segmentation_example_builds_and_trains() -> None:
    setup = setup_from_yaml("config/manifest_segmentation_example.yaml", device="cpu")

    assert isinstance(setup.model, CoreModel)
    assert setup.optimizer is None
    assert setup.loss_fn is None
    assert setup.train_loader is not None
    batch = next(iter(setup.train_loader))
    assert set(batch.keys()) >= {"image", "mask", "sample_id"}

    trainer = Trainer.from_setup(setup)
    trainer.fit(setup.create_model(), setup.train_loader, setup.val_loader)


def test_official_manifest_detection_example_builds_and_trains() -> None:
    setup = setup_from_yaml("config/manifest_detection_example.yaml", device="cpu")

    assert isinstance(setup.model, CoreModel)
    assert setup.optimizer is None
    assert setup.loss_fn is None
    assert setup.train_loader is not None
    batch = next(iter(setup.train_loader))
    assert set(batch.keys()) >= {"image", "boxes", "labels", "sample_id"}
    assert isinstance(batch["boxes"], list)
    assert isinstance(batch["labels"], list)

    trainer = Trainer.from_setup(setup)
    trainer.fit(setup.create_model(), setup.train_loader, setup.val_loader)


def test_official_manifest_regression_example_builds_and_trains() -> None:
    setup = setup_from_yaml("config/manifest_regression_example.yaml", device="cpu")

    assert isinstance(setup.model, CoreModel)
    assert setup.optimizer is None
    assert setup.loss_fn is None
    assert setup.train_loader is not None
    batch = next(iter(setup.train_loader))
    assert len(batch) == 2

    trainer = Trainer.from_setup(setup)
    trainer.fit(setup.create_model(), setup.train_loader, setup.val_loader)


def test_official_manifest_pair_example_builds_and_trains() -> None:
    setup = setup_from_yaml("config/manifest_pair_example.yaml", device="cpu")

    assert isinstance(setup.model, CoreModel)
    assert setup.optimizer is None
    assert setup.loss_fn is None
    assert setup.train_loader is not None
    batch = next(iter(setup.train_loader))
    assert set(batch.keys()) >= {"image_a", "image_b", "label", "sample_id"}

    trainer = Trainer.from_setup(setup)
    trainer.fit(setup.create_model(), setup.train_loader, setup.val_loader)
