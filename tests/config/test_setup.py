from textwrap import dedent

from xdl.config import setup_from_yaml


def test_setup_from_yaml_supports_new_schema(tmp_path) -> None:
    config_path = tmp_path / "new_schema.yaml"
    config_path.write_text(
        dedent(
            """
            config_version: 1
            runtime:
              device: cpu
              experiment_name: smoke_new
            trainer:
              max_epochs: 3
              batch_size: 2
            model:
              type: Linear
              source: torch.nn
              params:
                in_features: 4
                out_features: 2
            data:
              transforms:
                train:
                  type: compose
                  items:
                    - type: ToTensor
                      source: torchvision.transforms
                      params: {}
              datasets:
                train:
                  type: FakeData
                  source: torchvision.datasets
                  params:
                    size: 4
                    image_size: [1, 2, 2]
                    num_classes: 2
                  transform: ${data.transforms.train}
              dataloaders:
                train:
                  dataset: ${data.datasets.train}
                  params:
                    batch_size: 2
                    shuffle: false
            optimization:
              optimizer:
                type: SGD
                source: torch.optim
                params:
                  lr: 0.01
              scheduler:
                type: StepLR
                source: torch.optim.lr_scheduler
                params:
                  step_size: 1
                  gamma: 0.5
            loss:
              - type: CrossEntropyLoss
                source: torch.nn
                params: {}
            metrics:
              - type: Accuracy
                source: registry
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


def test_setup_from_yaml_supports_legacy_schema(tmp_path) -> None:
    config_path = tmp_path / "legacy_schema.yaml"
    config_path.write_text(
        dedent(
            """
            training:
              device: cpu
              num_epochs: 4
              batch_size: 2
            core_config:
              model:
                backbone:
                  name: Linear
                  from_library: torch
                  params:
                    in_features: 4
                    out_features: 2
              optimizer:
                main_optimizer:
                  name: SGD
                  from_library: torch
                  params:
                    lr: 0.01
              scheduler:
                main_scheduler:
                  name: StepLR
                  from_library: torch
                  params:
                    step_size: 1
                    gamma: 0.5
              loss:
                - name: CrossEntropyLoss
                  from_library: torch
                  params: {}
              metrics:
                - name: Accuracy
                  from_library: local
                  params:
                    num_classes: 2
            data_config:
              transform:
                train_transform:
                  transforms:
                    - name: ToTensor
                      from_library: torchvision
                      params: {}
              dataset:
                train_dataset:
                  name: FakeData
                  from_library: torchvision
                  transform: train_transform
                  params:
                    size: 4
                    image_size: [1, 2, 2]
                    num_classes: 2
              dataloader:
                train_loader:
                  dataset: train_dataset
                  params:
                    batch_size: 2
                    shuffle: false
            """
        ),
        encoding="utf-8",
    )

    setup = setup_from_yaml(config_path, device="cpu")

    assert type(setup.model).__name__ == "Linear"
    assert type(setup.optimizer).__name__ == "SGD"
    assert type(setup.scheduler).__name__ == "StepLR"
    assert type(setup.loss_fn).__name__ == "CrossEntropyLoss"
    assert setup.num_epochs == 4
    assert setup.batch_size == 2
    assert setup.train_loader is not None
