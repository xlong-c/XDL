"""多卡集体采样协议测试."""

from typing import Any, List

import torch

from xdl.trainer import Trainer
from xdl.trainer.core_model import CoreModel


class RecordingAccelerator:
    def __init__(self, main_process: bool) -> None:
        self.is_main_process = main_process
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1


class CollectiveInferenceModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(1, 1)
        self.inference_calls: List[Any] = []

    def requires_collective_sampling(self) -> bool:
        return True

    def inference(self, data: Any) -> None:
        self.inference_calls.append(data)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class LocalInferenceModel(CollectiveInferenceModel):
    def requires_collective_sampling(self) -> bool:
        return False


def test_collective_inference_runs_on_non_main_rank() -> None:
    model = CollectiveInferenceModel()
    trainer = Trainer(max_epochs=1, device="cpu")
    accelerator = RecordingAccelerator(main_process=False)
    trainer._accelerator = accelerator
    model._accelerator = accelerator

    trainer._model = model
    trainer._inference_data = ["prompt"]
    trainer._validate_epoch()

    assert len(model.inference_calls) == 1
    assert accelerator.wait_calls >= 1


def test_local_inference_skips_non_main_rank() -> None:
    model = LocalInferenceModel()
    trainer = Trainer(max_epochs=1, device="cpu")
    accelerator = RecordingAccelerator(main_process=False)
    trainer._accelerator = accelerator
    model._accelerator = accelerator

    trainer._model = model
    trainer._inference_data = ["prompt"]
    trainer._validate_epoch()

    assert model.inference_calls == []
