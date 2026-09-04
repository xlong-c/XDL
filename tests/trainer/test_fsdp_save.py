"""CoreModel FSDP 聚合保存/加载行为测试 (使用假 accelerator, 不跑真分布式)."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
import pytest

from accelerate.utils import DistributedType

from xdl.errors import TrainingError
from xdl.trainer.core_model import CoreModel


class FakeFSDPAccelerator:
    def __init__(self, main_process: bool = True) -> None:
        self.distributed_type = DistributedType.FSDP
        self._main_process = main_process
        self.wait_calls = 0
        self.gather_calls = 0

    @property
    def is_main_process(self) -> bool:
        return self._main_process

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def get_state_dict(self, module: torch.nn.Module) -> Dict[str, Any]:
        self.gather_calls += 1
        state = module.state_dict().copy()
        state["weight"] = state["weight"] + 1.0
        return state


class SimpleModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(3, 2)

    def training_step(self, batch, batch_idx):
        del batch, batch_idx

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _load_checkpoint_dir(path: Path) -> Dict[str, Any]:
    return torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)


def test_fsdp_save_gathers_state_dict_and_writes_on_main_rank() -> None:
    model = SimpleModel()
    model._accelerator = FakeFSDPAccelerator(main_process=True)

    with tempfile.TemporaryDirectory() as tmp:
        saved = model.save_checkpoint(
            tmp,
            save_optimizer=False,
            save_scheduler=False,
        )
        assert saved
        ckpt = _load_checkpoint_dir(Path(saved))
        assert torch.equal(
            ckpt["state_dict"]["net"]["weight"],
            model.net.weight + 1.0,
        )


def test_fsdp_save_non_main_rank_participates_but_skips_write() -> None:
    model = SimpleModel()
    accelerator = FakeFSDPAccelerator(main_process=False)
    model._accelerator = accelerator

    with tempfile.TemporaryDirectory() as tmp:
        saved = model.save_checkpoint(
            tmp,
            save_optimizer=False,
            save_scheduler=False,
        )
        assert saved == ""
        assert accelerator.gather_calls == 1


def test_fsdp_save_with_optimizer_raises_instead_of_silent_corruption() -> None:
    model = SimpleModel()
    model._accelerator = FakeFSDPAccelerator()

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TrainingError, match="format='accelerator'"):
            model.save_checkpoint(tmp, save_optimizer=True)


def test_checkpoint_without_optimizer_or_scheduler_state_can_be_loaded() -> None:
    model = SimpleModel()
    model._ensure_optimizers_initialized()

    with tempfile.TemporaryDirectory() as tmp:
        saved = model.save_checkpoint(
            tmp,
            save_optimizer=False,
            save_scheduler=False,
        )
        model.load_checkpoint(saved)


def test_non_fsdp_save_keeps_main_only_behavior() -> None:
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as tmp:
        saved = model.save_checkpoint(
            tmp,
            save_optimizer=False,
            save_scheduler=False,
        )
        assert saved
        ckpt = _load_checkpoint_dir(Path(saved))
        assert torch.equal(ckpt["state_dict"]["net"]["weight"], model.net.weight)
