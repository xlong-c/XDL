"""公共 API 兼容性测试。"""

import subprocess
import sys


def test_top_level_usage_api_is_lightweight() -> None:
    script = (
        "import xdl; "
        "text = xdl.get_usage_text(); "
        "assert text.startswith('# XDL Agent Usage'); "
        "assert '先做什么' in text; "
        "assert '推荐默认' in text; "
        "assert 'trainer' not in xdl.__dict__"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_stable_config_api_imports() -> None:
    from xdl.config import TrainSetup, setup_from_yaml

    assert setup_from_yaml is not None
    assert TrainSetup is not None


def test_stable_trainer_api_imports() -> None:
    from xdl.trainer import CoreModel, Trainer, TrainSetupModel

    assert CoreModel is not None
    assert Trainer is not None
    assert TrainSetupModel is not None


def test_legacy_trainer_import_paths_still_work() -> None:
    from xdl.trainer import CoreModel as PublicCoreModel
    from xdl.trainer import Trainer as PublicTrainer
    from xdl.trainer import TrainSetupModel as PublicTrainSetupModel
    from xdl.trainer.coreModel import CoreModel
    from xdl.trainer.trainSetupModel import TrainSetupModel
    from xdl.trainer.trainer import Trainer

    assert CoreModel is PublicCoreModel
    assert Trainer is PublicTrainer
    assert TrainSetupModel is PublicTrainSetupModel


def test_stable_callback_api_imports() -> None:
    from xdl.callbacks import Callback, ModelCheckpoint, TqdmCallback

    assert Callback is not None
    assert ModelCheckpoint is not None
    assert TqdmCallback is not None


def test_stable_registry_api_imports() -> None:
    from xdl.utils.registry import (
        Registry,
        register_collate,
        register_dataset,
        register_loss,
        register_metric,
        register_model,
        register_optimizer,
        register_scheduler,
        register_transform,
    )

    assert Registry is not None
    assert register_model is not None
    assert register_dataset is not None
    assert register_optimizer is not None
    assert register_scheduler is not None
    assert register_loss is not None
    assert register_metric is not None
    assert register_transform is not None
    assert register_collate is not None


def test_stable_utils_api_imports() -> None:
    from xdl.utils import resolve_dtype, save_yaml, seed_everything

    assert resolve_dtype is not None
    assert save_yaml is not None
    assert seed_everything is not None
