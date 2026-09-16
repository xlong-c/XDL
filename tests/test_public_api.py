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


def test_train_setup_model_lives_in_config_and_is_reexported() -> None:
    from xdl.trainer import CoreModel as PublicCoreModel
    from xdl.trainer import Trainer as PublicTrainer
    from xdl.trainer import TrainSetupModel as PublicTrainSetupModel
    from xdl.trainer.core_model import CoreModel
    from xdl.trainer.trainer import Trainer
    from xdl.config import TrainSetupModel as ConfigTrainSetupModel
    from xdl.config.train_setup_model import TrainSetupModel as ModuleTrainSetupModel

    assert CoreModel is PublicCoreModel
    assert Trainer is PublicTrainer
    assert PublicTrainSetupModel is ConfigTrainSetupModel
    assert PublicTrainSetupModel is ModuleTrainSetupModel


def test_stable_callback_api_imports() -> None:
    from xdl.callbacks import (
        AttentionRolloutCallback,
        Callback,
        FeatureCaptureCallback,
        ModelCheckpoint,
        TqdmCallback,
    )

    assert AttentionRolloutCallback is not None
    assert Callback is not None
    assert FeatureCaptureCallback is not None
    assert ModelCheckpoint is not None
    assert TqdmCallback is not None


def test_stable_post_training_api_imports() -> None:
    from xdl.post_training import (
        ModelMergeCallback,
        ReferenceModelCallback,
        RolloutBatch,
        RolloutCallback,
        SaveTrainableStateCallback,
        dpo_loss,
        grpo_loss,
        stpo_loss,
        tdm_loss,
        tdm_loss_weighted,
    )

    assert ModelMergeCallback is not None
    assert ReferenceModelCallback is not None
    assert RolloutBatch is not None
    assert RolloutCallback is not None
    assert SaveTrainableStateCallback is not None
    assert dpo_loss is not None
    assert stpo_loss is not None
    assert grpo_loss is not None
    assert tdm_loss is not None
    assert tdm_loss_weighted is not None


def test_stable_analysis_api_imports() -> None:
    from xdl.analysis import (
        attention_rollout,
        attention_rollout_for_model,
        capture_activations,
        capture_attention_maps,
        fit_concept_probe,
        compute_module_tcav,
        compute_grad_cam,
        fit_linear_probe,
        tcav_score,
        write_analysis_bundle,
    )

    assert attention_rollout is not None
    assert attention_rollout_for_model is not None
    assert capture_activations is not None
    assert capture_attention_maps is not None
    assert compute_module_tcav is not None
    assert fit_concept_probe is not None
    assert compute_grad_cam is not None
    assert fit_linear_probe is not None
    assert tcav_score is not None
    assert write_analysis_bundle is not None


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


def test_task_submodule_is_lazily_exported() -> None:
    import xdl

    assert xdl.task.__name__ == "xdl.task"
    assert "task" in xdl.__all__
