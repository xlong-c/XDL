"""
集中验证 XDL 训练框架各模块缺陷修复与稳定性的回归测试集.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from xdl.loss.generative_loss import KLDivergenceLoss
from xdl.post_training.losses.preference_loss import grpo_loss, stpo_loss
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.config.builder import _looks_like_component_config, _get_registry
from xdl.utils.registry import (
    CALLBACK_REGISTRY,
    TASK_REGISTRY,
    register_callback,
    register_task,
)
from xdl.callbacks.model_checkpoint import ModelCheckpoint
from xdl.callbacks.base import Callback
from xdl.optimizer.muon import Muon


class DummyModel(CoreModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self.fc(batch).sum()
        self.manual_optimization_step(loss)


def test_grpo_loss_penalty_direction() -> None:
    """验证 GRPO 的 KL 惩罚方向: 策略偏离 reference 时应增加 loss (惩罚)."""
    B = 2
    advantages = torch.zeros(B)
    ref = torch.zeros(B)

    # 当 policy 与 ref 相同
    loss_equal = grpo_loss(ref.clone(), advantages, ref, kl_beta=1.0)
    # 当 policy 发生漂移
    policy_deviated = ref + 2.0
    loss_deviated = grpo_loss(policy_deviated, advantages, ref, kl_beta=1.0)

    # 偏离后的 KL 惩罚应大于等于零,且总 loss 大于一致时的 loss
    assert loss_deviated.kl_penalty.item() > loss_equal.kl_penalty.item()
    assert loss_deviated.total.item() > loss_equal.total.item()


def test_stpo_loss_reduction_sum_exact() -> None:
    """验证 STPO 在 reduction='sum' 下 total 等于 dpo + auxiliary_weight * auxiliary."""
    pw = torch.tensor([1.0, 0.5])
    pl = torch.tensor([-0.5, 0.2])
    rw = torch.tensor([0.8, 0.4])
    rl = torch.tensor([-0.4, 0.1])

    breakdown = stpo_loss(pw, pl, rw, rl, auxiliary_weight=0.2, reduction="sum")
    expected = breakdown.dpo + 0.2 * breakdown.auxiliary
    assert torch.allclose(breakdown.total, expected, atol=1e-5)


def test_kl_divergence_loss_clamp_prevents_inf() -> None:
    """验证 KLDivergenceLoss 对极大 logvar 具备截断保护, 不产生 inf/NaN."""
    mu = torch.zeros(2, 4)
    logvar_extreme = torch.full((2, 4), 100.0)

    loss_fn = KLDivergenceLoss()
    loss = loss_fn(mu, logvar_extreme)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_configure_optimizers_single_scheduler_unpacking() -> None:
    """验证返回 ([opt], scheduler) 单一对象时不会抛出 TypeError."""
    model = DummyModel()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=10)

    # 模拟用户在 configure_optimizers 中返回 ([opt], sched)
    model._configure_optimizers_from_return(([opt], sched))
    assert len(model.optimizers) == 1
    assert len(model.schedulers) == 1
    assert model.schedulers[0] is sched


def test_component_config_strict_recognition() -> None:
    """验证普通参数字典不被误判为组件配置."""
    # 普通参数包含 target 键但不符合 source:name 格式
    plain_dict = {"target": 1.0, "weight": 0.5}
    assert not _looks_like_component_config(plain_dict)

    # 合法的组件格式
    component_dict = {"target": "registry:AdamW", "params": {"lr": 0.001}}
    assert _looks_like_component_config(component_dict)


def test_callback_and_task_registries() -> None:
    """验证 CALLBACK_REGISTRY 和 TASK_REGISTRY 正常注册和通过 builder 映射."""
    @register_callback("MockCallbackForTest")
    class MockCallback(Callback):
        pass

    @register_task("MockTaskForTest")
    class MockTask:
        pass

    assert CALLBACK_REGISTRY.get("MockCallbackForTest") is MockCallback
    assert TASK_REGISTRY.get("MockTaskForTest") is MockTask

    assert _get_registry("callback") is CALLBACK_REGISTRY
    assert _get_registry("task") is TASK_REGISTRY


def test_fsdp_collective_sampling_guard() -> None:
    """验证当模型声明或检测到 FSDP 活跃时, 强制集体采样以防死锁."""
    model = DummyModel()
    model._is_fsdp_active = lambda: True  # 模拟 FSDP 活跃状态
    assert model.requires_collective_sampling is False

    trainer = Trainer(max_epochs=1)
    assert trainer._requires_collective_sampling(model) is True


def test_model_checkpoint_pending_delete_cleanup() -> None:
    """验证 ModelCheckpoint 淘汰策略正确追踪与延迟清理与 last_model_path 冲突的文件."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt = ModelCheckpoint(dirpath=temp_dir, save_top_k=1, mode="min", monitor="val_loss")
        ckpt.last_model_path = str(Path(temp_dir) / "last.pt")
        # 创建假文件
        Path(ckpt.last_model_path).touch()

        # 模拟 worst 等于 last_model_path 时被加入待删除队列
        worst_item = {"path": ckpt.last_model_path, "score": 2.0}
        ckpt.best_k_models = [worst_item, {"path": str(Path(temp_dir) / "best.pt"), "score": 0.5}]
        ckpt._update_best_models(str(Path(temp_dir) / "new_best.pt"), 0.3)

        assert ckpt.last_model_path in ckpt._pending_delete_paths
        assert Path(ckpt.last_model_path).exists()

        # 当 last_model_path 更新后,应执行清理
        new_last = str(Path(temp_dir) / "last_new.pt")
        Path(new_last).touch()
        ckpt.last_model_path = new_last
        ckpt._cleanup_pending_deletes()

        assert not Path(worst_item["path"]).exists()
        assert worst_item["path"] not in ckpt._pending_delete_paths


def test_muon_single_device_execution() -> None:
    """验证 Muon 优化器在非分布式单机环境下正常 step 更新且不崩溃."""
    param = nn.Parameter(torch.randn(8, 8))
    opt = Muon([param], lr=0.01)

    loss = param.sum()
    loss.backward()

    initial_val = param.clone()
    opt.step()

    assert not torch.equal(param, initial_val)
