"""
ModelCheckpoint 回调功能测试
"""

import shutil
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from xdl.trainer.coreModel import CoreModel
from xdl.trainer.trainer import Trainer
from xdl.callbacks.model_checkpoint import ModelCheckpoint
project_root = Path(__file__).parent.parent


class SimpleDataset(Dataset):
    """简单的数据集包装器，用于测试"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleModel(CoreModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers[0]
        opt.zero_grad()
        loss = self.loss_fn(self.layer(x), y)
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.layer.parameters(), lr=0.1)


class TestModelCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = project_root / "others" / "test_ckpt_callback"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_step_saving_and_no_optimizer(self):
        """测试每X步保存一次, 且不保存优化器状态"""
        print("\n>>> Testing: Step saving & No Optimizer")
        model = SimpleModel()
        checkpoint_dir = self.test_dir / "step_no_opt"

        # 每 2 步保存一次, 不保存优化器
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            every_n_train_steps=2,
            save_optimizer=False,
            verbose=True
        )

        trainer = Trainer(
            max_epochs=1,
            callbacks=[ckpt_cb],
            device="cpu",
            enable_checkpointing=False  # 禁用 Trainer 默认的行为, 由回调处理
        )

        # 模拟数据 (10 个 batch)
        data = [(torch.randn(8, 10), torch.randn(8, 1)) for _ in range(10)]
        train_loader = DataLoader(SimpleDataset(data), batch_size=None)
        trainer.fit(model, train_loader)

        # 检查保存的目录数量 (应该是 5 个, 步长 2, 4, 6, 8, 10)
        # 实际上还会有一个 on_train_end 的 last 保存
        dirs = list(checkpoint_dir.glob("step_*"))
        print(f"Saved step directories: {[d.name for d in dirs]}")
        self.assertTrue(len(dirs) >= 5)

        # 验证其中一个 checkpoint 是否没有优化器状态
        sample_ckpt = torch.load(dirs[0] / "checkpoint.pt")
        self.assertIsNone(sample_ckpt.get('optimizer_states'))
        self.assertIsNotNone(sample_ckpt.get('state_dict'))
        print("✅ Step saving and No-Optimizer check passed.")

    def test_top_k_management(self):
        """测试只保存最好的 3 个节点"""
        print("\n>>> Testing: Top-K management")
        model = SimpleModel()
        checkpoint_dir = self.test_dir / "top_k"

        # 监控 train_loss, 保留最好的 3 个
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="train_loss",
            save_top_k=3,
            mode="min",
            every_n_train_steps=1,  # 每步都检查保存
            verbose=True
        )

        trainer = Trainer(max_epochs=1, callbacks=[ckpt_cb], device="cpu")

        # 模拟数据
        data = [(torch.randn(8, 10), torch.randn(8, 1)) for _ in range(10)]
        train_loader = DataLoader(SimpleDataset(data), batch_size=None)
        trainer.fit(model, train_loader)

        # 最终目录里应该只有 3 个监控指标相关的目录
        all_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        step_dirs = [d for d in all_dirs if "train_loss" in d.name]

        print(f"Top-K directories: {[d.name for d in step_dirs]}")
        self.assertLessEqual(len(step_dirs), 3)
        print("✅ Top-K management check passed.")

    def test_safetensors_format(self):
        """测试使用 safetensors 格式保存"""
        print("\n>>> Testing: Safetensors format")
        try:
            import safetensors
        except ImportError:
            print("⚠️ Skip: safetensors not installed")
            return

        model = SimpleModel()
        checkpoint_dir = self.test_dir / "safetensors_test"

        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            every_n_train_steps=5,
            format="safetensors",
            verbose=True
        )

        trainer = Trainer(max_epochs=1, callbacks=[ckpt_cb], device="cpu")
        data = [(torch.randn(8, 10), torch.randn(8, 1)) for _ in range(5)]
        train_loader = DataLoader(SimpleDataset(data), batch_size=None)
        trainer.fit(model, train_loader)

        # 检查是否生成了 .safetensors 文件
        st_dirs = list(checkpoint_dir.glob("step_*"))
        self.assertTrue((st_dirs[0] / "model.safetensors").exists())
        self.assertTrue((st_dirs[0] / "meta.pt").exists())
        print("✅ Safetensors format check passed.")

    def test_component_filtering(self):
        """测试组件过滤保存 (仅保存 encoder)"""
        print("\n>>> Testing: Component filtering")

        class MultiPartModel(CoreModel):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 5)
                self.decoder = nn.Linear(5, 1)

            def training_step(self, batch, idx):
                opt = self.optimizers[0]
                opt.zero_grad()
                loss = self.decoder(self.encoder(batch[0])).mean()
                self.manual_backward(loss)
                opt.step()

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.1)

        model = MultiPartModel()
        checkpoint_dir = self.test_dir / "filter_test"

        # 仅保存 encoder
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            include_components=["encoder"],
            every_n_train_steps=1,
            verbose=True
        )

        trainer = Trainer(max_epochs=1, callbacks=[ckpt_cb], device="cpu")
        data = [(torch.randn(8, 10), torch.randn(8, 1))]
        train_loader = DataLoader(SimpleDataset(data), batch_size=None)
        trainer.fit(model, train_loader)

        # 加载并检查 state_dict
        dirs = list(checkpoint_dir.glob("step_*"))
        ckpt = torch.load(dirs[0] / "checkpoint.pt")
        state_dict = ckpt['state_dict']

        self.assertIn("encoder", state_dict)
        self.assertNotIn("decoder", state_dict)
        print("✅ Component filtering check passed.")

    def test_max_mode_monitoring(self):
        """测试 Max 模式 (模拟监控准确率)"""
        print("\n>>> Testing: Max mode monitoring")
        model = SimpleModel()
        checkpoint_dir = self.test_dir / "max_mode"

        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="train_acc",
            mode="max",
            save_top_k=2,
            every_n_train_steps=1
        )

        # 手动注入一些不同的指标值
        trainer = Trainer(max_epochs=1, callbacks=[ckpt_cb], device="cpu")

        # 我们稍微修改一下 training_step 来模拟 acc
        original_training_step = model.training_step

        def training_step_with_acc(batch, batch_idx):
            model.log("train_acc", 0.1 * batch_idx)  # 递增的 acc
            original_training_step(batch, batch_idx)

        model.training_step = training_step_with_acc

        data = [(torch.randn(8, 10), torch.randn(8, 1)) for _ in range(5)]
        train_loader = DataLoader(SimpleDataset(data), batch_size=None)
        trainer.fit(model, train_loader)

        # 应该保留 acc 最大的两个 (0.4 和 0.3)
        dirs = [d.name for d in checkpoint_dir.glob("*train_acc*")]
        print(f"Max mode directories: {dirs}")
        self.assertTrue(any("0p4" in d for d in dirs))
        self.assertTrue(any("0p3" in d for d in dirs))
        self.assertFalse(any("0p1" in d for d in dirs))
        print("✅ Max mode monitoring check passed.")


if __name__ == "__main__":
    unittest.main()
