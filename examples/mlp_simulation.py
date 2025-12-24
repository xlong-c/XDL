"""
双层 MLP 模拟训练测试
验证 Trainer, CoreModel 与 ModelCheckpoint 的协同工作
"""

from xdl.callbacks.model_checkpoint import ModelCheckpoint
from xdl.trainer.trainer import Trainer
from xdl.trainer.coreModel import CoreModel
import shutil
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from pathlib import Path

# ==========================================================
# 必须在导入 xdl 之前设置路径
# ==========================================================
project_root = Path(__file__).resolve().parent.parent
# ==========================================================


# 1. 定义双层 MLP 模型

class TwoLayerMLP(CoreModel):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=1):
        super().__init__()
        # 使用 Sequential 定义双层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer = self.optimizers[0]

        # 前向传播
        preds = self.mlp(x)
        loss = self.loss_fn(preds, y)

        # 反向传播
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # 记录指标
        self.log("train_loss", loss.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.mlp(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def run_simulation():
    # 2. 准备数据 (10 步/轮, batch_size=8, 总计 80 样本)
    batch_size = 8
    num_steps_per_epoch = 10
    total_samples = batch_size * num_steps_per_epoch

    train_x = torch.randn(total_samples, 16)
    train_y = torch.randn(total_samples, 1)
    val_x = torch.randn(batch_size * 2, 16)  # 验证集小一点
    val_y = torch.randn(batch_size * 2, 1)

    train_loader = DataLoader(TensorDataset(
        train_x, train_y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)

    # 3. 设置保存路径
    ckpt_dir = project_root / "others" / "mlp_sim_checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    # 4. 配置 ModelCheckpoint
    # 每 5 步保存一次, 监控 train_loss, 保留最好的 3 个, 不带优化器
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="train_loss",
        save_top_k=3,
        mode="min",
        every_n_train_steps=5,
        save_optimizer=False,
        verbose=True
    )

    # 5. 初始化 Trainer 并运行
    trainer = Trainer(
        max_epochs=10,
        device="cpu",
        callbacks=[checkpoint_callback],
    )

    print("开始双层 MLP 模拟训练 (10 Epochs, 10 Steps/Epoch)...")
    model = TwoLayerMLP()
    trainer.fit(model, train_loader, val_loader)

    # 6. 验证结果
    print("\n" + "="*50)
    print("模拟训练完成。检查保存的节点：")
    saved_dirs = list(ckpt_dir.glob("step_*"))
    for d in sorted(saved_dirs):
        # 检查内部是否有优化器状态
        ckpt_data = torch.load(d / "checkpoint.pt", weights_only=False)
        has_opt = "包含" if ckpt_data.get("optimizer_states") else "不包含"
        print(f"目录: {d.name} | 优化器状态: {has_opt}")

    print(f"总计保留节点数: {len(saved_dirs)} (预期应 <= 3 个监控节点 + 可能的 last)")
    print("="*50)


if __name__ == "__main__":
    run_simulation()
