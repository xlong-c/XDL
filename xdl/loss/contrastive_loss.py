"""对比学习损失 — InfoNCE / NTXentLoss。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """InfoNCE 对比损失，常用于 SimCLR 等自监督学习。

    Args:
        temperature: 温度参数 τ，默认 0.1。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """计算 InfoNCE 损失。

        Args:
            features: (2N, D) 特征矩阵，前 N 为 anchor，后 N 为正样本。
            labels: 可选标签，用于监督对比学习。
        Returns:
            scalar loss。
        """
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0] // 2

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / self.temperature

        # 正样本：每对 (i, i+batch_size) 和 (i+batch_size, i)
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True

        # 负样本：除去自身和正样本
        neg_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        neg_mask = neg_mask & ~pos_mask

        pos_sim = sim[pos_mask].view(sim.shape[0], 1)
        neg_sim = sim[neg_mask].view(sim.shape[0], -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels_t = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels_t)


class NTXentLoss(InfoNCE):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失。

    InfoNCE 的别名，保留 SimCLR 论文命名习惯。
    """

    pass
