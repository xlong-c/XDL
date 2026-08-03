"""
简单 MLP 模型。
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """适合配置系统和分类任务 smoke test 的简单 MLP。"""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(input_size), int(hidden_size)),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(hidden_size), int(num_classes)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > 2:
            inputs = torch.flatten(inputs, start_dim=1)
        return self.classifier(inputs)


def simple_mlp(
    input_size: int = 784,
    hidden_size: int = 128,
    num_classes: int = 10,
    dropout_rate: float = 0.0,
) -> SimpleMLP:
    """简单 MLP 工厂函数。"""

    return SimpleMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
