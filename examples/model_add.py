from xdl.utils import register_model

@register_model('mymodel')
class MyModel:
    def __init__(self, input_dim, output_dim):
        import torch.nn as nn
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 现在可以通过构建器来创建模型实例
from xdl.utils import build_model
model_instance = build_model('mymodel', input_dim=10, output_dim=1)