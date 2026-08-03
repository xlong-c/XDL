import torch
from torch import nn

from xdl.analysis import compute_grad_cam


class SmallConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def test_compute_grad_cam_returns_normalized_heatmap() -> None:
    model = SmallConvClassifier()
    x = torch.randn(2, 1, 8, 8, requires_grad=True)

    result = compute_grad_cam(model, "features.0", x)

    assert result.heatmap.shape == (2, 8, 8)
    assert result.activations.shape == (2, 4, 8, 8)
    assert result.gradients.shape == (2, 4, 8, 8)
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0 + 1e-6
