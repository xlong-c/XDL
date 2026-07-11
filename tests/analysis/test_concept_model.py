import torch
from torch import nn

from xdl.analysis import compute_module_tcav, fit_concept_probe


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2))
            self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.head(hidden)


def test_compute_module_tcav_returns_score_and_derivatives() -> None:
    model = TinyClassifier()
    concept_features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [-1.0, 0.0]])
    concept_labels = torch.tensor([1, 1, 0, 0])
    probe = fit_concept_probe(concept_features, concept_labels, epochs=250, lr=0.05)

    batch = torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    score, derivatives = compute_module_tcav(model, "backbone", probe.direction, batch, target_index=0)

    assert derivatives.shape == (3,)
    assert 0.0 <= score <= 1.0
    assert float(derivatives[0]) > 0.0
