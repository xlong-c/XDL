import torch
from torch import nn

from xdl.analysis import ActivationCapture, capture_activations


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.stem(x))
        return self.head(hidden)


def test_capture_activations_returns_named_records() -> None:
    model = TinyNet()
    x = torch.randn(2, 4)

    records = capture_activations(model, ["stem"], x)

    assert set(records) == {"stem"}
    assert records["stem"].shape == (2, 3)
    assert isinstance(records["stem"].value, torch.Tensor)


def test_activation_capture_missing_module_raises_key_error() -> None:
    model = TinyNet()

    try:
        with ActivationCapture(model, ["missing"]):
            pass
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing module")
