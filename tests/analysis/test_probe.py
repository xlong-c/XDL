import torch

from xdl.analysis import fit_linear_probe, score_linear_probe


def test_fit_linear_probe_learns_separable_features() -> None:
    features = torch.tensor(
        [
            [2.0, 1.0],
            [1.8, 0.8],
            [-2.0, -1.0],
            [-1.5, -0.7],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])

    probe = fit_linear_probe(features, labels, epochs=300, lr=0.05)
    accuracy = score_linear_probe(probe, features, labels)

    assert probe.feature_dim == 2
    assert probe.num_classes == 2
    assert accuracy >= 0.99
