import torch

from xdl.analysis import concept_activation_vector, fit_concept_probe, tcav_from_probe, tcav_score


def test_fit_concept_probe_learns_binary_concept_direction() -> None:
    features = torch.tensor(
        [
            [2.0, 2.0],
            [1.5, 1.2],
            [-2.0, -1.8],
            [-1.2, -1.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])

    probe = fit_concept_probe(features, labels, epochs=300, lr=0.05)

    assert probe.accuracy(features, labels) >= 0.99
    direction = concept_activation_vector(probe)
    assert direction.shape == (2,)


def test_tcav_score_is_positive_for_aligned_gradients() -> None:
    gradients = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 0.5],
            [-1.0, -1.0],
            [0.5, 0.5],
        ]
    )
    direction = torch.tensor([1.0, 0.0])

    score = tcav_score(gradients, direction)

    assert score == 0.75


def test_tcav_from_probe_uses_probe_direction() -> None:
    features = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [-3.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])
    probe = fit_concept_probe(features, labels, epochs=250, lr=0.05)
    gradients = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    score = tcav_from_probe(gradients, probe)

    assert score == 0.75
