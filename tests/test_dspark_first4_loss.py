"""Ensure slot emphasis changes only valid-token gradient allocation."""

import torch
import torch.nn.functional as F

from train.core.posttrain.train_dspark_first4 import weighted_slot_ce


def test_causal_draft_has_no_future_or_other_block_leakage() -> None:
    from train.core.posttrain.train_dspark_first4 import (
        causal_mask_predicate,
        causal_noise_embed,
    )

    anchors = torch.tensor([[2, 5]])
    keep = torch.tensor([[True, True]])
    ids = torch.arange(10).reshape(1, 10)
    embedding = torch.nn.Embedding(20, 1)
    embedding.weight.data[:, 0] = torch.arange(20)
    values = causal_noise_embed(
        embedding, ids, anchors, keep, mask_token_id=19, block_size=3
    )
    assert values.flatten().tolist() == [2, 3, 4, 5, 6, 7]
    mask = causal_mask_predicate(anchors, keep, 10, 3)
    q = torch.arange(6)[:, None]
    kv = torch.arange(16)[None, :]
    allowed = mask(torch.tensor(0), torch.tensor(0), q, kv)
    for row in range(6):
        block, slot = divmod(row, 3)
        expected = set(range(int(anchors[0, block]))) | set(
            range(10 + block * 3, 10 + block * 3 + slot + 1)
        )
        assert set(allowed[row].nonzero().flatten().tolist()) == expected
    # Changing future teacher tokens must not change embeddings visible to slot0.
    changed = ids.clone()
    changed[0, 3:5] = 18
    other = causal_noise_embed(
        embedding, changed, anchors, keep, mask_token_id=19, block_size=3
    )
    assert torch.equal(values[:, :1], other[:, :1])


def test_residual_rnn_initially_preserves_markov_and_learns_context() -> None:
    from train.core.posttrain.train_dspark_first4 import residual_rnn_step

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.joint_proj = torch.nn.Linear(10, 9)
            self.project_bias = torch.nn.Linear(3, 5, bias=False)

    head = Head()
    with torch.no_grad():
        head.joint_proj.weight[6:].zero_()
        head.joint_proj.bias[6:].zero_()
    prev, hidden = torch.randn(2, 3), torch.randn(2, 4)
    state, out = residual_rnn_step(head, torch.zeros_like(prev), prev, hidden)
    torch.testing.assert_close(out, head.project_bias(prev), rtol=0, atol=0)
    assert state.abs().sum() > 0
    out.square().mean().backward()
    assert head.joint_proj.weight.grad[6:].abs().sum() > 0
    with torch.no_grad():
        head.joint_proj.weight[6:].add_(0.1)
    _, a = residual_rnn_step(head, state, prev, hidden)
    _, b = residual_rnn_step(head, torch.zeros_like(state), prev, hidden + 1)
    assert not torch.equal(a, b)


def test_prefix_loss_matches_explicit_survival_and_masks_tail() -> None:
    from train.core.posttrain.train_dspark_first4 import expected_prefix_loss

    logits = torch.zeros(1, 2, 3, 2, requires_grad=True)
    targets = torch.zeros(1, 2, 3, dtype=torch.long)
    mask = torch.tensor([[[True, True, False], [False, False, False]]])
    loss = expected_prefix_loss(logits, targets, mask)
    torch.testing.assert_close(loss, torch.tensor(-0.75))
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert logits.grad[0, 0, 0, 0] < logits.grad[0, 0, 1, 0] < 0


def test_prefix_loss_empty_mask_has_zero_gradient() -> None:
    from train.core.posttrain.train_dspark_first4 import expected_prefix_loss

    logits = torch.randn(1, 2, 3, 5, requires_grad=True)
    targets = torch.zeros(1, 2, 3, dtype=torch.long)
    loss = expected_prefix_loss(logits, targets, torch.zeros_like(targets).bool())
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_uniform_weights_match_valid_token_ce() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 2, 7, 5, requires_grad=True)
    target = torch.randint(5, (1, 2, 7))
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 1, 4:] = False
    loss = weighted_slot_ce(logits, target, mask, torch.ones(7))
    torch.testing.assert_close(loss, F.cross_entropy(logits[mask], target[mask]))
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0


def test_first_four_emphasis_and_scale_invariance() -> None:
    logits = torch.zeros(1, 1, 7, 2, requires_grad=True)
    target = torch.zeros(1, 1, 7, dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    loss = weighted_slot_ce(logits, target, mask, weights)
    torch.testing.assert_close(
        loss, weighted_slot_ce(logits, target, mask, weights * 7)
    )
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 0, 0], logits.grad[0, 0, 4] * 2)


def test_empty_mask_has_zero_loss_and_gradient() -> None:
    logits = torch.randn(1, 1, 7, 3, requires_grad=True)
    target = torch.zeros(1, 1, 7, dtype=torch.long)
    loss = weighted_slot_ce(logits, target, torch.zeros_like(target), torch.ones(7))
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_five_epochs_cover_every_sample_without_padding() -> None:
    from train.core.posttrain.train_dspark_first4 import epoch_orders

    orders = epoch_orders(13, 5, 42)
    assert len(orders) == 5
    assert orders == epoch_orders(13, 5, 42)
    assert all(sorted(order) == list(range(13)) for order in orders)
    assert len({tuple(order) for order in orders}) == 5


def test_partial_accumulation_matches_actual_window_mean() -> None:
    from train.core.posttrain.train_dspark_first4 import accumulation_window

    value = torch.tensor(0.7, requires_grad=True)
    samples = torch.arange(1.0, 8.0)
    measured = []
    for i, target in enumerate(samples):
        start, boundary, divisor = accumulation_window(i, len(samples), 4)
        if start:
            value.grad = None
        ((value - target).square() / divisor).backward()
        if boundary:
            measured.append(value.grad.clone())
    expected = [2 * (value.detach() - part).mean() for part in samples.split(4)]
    torch.testing.assert_close(torch.stack(measured), torch.stack(expected))
