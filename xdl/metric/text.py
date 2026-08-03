"""Text, ranking, and sequence metrics."""

from __future__ import annotations

import math

import torch

from xdl.metric._utils import (
    _lcs_length,
    _ngram_counts,
    _normalize_sequence_batch,
    _rouge_l_f1
)


class Perplexity:
    """Perplexity for language modeling logits."""

    def __init__(
        self, ignore_index: int = -100, max_value: float | None = None
    ) -> None:
        self.ignore_index = ignore_index
        self.max_value = max_value
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.token_count = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != target.dim() + 1:
            raise ValueError("logits must have one more dimension than target")
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_target = target.reshape(-1).long()
        valid = flat_target != self.ignore_index
        if not torch.any(valid):
            return
        losses = torch.nn.functional.cross_entropy(
            flat_logits[valid],
            flat_target[valid],
            reduction="sum",
        )
        self.loss_sum += float(losses.item())
        self.token_count += int(valid.sum().item())

    def compute(self) -> float:
        if self.token_count == 0:
            return 0.0
        value = math.exp(self.loss_sum / self.token_count)
        if self.max_value is not None:
            return min(value, self.max_value)
        return value

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(logits, target)
        return self.compute()


class TokenAccuracy:
    """Token-level accuracy for sequence labeling or language modeling."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != target.dim() + 1:
            raise ValueError("logits must have one more dimension than target")
        pred = torch.argmax(logits, dim=-1)
        valid = target != self.ignore_index
        if not torch.any(valid):
            return
        self.correct += int((pred[valid] == target[valid]).sum().item())
        self.total += int(valid.sum().item())

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        self.reset()
        self.update(logits, target)
        return self.compute()


class SequenceExactMatch:
    """Exact-match ratio for generated or labeled sequences."""

    def __init__(self, ignore_index: int | None = None) -> None:
        self.ignore_index = ignore_index

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() == target.dim() + 1:
            pred_labels = torch.argmax(pred, dim=-1)
        else:
            pred_labels = pred.long()
        if pred_labels.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred_labels.shape} vs {target.shape}"
            )
        if self.ignore_index is None:
            matches = (pred_labels == target).all(dim=-1)
            return matches.float().mean().item()
        valid = target != self.ignore_index
        token_matches = (pred_labels == target) | ~valid
        sequence_has_token = valid.any(dim=-1)
        exact = token_matches.all(dim=-1) & sequence_has_token
        if not torch.any(sequence_has_token):
            return 0.0
        return exact.float().sum().item() / sequence_has_token.float().sum().item()


class MeanReciprocalRank:
    """Mean reciprocal rank for retrieval or ranking logits."""

    def __call__(self, scores: torch.Tensor, target: torch.Tensor) -> float:
        if scores.dim() != 2 or target.dim() != 1:
            raise ValueError("scores must be (N, C) and target must be (N,)")
        order = torch.argsort(scores, dim=1, descending=True)
        target_expanded = target.long().unsqueeze(1)
        matches = order == target_expanded
        if not torch.all(matches.any(dim=1)):
            raise ValueError("every target must be a valid class index in scores")
        ranks = matches.float().argmax(dim=1).float() + 1.0
        return (1.0 / ranks).mean().item()


class HitRateAtK:
    """Hit rate@K for retrieval or ranking logits."""

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def __call__(self, scores: torch.Tensor, target: torch.Tensor) -> float:
        if scores.dim() != 2 or target.dim() != 1:
            raise ValueError("scores must be (N, C) and target must be (N,)")
        k = min(self.k, scores.shape[1])
        topk = torch.topk(scores, k=k, dim=1).indices
        hits = (topk == target.long().unsqueeze(1)).any(dim=1)
        return hits.float().mean().item()


class BLEUScore:
    """Corpus BLEU score for tokenized generated text.

    Inputs are batches of token sequences. A sequence can be a list of tokens,
    a whitespace-separated string, or a 1D tensor of token ids.
    """

    def __init__(
        self,
        max_n: int = 4,
        smooth: float = 1.0,
        lowercase: bool = False,
    ) -> None:
        if max_n <= 0:
            raise ValueError("max_n must be positive")
        self.max_n = max_n
        self.smooth = smooth
        self.lowercase = lowercase

    def __call__(self, pred: Sequence[Any], target: Sequence[Any]) -> float:
        pred_sequences = _normalize_sequence_batch(pred, self.lowercase)
        target_sequences = _normalize_sequence_batch(target, self.lowercase)
        if len(pred_sequences) != len(target_sequences):
            raise ValueError(
                "pred and target must contain the same number of sequences"
            )
        if not pred_sequences:
            return 0.0

        precisions = []
        for ngram_size in range(1, self.max_n + 1):
            overlap = 0.0
            total = 0.0
            for pred_tokens, target_tokens in zip(pred_sequences, target_sequences):
                pred_counts = _ngram_counts(pred_tokens, ngram_size)
                target_counts = _ngram_counts(target_tokens, ngram_size)
                overlap += sum(
                    min(count, target_counts.get(ngram, 0))
                    for ngram, count in pred_counts.items()
                )
                total += sum(pred_counts.values())
            precisions.append((overlap + self.smooth) / (total + self.smooth))

        pred_length = sum(len(tokens) for tokens in pred_sequences)
        target_length = sum(len(tokens) for tokens in target_sequences)
        if pred_length == 0:
            return 0.0
        brevity = (
            1.0
            if pred_length > target_length
            else math.exp(1.0 - target_length / pred_length)
        )
        score = brevity * math.exp(sum(math.log(p) for p in precisions) / self.max_n)
        return float(score)


class ROUGELScore:
    """Corpus ROUGE-L F1 score for tokenized generated text."""

    def __init__(self, beta: float = 1.2, lowercase: bool = False) -> None:
        self.beta = beta
        self.lowercase = lowercase

    def __call__(self, pred: Sequence[Any], target: Sequence[Any]) -> float:
        pred_sequences = _normalize_sequence_batch(pred, self.lowercase)
        target_sequences = _normalize_sequence_batch(target, self.lowercase)
        if len(pred_sequences) != len(target_sequences):
            raise ValueError(
                "pred and target must contain the same number of sequences"
            )
        if not pred_sequences:
            return 0.0
        scores = [
            _rouge_l_f1(pred_tokens, target_tokens, self.beta)
            for pred_tokens, target_tokens in zip(pred_sequences, target_sequences)
        ]
        return sum(scores) / len(scores)

