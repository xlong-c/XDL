"""Shared private helpers for metric implementations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _to_label_predictions(
    pred: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    if not torch.is_floating_point(pred):
        return pred.long()
    if pred.dim() > 1 and pred.shape[1] > 1:
        return torch.argmax(pred, dim=1)
    if pred.dim() > 1:
        pred = pred.squeeze(-1)
    return (pred >= threshold).long()


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=torch.finfo(torch.float32).eps),
        torch.zeros_like(numerator),
    )


def _confusion_matrix_from_labels(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if pred_labels.numel() != target_labels.numel():
        raise ValueError("pred and target must contain the same number of labels")
    valid = (target_labels >= 0) & (target_labels < num_classes)
    valid = valid & (pred_labels >= 0) & (pred_labels < num_classes)
    encoded = target_labels[valid] * num_classes + pred_labels[valid]
    counts = torch.bincount(encoded.cpu(), minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def _multilabel_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    from_logits: bool,
) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.dim() < 2:
        raise ValueError("multi-label tensors must have shape (N, C, ...)")
    prob = torch.sigmoid(pred) if from_logits else pred
    pred_binary = prob >= threshold
    target_binary = target.bool()
    dims = tuple(dim for dim in range(pred.dim()) if dim != 1)
    sample_dims = tuple(range(1, pred.dim()))
    return {
        "tp_class": (pred_binary & target_binary).sum(dim=dims).float(),
        "fp_class": (pred_binary & ~target_binary).sum(dim=dims).float(),
        "fn_class": (~pred_binary & target_binary).sum(dim=dims).float(),
        "tp_sample": (pred_binary & target_binary).sum(dim=sample_dims).float(),
        "fp_sample": (pred_binary & ~target_binary).sum(dim=sample_dims).float(),
        "fn_sample": (~pred_binary & target_binary).sum(dim=sample_dims).float(),
    }


def _multilabel_score(
    stats: dict[str, torch.Tensor],
    average: str,
    score: str,
    epsilon: float,
) -> float:
    if average == "micro":
        tp = stats["tp_class"].sum()
        fp = stats["fp_class"].sum()
        fn = stats["fn_class"].sum()
        return _precision_recall_f1(tp, fp, fn, score, epsilon).item()
    if average == "macro":
        values = _precision_recall_f1(
            stats["tp_class"], stats["fp_class"], stats["fn_class"], score, epsilon
        )
        return values.mean().item()
    values = _precision_recall_f1(
        stats["tp_sample"], stats["fp_sample"], stats["fn_sample"], score, epsilon
    )
    return values.mean().item()


def _precision_recall_f1(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    score: str,
    epsilon: float,
) -> torch.Tensor:
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    if score == "precision":
        return precision
    if score == "recall":
        return recall
    return 2.0 * precision * recall / (precision + recall + epsilon)


def _segmentation_labels(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if not torch.is_floating_point(pred):
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
            )
        return pred.long()
    if pred.dim() == target.dim() + 1 and pred.shape[1] > 1:
        return torch.argmax(pred, dim=1)
    if pred.dim() == target.dim() + 1 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
        )
    return (pred > threshold).long()


def _prediction_tensor(item: Any, key: str) -> torch.Tensor:
    if isinstance(item, Mapping):
        value = item[key]
    else:
        value = getattr(item, key)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end with 4 coordinates")
    if box_format == "xyxy":
        return boxes
    if box_format != "cxcywh":
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ),
        dim=-1,
    )


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0)
    return size[..., 0] * size[..., 1]


def _aligned_box_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    box_format: str,
    epsilon: float,
) -> torch.Tensor:
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"pred_boxes and target_boxes shape mismatch: {pred_boxes.shape} vs {target_boxes.shape}"
        )
    pred_xyxy = _boxes_to_xyxy(pred_boxes, box_format)
    target_xyxy = _boxes_to_xyxy(target_boxes, box_format)
    inter_top_left = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    inter_bottom_right = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    inter_size = (inter_bottom_right - inter_top_left).clamp(min=0.0)
    intersection = inter_size[..., 0] * inter_size[..., 1]
    union = _box_area(pred_xyxy) + _box_area(target_xyxy) - intersection
    return intersection / union.clamp(min=epsilon)


def _normalize_sequence_batch(batch: Sequence[Any], lowercase: bool) -> list[list[str]]:
    normalized = []
    for item in batch:
        if isinstance(item, str):
            tokens = item.split()
        elif isinstance(item, torch.Tensor):
            tokens = [str(token.item()) for token in item.reshape(-1)]
        else:
            tokens = [str(token) for token in item]
        if lowercase:
            tokens = [token.lower() for token in tokens]
        normalized.append(tokens)
    return normalized


def _ngram_counts(tokens: Sequence[str], ngram_size: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(tokens) < ngram_size:
        return counts
    for start in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[start : start + ngram_size])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _rouge_l_f1(
    pred_tokens: Sequence[str], target_tokens: Sequence[str], beta: float
) -> float:
    if not pred_tokens or not target_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, target_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    beta_sq = beta**2
    return ((1.0 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]

