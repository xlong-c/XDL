"""Detection metrics (BoxIoU, mAP). Postprocess helpers live in detection_utils.py."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from xdl.metric._utils import (
    _aligned_box_iou,
    _box_area,
    _boxes_to_xyxy,
    _prediction_tensor
)


class BoxIoU:
    """Aligned bounding box IoU metric."""

    def __init__(
        self,
        box_format: str = "xyxy",
        reduction: str = "mean",
        epsilon: float = 1e-7,
    ) -> None:
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.box_format = box_format
        self.reduction = reduction
        self.epsilon = epsilon

    def __call__(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> float | torch.Tensor:
        iou = _aligned_box_iou(pred_boxes, target_boxes, self.box_format, self.epsilon)
        if self.reduction == "none":
            return iou
        if self.reduction == "sum":
            return iou.sum().item()
        return iou.mean().item()


class DetectionMeanAveragePrecision:
    """COCO-style mean average precision over decoded detection predictions.

    ``predictions`` must provide per-image ``boxes``, ``scores`` and ``labels``
    fields, either as mappings or objects with attributes. ``targets`` must
    provide ``boxes`` and ``labels``.
    """

    def __init__(
        self,
        iou_thresholds: Sequence[float] | None = None,
        box_format: str = "xyxy",
        epsilon: float = 1e-7,
    ) -> None:
        self.iou_thresholds = list(
            iou_thresholds
            if iou_thresholds is not None
            else [0.5 + 0.05 * index for index in range(10)]
        )
        self.box_format = box_format
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.predictions: list[Any] = []
        self.targets: list[Mapping[str, Any]] = []

    def update(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> None:
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute(self) -> dict[str, float]:
        aps = [
            self._average_precision(self.predictions, self.targets, float(threshold))
            for threshold in self.iou_thresholds
        ]
        metric_map = {
            "map50_95": float(sum(aps) / len(aps)) if aps else 0.0,
            "map50": 0.0,
            "map75": 0.0,
        }
        for threshold, ap in zip(self.iou_thresholds, aps, strict=False):
            if abs(float(threshold) - 0.5) < 1e-6:
                metric_map["map50"] = float(ap)
            if abs(float(threshold) - 0.75) < 1e-6:
                metric_map["map75"] = float(ap)
        return metric_map

    def __call__(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        self.reset()
        self.update(predictions, targets)
        return self.compute()

    def _build_tp_flags(
        self,
        prediction: Any,
        target: Mapping[str, Any],
        iou_threshold: float,
    ) -> tuple[torch.Tensor, int]:
        pred_boxes = _prediction_tensor(prediction, "boxes").float()
        pred_scores = _prediction_tensor(prediction, "scores").float()
        pred_labels = _prediction_tensor(prediction, "labels").long()
        target_boxes = _prediction_tensor(target, "boxes").float()
        target_labels = _prediction_tensor(target, "labels").long()
        if pred_boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool), int(target_boxes.shape[0])

        matched_targets = torch.zeros((target_boxes.shape[0],), dtype=torch.bool)
        tp = torch.zeros((pred_boxes.shape[0],), dtype=torch.bool)
        order = pred_scores.argsort(descending=True)
        sorted_boxes = pred_boxes[order]
        sorted_labels = pred_labels[order]
        for pred_position, (box, label) in enumerate(
            zip(sorted_boxes, sorted_labels, strict=False)
        ):
            label_mask = target_labels == label
            if not bool(label_mask.any()):
                continue
            candidate_indices = torch.nonzero(label_mask, as_tuple=False).reshape(-1)
            ious = _aligned_box_iou(
                box.unsqueeze(0).expand(candidate_indices.shape[0], -1),
                target_boxes[candidate_indices],
                self.box_format,
                self.epsilon,
            )
            best_iou, best_index = ious.max(dim=0)
            candidate_index = int(candidate_indices[int(best_index.item())].item())
            if float(best_iou.item()) >= iou_threshold and not bool(
                matched_targets[candidate_index]
            ):
                tp[pred_position] = True
                matched_targets[candidate_index] = True
        reordered = torch.zeros_like(tp)
        reordered[order] = tp
        return reordered, int(target_boxes.shape[0])

    def _average_precision(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Mapping[str, Any]],
        iou_threshold: float,
    ) -> float:
        all_scores: list[torch.Tensor] = []
        all_tp: list[torch.Tensor] = []
        total_gt = 0
        for prediction, target in zip(predictions, targets, strict=False):
            tp_flags, gt_count = self._build_tp_flags(
                prediction,
                target,
                iou_threshold,
            )
            all_scores.append(_prediction_tensor(prediction, "scores").detach().cpu())
            all_tp.append(tp_flags.detach().cpu())
            total_gt += gt_count
        if total_gt == 0:
            return 0.0
        scores = torch.cat(all_scores) if all_scores else torch.empty((0,))
        tp = torch.cat(all_tp) if all_tp else torch.empty((0,), dtype=torch.bool)
        if scores.numel() == 0:
            return 0.0
        order = scores.argsort(descending=True)
        tp_sorted = tp[order].float()
        fp_sorted = 1.0 - tp_sorted
        tp_cum = tp_sorted.cumsum(dim=0)
        fp_cum = fp_sorted.cumsum(dim=0)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
        precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        for index in range(precision.numel() - 1, 0, -1):
            precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
        delta = recall[1:] - recall[:-1]
        return float((delta * precision[1:]).sum().item())

