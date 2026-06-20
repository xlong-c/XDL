"""Bounding box regression losses for object detection."""

import math

import torch
import torch.nn as nn

from .classification_loss import _reduce_loss, _validate_reduction


def _to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end with 4 coordinates")
    if box_format == "xyxy":
        return boxes
    if box_format != "cxcywh":
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    cx, cy, width, height = boxes.unbind(dim=-1)
    half_w = width / 2.0
    half_h = height / 2.0
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def _area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    return width * height


def _aligned_iou_terms(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inter_top_left = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    inter_bottom_right = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    inter_size = (inter_bottom_right - inter_top_left).clamp(min=0.0)
    intersection = inter_size[..., 0] * inter_size[..., 1]

    pred_area = _area_xyxy(pred_boxes)
    target_area = _area_xyxy(target_boxes)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp(min=eps)

    enclosing_top_left = torch.minimum(pred_boxes[..., :2], target_boxes[..., :2])
    enclosing_bottom_right = torch.maximum(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosing_size = (enclosing_bottom_right - enclosing_top_left).clamp(min=0.0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou, union, enclosing_area, enclosing_size


class BoxIoULoss(nn.Module):
    """IoU family loss for aligned bounding boxes.

    Args:
        mode: ``'iou'``, ``'giou'``, ``'diou'`` or ``'ciou'``.
        box_format: ``'xyxy'`` or ``'cxcywh'``.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        eps: Numerical stability epsilon.

    Shape:
        ``pred_boxes`` and ``target_boxes`` share shape ``(..., 4)``.
    """

    def __init__(
        self,
        mode: str = "giou",
        box_format: str = "xyxy",
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if mode not in {"iou", "giou", "diou", "ciou"}:
            raise ValueError("mode must be one of: 'iou', 'giou', 'diou', 'ciou'")
        _validate_reduction(reduction)
        self.mode = mode
        self.box_format = box_format
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(
                f"pred_boxes and target_boxes must share shape: {pred_boxes.shape} vs {target_boxes.shape}"
            )

        pred_xyxy = _to_xyxy(pred_boxes, self.box_format)
        target_xyxy = _to_xyxy(target_boxes, self.box_format)
        iou, union, enclosing_area, enclosing_size = _aligned_iou_terms(
            pred_xyxy,
            target_xyxy,
            self.eps,
        )

        score = iou
        if self.mode == "giou":
            score = iou - (enclosing_area - union) / enclosing_area.clamp(min=self.eps)
        elif self.mode in {"diou", "ciou"}:
            pred_center = (pred_xyxy[..., :2] + pred_xyxy[..., 2:]) / 2.0
            target_center = (target_xyxy[..., :2] + target_xyxy[..., 2:]) / 2.0
            center_distance = ((pred_center - target_center) ** 2).sum(dim=-1)
            diagonal = (enclosing_size**2).sum(dim=-1).clamp(min=self.eps)
            distance_penalty = center_distance / diagonal
            score = iou - distance_penalty
            if self.mode == "ciou":
                pred_wh = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=self.eps)
                target_wh = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp(
                    min=self.eps
                )
                v = (4.0 / math.pi**2) * (
                    torch.atan(target_wh[..., 0] / target_wh[..., 1])
                    - torch.atan(pred_wh[..., 0] / pred_wh[..., 1])
                ).pow(2)
                with torch.no_grad():
                    alpha = v / (1.0 - iou + v).clamp(min=self.eps)
                score = score - alpha * v

        loss = 1.0 - score
        return _reduce_loss(loss, self.reduction)
