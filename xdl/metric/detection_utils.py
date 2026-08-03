"""Detection output decoding utilities.

These are task-provider-level helpers: decode raw model outputs into
structured detection predictions (boxes / scores / labels). Moved out of
xqt/integrations/ per xqt core contract — XQT does not own task provider logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torchvision.ops import nms


def boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    """Convert cxcywh boxes to xyxy format."""
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


@dataclass
class DetectionPostprocessConfig:
    """Postprocess settings for detection model outputs."""

    format: str = "auto"
    box_format: str = "xyxy"
    score_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    score_activation: str = "identity"
    has_objectness: bool = False
    class_agnostic_nms: bool = False
    rescale_to_original: bool = True


@dataclass
class DetectionPrediction:
    """One image worth of decoded detections."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": self.boxes.detach().cpu().tolist(),
            "scores": self.scores.detach().cpu().tolist(),
            "labels": self.labels.detach().cpu().tolist(),
        }


def _score_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "identity":
        return logits
    if activation == "sigmoid":
        return logits.sigmoid()
    if activation == "softmax":
        return logits.softmax(dim=-1)
    raise ValueError(f"Unsupported score activation: {activation}")


def _apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    config: DetectionPostprocessConfig,
) -> DetectionPrediction:
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=boxes.device)
        return DetectionPrediction(
            boxes=boxes.reshape(0, 4),
            scores=empty,
            labels=torch.empty((0,), dtype=torch.long, device=boxes.device),
        )
    if config.class_agnostic_nms:
        keep = nms(boxes, scores, config.iou_threshold)
    else:
        keep_indices: list[torch.Tensor] = []
        for label in labels.unique(sorted=True):
            class_mask = labels == label
            class_indices = torch.nonzero(class_mask, as_tuple=False).reshape(-1)
            selected = nms(boxes[class_mask], scores[class_mask], config.iou_threshold)
            keep_indices.append(class_indices[selected])
        keep = (
            torch.cat(keep_indices, dim=0)
            if keep_indices
            else torch.empty((0,), dtype=torch.long)
        )
        keep = keep[scores[keep].argsort(descending=True)]
    keep = keep[: config.max_detections]
    return DetectionPrediction(
        boxes=boxes[keep],
        scores=scores[keep],
        labels=labels[keep],
    )


def decode_detection_output(
    output: Any,
    config: DetectionPostprocessConfig,
    *,
    image_sizes: Optional[Sequence[tuple[int, int]]] = None,
) -> list[DetectionPrediction]:
    """Decode provider/model outputs into per-image detections.

    Supports three input formats:

    - **RT-DETR style**: ``{logits: [B,N,C], pred_boxes: [B,N,4]}`` dict.
    - **end2end / [B,N,6]**: tensor with boxes + scores + labels per row.
    - **yolo_raw / [B,channel,anchor]**: raw yolo grid output, with sigmoid /
      softmax score activation and optional objectness.

    All paths go through score-threshold -> NMS -> max_detections truncation.
    """
    if isinstance(output, Mapping):
        if {"boxes", "scores", "labels"} <= set(output.keys()):
            return [
                DetectionPrediction(
                    boxes=output["boxes"].float(),
                    scores=output["scores"].float(),
                    labels=output["labels"].long(),
                )
            ]
        if {"logits", "pred_boxes"} <= set(output.keys()):
            logits = output["logits"]
            pred_boxes = output["pred_boxes"]
            if not isinstance(logits, torch.Tensor) or not isinstance(
                pred_boxes, torch.Tensor
            ):
                raise TypeError(
                    "RT-DETR style detection output must use tensor logits/pred_boxes"
                )
            if logits.ndim != 3 or pred_boxes.ndim != 3:
                raise TypeError(
                    "RT-DETR style detection output tensors must have shape [B, N, C]"
                )
            predictions: list[DetectionPrediction] = []
            for batch_index, (image_logits, image_boxes) in enumerate(
                zip(logits, pred_boxes, strict=False)
            ):
                scores_per_class = _score_activation(
                    image_logits.float(),
                    config.score_activation,
                )
                scores, labels = scores_per_class.max(dim=-1)
                boxes_input = image_boxes.float()
                if (
                    image_sizes is not None
                    and batch_index < len(image_sizes)
                    and boxes_input.numel() > 0
                    and float(boxes_input.max().item()) <= 1.5
                ):
                    height, width = image_sizes[batch_index]
                    scale = torch.tensor(
                        [float(width), float(height), float(width), float(height)],
                        dtype=boxes_input.dtype,
                        device=boxes_input.device,
                    )
                    boxes_input = boxes_input * scale
                boxes = boxes_to_xyxy(boxes_input, config.box_format)
                keep = scores >= config.score_threshold
                predictions.append(
                    _apply_nms(boxes[keep], scores[keep], labels[keep], config)
                )
            return predictions
        if "logits" in output:
            output = output["logits"]

    if isinstance(output, (tuple, list)):
        if len(output) == 1:
            output = output[0]
        elif output and isinstance(output[0], torch.Tensor):
            output = output[0]

    if not isinstance(output, torch.Tensor):
        raise TypeError("detection output must resolve to a tensor or mapping")
    if output.ndim == 2 and output.shape[-1] == 6:
        output = output.unsqueeze(0)
    if output.ndim != 3:
        raise TypeError(
            "detection output tensor must have shape [B, N, 6] or [B, C, N]"
        )

    predictions: list[DetectionPrediction] = []
    if config.format == "end2end" or (
        config.format == "auto" and output.shape[-1] == 6
    ):
        for image_output in output:
            boxes = image_output[:, :4].float()
            scores = image_output[:, 4].float()
            labels = image_output[:, 5].long()
            keep = scores >= config.score_threshold
            predictions.append(
                _apply_nms(boxes[keep], scores[keep], labels[keep], config)
            )
        return predictions

    if config.format not in {"auto", "yolo_raw"}:
        raise ValueError(f"Unsupported detection postprocess format: {config.format}")

    if output.shape[1] < 5:
        raise ValueError("yolo_raw output must have at least 5 channels")

    batch_predictions = output.permute(0, 2, 1).contiguous()
    for image_output in batch_predictions:
        boxes = boxes_to_xyxy(image_output[:, :4].float(), config.box_format)
        class_logits = image_output[:, 4:].float()
        if config.has_objectness:
            objectness = class_logits[:, :1].sigmoid()
            class_scores = _score_activation(
                class_logits[:, 1:],
                config.score_activation,
            )
            scores_per_class = objectness * class_scores
        else:
            scores_per_class = _score_activation(class_logits, config.score_activation)
        scores, labels = scores_per_class.max(dim=-1)
        keep = scores >= config.score_threshold
        predictions.append(
            _apply_nms(boxes[keep], scores[keep], labels[keep], config)
        )
    return predictions


__all__ = [
    "DetectionPostprocessConfig",
    "DetectionPrediction",
    "boxes_to_xyxy",
    "decode_detection_output",
]
