"""深度学习评估指标模块.

Barrel re-export for compatibility with ``from xdl.metric.metrics import ...``.
Implementations live in domain modules under ``xdl.metric``.
"""

from __future__ import annotations

from xdl.metric.classification import (
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    MultiLabelAccuracy,
    MultiLabelF1Score,
    MultiLabelPrecision,
    MultiLabelRecall,
    Precision,
    Recall,
    RootMeanSquaredError,
    TopKAccuracy,
)
from xdl.metric.detection import BoxIoU, DetectionMeanAveragePrecision
from xdl.metric.segmentation import (
    DiceCoefficient,
    FrequencyWeightedIoU,
    IoU,
    MeanIoU,
    PixelAccuracy,
)
from xdl.metric.text import (
    BLEUScore,
    HitRateAtK,
    MeanReciprocalRank,
    Perplexity,
    ROUGELScore,
    SequenceExactMatch,
    TokenAccuracy,
)

__all__ = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "TopKAccuracy",
    "IoU",
    "DiceCoefficient",
    "ConfusionMatrix",
    "BalancedAccuracy",
    "MatthewsCorrCoef",
    "MultiLabelAccuracy",
    "MultiLabelPrecision",
    "MultiLabelRecall",
    "MultiLabelF1Score",
    "PixelAccuracy",
    "MeanIoU",
    "FrequencyWeightedIoU",
    "BoxIoU",
    "DetectionMeanAveragePrecision",
    "Perplexity",
    "TokenAccuracy",
    "SequenceExactMatch",
    "MeanReciprocalRank",
    "HitRateAtK",
    "BLEUScore",
    "ROUGELScore"
]
