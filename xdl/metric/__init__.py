from ..utils.registry import register_metric
from .metrics import (
    Accuracy,
    DiceCoefficient,
    F1Score,
    IoU,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
    RootMeanSquaredError,
    TopKAccuracy,
)


# 注册名保持和类名一直大写开头
def _register_metrics():
    """统一注册所有评估指标到METRIC_REGISTRY"""
    register_metric("Accuracy")(Accuracy)
    register_metric("Precision")(Precision)
    register_metric("Recall")(Recall)
    register_metric("F1Score")(F1Score)
    register_metric("IoU")(IoU)
    register_metric("DiceCoefficient")(DiceCoefficient)
    register_metric("TopKAccuracy")(TopKAccuracy)
    register_metric("MeanAbsoluteError")(MeanAbsoluteError)
    register_metric("MeanSquaredError")(MeanSquaredError)
    register_metric("RootMeanSquaredError")(RootMeanSquaredError)


_register_metrics()

__all__ = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "IoU",
    "DiceCoefficient",
    "TopKAccuracy",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "register_metric",
]
