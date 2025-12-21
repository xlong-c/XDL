from ..utils.registry import register_metric


from .metrics import Accuracy, Precision, Recall, F1Score, IoU, DiceCoefficient, TopKAccuracy

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


_register_metrics()

__all__ = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "IoU",
    "DiceCoefficient",
    "TopKAccuracy",
    "register_metric",
]
