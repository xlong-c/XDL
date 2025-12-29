"""
深度学习评估指标模块
提供常用的评估指标实现
"""

import torch


class Accuracy:
    """
    准确率指标
    支持多分类和二分类
    """

    def __init__(self, num_classes: int | None = None, threshold: float = 0.5):
        """
        Args:
            num_classes: 分类数量,如果为None则自动判断为二分类或多分类
            threshold: 二分类时的阈值
        """
        self.num_classes = num_classes
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算准确率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes) 或 (batch_size,)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            准确率 (0-1之间的浮点数)
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            # 多分类情况
            pred_labels = torch.argmax(pred, dim=1)
        else:
            # 二分类情况
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            pred_labels = (pred >= self.threshold).long()

        correct = (pred_labels == target).sum().float()
        total = torch.tensor(target.shape[0], dtype=torch.float, device=target.device)
        return (correct / total).item()


class Precision:
    """
    精确率指标
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        """
        Args:
            num_classes: 分类数量
            average: 平均方式,'macro' 或 'micro'
        """
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算精确率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            精确率
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        precision_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_positive = pred_labels == class_idx
            true_positive = target == class_idx
            tp = (pred_positive & true_positive).sum().float()
            fp = (pred_positive & ~true_positive).sum().float()

            if tp + fp > 0:
                precision_sum += tp / (tp + fp)

        return (precision_sum / self.num_classes).item()


class Recall:
    """
    召回率指标
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        """
        Args:
            num_classes: 分类数量
            average: 平均方式,'macro' 或 'micro'
        """
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算召回率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            召回率
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        recall_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_positive = pred_labels == class_idx
            true_positive = target == class_idx
            tp = (pred_positive & true_positive).sum().float()
            fn = (~pred_positive & true_positive).sum().float()

            if tp + fn > 0:
                recall_sum += tp / (tp + fn)

        return (recall_sum / self.num_classes).item()


class F1Score:
    """
    F1分数指标
    """

    def __init__(self, num_classes: int | None = None, average: str = "macro"):
        """
        Args:
            num_classes: 分类数量
            average: 平均方式,'macro' 或 'micro'
        """
        self.num_classes = num_classes
        self.average = average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算F1分数

        Args:
            pred: 预测值,形状为 (batch_size, num_classes)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            F1分数
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = (pred >= 0.5).long()

        if self.num_classes is None:
            self.num_classes = len(torch.unique(target))

        f1_sum = torch.tensor(0.0, device=target.device)
        for class_idx in range(self.num_classes):
            pred_positive = pred_labels == class_idx
            true_positive = target == class_idx
            tp = (pred_positive & true_positive).sum().float()
            fp = (pred_positive & ~true_positive).sum().float()
            fn = (~pred_positive & true_positive).sum().float()

            precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0, device=target.device)
            recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0, device=target.device)

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_sum += f1

        return (f1_sum / self.num_classes).item()


class MeanAbsoluteError:
    """
    平均绝对误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算平均绝对误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            平均绝对误差
        """
        return torch.mean(torch.abs(pred - target)).item()


class MeanSquaredError:
    """
    均方误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算均方误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            均方误差
        """
        return torch.mean((pred - target) ** 2).item()


class RootMeanSquaredError:
    """
    均方根误差指标
    """

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算均方根误差

        Args:
            pred: 预测值
            target: 真实值

        Returns:
            均方根误差
        """
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()


class TopKAccuracy:
    """
    Top-K准确率指标
    """

    def __init__(self, k: int = 5):
        """
        Args:
            k: Top-K中的K值
        """
        self.k = k

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算Top-K准确率

        Args:
            pred: 预测值,形状为 (batch_size, num_classes)
            target: 真实标签,形状为 (batch_size,)

        Returns:
            Top-K准确率
        """
        _, pred_topk = torch.topk(pred, self.k, dim=1)
        target_expanded = target.unsqueeze(1).expand_as(pred_topk)
        correct = (pred_topk == target_expanded).any(dim=1).sum().float()
        total = torch.tensor(target.shape[0], dtype=torch.float, device=target.device)
        return (correct / total).item()


class IoU:
    """
    IoU(交并比)指标
    主要用于分割任务,计算预测区域和真实区域的交并比
    """

    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        """
        Args:
            threshold: 二值化阈值,将预测结果转换为二值掩码
            epsilon: 避免除以零的小常数
        """
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算IoU

        Args:
            pred: 预测值,形状为 (batch_size, ...) 或 (batch_size, 1, ...)
            target: 真实标签,形状与pred相同

        Returns:
            IoU分数 (0-1之间的浮点数)
        """
        # 确保输入形状一致
        if pred.shape != target.shape:
            raise ValueError(f"预测和真实标签形状不匹配: {pred.shape} vs {target.shape}")

        # 如果有通道维度且为1,则压缩
        if pred.dim() > 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() > 2 and target.shape[1] == 1:
            target = target.squeeze(1)

        # 二值化预测结果
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()

        # 计算交集和并集
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        # 计算IoU
        iou = intersection / (union + self.epsilon)

        return iou.item()

    def __call_multi_class__(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> float:
        """
        计算多类IoU的平均值

        Args:
            pred: 预测值,形状为 (batch_size, num_classes, height, width)
            target: 真实标签,形状为 (batch_size, height, width)
            num_classes: 类别数量

        Returns:
            平均IoU分数
        """
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(
                "多类IoU需要pred形状为(batch_size, num_classes, height, width),target形状为(batch_size, height, width)"
            )

        pred_labels = torch.argmax(pred, dim=1)
        iou_sum = torch.tensor(0.0, device=target.device)

        for class_idx in range(num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection

            if union > 0:
                iou_sum += intersection / (union + self.epsilon)

        return (iou_sum / num_classes).item()


class DiceCoefficient:
    """
    Dice系数指标
    主要用于分割任务,计算预测区域和真实区域的Dice系数
    """

    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        """
        Args:
            threshold: 二值化阈值,将预测结果转换为二值掩码
            epsilon: 避免除以零的小常数
        """
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算Dice系数

        Args:
            pred: 预测值,形状为 (batch_size, ...) 或 (batch_size, 1, ...)
            target: 真实标签,形状与pred相同

        Returns:
            Dice系数 (0-1之间的浮点数)
        """
        # 确保输入形状一致
        if pred.shape != target.shape:
            raise ValueError(f"预测和真实标签形状不匹配: {pred.shape} vs {target.shape}")

        # 如果有通道维度且为1,则压缩
        if pred.dim() > 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() > 2 and target.shape[1] == 1:
            target = target.squeeze(1)

        # 二值化预测结果
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()

        # 计算交集和预测区域与真实区域的总和
        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()

        # 计算Dice系数
        dice = (2 * intersection) / (total + self.epsilon)

        return dice.item()

    def __call_multi_class__(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> float:
        """
        计算多类Dice系数的平均值

        Args:
            pred: 预测值,形状为 (batch_size, num_classes, height, width)
            target: 真实标签,形状为 (batch_size, height, width)
            num_classes: 类别数量

        Returns:
            平均Dice系数
        """
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(
                "多类Dice系数需要pred形状为(batch_size, num_classes, height, width),target形状为(batch_size, height, width)"
            )

        pred_labels = torch.argmax(pred, dim=1)
        dice_sum = torch.tensor(0.0, device=target.device)

        for class_idx in range(num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()

            intersection = (pred_mask * target_mask).sum()
            total = pred_mask.sum() + target_mask.sum()

            if total > 0:
                dice_sum += (2 * intersection) / (total + self.epsilon)

        return (dice_sum / num_classes).item()
