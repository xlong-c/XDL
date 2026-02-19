"""
组件构建器
从配置字典构建各个训练组件
"""

import importlib
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def _get_class_from_library(name: str, from_library: str) -> Any:
    """
    从指定库获取类或函数
    
    Args:
        name: 类或函数的名称
        from_library: 库标识符 ("torch", "torchvision", "local")
    
    Returns:
        类或函数对象
    """
    if from_library == "torch":
        return getattr(torch, name)
    elif from_library == "torchvision":
        return getattr(torchvision, name)
    elif from_library == "local":
        # 本地注册的组件，通过 xdl.utils.registry 获取
        from xdl.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
        
        registries = {
            "model": MODEL_REGISTRY,
            "dataset": DATASET_REGISTRY,
            "loss": LOSS_REGISTRY,
            "metric": METRIC_REGISTRY,
            "optimizer": OPTIMIZER_REGISTRY,
            "scheduler": SCHEDULER_REGISTRY,
        }
        
        # 尝试从各个注册表获取
        for reg_name, registry in registries.items():
            try:
                return registry.get(name)
            except KeyError:
                continue
        
        raise ValueError(f"Cannot find '{name}' in any local registry")
    else:
        raise ValueError(f"Unknown library: {from_library}")


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    从配置构建模型
    
    Args:
        config: 模型配置字典，格式如下：
            {
                "backbone": {
                    "name": "VGG16",
                    "from_library": "local",
                    "params": {"num_classes": 100, "dropout": 0.5}
                }
            }
    
    Returns:
        构建的模型实例
    """
    if "backbone" in config:
        backbone_config = config["backbone"]
    else:
        backbone_config = config
    
    name = backbone_config.get("name")
    from_library = backbone_config.get("from_library", "local")
    params = backbone_config.get("params", {})
    
    # 处理特殊参数
    if "num_classes" in params and "num_classes" not in params.get("kwargs", {}):
        if "kwargs" not in params:
            params["kwargs"] = {}
        params["kwargs"]["num_classes"] = params.pop("num_classes")
    
    cls = _get_class_from_library(name, from_library)
    return cls(**params)


def build_transform(config: Dict[str, Any]) -> Any:
    """
    从配置构建数据变换
    
    Args:
        config: 变换配置字典
    """
    import torchvision
    from torchvision import transforms
    
    transforms_list = []
    combination_strategy = config.get("combination_strategy", "compose")
    
    for transform_config in config.get("transforms", []):
        name = transform_config.get("name")
        params = transform_config.get("params", {})
        
        # 获取变换类
        if hasattr(transforms, name):
            transform_cls = getattr(transforms, name)
            transforms_list.append(transform_cls(**params))
    
    # 组合变换
    if combination_strategy == "compose":
        return transforms.Compose(transforms_list)
    elif combination_strategy == "sequential":
        return transforms.Sequential(transforms_list)
    else:
        return transforms_list


def build_dataset(config: Dict[str, Any], transform: Optional[Any] = None) -> Any:
    """
    从配置构建数据集
    
    Args:
        config: 数据集配置字典
        transform: 数据变换（可选）
    
    Returns:
        构建的数据集实例
    """
    name = config.get("name")
    from_library = config.get("from_library", "torchvision")
    params = config.get("params", {}).copy()
    
    # 添加 transform 参数
    if transform is not None:
        params["transform"] = transform
    
    # 获取数据集类
    if from_library == "torchvision":
        import torchvision
        dataset_cls = getattr(torchvision.datasets, name)
    elif from_library == "local":
        from xdl.utils.registry import DATASET_REGISTRY
        dataset_cls = DATASET_REGISTRY.get(name)
    else:
        raise ValueError(f"Unknown library for dataset: {from_library}")
    
    return dataset_cls(**params)


def build_dataloader(
    dataset: Any,
    config: Dict[str, Any]
) -> DataLoader:
    """
    从配置构建数据加载器
    
    Args:
        dataset: 数据集实例
        config: 数据加载器配置字典
    
    Returns:
        DataLoader 实例
    """
    params = config.get("params", {})
    return DataLoader(dataset, **params)


def build_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    从配置构建优化器
    
    Args:
        model: 模型实例
        config: 优化器配置字典，格式如下：
            {
                "main_optimizer": {
                    "model": ["backbone"],  # 要优化的参数组
                    "name": "SGD",
                    "from_library": "torch",
                    "params": {"lr": 0.01, "momentum": 0.9}
                }
            }
    
    Returns:
        优化器实例
    """
    # 获取优化器配置
    if "main_optimizer" in config:
        opt_config = config["main_optimizer"]
    else:
        opt_config = config
    
    name = opt_config.get("name")
    from_library = opt_config.get("from_library", "torch")
    params = opt_config.get("params", {})
    
    # 获取要优化的参数
    param_groups = opt_config.get("model", ["model"])
    if isinstance(param_groups, list):
        # 简单处理：优化所有参数
        optimizer_cls = _get_class_from_library(name, from_library)
        return optimizer_cls(model.parameters(), **params)
    else:
        # 参数组处理
        optimizer_cls = _get_class_from_library(name, from_library)
        
        # 如果指定了参数组名称
        param_group_configs = opt_config.get("param_groups", {})
        groups = []
        
        for group_name, group_params in param_group_configs.items():
            if group_name == "backbone" or group_name == "all":
                groups.append({"params": model.parameters(), **group_params})
            elif hasattr(model, group_name):
                groups.append({"params": getattr(model, group_name).parameters(), **group_params})
        
        if not groups:
            groups = [{"params": model.parameters()}]
        
        return optimizer_cls(groups, **params)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[Any]:
    """
    从配置构建学习率调度器
    
    Args:
        optimizer: 优化器实例
        config: 调度器配置字典
    
    Returns:
        调度器实例，如果配置为空则返回 None
    """
    if not config:
        return None
    
    # 获取调度器配置
    if "main_scheduler" in config:
        sched_config = config["main_scheduler"]
    else:
        sched_config = config
    
    name = sched_config.get("name")
    from_library = sched_config.get("from_library", "torch")
    params = sched_config.get("params", {})
    
    if name is None:
        return None
    
    scheduler_cls = _get_class_from_library(name, from_library)
    return scheduler_cls(optimizer, **params)


def build_loss(config: List[Dict[str, Any]]) -> torch.nn.Module:
    """
    从配置构建损失函数
    
    Args:
        config: 损失函数配置列表，格式如下：
            [
                {
                    "name": "CrossEntropyLoss",
                    "from_library": "torch",
                    "weight": 1.0,
                    "params": {}
                }
            ]
    
    Returns:
        损失函数实例（如果是多个，返回加权组合）
    """
    if not config:
        raise ValueError("Loss configuration cannot be empty")
    
    if len(config) == 1:
        # 单个损失函数
        loss_config = config[0]
        name = loss_config.get("name")
        from_library = loss_config.get("from_library", "torch")
        params = loss_config.get("params", {})
        
        loss_cls = _get_class_from_library(name, from_library)
        return loss_cls(**params)
    else:
        # 多个损失函数 - 返回加权组合
        from .loss_weighted import WeightedLoss
        
        losses = []
        weights = []
        
        for loss_config in config:
            name = loss_config.get("name")
            from_library = loss_config.get("from_library", "torch")
            params = loss_config.get("params", {})
            weight = loss_config.get("weight", 1.0)
            
            loss_cls = _get_class_from_library(name, from_library)
            losses.append(loss_cls(**params))
            weights.append(weight)
        
        return WeightedLoss(losses, weights)


def build_metrics(config: List[Dict[str, Any]]) -> List[Any]:
    """
    从配置构建评估指标
    
    Args:
        config: 指标配置列表
    
    Returns:
        指标实例列表
    """
    metrics = []
    
    for metric_config in config:
        name = metric_config.get("name")
        from_library = metric_config.get("from_library", "local")
        params = metric_config.get("params", {})
        
        metric_cls = _get_class_from_library(name, from_library)
        
        # 指标可能有不同的初始化方式
        try:
            metrics.append(metric_cls(**params))
        except TypeError:
            # 如果初始化失败，尝试不传参数
            try:
                metrics.append(metric_cls())
            except Exception as e:
                print(f"Warning: Failed to build metric '{name}': {e}")
    
    return metrics
