"""
setup_from_yaml 函数
从 YAML 配置文件一键构建完整的训练配置
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

from .builder import (
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transform,
)
from .dataclass import TrainSetup


def setup_from_yaml(
    config_path: Union[str, Path],
    device: Optional[str] = None,
) -> TrainSetup:
    """
    从 YAML 配置文件构建完整的训练配置
    
    一行代码完成模型、数据加载器、优化器、调度器、损失函数、指标等所有组件的构建。
    
    Args:
        config_path: YAML 配置文件的路径
        device: 设备名称 (cuda/cpu)，如果为 None 则从配置文件中读取
    
    Returns:
        TrainSetup 对象，包含所有训练组件
    
    Example:
        >>> from xdl.config import setup_from_yaml
        >>> 
        >>> # 基本用法
        >>> setup = setup_from_yaml('config/vgg_cifar100.yaml')
        >>> 
        >>> # 使用 Trainer
        >>> from xdl.trainer import Trainer
        >>> trainer = Trainer(
        ...     model=setup.model,
        ...     train_dataloader=setup.train_loader,
        ...     val_dataloader=setup.val_loader,
        ...     optimizer=setup.optimizer,
        ...     loss_fn=setup.loss_fn,
        ...     metrics=setup.metrics,
        ...     max_epochs=setup.num_epochs,
        ... )
        >>> trainer.fit()
        
        >>> # 直接访问组件
        >>> model = setup.model
        >>> optimizer = setup.optimizer
        >>> scheduler = setup.scheduler
    """
    # 加载配置文件
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # ===== 1. 构建模型 =====
    model_config = config.get("core_config", {}).get("model", {})
    model = build_model(model_config)
    
    # ===== 2. 构建数据集和数据加载器 =====
    data_config = config.get("data_config", {})
    transform_config = data_config.get("transform", {})
    dataset_config = data_config.get("dataset", {})
    dataloader_config = data_config.get("dataloader", {})
    
    # 构建数据变换
    train_transform = None
    val_transform = None
    
    if "train_transform" in transform_config:
        train_transform = build_transform(transform_config["train_transform"])
    if "val_transform" in transform_config:
        val_transform = build_transform(transform_config["val_transform"])
    
    # 构建数据集
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if "train_dataset" in dataset_config:
        train_dataset = build_dataset(
            dataset_config["train_dataset"],
            transform=train_transform
        )
    if "val_dataset" in dataset_config:
        val_dataset = build_dataset(
            dataset_config["val_dataset"],
            transform=val_transform
        )
    if "test_dataset" in dataset_config:
        test_dataset = build_dataset(
            dataset_config["test_dataset"],
            transform=val_transform
        )
    
    # 构建数据加载器
    train_loader = None
    val_loader = None
    test_loader = None
    
    if "train_loader" in dataloader_config and train_dataset:
        train_loader = build_dataloader(
            train_dataset,
            dataloader_config["train_loader"]
        )
    if "val_loader" in dataloader_config and val_dataset:
        val_loader = build_dataloader(
            val_dataset,
            dataloader_config["val_loader"]
        )
    if "test_loader" in dataloader_config and test_dataset:
        test_loader = build_dataloader(
            test_dataset,
            dataloader_config["test_loader"]
        )
    
    # ===== 3. 构建优化器 =====
    optimizer_config = config.get("core_config", {}).get("optimizer", {})
    optimizer = build_optimizer(model, optimizer_config)
    
    # ===== 4. 构建调度器 =====
    scheduler_config = config.get("core_config", {}).get("scheduler", {})
    scheduler = build_scheduler(optimizer, scheduler_config)
    
    # ===== 5. 构建损失函数 =====
    loss_config = config.get("core_config", {}).get("loss", [])
    loss_fn = build_loss(loss_config)
    
    # ===== 6. 构建评估指标 =====
    metrics_config = config.get("core_config", {}).get("metrics", [])
    metrics = build_metrics(metrics_config)
    
    # ===== 7. 获取训练参数 =====
    training_config = config.get("training", {})
    
    # 设备
    if device is None:
        device = str(training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = str(device)
    
    # 训练参数
    num_epochs = training_config.get("num_epochs", 100)
    batch_size = training_config.get("batch_size", 128)
    
    # ===== 8. 构建 TrainSetup =====
    setup = TrainSetup(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
        full_config=config,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    
    return setup


__all__ = ["setup_from_yaml"]
