"""
注册表系统,用于管理模型注册和实例化。
"""

from typing import Dict, Type, Any, Optional, Tuple
import inspect
import fnmatch

# Optional imports for type hints
try:
    from torch import Tensor
    from torch.utils.data import Dataset

    Sample = Tuple[Tensor, int]  # 举例：图像张量 + 整数标签
except ImportError:
    # Fallback types if torch is not available
    Tensor = Any
    Dataset = Any
    Sample = Any

# 导入配置相关的dataclass
from .config import (
    TransformConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    MetricConfig
)


class Registry:
    """通用的注册表类,用于管理注册的组件。"""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None):
        """装饰器,用于注册类或函数。

        Args:
            name: 注册的名称。如果为None,则使用类/函数名称。
        """

        def decorator(cls_or_fn):
            register_name = name if name is not None else cls_or_fn.__name__
            if register_name in self._registry:
                raise ValueError(f"{register_name} already registered in {self.name}")
            self._registry[register_name] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, name: str) -> Type:
        """通过名称获取注册的组件。

        Args:
            name: 注册组件的名称。

        Returns:
            注册的类或函数。
        """
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self.name} registry")
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """构建注册组件的实例。

        Args:
            name: 注册组件的名称。
            **kwargs: 传递给构造函数的参数。

        Returns:
            注册组件的实例。
        """
        cls_or_fn = self.get(name)

        # Filter kwargs to match the signature
        if inspect.isclass(cls_or_fn):
            sig = inspect.signature(cls_or_fn.__init__)
            # Remove 'self' parameter
            valid_params = set(sig.parameters.keys()) - {"self"}
        else:
            sig = inspect.signature(cls_or_fn)
            valid_params = set(sig.parameters.keys())

        # Filter kwargs to only include valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        return cls_or_fn(**filtered_kwargs)

    def list_available(self):
        """列出所有可用的注册组件。"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """检查组件是否已注册。"""
        return name in self._registry

    def __len__(self) -> int:
        """获取注册组件的数量。"""
        return len(self._registry)


# 不同组件的全局注册表
MODEL_REGISTRY = Registry("MODEL")
DATASET_REGISTRY = Registry("DATASET")
OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
SCHEDULER_REGISTRY = Registry("SCHEDULER")
LOSS_REGISTRY = Registry("LOSS")
METRIC_REGISTRY = Registry("METRIC")
TRANSFORM_REGISTRY = Registry("TRANSFORM")


def register_model(name: Optional[str] = None):
    """注册模型类。"""
    return MODEL_REGISTRY.register(name)


def register_dataset(name: Optional[str] = None):
    """注册数据集类。"""
    return DATASET_REGISTRY.register(name)


def register_optimizer(name: Optional[str] = None):
    """注册优化器类。"""
    return OPTIMIZER_REGISTRY.register(name)


def register_scheduler(name: Optional[str] = None):
    """注册调度器类。"""
    return SCHEDULER_REGISTRY.register(name)


def register_loss(name: Optional[str] = None):
    """注册损失函数类。"""
    return LOSS_REGISTRY.register(name)


def register_metric(name: Optional[str] = None):
    """注册评估指标类。"""
    return METRIC_REGISTRY.register(name)


def register_transform(name: Optional[str] = None):
    """注册数据变换类。"""
    return TRANSFORM_REGISTRY.register(name)

def build_model(config: ModelConfig):
    """从注册表或外部库构建模型。

    Args:
        config: 模型配置, 包含name、from_library、params和weight_init

    Returns:
        模型实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params
    weight_init_fn = config.weight_init
    load_ckpt = config.load_ckpt

    # 构建模型
    if from_library is None or from_library == "local":
        # 使用本地注册的模型
        model = MODEL_REGISTRY.build(name, **params)
    elif from_library == "timm":
        # 使用timm库创建模型
        try:
            import timm
            model = timm.create_model(name, **params)
        except ImportError:
            raise ImportError("请安装timm库: pip install timm")
        except Exception as e:
            raise RuntimeError(f"从timm创建模型 '{name}' 失败: {e}")
    elif from_library == "torchvision":
        # 使用torch库创建模型
        try:
            import torchvision.models as models

            if not hasattr(models, name):
                raise AttributeError(f"torch库中未找到模型 '{name}'")
            model_fn = getattr(models, name)
            model = model_fn(**params)
        except ImportError:
            raise ImportError("请安装torchvision库: pip install torchvision")
        except Exception as e:
            raise RuntimeError(f"从torch库创建模型 '{name}' 失败: {e}")
    elif from_library == "torch":
        # 使用torch.nn创建层/模块
        try:
            import torch.nn as nn

            if not hasattr(nn, name):
                raise AttributeError(f"torch.nn中未找到模块 '{name}'")
            module_cls = getattr(nn, name)
            model = module_cls(**params)
        except ImportError:
            raise ImportError("请安装torch库")
        except Exception as e:
            raise RuntimeError(f"从torch.nn创建模块 '{name}' 失败: {e}")
    else:
        raise ValueError(
            f"不支持的库: {from_library}. 支持的库: 'timm', 'torchvision', 'torch', None"
        )

    # 应用权重初始化
    if hasattr(model, 'apply') and callable(model.apply):
        # 确保模型有apply方法才进行初始化
        from .weight import weight_init
        weight_init(model, weight_init_fn)
    else:
        print(f"警告: 模型 {name} 不支持权重初始化, 跳过初始化步骤")

    if load_ckpt is not None:
        if hasattr(model, 'load') and callable(model.load):
            model.load(load_ckpt)
        else:
            from .weight import load_weight
            load_weight(model,load_ckpt)

    return model


def build_dataset(
    config: DatasetConfig,
    transforms: Optional[Any] = None,
) -> Any:
    """从注册表或外部库构建数据集。

    Args:
        config: 数据集配置, 包含name、from_library和params
        transforms:
            - None: 不应用变换
            - 变换实例: 直接使用的变换对象

    Returns:
        数据集实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params.copy()  # 避免修改原始配置
    
    # 将变换应用到参数中
    if transforms is not None:
        params["transform"] = transforms

    # 构建数据集
    if from_library is None or from_library == "local":
        # 使用本地注册的数据集
        return DATASET_REGISTRY.build(name, **params)
    elif from_library == "torchvision":
        # 使用torchvision库创建数据集
        try:
            import torchvision.datasets as datasets

            if not hasattr(datasets, name):
                raise AttributeError(f"torchvision.datasets中未找到数据集 '{name}'")
            dataset_cls = getattr(datasets, name)
            return dataset_cls(**params)
        except ImportError:
            raise ImportError("请安装torchvision库: pip install torchvision")
        except Exception as e:
            raise RuntimeError(f"从torchvision库创建数据集 '{name}' 失败: {e}")
    elif from_library == "torch":
        # 使用torch库的基础数据集类
        try:
            import torch.utils.data as data

            if not hasattr(data, name):
                raise AttributeError(f"torch.utils.data中未找到数据集 '{name}'")
            dataset_cls = getattr(data, name)
            return dataset_cls(**params)
        except ImportError:
            raise ImportError("请安装torch库")
        except Exception as e:
            raise RuntimeError(f"从torch库创建数据集 '{name}' 失败: {e}")
    else:
        raise ValueError(
            f"不支持的库: {from_library}. 支持的库: 'torchvision', 'torch', None"
        )


def build_optimizer(
    config: OptimizerConfig,
    model_params,
):
    """
    根据配置构建一个 PyTorch 优化器,支持为不同参数组设置不同的超参数。

    Args:
        config (OptimizerConfig): 优化器配置, 包含name、from_library、params等
        model_params (iterable): 模型参数的迭代器,通常是 `model.named_parameters()` 的结果。
                                 它应该是一个 (name, param) 元组的序列.

    Returns:
        torch.optim.Optimizer: 配置好的 PyTorch 优化器实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params

    # 1. 获取优化器类
    if from_library is None or from_library == "local":
        opt_cls = OPTIMIZER_REGISTRY.get(name)
        if opt_cls is None:
            raise ValueError(f"Optimizer '{name}' not found in local registry")
    elif from_library == "torch":
        try:
            from torch import optim

            opt_cls = getattr(optim, name)
        except AttributeError:
            raise ValueError(f"Optimizer '{name}' not found in torch.optim")
    else:
        raise ValueError(f"不支持的库: {from_library}. 支持的库: 'torch', None")

    # 复制配置字典,以防修改原始字典
    optimizer_config = params.copy() if params else {}

    # 提取并移除特殊组的定义,剩下的是优化器的默认参数
    special_groups_rules = optimizer_config.pop("groups", [])

    # 将模型参数转换为列表,以便多次遍历
    all_params = list(model_params)

    # 用于存储已分配到特殊组的参数名称,避免重复分配
    assigned_params = set()

    # 最终传递给优化器的参数组列表
    final_param_groups = []

    # 2. 处理特殊参数组
    for rule in special_groups_rules:
        pattern = rule["params"]

        # 找到所有匹配当前规则的参数
        matching_params = []
        for param_name, param in all_params:
            # 如果参数已经被分配,则跳过
            if param_name in assigned_params:
                continue

            # 使用 fnmatch 进行通配符匹配
            if fnmatch.fnmatch(param_name, pattern):
                matching_params.append(param)
                assigned_params.add(param_name)

        # 如果找到了匹配的参数,则创建新组
        if matching_params:
            new_group = rule.copy()
            new_group["params"] = matching_params
            final_param_groups.append(new_group)

    # 3. 处理默认参数组(未被任何特殊规则匹配的参数)
    default_params = []
    for param_name, param in all_params:
        if param_name not in assigned_params:
            default_params.append(param)

    if default_params:
        final_param_groups.append({"params": default_params})

    # 4. 实例化优化器
    # final_param_groups 是优化器的第一个参数
    # optimizer_config 现在只包含默认参数 (如 lr, momentum)
    optimizer = opt_cls(final_param_groups, **optimizer_config)

    return optimizer


def build_scheduler(optimizer, config: SchedulerConfig):
    """从注册表或外部库构建调度器。

    Args:
        optimizer: 优化器实例
        config: 调度器配置, 包含name、from_library和params

    Returns:
        调度器实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params

    if from_library is None or from_library == "local":
        # 使用本地注册的调度器
        return SCHEDULER_REGISTRY.build(name, **params)
    elif from_library == "torch":
        # 使用torch库创建学习率调度器
        try:
            import torch.optim.lr_scheduler as lr_scheduler

            if not hasattr(lr_scheduler, name):
                raise AttributeError(f"torch.optim.lr_scheduler中未找到调度器 '{name}'")
            scheduler_cls = getattr(lr_scheduler, name)
            return scheduler_cls(optimizer, **params)
        except ImportError:
            raise ImportError("请安装torch库")
        except Exception as e:
            raise RuntimeError(f"从torch库创建调度器 '{name}' 失败: {e}")
    else:
        raise ValueError(f"不支持的库: {from_library}. 支持的库: 'torch', None")


def build_loss(config: LossConfig):
    """从注册表或外部库构建损失函数。

    Args:
        config: 损失函数配置, 包含name、from_library和params

    Returns:
        损失函数实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params

    if from_library is None or from_library == "local":
        # 使用本地注册的损失函数
        return LOSS_REGISTRY.build(name, **params)
    elif from_library == "torch":
        # 使用torch库创建损失函数
        try:
            import torch.nn as nn

            if not hasattr(nn, name):
                raise AttributeError(f"torch.nn中未找到损失函数 '{name}'")
            loss_cls = getattr(nn, name)
            return loss_cls(**params)
        except ImportError:
            raise ImportError("请安装torch库")
        except Exception as e:
            raise RuntimeError(f"从torch库创建损失函数 '{name}' 失败: {e}")
    else:
        raise ValueError(f"不支持的库: {from_library}. 支持的库: 'torch', None")


def build_metric(config: MetricConfig):
    """从注册表构建评估指标。"""
    name = config.name
    from_library = config.from_library
    params = config.params

    if from_library is None or from_library == "local":
        # 使用本地注册的指标
        return METRIC_REGISTRY.build(name, **params)
    elif from_library == "torch":
        # 使用torch库创建指标
        try:
            import torch.nn as nn

            if not hasattr(nn, name):
                raise AttributeError(f"torch.nn中未找到指标 '{name}'")
            metric_cls = getattr(nn, name)
            return metric_cls(**params)
        except ImportError:
            raise ImportError("请安装torch库")
        except Exception as e:
            raise RuntimeError(f"从torch库创建指标 '{name}' 失败: {e}")

    else:
        raise ValueError(f"不支持的库: {from_library}. 支持的库: 'torch', local")


def build_transform(config: TransformConfig):
    """从注册表或外部库构建数据变换。

    Args:
        config: 数据变换配置, 包含name、from_library和params

    Returns:
        数据变换实例。
    """
    name = config.name
    from_library = config.from_library
    params = config.params

    if from_library is None or from_library == "local":
        # 使用本地注册的数据变换
        return TRANSFORM_REGISTRY.build(name, **params)
    elif from_library in ["torchvision", "torch"]:
        # 使用torchvision库创建数据变换
        try:
            import torchvision.transforms as transforms

            if not hasattr(transforms, name):
                raise AttributeError(f"torchvision.transforms中未找到数据变换 '{name}'")
            transform_cls = getattr(transforms, name)
            return transform_cls(**params)
        except ImportError:
            raise ImportError("请安装torchvision库: pip install torchvision")
        except Exception as e:
            raise RuntimeError(f"从torchvision库创建数据变换 '{name}' 失败: {e}")
    elif from_library in ["albumentations", "A"]:
        # 使用albumentations库创建数据变换
        try:
            import albumentations as A

            if not hasattr(A, name):
                raise AttributeError(f"albumentations中未找到数据变换 '{name}'")
            transform_cls = getattr(A, name)
            return transform_cls(**params)
        except ImportError:
            raise ImportError("请安装albumentations库: pip install albumentations")
        except Exception as e:
            raise RuntimeError(f"从albumentations库创建数据变换 '{name}' 失败: {e}")
    else:
        raise ValueError(
            f"不支持的库: {from_library}. 支持的库: 'torchvision', 'albumentations', None"
        )   