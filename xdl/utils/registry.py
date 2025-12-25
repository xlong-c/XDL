"""
注册表系统, 用于管理组件注册和实例化。
支持从本地注册表、PyTorch 原生库以及第三方库 (如 timm, torchvision, albumentations) 构建组件。
"""

import inspect
import fnmatch
import importlib
from typing import Dict,  Any, Optional, List,  Callable

# 导入配置类, 用于类型提示
from .config import (
    TransformConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    MetricConfig
)

# 可选导入
try:
    import torch
    import torch.nn as nn
    from torch import optim
except ImportError:
    torch = None
    nn = None
    optim = None


class Registry:
    """通用的注册表类, 用于管理注册的组件及其构建逻辑。"""

    def __init__(self, name: str, default_lib_map: Optional[Dict[str, str]] = None):
        """
        Args:
            name: 注册表名称
            default_lib_map: 库名到子模块路径的默认映射, 例如 {"torchvision": "models"}
        """
        self.name = name
        self._registry: Dict[str, Any] = {}
        self.default_lib_map = default_lib_map or {}

    def register(self, name: Optional[str] = None) -> Callable:
        """装饰器, 用于注册类或函数。"""
        def decorator(cls_or_fn):
            register_name = name if name is not None else cls_or_fn.__name__
            if register_name in self._registry:
                raise ValueError(
                    f"'{register_name}' already registered in {self.name} registry")
            self._registry[register_name] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, name: str, from_library: Optional[str] = None) -> Any:
        """获取注册的组件或从外部库获取。"""
        # 1. 优先从本地注册表查找 (当从库为 None 或 "local" 时)
        if (from_library is None or from_library == "local") and name in self._registry:
            return self._registry[name]

        # 2. 如果指定了外部库, 或者本地没找到且指定了默认库
        if from_library and from_library != "local":
            module_path = self.default_lib_map.get(from_library, "")
            return self._get_obj_from_lib(from_library, name, module_path)

        # 3. 未找到
        raise KeyError(
            f"'{name}' not found in {self.name} registry (lib={from_library}). "
            f"Available local: {self.list_available()}"
        )

    @staticmethod
    def _get_obj_from_lib(lib_name: str, obj_name: str, module_path: str = ""):
        """从指定库中动态获取对象。"""
        try:
            if not module_path:
                module = importlib.import_module(lib_name)
            else:
                module = importlib.import_module(f"{lib_name}.{module_path}")

            if not hasattr(module, obj_name):
                raise AttributeError(
                    f"Module '{module.__name__}' has no attribute '{obj_name}'")
            return getattr(module, obj_name)
        except ImportError:
            raise ImportError(f"Please install library: {lib_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to get '{obj_name}' from {lib_name}: {e}")

    def build(self, cfg: Any = None, **kwargs) -> Any:
        """
        通用构建方法。
        支持从配置对象或字典构建。
        """
        if cfg is None:
            # 如果没有配置对象, 尝试从 kwargs 中提取 name 和 params
            name = kwargs.pop("name", None)
            from_lib = kwargs.pop("from_library", "local")
            params = kwargs
        elif isinstance(cfg, dict):
            name = cfg.get("name")
            from_lib = cfg.get("from_library", "local")
            params = {**cfg.get("params", {}), **kwargs}
        else:
            # 假设是 Dataclass 配置对象
            name = getattr(cfg, "name", None)
            from_lib = getattr(cfg, "from_library", "local")
            params = {**getattr(cfg, "params", {}), **kwargs}

        if not name:
            raise ValueError(
                f"Build failed: 'name' is required for {self.name} registry")

        # 特殊处理 timm 库
        if from_lib == "timm":
            try:
                import timm
                return timm.create_model(name, **params)
            except ImportError:
                raise ImportError(
                    "Please install 'timm' library to use timm models")

        # 获取对象并实例化
        obj = self.get(name, from_lib)
        return self.instantiate(obj, **params)

    @staticmethod
    def instantiate(obj: Any, **kwargs) -> Any:
        """智能实例化对象, 自动过滤不支持的参数。"""
        if not (inspect.isclass(obj) or callable(obj)):
            return obj

        try:
            if inspect.isclass(obj):
                # 优先检查 __init__, 如果没有定义则可能在父类
                sig = inspect.signature(obj.__init__)
                valid_params = set(sig.parameters.keys()) - {"self"}
            else:
                sig = inspect.signature(obj)
                valid_params = set(sig.parameters.keys())

            # 如果函数支持 **kwargs, 则不进行过滤
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                filtered_kwargs = kwargs
            else:
                filtered_kwargs = {k: v for k,
                                   v in kwargs.items() if k in valid_params}
        except (ValueError, TypeError):
            # 对于一些内置函数、C扩展或无签名的对象, 可能无法获取 signature
            filtered_kwargs = kwargs

        return obj(**filtered_kwargs)

    def list_available(self) -> List[str]:
        """列出所有本地注册的组件名称。"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# --- 全局注册表实例 ---

MODEL_REGISTRY = Registry("MODEL", {"torchvision": "models", "torch": "nn"})
DATASET_REGISTRY = Registry(
    "DATASET", {"torchvision": "datasets", "torch": "utils.data"})
OPTIMIZER_REGISTRY = Registry("OPTIMIZER", {"torch": "optim"})
SCHEDULER_REGISTRY = Registry("SCHEDULER", {"torch": "optim.lr_scheduler"})
LOSS_REGISTRY = Registry("LOSS", {"torch": "nn"})
METRIC_REGISTRY = Registry("METRIC", {"torch": "nn"})
TRANSFORM_REGISTRY = Registry(
    "TRANSFORM", {"torchvision": "transforms", "torch": "transforms"})

# --- 便捷注册装饰器 ---
register_model = MODEL_REGISTRY.register
register_dataset = DATASET_REGISTRY.register
register_optimizer = OPTIMIZER_REGISTRY.register
register_scheduler = SCHEDULER_REGISTRY.register
register_loss = LOSS_REGISTRY.register
register_metric = METRIC_REGISTRY.register
register_transform = TRANSFORM_REGISTRY.register


# --- 统一构建入口 ---

def build_from_cfg(cfg: Any, registry: Registry, **extra_params) -> Any:
    """从配置构建组件的统一入口。"""
    return registry.build(cfg, **extra_params)


# --- 具体组件构建函数 (向后兼容并增强) ---

def build_model(config: ModelConfig) -> Any:
    """构建模型。"""
    model = MODEL_REGISTRY.build(config)

    # 权重初始化
    if hasattr(model, 'apply'):
        from .weight import weight_init
        weight_init(model, config.weight_init)

    # 加载权重
    if config.load_ckpt:
        if hasattr(model, 'load') and callable(model.load):
            model.load(config.load_ckpt)
        else:
            from .weight import load_weight
            load_weight(model, config.load_ckpt)

    return model


def build_dataset(config: DatasetConfig, transforms: Optional[Any] = None) -> Any:
    """构建数据集。"""
    extra = {"transform": transforms} if transforms is not None else {}
    return DATASET_REGISTRY.build(config, **extra)


def build_optimizer(config: OptimizerConfig, model_params: Any) -> Any:
    """构建优化器, 支持复杂参数分组。"""
    # 提取基本参数
    params = config.params.copy()
    special_rules = params.pop("groups", [])

    # 获取优化器类
    opt_cls = OPTIMIZER_REGISTRY.get(config.name, config.from_library)

    # 如果没有特殊规则, 直接构建
    if not special_rules:
        # model_params 可能是参数迭代器或 named_parameters
        # 如果是迭代器, 转换成列表以支持多次遍历(如果需要)
        param_list = list(model_params)
        # 如果元素是元组 (name, param), 提取 param
        if param_list and isinstance(param_list[0], tuple):
            param_list = [p for n, p in param_list]
        return opt_cls(param_list, **params)

    # 处理参数分组
    all_named_params = list(model_params)
    if not (all_named_params and isinstance(all_named_params[0], tuple)):
        raise ValueError(
            "Parameter groups require named_parameters from model")

    assigned_names = set()
    param_groups = []

    # 1. 匹配特殊规则
    for rule in special_rules:
        pattern = rule["params"]
        matched = []
        for n, p in all_named_params:
            if n not in assigned_names and fnmatch.fnmatch(n, pattern):
                matched.append(p)
                assigned_names.add(n)

        if matched:
            group = rule.copy()
            group["params"] = matched
            param_groups.append(group)

    # 2. 剩余参数进入默认组
    default_params = [
        p for n, p in all_named_params if n not in assigned_names]
    if default_params:
        param_groups.append({"params": default_params})

    return opt_cls(param_groups, **params)


def build_scheduler(optimizer: Any, config: SchedulerConfig) -> Any:
    """构建学习率调度器。"""
    return SCHEDULER_REGISTRY.build(config, optimizer=optimizer)


def build_loss(config: LossConfig) -> Any:
    """构建损失函数。"""
    return LOSS_REGISTRY.build(config)


def build_metric(config: MetricConfig) -> Any:
    """构建评估指标。"""
    return METRIC_REGISTRY.build(config)


def build_transform(config: TransformConfig) -> Any:
    """构建数据变换。"""
    # 特殊映射: A 代表 albumentations
    from_lib = config.from_library
    if from_lib == "A":
        from_lib = "albumentations"

    return TRANSFORM_REGISTRY.build(config, from_library=from_lib)
