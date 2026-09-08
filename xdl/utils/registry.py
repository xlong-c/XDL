"""
至简注册表系统.
仅负责建立名称到对象的映射, 不再包含任何配置解析或权重处理逻辑.
"""

import difflib
import importlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from xdl.errors import RegistryError

# 设置日志
logger = logging.getLogger(__name__)


class Registry:
    """最简注册表, 仅提供映射管理."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """注册装饰器.

        同名重复注册同一对象时保持幂等 (模块 reload 场景); 同名注册不同
        对象时抛 ``RegistryError``, 不静默覆盖先注册者.
        """

        def decorator(cls_or_fn):
            register_name = name if name is not None else cls_or_fn.__name__
            existing = self._registry.get(register_name)
            if existing is not None:
                if existing is cls_or_fn:
                    logger.debug(
                        "'%s' already registered in %s with the same object. Skipping.",
                        register_name,
                        self.name,
                    )
                    return cls_or_fn
                raise RegistryError(
                    f"'{register_name}' already registered in {self.name} with a "
                    f"different object ({existing!r}); refusing to overwrite with "
                    f"{cls_or_fn!r}."
                )
            self._registry[register_name] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, name: str) -> Any:
        """根据名称获取对象.用法: Registry.get('name')(*args, **kwargs)"""
        if name in self._registry:
            return self._registry[name]

        _ensure_builtin_registries_for(self)
        if name in self._registry:
            return self._registry[name]

        available = list(self._registry.keys())
        matches = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        msg = f"'{name}' not found in {self.name} registry."
        if matches:
            msg += f" Did you mean: {matches}?"
        raise RegistryError(msg)

    def get_signature(self, name: str) -> str:
        """获取并格式化组件的参数签名."""
        obj = self.get(name)
        try:
            if inspect.isclass(obj):
                target = obj.__init__
            elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
                target = obj
            else:
                target = getattr(obj, "__call__", obj)
            sig = inspect.signature(target)

            params = []
            for p_name, param in sig.parameters.items():
                if p_name == "self":
                    continue

                p_str = f"{p_name}"
                if param.annotation is not inspect.Parameter.empty:
                    p_str += f": {getattr(param.annotation, '__name__', str(param.annotation))}"
                if param.default is not inspect.Parameter.empty:
                    p_str += f" = {repr(param.default)}"
                params.append(p_str)

            return f"{name}({', '.join(params)})"
        except Exception as e:
            return f"Could not get signature for {name}: {e}"

    def list_available(self) -> List[str]:
        _ensure_builtin_registries_for(self)
        return list(self._registry.keys())


# --- 全局注册表实例 ---
MODEL_REGISTRY = Registry("MODEL")
DATASET_REGISTRY = Registry("DATASET")
OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
SCHEDULER_REGISTRY = Registry("SCHEDULER")
LOSS_REGISTRY = Registry("LOSS")
METRIC_REGISTRY = Registry("METRIC")
TRANSFORM_REGISTRY = Registry("TRANSFORM")
COLLATE_REGISTRY = Registry("COLLATE")
CALLBACK_REGISTRY = Registry("CALLBACK")
TASK_REGISTRY = Registry("TASK")

# --- 快捷装饰器 ---
register_model = MODEL_REGISTRY.register
register_dataset = DATASET_REGISTRY.register
register_optimizer = OPTIMIZER_REGISTRY.register
register_scheduler = SCHEDULER_REGISTRY.register
register_loss = LOSS_REGISTRY.register
register_metric = METRIC_REGISTRY.register
register_transform = TRANSFORM_REGISTRY.register
register_collate = COLLATE_REGISTRY.register
register_callback = CALLBACK_REGISTRY.register
register_task = TASK_REGISTRY.register

_BOOTSTRAPPED_MODULES: Dict[str, bool] = {}
_BOOTSTRAPPING_MODULES: Set[str] = set()


def _bootstrap_modules_for_registry(registry: Registry) -> List[str]:
    if registry is MODEL_REGISTRY:
        return ["xdl.model"]
    if registry in (DATASET_REGISTRY, COLLATE_REGISTRY, TRANSFORM_REGISTRY):
        return ["xdl.dataset"]
    if registry is OPTIMIZER_REGISTRY:
        return ["xdl.optimizer"]
    if registry is SCHEDULER_REGISTRY:
        return ["xdl.scheduler"]
    if registry is LOSS_REGISTRY:
        return ["xdl.loss", "xdl.post_training._registry"]
    if registry is METRIC_REGISTRY:
        return ["xdl.metric"]
    if registry is CALLBACK_REGISTRY:
        return ["xdl.callbacks"]
    return []


def _ensure_builtin_registries_for(registry: Registry) -> None:
    """按需导入内置组件模块,避免依赖顶层 xdl import 的副作用."""
    for module_path in _bootstrap_modules_for_registry(registry):
        if _BOOTSTRAPPED_MODULES.get(module_path) or module_path in _BOOTSTRAPPING_MODULES:
            continue
        _BOOTSTRAPPING_MODULES.add(module_path)
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            raise RegistryError(
                f"Failed to bootstrap '{module_path}' for {registry.name} registry: {exc}"
            ) from exc
        else:
            _BOOTSTRAPPED_MODULES[module_path] = True
        finally:
            _BOOTSTRAPPING_MODULES.discard(module_path)


# --- 检查/帮助辅助函数 ---
def inspect_model(name: str):
    """打印构建模型所需的参数."""
    print(f"[MODEL] {MODEL_REGISTRY.get_signature(name)}")


def inspect_optimizer(name: str):
    print(f"[OPTIMIZER] {OPTIMIZER_REGISTRY.get_signature(name)}")


def inspect_dataset(name: str):
    print(f"[DATASET] {DATASET_REGISTRY.get_signature(name)}")


# --- 常用组件预注册 (PyTorch 原生) ---
try:
    import torch

    for opt in ["Adam", "AdamW", "SGD", "RMSprop"]:
        if hasattr(torch.optim, opt):
            OPTIMIZER_REGISTRY.register(opt)(getattr(torch.optim, opt))
    for loss in ["CrossEntropyLoss", "MSELoss", "L1Loss", "BCEWithLogitsLoss"]:
        if hasattr(torch.nn, loss):
            LOSS_REGISTRY.register(loss)(getattr(torch.nn, loss))
except ImportError:
    pass


# --- 统一入口 (仅查找, 实例化由调用方完成) ---
def build_model(name: str):
    """用法: model = build_model('vgg19')(num_classes=10)"""
    return MODEL_REGISTRY.get(name)


def build_dataset(name: str):
    return DATASET_REGISTRY.get(name)


def build_optimizer(name: str):
    return OPTIMIZER_REGISTRY.get(name)


def build_scheduler(name: str):
    return SCHEDULER_REGISTRY.get(name)


def build_loss(name: str):
    return LOSS_REGISTRY.get(name)


def build_metric(name: str):
    return METRIC_REGISTRY.get(name)


def build_transform(name: str):
    return TRANSFORM_REGISTRY.get(name)


def build_callback(name: str):
    return CALLBACK_REGISTRY.get(name)
