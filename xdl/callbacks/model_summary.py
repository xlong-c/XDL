"""
Model Summary Callback
显示模型结构摘要信息

参考 PyTorch Lightning 的 ModelSummary 实现。
"""

import logging
from typing import Any, Dict, Optional

from .base import Callback

# 尝试导入torch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class ModelSummary(Callback):
    """
    模型结构摘要回调

    参考 Lightning 的 ModelSummary, 提供：
    - 模型层数统计
    - 参数数量统计
    - 模型大小估算
    - 可配置的显示深度

    Args:
        max_depth: 最大显示深度(-1为全部显示)
        verbose: 是否输出详细信息
    """

    def __init__(self, max_depth: int = -1, verbose: bool = True):
        super().__init__(priority=1)  # 高优先级, 尽早执行

        self.max_depth = max_depth
        self.verbose = verbose

        # 状态管理
        self._state.update(
            {"model_summary": {}, "parameter_count": 0, "layer_count": 0, "model_size_mb": 0}
        )

        self._logger = logging.getLogger(__name__)

    def on_train_start(self, trainer, core_module):
        """训练开始时显示模型摘要"""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, model summary disabled")
            return

        try:
            summary = self._analyze_model(core_module)
            self._display_summary(summary)
            self._state["model_summary"] = summary

        except Exception as e:
            self._logger.error(f"Error generating model summary: {e}")

    def _analyze_model(self, core_module) -> Dict[str, Any]:
        """分析模型结构"""
        summary = {
            "total_params": 0,
            "trainable_params": 0,
            "non_trainable_params": 0,
            "total_layers": 0,
            "model_size_mb": 0,
            "layer_info": [],
        }

        # 查找模型
        model = self._find_model(core_module)
        if model is None:
            self._logger.warning("No PyTorch model found in core_module")
            return summary

        # 统计参数
        param_info = self._count_parameters(model)
        summary.update(param_info)

        # 统计层数和结构
        layer_info = self._analyze_layers(model)
        summary.update(layer_info)

        # 计算模型大小
        summary["model_size_mb"] = summary["total_params"] * 4 / (1024**2)  # 假设float32

        return summary

    def _find_model(self, core_module) -> Optional[nn.Module]:
        """在core_module中查找PyTorch模型"""
        # 尝试常见的模型属性名
        model_names = ["model", "net", "network", "module"]

        for name in model_names:
            if hasattr(core_module, name):
                model = getattr(core_module, name)
                if isinstance(model, nn.Module):
                    return model

        # 尝试直接检查core_module本身是否是模型
        if isinstance(core_module, nn.Module):
            return core_module

        # 遍历所有属性寻找nn.Module
        for attr_name in dir(core_module):
            if not attr_name.startswith("_"):
                attr = getattr(core_module, attr_name)
                if isinstance(attr, nn.Module):
                    return attr

        return None

    def _count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """统计模型参数"""
        total_params = 0
        trainable_params = 0

        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params,
        }

    def _analyze_layers(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型层结构"""
        layer_info = []
        total_layers = 0

        def _analyze_recursive(module: nn.Module, prefix: str = "", depth: int = 0):
            nonlocal total_layers

            if self.max_depth >= 0 and depth > self.max_depth:
                return

            for name, child in module.named_children():
                layer_name = f"{prefix}.{name}" if prefix else name
                layer_type = type(child).__name__

                # 获取层的参数数量
                layer_params = sum(p.numel() for p in child.parameters())

                # 获取输出形状(如果可能)
                output_shape = "Unknown"
                if hasattr(child, "output_shape"):
                    output_shape = str(child.output_shape)
                elif hasattr(child, "_modules") and not child._modules:
                    # 这是一个叶子层, 可能能推断形状
                    try:
                        # 这里可以添加形状推断逻辑
                        output_shape = "Inferred"
                    except Exception:
                        pass

                layer_info.append(
                    {
                        "name": layer_name,
                        "type": layer_type,
                        "params": layer_params,
                        "output_shape": output_shape,
                        "depth": depth,
                    }
                )

                total_layers += 1
                _analyze_recursive(child, layer_name, depth + 1)

        _analyze_recursive(model)

        return {"total_layers": total_layers, "layer_info": layer_info}

    def _display_summary(self, summary: Dict[str, Any]):
        """显示模型摘要"""
        if not self.verbose:
            return

        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)

        # 基本信息
        print(f"Total parameters: {summary['total_params']:,}")
        print(f"Trainable parameters: {summary['trainable_params']:,}")
        print(f"Non-trainable parameters: {summary['non_trainable_params']:,}")
        print(f"Total layers: {summary['total_layers']}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")

        # 参数分布
        if summary["trainable_params"] > 0:
            trainable_ratio = (summary["trainable_params"] / summary["total_params"]) * 100
            print(f"Trainable parameter ratio: {trainable_ratio:.1f}%")

        # 层结构(如果深度允许)
        if self.max_depth != 0:
            print("\n" + "-" * 80)
            print("LAYER STRUCTURE")
            print("-" * 80)
            print(f"{'Layer Name':<40} {'Type':<20} {'Params':<15} {'Output Shape':<15}")
            print("-" * 80)

            for layer in summary["layer_info"]:
                indent = "  " * layer["depth"]
                name = (indent + layer["name"])[:40]
                layer_type = layer["type"][:20]
                params = f"{layer['params']:,}"[:15]
                output_shape = layer["output_shape"][:15]

                print(f"{name:<40} {layer_type:<20} {params:<15} {output_shape:<15}")

        print("=" * 80)

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量统计"""
        return {
            "total": self._state.get("parameter_count", 0),
            "trainable": 0,  # 可以从模型摘要中获取
            "non_trainable": 0,
        }

    def get_model_size_mb(self) -> float:
        """获取模型大小(MB)"""
        return self._state.get("model_size_mb", 0)

    def get_layer_count(self) -> int:
        """获取层数"""
        return self._state.get("layer_count", 0)
