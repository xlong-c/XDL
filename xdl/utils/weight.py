"""
负责初始化模型权重,包含自定义初始和权重加载
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            ema_name = (
                model_name.replace("module.", "")
                if model_name.startswith("module.")
                else f"module.{model_name}"
            )
            if ema_name in ema_params:
                ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                raise KeyError(f"Parameter name {model_name} not found in EMA model!")


try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    """
    使用正态分布初始化模块的权重.

    Args:
        module (nn.Module): 需要初始化的模块.
        mean (float): 正态分布的均值.
        std (float): 正态分布的标准差.
        bias (float): 偏置项的初始化值.
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    """
    使用常数初始化模块的权重.

    Args:
        module (nn.Module): 需要初始化的模块.
        val (float): 权重的初始化常数值.
        bias (float): 偏置项的初始化值.
    """
    nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """
    使用Xavier初始化方法初始化模块的权重.

    Args:
        module (nn.Module): 需要初始化的模块.
        gain (float): 缩放因子.
        bias (float): 偏置项的初始化值.
        distribution (str): 'normal' 或 'uniform', 选择不同的Xavier初始化分布.
    """
    assert distribution in ["normal", "uniform"]
    if distribution == "normal":
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_in", nonlinearity="leaky_relu", bias=0, distribution="normal"
):
    """
    使用Kaiming初始化方法初始化模块的权重.

    Args:
        module (nn.Module): 需要初始化的模块.
        a (float): leaky_relu的负斜率.
        mode (str): 'fan_in'或'fan_out'.
        nonlinearity (str): 非线性激活函数的名称.
        bias (float): 偏置项的初始化值.
        distribution (str): 'normal' 或 'uniform'.
    """
    assert distribution in ["normal", "uniform"]
    # 验证mode参数的有效性
    assert mode in ["fan_in", "fan_out"], f"mode参数必须是'fan_in'或'fan_out',当前值为'{mode}'"
    if distribution == "normal":
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)  # type: ignore
    else:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def default_init(model):
    """
    默认权重初始化方法.
    对卷积层使用Kaiming初始化,对线性层使用Xavier初始化.
    """

    def init_fn(m):
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            xavier_init(m, distribution="normal")
        elif hasattr(m, "weight") and hasattr(m.weight, "data"):
            # 对于其他有权重的层,使用正态分布初始化
            if m.weight is not None:
                nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_fn)


def vit_init(model):
    """
    Vision Transformer专用的权重初始化方法.
    """

    def vit_init_fn(m):
        if isinstance(m, nn.Linear):
            # 对于Linear层使用截断正态分布初始化
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm使用常数初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 对于patch embedding的卷积层
            kaiming_init(m, mode="fan_in", nonlinearity="relu")
        elif hasattr(m, "weight") and hasattr(m.weight, "data"):
            # 其他有权重的层
            if m.weight is not None:
                nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(vit_init_fn)


def cnn_init(model):
    """
    CNN专用的权重初始化方法.
    """

    def cnn_init_fn(m):
        if isinstance(m, nn.Conv2d):
            # 卷积层使用Kaiming初始化
            kaiming_init(m, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            # 全连接层使用Xavier初始化
            xavier_init(m, distribution="normal")
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # 批归一化层初始化
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif hasattr(m, "weight") and hasattr(m.weight, "data"):
            # 其他有权重的层
            if m.weight is not None:
                nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(cnn_init_fn)


def weight_init(model, init_fn: str = "default"):
    """
    对模型进行权重初始化.

    Args:
        model (nn.Module): 需要初始化的模型.
        init_fn (str): 初始化函数名称, 支持 'default', 'vit', 'cnn' 等.
    """
    if init_fn == "default":
        default_init(model)
    elif init_fn == "vit":
        vit_init(model)
    elif init_fn == "cnn":
        cnn_init(model)
    else:
        raise ValueError(f"不支持的初始化方法: {init_fn}")


def load_weight(model: nn.Module, ckpt_path: str) -> None:
    """
    加载预训练权重到模型中.

    Args:
        model (nn.Module): 目标模型.
        ckpt_path (str): 权重文件路径.
            - 支持 .pt, .pth 格式 (torch格式)
            - 支持 .safetensors 格式 (safetensors格式)

    Raises:
        FileNotFoundError: 权重文件不存在.
        ValueError: 不支持的文件格式或文件加载失败.
        ImportError: safetensors库未安装但需要加载safetensors文件.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    # 获取文件扩展名
    file_ext = os.path.splitext(ckpt_path)[1].lower()

    try:
        if file_ext in [".pt", ".pth"]:
            # 使用torch加载权重文件
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                # 如果是字典格式, 尝试常见的键名
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "net" in checkpoint:
                    state_dict = checkpoint["net"]
                else:
                    # 如果没有常见键名, 假设整个字典就是state_dict
                    state_dict = checkpoint
            else:
                # 如果不是字典, 直接作为state_dict使用
                state_dict = checkpoint

        elif file_ext == ".safetensors":
            # 使用safetensors加载权重文件
            if load_safetensors is None:
                raise ImportError(
                    "safetensors库未安装, 无法加载.safetensors文件。请安装: pip install safetensors"
                )

            state_dict = load_safetensors(ckpt_path)

        else:
            raise ValueError(
                f"不支持的权重文件格式: {file_ext}. 支持的格式: .pt, .pth, .safetensors"
            )

        # 加载权重到模型中
        model.load_state_dict(state_dict, strict=False)
        print(f"成功加载权重: {ckpt_path}")

    except Exception as e:
        raise ValueError(f"加载权重文件失败: {ckpt_path}. 错误信息: {str(e)}")
