from torch import nn
import torch


def count_trainable_parameters(model: nn.Module) -> int:
    """
    计算模型可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """
    计算模型总参数数量

    Args:
        model: PyTorch模型

    Returns:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters())


def count_non_trainable_parameters(model: nn.Module) -> int:
    """
    计算模型非可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def format_number(num: int) -> str:
    """
    将数字格式化为人类可读的形式 (K, M, B)

    Args:
        num: 需要格式化的数字

    Returns:
        str: 格式化后的字符串
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_model_parameters(obj: nn.Module):
    """
    打印对象(通常是CoreModel或nn.Module)中包含的所有模型组件的参数统计信息

    Args:
        obj: 包含nn.Module属性的对象
    """
    print("=" * 91)
    print("模型参数统计")
    print("=" * 91)

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    # 中文表头手动对齐 (考虑到中文字符视觉宽度为2)
    # 模块名称(30宽, 4中文字=8宽, 补22空) | 总参数量(15宽, 4中文字=8宽, 补7空) | 可训练(15宽, 3中文字=6宽, 补9空) | 冻结(15宽, 2中文字=4宽, 补11空) | 比例(12宽, 5中文字=10宽, 补2空)
    header = f"{'模块名称' + ' ' * 22} {' ' * 7 + '总参数量'} {' ' * 9 + '可训练'} {' ' * 11 + '冻结'} {' ' * 2 + '可训练比例'}"
    print(header)
    print("-" * 91)

    # 收集要检查的模块
    modules_to_check = []
    
    # 1. 如果对象本身是 nn.Module, 尝试获取它的子模块
    if isinstance(obj, torch.nn.Module):
        # 使用 named_children 获取第一层子模块
        modules_to_check.extend(list(obj.named_children()))
        
    # 2. 检查对象的属性中是否还有其他 nn.Module (比如未注册为子模块的属性)
    if hasattr(obj, "__dict__"):
        for name, module in obj.__dict__.items():
            if isinstance(module, torch.nn.Module):
                # 避免重复添加已在 named_children 中的模块
                if not any(m is module for _, m in modules_to_check):
                    modules_to_check.append((name, module))
    elif isinstance(obj, dict):
        for name, module in obj.items():
            if isinstance(module, torch.nn.Module):
                modules_to_check.append((name, module))

    # 遍历并打印每个模块的参数
    for name, module in modules_to_check:
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = module_params - module_trainable

        if module_params > 0:
            trainable_ratio = module_trainable / \
                module_params * 100 if module_params > 0 else 0
            # 每一行数据使用固定宽度对齐
            print(f"{name:<30} {module_params:>15,} {module_trainable:>15,} {module_frozen:>15,} {trainable_ratio:>11.2f}%")

            total_params += module_params
            trainable_params += module_trainable
            frozen_params += module_frozen

    # 如果没有找到任何子模块, 但对象本身有参数
    if total_params == 0 and isinstance(obj, torch.nn.Module):
        total_params = sum(p.numel() for p in obj.parameters())
        trainable_params = sum(p.numel() for p in obj.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        if total_params > 0:
            total_trainable_ratio = trainable_params / total_params * 100
            print(f"{'self':<30} {total_params:>15,} {trainable_params:>15,} {frozen_params:>15,} {total_trainable_ratio:>11.2f}%")

    # 打印总计
    print("-" * 91)
    total_trainable_ratio = trainable_params / \
        total_params * 100 if total_params > 0 else 0
    # "总计" (2中文字=4宽, 补26空)
    print(f"{'总计' + ' ' * 26} {total_params:>15,} {trainable_params:>15,} {frozen_params:>15,} {total_trainable_ratio:>11.2f}%")
    print("=" * 91)

    # 打印人类可读的参数量
    print("\n参数量概览:")
    print(f"  总参数量: {format_number(total_params)}")
    print(f"  可训练参数量: {format_number(trainable_params)}")
    print(f"  冻结参数量: {format_number(frozen_params)}")
    print(f"  可训练比例: {total_trainable_ratio:.2f}%")
    print()
