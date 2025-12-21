"""
Checkpoint 工具函数模块
提供格式无关的保存、加载、路径生成等工具
"""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch

try:
    from safetensors.torch import save_file as safetensors_save_file, load_file as safetensors_load_file
except ImportError:
    safetensors_save_file = safetensors_load_file = None

def format_value_for_dirname(value: Any) -> str:
    """将数值转换为目录安全格式 (e.g., 0.123 -> 0p123)"""
    if isinstance(value, float):
        s = f"{value:.6f}".rstrip('0').rstrip('.')
        return ("n" if value < 0 else "") + s.replace("-", "").replace(".", "p")
    return str(value)

def generate_checkpoint_dirname(naming_keys: List[str], values_dict: Dict[str, Any]) -> str:
    """生成语义化的目录名"""
    parts = [f"{k}_{format_value_for_dirname(values_dict[k])}" for k in naming_keys if k in values_dict]
    return "_".join(parts) if parts else "checkpoint"

def flatten_state_dict(state_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """嵌套 dict -> 扁平 dict (module.param)"""
    return {f"{m}.{p}": v.detach().cpu() for m, params in state_dict.items() for p, v in params.items()}

def unflatten_state_dict(flat_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """扁平 dict -> 嵌套 dict"""
    res = {}
    for k, v in flat_dict.items():
        m, p = k.split('.', 1) if '.' in k else (k, '')
        res.setdefault(m, {})[p] = v
    return res

def save_checkpoint(save_dir: Path, checkpoint: Dict[str, Any], format: str = "pt") -> None:
    """统一保存入口"""
    save_dir.mkdir(parents=True, exist_ok=True)
    if format in ("st", "safetensors"):
        if not safetensors_save_file:
            raise ImportError("请安装 safetensors 以使用该格式")
        state_dict = checkpoint.pop('state_dict', {})
        safetensors_save_file(flatten_state_dict(state_dict), str(save_dir / "model.safetensors"))
        torch.save(checkpoint, save_dir / "meta.pt")
    else:
        torch.save(checkpoint, save_dir / "checkpoint.pt")

def detect_and_load_checkpoint(ckpt_path: Path, map_location: Union[str, torch.device] = 'cpu') -> Dict[str, Any]:
    """自动识别格式并加载"""
    if not ckpt_path.is_dir():
        raise ValueError(f"需要目录路径: {ckpt_path}")
    
    if (ckpt_path / "checkpoint.pt").exists():
        return torch.load(str(ckpt_path / "checkpoint.pt"), map_location=map_location)
    
    if (ckpt_path / "model.safetensors").exists():
        if not safetensors_load_file:
            raise ImportError("请安装 safetensors 以加载该格式")
        flat = safetensors_load_file(str(ckpt_path / "model.safetensors"), device=str(map_location))
        ckpt = torch.load(str(ckpt_path / "meta.pt"), map_location=map_location) if (ckpt_path / "meta.pt").exists() else {}
        ckpt['state_dict'] = unflatten_state_dict(flat)
        return ckpt
        
    raise FileNotFoundError(f"在 {ckpt_path} 中未找到有效节点文件")