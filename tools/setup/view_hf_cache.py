"""
Hugging Face 缓存查看工具

功能：
1. 显示默认缓存位置
2. 列出缓存目录中的所有内容
3. 显示各个缓存目录的大小
"""

import os
from pathlib import Path
from collections import defaultdict


def get_cache_size(path):
    """计算目录大小"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    except (PermissionError, OSError):
        return 0


def format_size(size_bytes):
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def scan_hub_cache(hub_path):
    """扫描 hub 缓存目录"""
    if not hub_path.exists():
        return

    print("\n" + "="*60)
    print("📦 Hub 缓存内容 (模型和数据集)")
    print("="*60)

    # 获取所有子目录
    all_items = []
    try:
        for item in hub_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                all_items.append(item)
    except PermissionError:
        print("\n⚠️  权限不足，无法访问缓存目录")
        return

    if not all_items:
        print("\n  (无)")
        return

    # 分类统计
    models = []
    datasets = []
    others = []

    for item_dir in all_items:
        # 提取名称 (格式: models--org--name 或 datasets--org--name)
        name_parts = item_dir.name.split("--")

        if len(name_parts) >= 3:
            # 重建原始名称
            original_name = "/".join(name_parts[1:])

            if name_parts[0] == "models":
                models.append((original_name, item_dir))
            elif name_parts[0] == "datasets":
                datasets.append((original_name, item_dir))
            else:
                others.append((item_dir.name, item_dir))
        else:
            others.append((item_dir.name, item_dir))

    # 显示模型
    if models:
        print(f"\n🤖 模型 (Models) - 共 {len(models)} 个:")
        for i, (name, path) in enumerate(sorted(models), 1):
            size = get_cache_size(path)
            print(f"  {i}. {name}")
            print(f"     大小: {format_size(size)}")

    # 显示数据集
    if datasets:
        print(f"\n📊 数据集 (Datasets) - 共 {len(datasets)} 个:")
        for i, (name, path) in enumerate(sorted(datasets), 1):
            size = get_cache_size(path)
            print(f"  {i}. {name}")
            print(f"     大小: {format_size(size)}")

    # 显示其他
    if others:
        print(f"\n📁 其他项 - 共 {len(others)} 个:")
        for i, (name, path) in enumerate(sorted(others)[:10], 1):  # 只显示前10个
            size = get_cache_size(path)
            print(f"  {i}. {name}")
            print(f"     大小: {format_size(size)}")
        if len(others) > 10:
            print(f"  ... 还有 {len(others) - 10} 项")


def main():
    print("="*60)
    print("   Hugging Face 缓存查看工具")
    print("="*60)

    # 1. 显示环境变量
    print("\n🔍 环境变量:")
    print(f"  HF_HOME      = {os.environ.get('HF_HOME', '未设置')}")
    print(f"  HF_ENDPOINT  = {os.environ.get('HF_ENDPOINT', '未设置')}")
    print(f"  TRANSFORMERS_CACHE = {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")

    # 2. 确定缓存目录
    if "HF_HOME" in os.environ:
        cache_dir = Path(os.environ["HF_HOME"])
    elif "TRANSFORMERS_CACHE" in os.environ:
        cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    else:
        # 默认缓存位置
        if os.name == 'nt':  # Windows
            cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface"
        else:  # Linux/Mac
            cache_dir = Path.home() / ".cache" / "huggingface"

    print(f"\n📁 默认缓存目录: {cache_dir}")
    print(f"   目录存在: {'✓ 是' if cache_dir.exists() else '✗ 否'}")

    if not cache_dir.exists():
        print("\n💡 提示: 缓存目录不存在，可能是还没有下载过模型或数据集")
        return

    # 3. 扫描各个子目录
    print("\n" + "="*60)
    print("📂 缓存目录结构")
    print("="*60)

    subdirs = {
        'hub': 'Hub 缓存 (模型/数据集)',
        'assets': '资源文件',
        'processors': '处理器',
    }

    total_size = 0
    for subdir_name, description in subdirs.items():
        subdir_path = cache_dir / subdir_name
        if subdir_path.exists():
            size = get_cache_size(subdir_path)
            total_size += size
            print(f"\n📂 {subdir_name}/ - {description}")
            print(f"   大小: {format_size(size)}")
            print(f"   路径: {subdir_path}")

            # 列出该目录下的文件/文件夹数量
            if subdir_name == 'hub':
                # hub 目录单独处理
                scan_hub_cache(subdir_path)
            else:
                items = list(subdir_path.iterdir())
                print(f"   包含: {len(items)} 个项目")

    # 4. 详细扫描 hub 缓存
    hub_path = cache_dir / "hub"
    if hub_path.exists():
        scan_hub_cache(hub_path)

    # 5. 统计总结
    print("\n" + "="*60)
    print("📊 总结")
    print("="*60)
    print(f"缓存位置: {cache_dir}")
    print(f"总大小: {format_size(total_size)}")

    # 列出所有文件类型
    print("\n📝 缓存的文件类型统计:")
    file_types = defaultdict(int)
    for suffix in ["*.bin", "*.safetensors", "*.json", "*.txt", "*.py"]:
        # 这里简化处理，实际可以遍历所有文件
        pass

    print("\n💡 提示:")
    print("  - 可以通过设置环境变量 HF_HOME 来改变缓存位置")
    print("  - 可以使用 huggingface-cli delete-cache 来清理缓存")
    print("  - 模型通常存储在 hub/models--{org}--{name} 目录中")


if __name__ == "__main__":
    main()
