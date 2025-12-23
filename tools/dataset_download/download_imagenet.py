"""
ImageNet 数据集下载脚本(从 Hugging Face)
支持完整 ImageNet、Tiny ImageNet 和子集下载
"""

import os
import sys
import argparse


def setup_hf_mirror(mirror_url=None):
    """
    配置 Hugging Face 镜像站点

    Args:
        mirror_url: 镜像 URL,None 则使用默认镜像
    """
    if mirror_url is None:
        # 默认使用国内镜像
        mirror_url = "https://hf-mirror.com"

    os.environ['HF_ENDPOINT'] = mirror_url
    # 禁用 symlink 警告(Windows 不支持)
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    print(f"✓ 已配置 Hugging Face 镜像: {mirror_url}")


def setup_hf_token(token=None):
    """
    配置 Hugging Face 认证 Token

    Args:
        token: HF Token,如果为 None 则尝试从环境变量读取
    """
    from huggingface_hub import login

    if token is None:
        # 尝试从环境变量读取
        token = os.environ.get('HF_TOKEN')
        if token:
            print("✓ 从环境变量读取到 HF Token")
        else:
            print("⚠ 未配置 HF Token")
            print("提示：ImageNet-1K 是 gated dataset,需要认证才能访问")
            print("获取 Token 步骤：")
            print("  1. 访问: https://huggingface.co/settings/tokens")
            print("  2. 创建新 Token(需要 Read 权限)")
            print("  3. 访问: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print("  4. 同意使用条款并申请访问权限")
            print("\n使用方法：")
            print("  python download_imagenet.py --token hf_xxx --dataset full --split validation --convert")
            return False

    try:
        # 使用新的 login API
        login(token=token, add_to_git_credential=False)
        print("✓ HF Token 认证成功")
        return True
    except Exception as e:
        print(f"✗ HF Token 认证失败: {e}")
        return False


def check_dependencies():
    """检查并安装必要的依赖"""
    print("正在检查依赖...")
    try:
        import datasets
        print(f"✓ datasets 库已安装 (版本: {datasets.__version__})")
    except ImportError:
        print("✗ datasets 库未安装")
        print("正在安装 datasets 库...")
        os.system("pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple")
        import datasets
        print(f"✓ datasets 库安装成功 (版本: {datasets.__version__})")


def download_imagenet_full(save_path, split="validation"):
    """
    下载完整 ImageNet-1K 数据集(约 150GB)

    Args:
        save_path: 保存路径
        split: 'validation' 或 'train'
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"下载完整 ImageNet-1K 数据集 ({split} 集)")
    print(f"目标路径: {save_path}")
    print("预计大小: 验证集 ~6.5GB | 训练集 ~140GB")
    print(f"{'='*60}\n")

    dataset = load_dataset("ILSVRC/imagenet-1k", split=split)

    # 保存为文件夹格式(兼容 torchvision.datasets.ImageNet)
    print(f"\n正在保存数据集到 {save_path}...")
    dataset.save_to_disk(save_path)
    print("✓ 数据集保存成功！")

    return dataset


def download_imagenet_wds(save_path):
    """
    下载 ImageNet WebDataset 格式(推荐,timm 维护)

    Args:
        save_path: 保存路径
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print("下载 ImageNet-1K (WebDataset 格式)")
    print(f"目标路径: {save_path}")
    print("预计大小: ~150GB")
    print(f"{'='*60}\n")

    dataset = load_dataset("timm/imagenet-1k-wds")

    print("\n正在保存数据集...")
    dataset.save_to_disk(save_path)
    print("✓ 数据集保存成功！")

    return dataset


def download_tiny_imagenet(save_path):
    """
    下载 Tiny ImageNet(200 类,64x64 图像)

    Args:
        save_path: 保存路径
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print("下载 Tiny ImageNet 数据集")
    print(f"目标路径: {save_path}")
    print("预计大小: ~500MB")
    print(f"{'='*60}\n")

    dataset = load_dataset("zh-plus/tiny-imagenet")

    print("\n正在保存数据集...")
    dataset.save_to_disk(save_path)
    print("✓ 数据集保存成功！")

    return dataset


def download_imagenet_subset(save_path, num_samples=10000):
    """
    下载 ImageNet 子集(用于快速测试)

    Args:
        save_path: 保存路径
        num_samples: 采样数量
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print("下载 ImageNet 子集 (测试用)")
    print(f"目标路径: {save_path}")
    print(f"样本数量: {num_samples}")
    print(f"{'='*60}\n")

    # 加载并采样子集
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
    subset = dataset.select(range(min(num_samples, len(dataset))))

    print("\n正在保存子集...")
    subset.save_to_disk(save_path)
    print("✓ 子集保存成功！")

    return subset


def convert_to_torchvision_format(huggingface_path, output_path):
    """
    将 Hugging Face 格式转换为 torchvision.datasets.ImageNet 格式

    Args:
        huggingface_path: Hugging Face 数据集路径
        output_path: 输出路径(应该是 F:/dataset/imagenet)
    """
    print(f"\n{'='*60}")
    print("转换数据集格式为 torchvision 兼容格式")
    print(f"源路径: {huggingface_path}")
    print(f"目标路径: {output_path}")
    print(f"{'='*60}\n")

    from datasets import load_from_disk

    # 加载 Hugging Face 数据集
    dataset = load_from_disk(huggingface_path)

    # 创建 torchvision 标准目录结构
    if isinstance(dataset, dict):
        # 如果是 DatasetDict(包含 train/validation)
        for split_name, split_data in dataset.items():
            split_path = os.path.join(output_path, str(split_name))
            print(f"\n正在处理 {split_name} 集...")
            _convert_split(split_data, split_path)
    else:
        # 如果是单个 Dataset
        _convert_split(dataset, output_path)

    print("\n✓ 格式转换完成！")
    print("\n转换后的目录结构:")
    print(f"{output_path}/")
    print("├── train/")
    print("│   ├── n01440764/")
    print("│   │   ├── xxx.JPEG")
    print("│   │   └── ...")
    print("└── val/")
    print("    ├── n01440764/")
    print("    │   ├── xxx.JPEG")
    print("    │   └── ...")


def _convert_split(dataset, output_path):
    """转换单个数据集分割"""
    from tqdm import tqdm

    os.makedirs(output_path, exist_ok=True)

    # 按类别组织
    print(f"正在处理 {len(dataset)} 张图片...")

    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        img = item['image']
        label = item['label']

        # 获取类别文件夹名称(如果是数值标签,需要映射)
        # ImageNet 使用 WordNet ID(如 n01440764)
        if isinstance(label, int):
            # 对于没有 WordNet ID 的数据集,使用数字标签
            class_folder = f"class_{label:04d}"
        else:
            class_folder = label

        class_path = os.path.join(output_path, class_folder)
        os.makedirs(class_path, exist_ok=True)

        # 保存图片
        img_filename = f"img_{idx:06d}.JPEG"
        img_path = os.path.join(class_path, img_filename)

        # 如果 img 是 PIL Image,直接保存
        if hasattr(img, 'save'):
            img.save(img_path)
        else:
            # 如果是其他格式(如 numpy array),转换后保存
            from PIL import Image
            Image.fromarray(img).save(img_path)


def main():
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载 ImageNet 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["full", "wds", "tiny", "subset"],
        default="full",
        help="数据集类型: full(完整), wds(WebDataset格式), tiny(Tiny ImageNet), subset(子集)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=r"F:\dataset\imagenet_hf",
        help="HuggingFace 格式保存路径"
    )
    parser.add_argument(
        "--torchvision-path",
        type=str,
        default=r"F:\dataset\imagenet",
        help="torchvision 格式输出路径(可选)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="下载的数据集分割(仅对 full 模式有效)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="子集模式的样本数量"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="下载后转换为 torchvision 格式"
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="HuggingFace 镜像站点 (默认: https://hf-mirror.com)。可用镜像: hf-mirror.com, 无: None"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像,直接从官方源下载"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 访问 Token(ImageNet-1K 需要)。获取方式: https://huggingface.co/settings/tokens"
    )

    args = parser.parse_args()

    # 配置镜像(如果未指定 --no-mirror)
    if not args.no_mirror:
        setup_hf_mirror(args.mirror)
    else:
        print("⚠ 未使用镜像,直接从官方源下载(可能较慢)")

    # 检查依赖
    check_dependencies()

    # 配置 Token(ImageNet-1K 需要)
    if not setup_hf_token(args.token):
        print("\n✗ 认证失败,无法继续下载")
        print("请按照上述提示获取 Token 后重试")
        sys.exit(1)

    # 根据选择下载数据集
    if args.dataset == "full":
        _ = download_imagenet_full(args.save_path, args.split)
    elif args.dataset == "wds":
        _ = download_imagenet_wds(args.save_path)
    elif args.dataset == "tiny":
        _ = download_tiny_imagenet(args.save_path)
    elif args.dataset == "subset":
        _ = download_imagenet_subset(args.save_path, args.num_samples)

    # 可选：转换为 torchvision 格式
    if args.convert:
        convert_to_torchvision_format(args.save_path, args.torchvision_path)

    print(f"\n{'='*60}")
    print("✓ 所有操作完成！")
    print(f"{'='*60}")
    print("\n使用方法:")
    print("\n1. HuggingFace 格式:")
    print("   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{args.save_path}')")

    if args.convert:
        print("\n2. torchvision 格式 (推荐用于 torchao_vit.py):")
        print("   from torchvision.datasets import ImageNet")
        print(f"   dataset = ImageNet(root='{args.torchvision_path}', split='val')")
        print("\n修改 torchao_vit.py 中的路径:")
        print(f"   IMAGE_NET_ROOT = r'{os.path.join(args.torchvision_path, 'val')}'")


if __name__ == "__main__":
    main()
