import os
import csv
from pathlib import Path


def generate_image_mask_csv(base_dir, image_folder, mask_folder):
    """
    生成图像和mask的配对CSV文件

    参数:
        base_dir: 上级路径, CSV文件将保存在此目录下
        image_folder: 图片文件夹路径(相对于base_dir或绝对路径)
        mask_folder: mask文件夹路径(相对于base_dir或绝对路径)
    """
    # 支持的图片格式
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    # 支持的mask格式
    mask_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".txt"}

    # 处理路径
    base_path = Path(base_dir)
    image_path = (
        Path(image_folder)
        if Path(image_folder).is_absolute()
        else base_path / image_folder
    )
    mask_path = (
        Path(mask_folder)
        if Path(mask_folder).is_absolute()
        else base_path / mask_folder
    )

    # 检查文件夹是否存在
    if not image_path.exists():
        raise ValueError(f"图片文件夹不存在: {image_path}")
    if not mask_path.exists():
        raise ValueError(f"Mask文件夹不存在: {mask_path}")

    # 扫描图片文件
    image_files = {}
    for file_path in image_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            name = file_path.stem  # 不带扩展名的文件名
            image_files[name] = file_path

    # 扫描mask文件
    mask_files = {}
    for file_path in mask_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in mask_exts:
            name = file_path.stem  # 不带扩展名的文件名
            mask_files[name] = file_path

    # 找到匹配的文件对
    matched_pairs = []
    for name in image_files:
        if name in mask_files:
            # 使用相对路径记录
            image_rel = str(image_files[name].relative_to(base_path))
            mask_rel = str(mask_files[name].relative_to(base_path))
            matched_pairs.append([image_rel, mask_rel])

    # 生成CSV文件
    csv_path = base_path / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["image", "mask"])
        # 写入数据对
        writer.writerows(matched_pairs)

    print(f"成功生成数据对CSV文件: {csv_path}")
    print(f"共找到 {len(matched_pairs)} 个匹配的数据对")
    print(f"图片文件夹: {image_path} ({len(image_files)} 个文件)")
    print(f"Mask文件夹: {mask_path} ({len(mask_files)} 个文件)")


if __name__ == "__main__":
    # 示例用法
    generate_image_mask_csv(
        base_dir=r"F:\dataset\polyp\TrainDataset", image_folder="images", mask_folder="masks"
    )
