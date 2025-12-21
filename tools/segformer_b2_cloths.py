import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import warnings
import glob
import cv2
from typing import Tuple, List, Optional

# 忽略警告
warnings.filterwarnings("ignore")

# 上半身相关的类别索引
UPPER_BODY_CLASSES = [
    1,   # Hat
    2,   # Hair
    3,   # Sunglasses
    4,   # Upper-clothes
    8,   # Belt
    11,  # Face
    14,  # Left-arm
    15,  # Right-arm
    16,  # Bag
    17   # Scarf
]

# 加载模型, 优化配置提高质量
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").cuda()

print("正在加载服装分割模型...")

def resize_and_pad_image(image: Image.Image, target_size: int = 1024) -> Tuple[Image.Image, Tuple[float, float], Tuple[int, int]]:
    """
    将图像长边缩放到target_size, 短边填充

    Args:
        image: 输入图像
        target_size: 目标尺寸(长边)

    Returns:
        tuple: (处理后图像, 缩放比例, 填充偏移量)
    """
    w, h = image.size
    # 计算缩放比例, 使长边等于target_size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放图像
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 创建目标大小的黑色背景
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))

    # 计算填充偏移量(居中)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2

    # 将缩放后的图像粘贴到中心
    padded_image.paste(resized_image, (offset_x, offset_y))

    return padded_image, (scale, offset_x, offset_y), (new_w, new_h)

def map_mask_to_original(mask: np.ndarray, scale_info: Tuple[float, float],
                        original_size: Tuple[int, int], resized_size: Tuple[int, int]) -> np.ndarray:
    """
    将缩放图像上的mask映射回原始图像坐标

    Args:
        mask: 缩放图像上的mask
        scale_info: (scale, offset_x, offset_y)
        original_size: 原始图像尺寸 (w, h)
        resized_size: 缩放后图像尺寸 (new_w, new_h)

    Returns:
        映射到原图坐标的mask
    """
    scale, offset_x, offset_y = scale_info
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    # 创建原图大小的mask
    original_mask = np.zeros((orig_h, orig_w), dtype=mask.dtype)

    # 找到缩放图像中有效区域的边界
    valid_mask = mask[offset_y:offset_y+resized_h, offset_x:offset_x+resized_w]

    if np.sum(valid_mask) == 0:
        return original_mask

    # 将有效区域缩放回原图尺寸
    resized_mask = cv2.resize(valid_mask.astype(np.uint8),
                             (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)

    # 设置阈值, 转换为布尔mask
    original_mask = resized_mask > 0

    return original_mask

def get_bounding_box_opencv(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    使用OpenCV快速获取边界框

    Args:
        mask: 二值mask

    Returns:
        边界框 (cmin, rmin, cmax, rmax) 或 None
    """
    # 使用OpenCV的findContours寻找边界
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 合并所有轮廓的点
    all_points = np.vstack(contours)

    # 获取边界框
    cmin, rmin, w, h = cv2.boundingRect(all_points)
    cmax = cmin + w
    rmax = rmin + h

    return cmin, rmin, cmax, rmax

def process_batch_images(image_batch_info, output_dir):
    """批量处理图片, 返回处理结果列表"""
    batch_size = len(image_batch_info)
    original_images = []
    processed_images = []
    scale_info_list = []
    resized_size_list = []
    image_info = []

    # 准备批次数据
    for image_path, relative_path, filename in image_batch_info:
        try:
            # 加载原始图像
            original_image = Image.open(image_path)
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            if original_image.size[0] == 0 or original_image.size[1] == 0:
                raise ValueError("图片尺寸无效")

            # 缩放和填充到1024x1024
            processed_image, scale_info, resized_size = resize_and_pad_image(original_image, 1024)

            original_images.append(original_image)
            processed_images.append(processed_image)
            scale_info_list.append(scale_info)
            resized_size_list.append(resized_size)
            image_info.append({
                'path': image_path,
                'relative_path': relative_path,
                'filename': filename,
                'original_size': original_image.size,
                'original_ext': os.path.splitext(image_path)[1]
            })
        except Exception as e:
            print(f"  加载图片 {image_path} 时出错: {e}")
            continue

    if not processed_images:
        return []

    print(f"正在批量处理 {len(processed_images)} 张图片(1024x1024预处理)...")

    # 批量预处理处理后的图像
    batch_inputs = processor(images=processed_images, return_tensors="pt")

    # 将数据移到GPU
    batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}

    # 批量推理
    with torch.no_grad():
        batch_outputs = model(**batch_inputs)
        batch_logits = batch_outputs.logits.cpu()  # 移回CPU用于后续处理

    results = []

    # 处理每张图片的结果
    for i, (original_image, processed_image, info) in enumerate(zip(original_images, processed_images, image_info)):
        try:
            # 获取单张图片的分割结果
            logits = batch_logits[i:i+1]  # 保持batch维度

            # 上采样到1024x1024(处理图像的尺寸)
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]

            # 提取上半身mask
            upper_body_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
            for class_idx in UPPER_BODY_CLASSES:
                upper_body_mask |= (pred_seg == class_idx)
            upper_body_mask_np = upper_body_mask.numpy()

            # 将处理图像的mask映射回原始图像坐标
            scale_info = scale_info_list[i]
            resized_size = resized_size_list[i]
            original_mask = map_mask_to_original(upper_body_mask_np,
                                                  scale_info,
                                                  info['original_size'],
                                                  resized_size)

            # 计算掩码区域占比(基于原图)
            total_pixels = original_mask.size
            mask_pixels = np.sum(original_mask)
            mask_ratio = mask_pixels / total_pixels * 100

            # 检查占比是否低于10%
            if mask_ratio < 10.0:
                results.append({
                    'success': False,
                    'reason': f'上半身区域占比仅为 {mask_ratio:.2f}%, 低于10%阈值',
                    'info': info
                })
                continue

            # 使用OpenCV获取上半身边界框(基于原图坐标)
            bbox = get_bounding_box_opencv(original_mask.astype(np.uint8))
            if bbox is None:
                results.append({
                    'success': False,
                    'reason': '未检测到上半身, 无法裁切',
                    'info': info
                })
                continue

            cmin, rmin, cmax, rmax = bbox

            # 计算10%外延(基于原图坐标)
            bbox_width = cmax - cmin
            bbox_height = rmax - rmin
            width_extension = int(bbox_width * 0.1)
            height_extension = int(bbox_height * 0.1)

            orig_h, orig_w = original_mask.shape
            cmin = max(0, cmin - width_extension)
            cmax = min(orig_w, cmax + width_extension)
            rmin = max(0, rmin - height_extension)
            rmax = min(orig_h, rmax + height_extension)

            # 裁切原图(保持原图质量)
            cropped_image = original_image.crop((cmin, rmin, cmax, rmax))

            # 创建输出目录
            if info['relative_path']:
                target_dir = os.path.join(output_dir, info['relative_path'])
            else:
                target_dir = output_dir
            os.makedirs(target_dir, exist_ok=True)

            # 保存图片
            original_ext = info['original_ext']
            if original_ext.lower() in ['.jpg', '.jpeg']:
                output_path = os.path.join(target_dir, f"{info['filename']}.jpg")
                cropped_image.save(output_path, quality=95, optimize=True, dpi=(300, 300))
            else:
                output_path = os.path.join(target_dir, f"{info['filename']}{original_ext}")
                cropped_image.save(output_path)

            results.append({
                'success': True,
                'output_path': output_path,
                'mask_ratio': mask_ratio,
                'cropped_size': cropped_image.size,
                'original_size': original_image.size,
                'bbox': (cmin, rmin, cmax, rmax),
                'info': info
            })

        except Exception as e:
            results.append({
                'success': False,
                'reason': f'处理时出错: {e}',
                'info': info
            })

    return results


def collect_image_files(input_dir):
    """递归收集所有图片文件, 返回(文件路径, 相对路径)的列表"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    seen_files = set()  # 避免重复计数

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()

            # 检查是否是支持的图片格式
            if any(file_lower.endswith(ext.lower().replace('*', '')) for ext in image_extensions):
                # 使用绝对路径作为唯一标识, 避免重复
                abs_path = os.path.abspath(file_path)
                if abs_path not in seen_files:
                    seen_files.add(abs_path)
                    relative_path = os.path.relpath(file_path, input_dir)
                    relative_dir = os.path.dirname(relative_path)
                    image_files.append((file_path, relative_dir))

    return image_files

def batch_process_images(input_dir, output_dir, batch_size=8):
    """批量处理图片"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 递归收集所有图片文件
    image_files = collect_image_files(input_dir)

    if not image_files:
        print(f"在目录 {input_dir} 及其子目录中未找到任何图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"输出目录: {output_dir}")
    print(f"批量大小: {batch_size}")

    success_count = 0
    skip_count = 0
    error_count = 0

    # 将图片分批处理
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        print(f"\n[批次 {batch_idx + 1}/{total_batches}] 处理 {len(batch_files)} 张图片")

        # 准备批次数据
        batch_info = []
        for image_path, relative_dir in batch_files:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            batch_info.append((image_path, relative_dir, filename))

        # 批量处理
        batch_results = process_batch_images(batch_info, output_dir)

        # 统计结果
        for result in batch_results:
            if result['success']:
                success_count += 1
                info = result['info']
                rel_path = os.path.join(info['relative_path'], info['filename'] + info['original_ext']) if info['relative_path'] else info['filename'] + info['original_ext']
                print(f"  ✓ {rel_path} -> 上半身占比: {result['mask_ratio']:.1f}%")
            else:
                skip_count += 1
                info = result['info']
                rel_path = os.path.join(info['relative_path'], info['filename'] + info['original_ext']) if info['relative_path'] else info['filename'] + info['original_ext']
                print(f"  ✗ {rel_path}: {result['reason']}")

    print(f"\n" + "="*50)
    print("批量处理完成!")
    print(f"总共处理: {len(image_files)} 张图片")
    print(f"成功处理: {success_count} 张")
    print(f"跳过图片: {skip_count} 张")
    print(f"处理出错: {error_count} 张")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    input_dir = r"F:\BaiduNetDiskDownload\001"
    output_dir = r"F:\BaiduNetDiskDownload\001_crop"

    print("=" * 60)
    print("服装分割图像批量裁切工具 (优化版)")
    print("=" * 60)
    print("优化特性:")
    print("- 图像预处理：长边缩放至1024, 短边填充")
    print("- OpenCV边界框检测：更快的轮廓查找")
    print("- 原图裁切：保持原始图像质量")
    print("- 批量处理：提高GPU利用率")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 使用批量处理, 默认批大小为8, 可以根据GPU内存调整
    batch_process_images(input_dir, output_dir, batch_size=32)