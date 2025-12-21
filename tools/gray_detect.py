#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片黑白检测工具
用于检测给定图片是否为黑白(灰度)图像
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path


def is_grayscale_image(image_path, threshold=0.05):
    """
    检测图片是否为黑白(灰度)图像

    Args:
        image_path (str): 图片路径
        threshold (float): 判断阈值, 当RGB通道间的标准差小于此值时认为是灰度图

    Returns:
        bool: True表示是黑白图像, False表示是彩色图像
        float: RGB通道间的平均标准差
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 如果图片本身就是灰度图(单通道)
        if len(img.shape) == 2:
            return True, 0.0

        # 如果是RGBA图片, 去掉alpha通道
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # 转换为RGB格式(OpenCV默认是BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 计算通道间的差异
        diff_rg = np.mean(np.abs(img_rgb[:, :, 0].astype(float) - img_rgb[:, :, 1].astype(float)))
        diff_rb = np.mean(np.abs(img_rgb[:, :, 0].astype(float) - img_rgb[:, :, 2].astype(float)))
        diff_gb = np.mean(np.abs(img_rgb[:, :, 1].astype(float) - img_rgb[:, :, 2].astype(float)))

        # 计算平均差异
        avg_diff = (diff_rg + diff_rb + diff_gb) / 3

        # 如果平均差异小于阈值, 认为是灰度图
        is_grayscale = avg_diff < threshold

        return is_grayscale, avg_diff

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return False, 0.0


def batch_detect_images(input_dir, output_file=None, threshold=0.05, copy_grayscale_to=None):
    """
    批量检测目录中的所有图片

    Args:
        input_dir (str): 输入目录路径
        output_file (str): 输出结果文件路径, 如果为None则只打印结果
        threshold (float): 判断阈值
        copy_grayscale_to (str): 黑白图片复制目标目录, 如果为None则不复制
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 目录 {input_dir} 不存在")
        return

    # 如果需要复制黑白图片, 创建目标目录
    if copy_grayscale_to:
        copy_path = Path(copy_grayscale_to)
        copy_path.mkdir(parents=True, exist_ok=True)
        print(f"黑白图片将复制到: {copy_grayscale_to}")

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))

    if not image_files:
        print(f"在目录 {input_dir} 中没有找到图片文件")
        return

    results = []
    grayscale_count = 0
    color_count = 0

    print(f"开始检测 {len(image_files)} 张图片...")
    print("-" * 60)

    for i, image_file in enumerate(image_files, 1):
        is_grayscale, avg_diff = is_grayscale_image(str(image_file), threshold)

        result = {
            'filename': image_file.name,
            'is_grayscale': is_grayscale,
            'avg_diff': avg_diff
        }
        results.append(result)

        if is_grayscale:
            grayscale_count += 1
            status = "黑白"
            # 如果需要复制黑白图片
            if copy_grayscale_to:
                try:
                    dest_path = Path(copy_grayscale_to) / image_file.name
                    shutil.copy2(image_file, dest_path)
                    print(f"           已复制到: {dest_path}")
                except Exception as e:
                    print(f"           复制失败: {e}")
        else:
            color_count += 1
            status = "彩色"

        print(f"[{i:3d}/{len(image_files)}] {image_file.name:30s} - {status:4s} (差异值: {avg_diff:.6f})")

    print("-" * 60)
    print(f"检测完成: 黑白图片 {grayscale_count} 张, 彩色图片 {color_count} 张")
    print(f"黑白图片比例: {grayscale_count/len(image_files)*100:.1f}%")

    # 保存结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("文件名,是否黑白,平均差异值\n")
            for result in results:
                status = "是" if result['is_grayscale'] else "否"
                f.write(f"{result['filename']},{status},{result['avg_diff']:.6f}\n")
        print(f"结果已保存到: {output_file}")


def main():
    # 配置参数 - 直接在此处修改
    INPUT_FILE = r"F:\protrait (379).jpg"  # 输入单个图片文件路径
    INPUT_PATH = r"F:\1124_nog"  # 输入目录路径
    OUTPUT_FILE = None  # 输出结果文件路径, 设为None则不保存文件
    THRESHOLD = 0.15  # 判断阈值
    DETECT_SINGLE_FILE = False  # 是否检测单个文件(False表示检测整个目录)
    COPY_GRAYSCALE_TO = r"F:\1124_nog_gray"  # 黑白图片复制目标目录, 设为None则不复制

    print("=" * 60)
    print("图片黑白检测工具")
    print("=" * 60)

    # 根据DETECT_SINGLE_FILE选择使用INPUT_FILE还是INPUT_PATH
    input_source = INPUT_FILE if DETECT_SINGLE_FILE else INPUT_PATH
    print(f"输入路径: {input_source}")
    print(f"输出文件: {OUTPUT_FILE if OUTPUT_FILE else '仅控制台输出'}")
    print(f"检测阈值: {THRESHOLD}")
    print(f"检测模式: {'单文件' if DETECT_SINGLE_FILE else '目录批量'}")
    print(f"黑白图片复制到: {COPY_GRAYSCALE_TO if COPY_GRAYSCALE_TO else '不复制'}")
    print("=" * 60)

    if DETECT_SINGLE_FILE:
        # 检测单个文件
        if not os.path.isfile(input_source):
            print(f"错误: 文件 {input_source} 不存在")
            return

        is_grayscale, avg_diff = is_grayscale_image(input_source, THRESHOLD)
        status = "黑白" if is_grayscale else "彩色"

        print(f"文件: {input_source}")
        print(f"检测结果: {status}")
        print(f"RGB通道间平均差异: {avg_diff:.6f}")
        print(f"阈值: {THRESHOLD}")

        # 保存结果
        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(f"文件名,是否黑白,平均差异值\n")
                status_text = "是" if is_grayscale else "否"
                f.write(f"{os.path.basename(input_source)},{status_text},{avg_diff:.6f}\n")
            print(f"结果已保存到: {OUTPUT_FILE}")
    else:
        # 批量检测目录
        batch_detect_images(input_source, OUTPUT_FILE, THRESHOLD, COPY_GRAYSCALE_TO)


if __name__ == '__main__':
    main()