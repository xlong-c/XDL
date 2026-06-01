"""MP4 随机抽帧工具。

直接修改 `CONFIG` 后运行，不使用命令行参数解析库。
"""

from __future__ import annotations

import cv2
import os
import random
from pathlib import Path
from typing import Dict, List


CONFIG = {
    "input_dir": "",
    "output_dir": "",
    "num_frames": 10,
    "dry_run": True,
}


def find_mp4_files(directory: str) -> List[str]:
    """
    在指定目录下递归搜索所有MP4文件
    
    Args:
        directory: 要搜索的目录路径
        
    Returns:
        所有MP4文件的完整路径列表
    """
    mp4_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def extract_random_frames(video_path: str, output_dir: str, num_frames: int) -> int:
    """
    从MP4视频中随机抽取指定数量的帧并保存为图片
    
    Args:
        video_path: MP4视频文件路径
        output_dir: 输出目录
        num_frames: 要抽取的帧数量
        
    Returns:
        成功保存的图片数量
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return 0
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"视频文件没有帧: {video_path}")
            cap.release()
            return 0
        
        # 如果请求的帧数超过总帧数,则使用总帧数
        actual_num_frames = min(num_frames, total_frames)
        
        # 随机选择帧索引
        frame_indices = random.sample(range(total_frames), actual_num_frames)
        frame_indices.sort()  # 按顺序处理以提高效率
        
        saved_count = 0
        video_name = Path(video_path).stem
        
        for frame_idx in frame_indices:
            # 设置到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 创建输出文件名
                output_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存帧为图片
                success = cv2.imwrite(output_path, frame)
                if success:
                    saved_count += 1
                    print(f"已保存: {output_filename}")
                else:
                    print(f"保存失败: {output_filename}")
            else:
                print(f"读取帧失败: 索引 {frame_idx}")
        
        cap.release()
        return saved_count
        
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {e}")
        return 0


def main(config: Dict[str, object] = CONFIG) -> None:
    input_dir = str(config["input_dir"])
    output_dir = str(config["output_dir"])
    num_frames = int(config["num_frames"])
    dry_run = bool(config["dry_run"])

    if not input_dir or not output_dir:
        raise ValueError("请先配置 CONFIG['input_dir'] 和 CONFIG['output_dir']")

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有MP4文件
    print(f"正在搜索目录: {input_dir}")
    mp4_files = find_mp4_files(input_dir)
    
    if not mp4_files:
        print("未找到任何MP4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个MP4文件:")
    for file in mp4_files:
        print(f"  - {file}")
    
    if dry_run:
        print("\nDry run模式: 只显示操作,不实际执行")
        print(f"将在 {output_dir} 下为每个视频创建文件夹")
        print(f"每个视频将抽取 {num_frames} 帧")
        return
    
    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)
    
    total_saved = 0
    total_processed = 0
    
    for video_path in mp4_files:
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_dir, video_name)
        
        # 为每个视频创建单独的文件夹
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"\n处理视频: {video_name}")
        print(f"输出目录: {video_output_dir}")
        
        # 抽取帧
        saved = extract_random_frames(video_path, video_output_dir, num_frames)
        
        total_processed += 1
        total_saved += saved
        
        print(f"从 {video_name} 中成功保存了 {saved} 张图片")
    
    print("处理完成!")
    print(f"总共处理了 {total_processed} 个视频")
    print(f"总共保存了 {total_saved} 张图片")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
