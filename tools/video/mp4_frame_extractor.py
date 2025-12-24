import os
import sys
import cv2
import argparse
import random
from pathlib import Path
from typing import List


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


def main():
    parser = argparse.ArgumentParser(description='从MP4视频中随机抽取帧并保存为图片')
    parser.add_argument('input_dir', help='要搜索MP4文件的输入目录')
    parser.add_argument('output_dir', help='输出目录,将在此创建对应视频名的文件夹')
    parser.add_argument('-n', '--num_frames', type=int, default=10, 
                       help='从每个视频中抽取的帧数量 (默认: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要处理的操作,不实际执行')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 查找所有MP4文件
    print(f"正在搜索目录: {args.input_dir}")
    mp4_files = find_mp4_files(args.input_dir)
    
    if not mp4_files:
        print("未找到任何MP4文件")
        sys.exit(0)
    
    print(f"找到 {len(mp4_files)} 个MP4文件:")
    for file in mp4_files:
        print(f"  - {file}")
    
    if args.dry_run:
        print("\nDry run模式: 只显示操作,不实际执行")
        print(f"将在 {args.output_dir} 下为每个视频创建文件夹")
        print(f"每个视频将抽取 {args.num_frames} 帧")
        sys.exit(0)
    
    # 创建输出目录(如果不存在)
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_saved = 0
    total_processed = 0
    
    for video_path in mp4_files:
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(args.output_dir, video_name)
        
        # 为每个视频创建单独的文件夹
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"\n处理视频: {video_name}")
        print(f"输出目录: {video_output_dir}")
        
        # 抽取帧
        saved = extract_random_frames(video_path, video_output_dir, args.num_frames)
        
        total_processed += 1
        total_saved += saved
        
        print(f"从 {video_name} 中成功保存了 {saved} 张图片")
    
    print("处理完成!")
    print(f"总共处理了 {total_processed} 个视频")
    print(f"总共保存了 {total_saved} 张图片")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()