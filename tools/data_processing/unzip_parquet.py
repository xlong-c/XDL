import pandas as pd
import os
from tqdm import tqdm

def extract_parquet_images(parquet_paths, output_dir):
    """
    从 Parquet 文件中提取图像并保存到指定目录。
    支持处理 Hugging Face 格式的图像数据集（通常包含 'bytes' 键）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    for file_path in parquet_paths:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在，跳过: {file_path}")
            continue
        
        print(f"正在处理: {file_path}")
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(file_path)
            
            # 获取列名以确定图像数据所在位置
            columns = df.columns.tolist()
            img_col = 'image' if 'image' in columns else None
            if not img_col:
                # 尝试寻找包含字节数据的列
                for col in columns:
                    if isinstance(df[col].iloc[0], (bytes, dict)):
                        img_col = col
                        break
            
            if not img_col:
                print(f"错误: 在 {file_path} 中未找到图像列。列名为: {columns}")
                continue

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {os.path.basename(file_path)}"):
                img_data = row[img_col]
                
                # 处理常见的字典格式 {'bytes': b'...', 'path': '...'}
                if isinstance(img_data, dict):
                    img_bytes = img_data.get('bytes')
                    original_filename = img_data.get('path') or f"{idx}.jpg"
                else:
                    img_bytes = img_data
                    original_filename = f"{idx}.jpg"
                
                if img_bytes is None:
                    continue

                # 如果有 label，按类别建立子目录
                label = row.get('label', '')
                current_save_dir = output_dir
                if label != '':
                    current_save_dir = os.path.join(output_dir, str(label))
                    os.makedirs(current_save_dir, exist_ok=True)
                
                # 构造唯一文件名：分卷名 + 索引 + 原始名
                file_prefix = os.path.basename(file_path).replace('.parquet', '')
                final_filename = f"{file_prefix}_{idx}_{os.path.basename(original_filename)}"
                full_path = os.path.join(current_save_dir, final_filename)
                
                with open(full_path, 'wb') as f:
                    f.write(img_bytes)
                    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

if __name__ == "__main__":
    # 配置路径
    # 使用原始字符串处理 Windows 路径
    source_base_path = r"F:\dataset\miniimagenet"
    
    # 自动生成 3 个分卷的文件名
    target_files = [
        os.path.join(source_base_path, f"validation-0000{i}-of-00003.parquet") 
        for i in range(3)
    ]
    
    # 输出目录
    out_dir = os.path.join(source_base_path, "validation_extracted")
    
    print("开始提取 miniimagenet 验证集图像...")
    extract_parquet_images(target_files, out_dir)
    print(f"\n提取完成！图像已保存至: {out_dir}")
