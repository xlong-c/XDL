import os
import csv
from PIL import Image
from tqdm import tqdm

def process_images_to_csv():
    """
    Scans input folder for all JPG images and writes relative paths to CSV
    """
    input_folder = r"F:\1203_b_jpg"
    output_csv = r"F:\1203b.csv"
    
    print(f"Processing images in folder: {input_folder}")
    
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    # Find all JPG files in the folder
    jpg_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.append(file)
    
    print(f"Found {len(jpg_files)} JPG files")
    
    # Find all PNG files as well
    png_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.png'):
            png_files.append(file)

    # Combine all image files
    all_image_files = jpg_files + png_files
    print(f"Found {len(jpg_files)} JPG files and {len(png_files)} PNG files")
    print(f"Total images: {len(all_image_files)}")

    # Prepare data for CSV - include all valid images with relative paths
    csv_data = []
    skipped_count = 0

    print(f"\n开始处理 {len(all_image_files)} 个图像文件...")

    for filename in tqdm(all_image_files, desc="处理图片"):
        # Construct relative path
        relative_path = os.path.join(os.path.basename(input_folder), filename)
        csv_data.append([relative_path])

    # Write data to CSV
    print(f"\n写入CSV文件: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['image_path'])
        # Write data
        writer.writerows(csv_data)
    
    print(f"CSV file created: {output_csv}")
    print(f"Total included images: {len(csv_data)}")
    print(f"Total skipped images (errors): {skipped_count}")


def main():
    try:
        process_images_to_csv()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
