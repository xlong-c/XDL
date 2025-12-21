import json
import urllib.request
import random
import time
import os
import glob

# === 配置区域 ===
COMFYUI_SERVER = "127.0.0.1:8188"
WORKFLOW_FILE = "workflow_api.json"
INPUT_FOLDER = "F:\\1203_crop"  # 从JSON中获取的输入文件夹
OUTPUT_FOLDER = "F:\\1203_g"    # 从JSON中获取的输出文件夹

# === 计数器配置 ===
# 用于控制图片索引的自增
INCREMENT_CONFIG = {
    "220": "start",   # Number Counter节点, 控制图片索引
}

def get_random_seed():
    return random.randint(1, 18446744073709551615)

def randomize_workflow_seeds(workflow_data):
    """自动随机化所有种子"""
    seed_keys = ["seed", "noise_seed", "seed_int", "control_net_seed"]
    for node_data in workflow_data.values():
        if "inputs" in node_data:
            for key in node_data["inputs"]:
                if key in seed_keys and isinstance(node_data["inputs"][key], (int, float)):
                    node_data["inputs"][key] = get_random_seed()
    return workflow_data

def apply_increment(workflow_data, iteration_index):
    """
    根据循环次数, 更新自增节点的值
    iteration_index: 当前是第几次循环 (0, 1, 2...)
    """
    for node_id, param_name in INCREMENT_CONFIG.items():
        if node_id in workflow_data:
            try:
                # 获取原始值
                original_value = workflow_data[node_id]["inputs"][param_name]
                
                # 确保原始值是数字
                if isinstance(original_value, (int, float)):
                    # 新值 = 原始值 + 当前循环次数
                    workflow_data[node_id]["inputs"][param_name] = original_value + iteration_index
                    print(f"  > 计数器更新: 节点[{node_id}]的[{param_name}] -> {original_value + iteration_index}")
                else:
                    print(f"警告: 节点[{node_id}]的参数[{param_name}]不是数字, 无法自增。")
            except KeyError:
                print(f"警告: 节点[{node_id}]中找不到参数[{param_name}]。")
        else:
            print(f"警告: 找不到 ID 为 [{node_id}] 的节点。")
    return workflow_data

def update_image_index(workflow_data, image_index):
    """
    更新LoadImagesFromFolder节点的image_index参数
    根据你的JSON, 节点183是LoadImagesFromFolder
    """
    node_id = "183"
    if node_id in workflow_data:
        try:
            # 在widgets_values中, image_index在第4个位置(索引3)
            workflow_data[node_id]["widgets_values"][3] = image_index
            print(f"  > 图片索引更新: 节点[{node_id}] -> {image_index}")
        except IndexError:
            print(f"警告: 节点[{node_id}]的widgets_values格式不正确")
    return workflow_data

def update_file_paths(workflow_data, input_folder, output_folder):
    """
    更新输入和输出文件夹路径
    """
    # 更新LoadImagesFromFolder节点的输入文件夹(节点183)
    if "183" in workflow_data:
        workflow_data["183"]["widgets_values"][6] = input_folder
        print(f"  > 输入文件夹更新: {input_folder}")
    
    # 更新SaveImageKJ节点的输出文件夹(节点179)
    if "179" in workflow_data:
        workflow_data["179"]["widgets_values"][1] = output_folder
        print(f"  > 输出文件夹更新: {output_folder}")
    
    return workflow_data

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_SERVER}/prompt", data=data)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(f"API 错误: {e}")
        return None

def get_image_files(folder_path):
    """获取文件夹中的所有图片文件"""
    # 支持的图片格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
    
    # 按文件名排序, 确保顺序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图片:")
    for i, img in enumerate(image_files):
        print(f"  [{i}] {os.path.basename(img)}")
    
    return image_files

def main():
    print(f"加载工作流: {WORKFLOW_FILE}")
    try:
        with open(WORKFLOW_FILE, 'r', encoding='utf-8') as f:
            base_workflow = json.load(f)
    except FileNotFoundError:
        print("找不到工作流文件！")
        return
    
    # 检查节点是否存在
    if "220" not in base_workflow:
        print("警告: 工作流中未找到计数器节点(ID:220)")
    
    # 获取所有图片文件
    if not os.path.exists(INPUT_FOLDER):
        print(f"输入文件夹不存在: {INPUT_FOLDER}")
        return
    
    image_files = get_image_files(INPUT_FOLDER)
    if not image_files:
        print("输入文件夹中没有找到图片文件！")
        return
    
    print(f"开始处理 {len(image_files)} 张图片...")
    
    # 创建输出文件夹(如果不存在)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"创建输出文件夹: {OUTPUT_FOLDER}")
    
    for i, image_path in enumerate(image_files):
        print(f"\n=== 处理第 {i+1}/{len(image_files)} 张图片 ===")
        print(f"图片: {os.path.basename(image_path)}")
        
        # 深拷贝工作流
        current_workflow = json.loads(json.dumps(base_workflow))
        
        # 1. 更新文件夹路径
        current_workflow = update_file_paths(current_workflow, INPUT_FOLDER, OUTPUT_FOLDER)
        
        # 2. 更新图片索引
        current_workflow = update_image_index(current_workflow, i)
        
        # 3. 处理自增计数器
        current_workflow = apply_increment(current_workflow, i)
        
        # 4. 处理随机种子
        current_workflow = randomize_workflow_seeds(current_workflow)
        
        # 5. 发送到ComfyUI
        result = queue_prompt(current_workflow)
        
        if result:
            print(f"  ✓ 提交成功 | 图片索引: {i} | ID: {result.get('prompt_id')}")
        else:
            print(f"  ✗ 提交失败 | 图片索引: {i}")
        
        # 等待一段时间, 避免服务器过载
        time.sleep(0.1)
    
    print(f"\n=== 处理完成 ===")
    print(f"已处理 {len(image_files)} 张图片")
    print(f"结果保存到: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()