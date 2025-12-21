import os
import subprocess
import sys

def set_env(name, value):
    """
    使用 setx 永久设置 Windows 用户环境变量
    """
    try:
        # setx 默认设置用户变量
        result = subprocess.run(['setx', name, value], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"设置 {name} 失败: {e.stderr}")
        return False

def main():
    print("="*40)
    print("   Hugging Face Windows 环境配置工具")
    print("="*40)
    
    # 1. 配置下载位置 (HF_HOME)
    # HF 默认使用 HF_HOME 替代旧的 TRANSFORMERS_CACHE
    current_home = os.environ.get("HF_HOME", "未设置")
    print(f"当前 HF_HOME: {current_home}")
    
    default_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    path = input(f"请输入新的缓存路径 (直接回车使用默认 {default_path}): ").strip()
    
    if not path:
        path = default_path
    
    path = os.path.abspath(path)
    
    # 2. 配置镜像站 (HF_ENDPOINT)
    print("-" * 20)
    mirror = "https://hf-mirror.com"
    use_mirror = input(f"是否配置镜像站 {mirror} ? (y/n, 默认 y): ").strip().lower()

    # 3. 配置 Token (HF_TOKEN)
    print("-" * 20)
    print("提示: Token 可以在 https://huggingface.co/settings/tokens 获取")
    token = input("请输入 Hugging Face Token (可选，直接回车跳过): ").strip()
    
    print("-" * 20)
    # 执行设置
    success = True
    if not set_env("HF_HOME", path):
        success = False
        
    if use_mirror != 'n':
        if not set_env("HF_ENDPOINT", mirror):
            success = False
            
    if token:
        if not set_env("HF_TOKEN", token):
            success = False

    if success:
        print("\n[成功] 配置已完成！")
        print(f"1. HF_HOME = {path}")
        if use_mirror != 'n':
            print(f"2. HF_ENDPOINT = {mirror}")
        if token:
            print(f"3. HF_TOKEN = {'*'*len(token[:-4]) + token[-4:] if len(token) > 4 else '********'}")
        print("\n[重要] 请注意：")
        print("1. 你必须【重启】当前的终端、编译器 (如 PyCharm/VSCode) 才能看到变化。")
        print("2. 在代码中可以通过 os.environ['HF_ENDPOINT'] 验证。")
    else:
        print("\n[错误] 部分配置失败，请尝试以管理员权限运行。")

if __name__ == "__main__":
    main()
