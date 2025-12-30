import os
import requests
import json
import base64
import re
from io import BytesIO
from PIL import Image

# ==================== 核心配置区 ====================
# 1. API 密钥
API_KEY = "sk-nsZQxag0NYOAdxkS2AddfoHpPrmMPvyjbMagu3AFvHkFh3NW"     
# 第三方镜像地址
API_URL = "https://yunwu.ai/v1/chat/completions"

# 2. 图片路径设置
# 如果要编辑现有图片，请填写路径（例如 "bald.png"）；如果仅生成图片，请设为 None
SOURCE_IMAGE_PATH = None 

# 3. 提示词
PROMPT = "请生成一张图片, 一个人物的正脸照片,她有着一头如下描述的头发:•经典纯黑法式鲍伯：Classic Jet Black French Bob，2k清晰度，纯正黑色发质，长度在下巴至锁骨之间，法式剪裁强调简约利落，发身无过多层次，发尾整齐一刀切，线条流畅，整体造型复古优雅，兼具高级感与日常实用性"

# 4. 模型与参数
MODEL = "gemini-3-pro-image-preview" 
ASPECT_RATIO = "1:1"
IMAGE_SIZE = "1k"
BATCH_SIZE = 1
# ====================================================

def encode_image(image_path):
    """将本地图片编码为 base64 用于上传"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None

def save_base64_image(base64_str, filename):
    """将 Base64 字符串保存为图片文件"""
    try:
        # 去掉可能存在的 data:image/png;base64, 前缀
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # 过滤掉非 base64 字符（如换行符、括号等）
        base64_str = re.sub(r'[^A-Za-z0-9+/=]', '', base64_str)
        
        img_data = base64.b64decode(base64_str)
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"成功从 Base64 保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存 Base64 图片失败: {e}")
        return False

def run_task():
    if not API_KEY:
        print("错误: 请填入有效的 API_KEY。")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    content = [{"type": "text", "text": PROMPT}]

    if SOURCE_IMAGE_PATH and os.path.exists(SOURCE_IMAGE_PATH):
        print(f"--- 模式: 图片编辑 (输入: {SOURCE_IMAGE_PATH}) ---")
        base64_image = encode_image(SOURCE_IMAGE_PATH)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
    else:
        print("--- 模式: 纯文字生成 ---")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "provider": {
            "image_config": {
                "aspect_ratio": ASPECT_RATIO,
                "image_size": IMAGE_SIZE
            }
        }
    }

    print(f"正在发送请求至镜像站 ({API_URL})...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"API 错误 (状态码: {response.status_code})")
            print(f"返回内容: {response.text}")
            return

        result = response.json()
        msg_content = result['choices'][0]['message']['content']
        
        saved = False

        # 1. 尝试匹配 URL
        urls = re.findall(r'https?://[^\s)\]]+', msg_content)
        if urls:
            for i, url in enumerate(urls):
                filename = f"output_image_{i}.png"
                print(f"正在下载图片: {url}")
                img_data = requests.get(url).content
                with open(filename, 'wb') as f:
                    f.write(img_data)
                print(f"成功保存到: {filename}")
                saved = True

        # 2. 尝试匹配 Base64 数据 (支持 Markdown 格式 ![image](data:...))
        base64_matches = re.findall(r'data:image/[^;]+;base64,([^)\s\]]+)', msg_content)
        if not base64_matches:
            # 尝试匹配纯 Base64 字符串（如果模型直接返回一堆字符）
            # 我们找长度超过 1000 的连续 Base64 字符
            base64_matches = re.findall(r'([A-Za-z0-9+/]{1000,}=*)', msg_content)

        if base64_matches:
            for i, b64_data in enumerate(base64_matches):
                filename = f"output_b64_image_{i}.png"
                if save_base64_image(b64_data, filename):
                    saved = True

        if not saved:
            print("未能识别到图片链接或 Base64 数据。内容如下:")
            print(msg_content[:500] + "..." if len(msg_content) > 500 else msg_content)

    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    run_task()