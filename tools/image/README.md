# tools/image 工具集文档

本文档对 `tools/image` 目录下的图片处理工具进行解析说明。

## 目录

1. [rename_jpg.py](#rename_jpgpy) - 图片格式转换与重命名工具
2. [batch_viewer.py](#batch_viewerpy) - 批量图片查看器
3. [corpimage.py](#corpimagepy) - 图片裁切工具
4. [gray_detect.py](#gray_detectpy) - 图片黑白检测工具
5. [image_select.py](#image_selectpy) - 图片行删除处理器
6. [nanobanan.py](#nanobananpy) - AI 图像生成工具
7. [png2jpg.py](#png2jpgpy) - PNG 转 JPG 工具

---

## rename_jpg.py

**功能**: 批量将图片转换为 JPG 格式，并支持自定义命名规则

### 支持的源格式
JPG, JPEG, PNG, WEBP, BMP, TIFF, TIF

### 命名模式
1. **数字序号模式**: 00001.jpg, 00002.jpg, ...
2. **保留原文件名模式**: 原文件名.jpg

### 核心特性
- 自动解决文件名冲突（自动添加 _1, _2 后缀）
- JPG 源文件直接复制/移动（保持原质量，避免重复压缩）
- 支持递归处理子目录
- 支持预览模式(dry_run)
- 可选处理完成后删除源文件
- 可配置输出 JPG 质量

### 使用方法
修改 `main()` 函数中的配置参数：

```python
source_path = r"/path/to/source/"      # 源图片文件夹路径
target_path = r"/path/to/target/"      # 目标文件夹路径
delete_source = False                   # 处理成功后是否删除源文件
recursive = False                       # 是否递归处理子文件夹
dry_run = False                         # 预览模式
number_mode = True                      # 数字序号模式
quality = 99                            # JPG 质量
```

---

## batch_viewer.py

**功能**: 基于 tkinter 的 GUI 批量图片查看器，支持网格布局和选择删除

### 核心功能
- **网格布局显示**: 可配置列数，自动计算行数以填充屏幕
- **图片选择**: 点击选择/取消选择，红色 X 标记已选图片
- **批量删除**: 支持删除选中的图片文件
- **缩放控制**: 通过列数调整实现缩放（列数越少，图片越大）
- **翻页浏览**: 支持上一页/下一页，可跳转到指定索引

### 快捷键
| 按键 | 功能 |
|------|------|
| ← (左箭头) | 上一页 |
| → (右箭头) | 下一页 |
| ↑ (上箭头) | 放大（减少列数） |
| ↓ (下箭头) | 缩小（增加列数） |
| Delete | 删除选中的图片 |
| Escape | 退出程序 |

### 使用方法
修改脚本底部的 `start_dir` 和 `aspect_ratio` 参数：

```python
app = BatchImageViewer(
    root, 
    start_dir="/path/to/images",  # 图片文件夹路径
    aspect_ratio=4/1              # 图片显示比例
)
```

---

## corpimage.py

**功能**: 基于 tkinter 的 GUI 图片裁切工具，支持可视化拖拽和批量处理

### 核心功能
- **可视化裁切**: 固定 3:4 比例的裁切框，居中显示
- **图片操作**: 支持拖拽移动、滚轮缩放
- **批量处理**: 可快速切换上一张/下一张图片
- **沿用设置**: 可选择保留上一张的缩放和位置设置

### 快捷键
| 按键 | 功能 |
|------|------|
| W/A/S/D | 移动图片位置 |
| Q/E | 放大/缩小 |
| 空格 | 保存并切换下一张 |
| ←/→ | 上一张/下一张 |

### 使用方法
1. 在输入框中填写输入文件夹和输出文件夹路径
2. 点击"加载图片"按钮
3. 调整图片位置和缩放，使裁切区域对准目标
4. 点击"保存裁切"或按空格键保存

---

## gray_detect.py

**功能**: 检测图片是否为黑白（灰度）图像，支持批量检测

### 检测原理
- 计算 RGB 三个通道之间的平均差异值
- 当平均差异小于阈值时，判定为灰度图
- 支持单通道灰度图直接识别

### 核心功能
- **单文件检测**: 检测指定图片是否为黑白
- **批量检测**: 检测整个目录中的所有图片
- **结果导出**: 可将检测结果保存为 CSV 文件
- **自动分类**: 可将检测到的黑白图片复制到指定目录

### 使用方法
修改 `main()` 函数中的配置参数：

```python
INPUT_FILE = r"path/to/image.jpg"       # 单文件路径
INPUT_PATH = r"path/to/folder"          # 目录路径
OUTPUT_FILE = "results.csv"             # 结果输出文件
THRESHOLD = 0.15                        # 判断阈值
DETECT_SINGLE_FILE = False              # 单文件/批量模式
COPY_GRAYSCALE_TO = r"path/to/output"   # 黑白图片复制目标目录
```

---

## image_select.py

**功能**: 基于 tkinter 的 GUI 图片行删除处理器

### 核心功能
- **行删除**: 将图片分为 4 行，可选择性删除任意行
- **点击删除**: 点击画布上的行区域即可删除该行
- **按钮删除**: 左侧面板提供按钮删除对应行
- **缩放控制**: 支持缩放查看细节
- **进度追踪**: 显示处理进度，自动跳过已处理文件

### 使用场景
适用于需要将多张图片按行拼接后，删除不需要的行。

### 使用方法
1. 设置图片文件夹和输出文件夹
2. 点击"加载图片"
3. 点击画布或按钮删除不需要的行
4. 点击"保存并下一张"或按空格键保存

---

## nanobanan.py

**功能**: 调用 AI API 生成图像，支持文生图和图生图

### 核心功能
- **文本生成图像**: 根据提示词生成图片
- **图像编辑**: 基于现有图片进行编辑
- **多格式支持**: 支持 URL 下载和 Base64 解码保存

### 配置参数
```python
API_KEY = "your-api-key"                # API 密钥
API_URL = "https://yunwu.ai/v1/chat/completions"  # API 地址
SOURCE_IMAGE_PATH = None                # 源图片路径（图生图模式）
PROMPT = "生成图片的描述..."             # 提示词
MODEL = "gemini-3-pro-image-preview"    # 模型名称
ASPECT_RATIO = "1:1"                    # 图片比例
IMAGE_SIZE = "1k"                       # 图片尺寸
```

### 使用方法
1. 配置 API 密钥和参数
2. 设置提示词 PROMPT
3. 如需图生图，设置 SOURCE_IMAGE_PATH
4. 运行脚本，生成的图片将保存到当前目录

---

## png2jpg.py

**功能**: 将 PNG 文件转换为 JPG 格式，支持递归处理

### 核心功能
- **格式转换**: 将 PNG 转换为 JPG，处理透明通道（白色背景填充）
- **递归处理**: 支持处理子文件夹中的所有 PNG
- **文件名过滤**: 支持按文件名关键字过滤
- **自动删除**: 可选转换成功后删除原 PNG 文件
- **质量配置**: 可设置输出 JPG 质量

### 使用方法
修改 `main()` 函数中的配置参数：

```python
path = r"/path/to/folder"               # 要处理的文件夹路径
filter_str = None                       # 文件名过滤字符串
quality = 99                            # JPG 质量 (1-100)
delete_original = True                  # 转换成功后删除原文件
dry_run = False                         # 预览模式
```

---

## 工具对比表

| 工具 | 类型 | 主要功能 | GUI | 批量处理 |
|------|------|----------|-----|----------|
| rename_jpg.py | 转换工具 | 多格式转JPG + 重命名 | ❌ | ✅ |
| batch_viewer.py | 查看工具 | 网格查看 + 选择删除 | ✅ | ✅ |
| corpimage.py | 编辑工具 | 可视化裁切 | ✅ | ✅ |
| gray_detect.py | 检测工具 | 黑白图片检测 | ❌ | ✅ |
| image_select.py | 编辑工具 | 删除图片行 | ✅ | ✅ |
| nanobanan.py | 生成工具 | AI 图像生成 | ❌ | ❌ |
| png2jpg.py | 转换工具 | PNG 转 JPG | ❌ | ✅ |

## 依赖要求

### 公共依赖
```bash
pip install Pillow tqdm
```

### GUI 工具依赖
```bash
pip install opencv-python numpy
```

### AI 生成工具依赖
```bash
pip install requests
```

---

*文档生成时间: 2025-01-30*
