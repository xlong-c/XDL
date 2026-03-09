# tools/image 工具集文档

本文档对 `tools/image` 目录下的图片处理工具进行解析说明。

## 目录

1. [rename_jpg.py](#rename_jpgpy) - 图片格式转换与重命名工具
2. [batch_viewer.py](#batch_viewerpy) - 批量图片查看器
3. [corpimage.py](#corpimagepy) - 图片裁切工具
4. [gray_detect.py](#gray_detectpy) - 图片黑白检测工具
5. [detect_person_yolo.py](#detect_person_yolopy) - YOLO 人物检测工具
6. [dedup_images.py](#dedup_imagespy) - 图片去重工具
7. [resize_jpg_resolution.py](#resize_jpg_resolutionpy) - JPG 分辨率批量调整工具

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

## resize_jpg_resolution.py

**功能**: 批量将 `.jpg/.jpeg` 图片缩放到指定分辨率，并尽量保留原始 JPEG 编码参数

### 核心特性
- **仅处理 JPG/JPEG**: 递归扫描输入目录中的 `.jpg` 与 `.jpeg` 文件
- **指定目标分辨率**: 直接配置 `TARGET_SIZE = (宽, 高)`
- **多种缩放模式**: 支持 `exact`、`fit_pad`、`fit_crop`
- **尽量保持原质量**: 优先继承原图的量化表、subsampling、EXIF、ICC profile
- **安全批处理**: 支持预览确认、多进程处理、失败日志 `_resize_failed.log`
- **WSL/Windows 兼容**: 支持 `F:\\...` 路径自动转换为 WSL 路径

### 缩放模式
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `exact` | 直接拉伸/压缩到指定宽高 | 必须输出固定尺寸时 |
| `fit_pad` | 保持比例缩放，空白区域补背景色 | 不想变形时 |
| `fit_crop` | 保持比例缩放后居中裁切 | 需要铺满目标画布时 |

### 使用方法
修改 `main()` 函数中的配置参数：

```python
INPUT = path_win2wsl(r"F:\\dataset\\select_clein_reblad_data")
OUTPUT = path_win2wsl(r"F:\\dataset\\select_clein_reblad_data_resized")
TARGET_SIZE = (1024, 1024)              # 目标分辨率
QUALITY = 99                            # 原图无量化表时的回退质量
RESIZE_MODE = "exact"                 # exact / fit_pad / fit_crop
BACKGROUND_COLOR = (255, 255, 255)      # fit_pad 模式背景色
KEEP_STRUCTURE = True                   # 是否保持原目录结构
PREVIEW = True                          # 执行前预览确认
PREVIEW_LIMIT = 5                       # 预览显示数量
WORKERS = None                          # 进程数，None 表示自动
```

### 运行方式
```bash
python tools/image/resize_jpg_resolution.py
```

### 说明
- JPEG 在缩放后无法做到真正“原质量无损”，脚本采用的是 **best effort** 策略。
- 如果你希望图片不变形，优先使用 `fit_pad` 或 `fit_crop`，不要用 `exact`。

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

## detect_person_yolo.py

**功能**: 使用 YOLO 批量检测图片中的人物，并导出 CSV 结果

### 核心特性
- **仅检测 person 类别**: 调用 YOLO 并固定 `classes=[0]`
- **批量处理图片**: 递归扫描常见图片格式并并行检测
- **CSV 导出**: 输出 `detection_results.csv`，包含文件路径、人数、box 坐标和置信度
- **调试图片输出**: 可指定 `DEBUG_IMAGE_IDX` 保存带框可视化结果
- **WSL/Windows 兼容**: 支持 `F:\\...` 路径自动转换为 WSL 路径
- **预览与容错**: 支持预览确认、失败不中断、失败日志输出

### 输出结果
| 文件 | 说明 |
|------|------|
| `detection_results.csv` | 每张图片的人物检测结果明细 |
| `_process_failed.log` | 处理失败文件日志 |
| `debug_XXXX_xxx.jpg` | 指定调试序号生成的带框可视化图片 |

### 使用方法
修改 `main()` 函数中的配置参数：

```python
INPUT = path_win2wsl(r"F:\\raw_pics\\cloths\\773")
OUTPUT = path_win2wsl(r"F:\\raw_pics\\cloths")
MODEL_PATH = "downloads/yolo26m.pt"     # YOLO 模型路径
CONF_THRESHOLD = 0.75                    # 置信度阈值
END2END = True                           # 是否启用 End2End 模式
PREVIEW = True                           # 执行前预览确认
PREVIEW_LIMIT = 5                        # 预览显示数量
WORKERS = None                           # 进程数，None 表示自动
DEBUG_IMAGE_IDX = 0                      # 调试图片序号，0 表示关闭
```

### 运行方式
```bash
python tools/image/detect_person_yolo.py
```

### 依赖
```bash
pip install ultralytics Pillow
```

---

## dedup_images.py

**功能**: 检测并删除文件夹中的重复图片，支持精确匹配和相似图片检测

### 核心功能
- **精确匹配**: 使用 MD5/SHA256 哈希检测完全相同的图片
- **相似图片检测**: 使用感知哈希(pHash)检测相似度高的图片
- **分阶段处理**: 先按文件大小分组，再计算哈希，提高效率
- **并行计算**: 使用多线程并行计算哈希值
- **智能删除**: 支持保留第一个或保留最新修改的文件

### 哈希模式
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| 感知哈希 | 检测视觉相似的图片 | 查找相似但不完全相同的图片 |
| MD5/SHA256 | 仅检测完全相同的文件 | 查找完全重复的文件 |

### 使用方法
修改 `main()` 函数中的配置参数：

```python
USE_PERCEPTUAL_HASH = True              # True=感知哈希, False=MD5精确匹配
folder = "/path/to/images"              # 图片文件夹路径
delete_duplicates = True                # 是否执行删除操作
dry_run = False                         # True=仅预览不删除
keep_latest = False                     # True=保留最新文件, False=保留第一个
workers = None                          # 并行进程数, None=使用CPU核心数
```

---

## 工具对比表

| 工具 | 类型 | 主要功能 | GUI | 批量处理 |
|------|------|----------|-----|----------|
| rename_jpg.py | 转换工具 | 多格式转JPG + 重命名 | ❌ | ✅ |
| batch_viewer.py | 查看工具 | 网格查看 + 选择删除 | ✅ | ✅ |
| corpimage.py | 编辑工具 | 可视化裁切 | ✅ | ✅ |
| gray_detect.py | 检测工具 | 黑白图片检测 | ❌ | ✅ |
| detect_person_yolo.py | 检测工具 | YOLO 人物检测 + CSV 导出 | ❌ | ✅ |
| dedup_images.py | 检测工具 | 重复图片检测与删除 | ❌ | ✅ |
| resize_jpg_resolution.py | 调整工具 | JPG 固定分辨率缩放 | ❌ | ✅ |

## 依赖要求

### 公共依赖
```bash
pip install Pillow tqdm
```

### 检测工具依赖
```bash
pip install ultralytics
```

### GUI 工具依赖
```bash
pip install opencv-python numpy
```


---

*文档更新时间: 2026-03-09*
