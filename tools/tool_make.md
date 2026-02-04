# Tools 脚本编写规范

## 快速开始模板

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def path_win2wsl(win_path: str) -> str:
    if not win_path:
        return win_path
    win_path = win_path.strip()
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path


def find_images(input_path: Path):
    for root, _, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.relative_to(input_path), p

def process_files(input_path: Path, output_path: Path, param: float, preview: bool = False, preview_limit: int = 5):
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return

    # 预览
    if preview:
        print("=" * 80)
        print(f"预览模式 (共 {len(files)} 个文件, 显示前 {preview_limit} 个):")
        print("=" * 80)
        for rel, src in files[:preview_limit]:
            dst = output_path / src.name
            print(f"\n[操作: 处理]")
            print(f"  源: {src}")
            print(f"  到: {dst}")
            print(f"  参数: {param}")
        if len(files) > preview_limit:
            print(f"\n... 还有 {len(files) - preview_limit} 个文件 ...")
        print("=" * 80)
        if input("\n确认执行? (y/n): ").lower() != "y":
            print("已取消")
            return

    # 处理
    ok = fail = 0
    output_path.mkdir(parents=True, exist_ok=True)
    for rel, src in files:
        try:
            dst = output_path / src.name
            _process_single(src, dst, param)
            print(f"✓ {src.name}")
            ok += 1
        except Exception as e:
            print(f"✗ {src.name}: {e}")
            fail += 1

    print(f"\n完成: 成功 {ok}, 失败 {fail}")


def _process_single(src: Path, dst: Path, param: float):
    pass  # 实现单文件处理逻辑


def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\input")      # 输入文件夹
    OUTPUT = path_win2wsl(r"F:\output")    # 输出文件夹
    PARAM = 1.0                            # 处理参数
    PREVIEW = True                         # 预览模式
    PREVIEW_LIMIT = 5                      # 预览条数
    # ==================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        param=PARAM,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
    )


if __name__ == "__main__":
    main()
```

## 参数配置规则

1. **紧凑布局**: 所有参数集中在 `main()` 顶部的注释块内
2. **直接指定**: 不使用命令行参数，直接修改代码中的值
3. **注释说明**: 每行参数后添加注释说明用途
4. **大写命名**: 配置参数使用全大写

```python
def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\input")      # 输入文件夹路径
    OUTPUT = path_win2wsl(r"F:\output")    # 输出文件夹路径
    SCALE = 0.5                            # 缩放比例 (0.0-1.0)
    ORDER = [3, 2, 4]                      # 拼接顺序
    KEEP_STRUCTURE = False                 # 保持原文件夹结构
    PREVIEW = True                         # 启用预览模式
    PREVIEW_LIMIT = 5                      # 预览显示条数
    # ==================================
```

## 预览模式要求

预览必须清晰显示完整的路径变化和操作信息：

```
================================================================================
预览模式 (共 100 个文件, 显示前 5 个):
================================================================================

[操作: 图片切割+拼接+缩放]
  源: /mnt/f/input/photo_001.jpg
  到: /mnt/f/output/photo_001.jpg
  参数: scale=0.5, order=[3,2,4]

[操作: 图片切割+拼接+缩放]
  源: /mnt/f/input/photo_002.jpg
  到: /mnt/f/output/photo_002.jpg
  参数: scale=0.5, order=[3,2,4]

... 还有 95 个文件 ...
================================================================================

确认执行? (y/n):
```

### 预览必须包含

| 项目 | 说明 | 示例 |
|------|------|------|
| 操作类型 | 明确说明操作 | `[操作: 图片切割+拼接+缩放]` |
| 源路径 | 完整绝对路径 | `源: /mnt/f/input/photo.jpg` |
| 目标路径 | 完整绝对路径 | `到: /mnt/f/output/photo.jpg` |
| 关键参数 | 影响结果的参数 | `参数: scale=0.5, order=[3,2,4]` |
| 统计信息 | 总数和预览数 | `共 100 个文件, 显示前 5 个` |

## 命名规范

| 类型 | 规则 | 示例 |
|------|------|------|
| 文件 | 小写+下划线 | `rename_images.py` |
| 函数 | snake_case, 动词开头 | `process_file()`, `is_valid()` |
| 配置参数 | 全大写 | `INPUT_PATH`, `SCALE`, `PREVIEW_LIMIT` |
| 局部变量 | 小写 | `src_path`, `dst_path` |

## 核心功能清单

- [ ] **参数配置**: 所有参数放在 `main()` 顶部注释块内
- [ ] **路径转换**: `path_win2wsl()` Windows路径转WSL
- [ ] **预览模式**: 显示完整路径变化和操作类型
- [ ] **错误处理**: try-except包裹, 失败不中断
- [ ] **进度显示**: 打印处理状态和统计
- [ ] **自动创建目录**: `mkdir(parents=True, exist_ok=True)`

## 参考脚本

| 脚本 | 功能 |
|------|------|
| `rename_files.py` | 批量重命名/转换 |
| `slice_reorder_image.py` | 图片切割拼接 |
| `dedup_images.py` | 重复图片检测 |
| `compare_folders.py` | 文件夹比对 |
| `delete_files.py` | 删除重复文件 |
