# Tools 脚本编写规范

## 注意事项
- 避免出现 f-string without any placeholders。
- 图像保存默认质量分数为 **99** (95质量太低)。
- 路径处理必须兼容 Windows 和 WSL。

## 统一标准模板

此模板集成了 **路径转换、预览确认、多进程加速、错误日志记录** 等核心功能。

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def path_win2wsl(win_path: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not win_path: return win_path
    win_path = win_path.strip().replace('"', '')
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path

def find_images(input_path: Path):
    """递归查找所有图片文件"""
    for root, _, files in os.walk(input_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p.relative_to(input_path), p

def _process_single(src: Path, dst: Path, param: float, quality: int):
    """
    单文件处理核心逻辑 (子进程执行)
    """
    # 示例逻辑：创建目录并模拟处理
    dst.parent.mkdir(parents=True, exist_ok=True)
    # result = some_processing_func(src, param)
    # result.save(dst, quality=quality)
    pass

def _worker_wrapper(args, param, quality, keep_structure, output_path):
    """多进程包装函数"""
    idx, (rel, src) = args
    try:
        if keep_structure:
            dst = output_path / rel
        else:
            # 默认打平目录结构并编号
            dst = output_path / f"{idx:06d}{src.suffix}"
        
        _process_single(src, dst, param, quality)
        return (src.name, True, None)
    except Exception as e:
        return (src.name, False, str(e))

def process_files(input_path: Path, output_path: Path, **config):
    """批量处理主逻辑"""
    files = list(find_images(input_path))
    if not files:
        print("[-] 未找到图片文件")
        return

    # 1. 预览模式
    if config.get('PREVIEW'):
        print("=" * 80)
        print(f"预览模式 (总计: {len(files)} | 显示前 {config.get('PREVIEW_LIMIT')} 个)")
        print("=" * 80)
        for idx, (rel, src) in enumerate(files[:config.get('PREVIEW_LIMIT')], 1):
            dst = output_path / (rel if config.get('KEEP_STRUCTURE') else f"{idx:06d}{src.suffix}")
            print(f"\n[操作: 批量处理]")
            print(f"  源: {src}")
            print(f"  到: {dst}")
            print(f"  参数: {config.get('PARAM')}, Quality: {config.get('QUALITY')}")
        
        print("\n" + "=" * 80)
        if input("确认执行? (y/n): ").lower() != "y":
            print("[!] 已取消")
            return

    # 2. 执行处理 (多进程)
    output_path.mkdir(parents=True, exist_ok=True)
    worker_count = config.get('WORKERS') or cpu_count()
    print(f"[*] 启动 {worker_count} 个进程处理 {len(files)} 个文件...")
    
    args_list = list(enumerate(files, 1))
    worker_func = partial(
        _worker_wrapper, 
        param=config.get('PARAM'), 
        quality=config.get('QUALITY'),
        keep_structure=config.get('KEEP_STRUCTURE'),
        output_path=output_path
    )
    
    ok = fail = 0
    failed_log = []
    
    with Pool(processes=worker_count) as pool:
        # 使用 imap_unordered 实时获取进度
        for i, (name, success, error) in enumerate(pool.imap_unordered(worker_func, args_list), 1):
            if success:
                print(f"[{i}/{len(files)}] ✓ {name}")
                ok += 1
            else:
                print(f"[{i}/{len(files)}] ✗ {name}: {error}")
                fail += 1
                failed_log.append((name, error))

    # 3. 统计与日志
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    if failed_log:
        log_file = output_path / "_process_failed.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for name, err in failed_log: f.write(f"{name}: {err}\n")
        print(f"[!] 失败详情已记录至: {log_file}")

def main():
    # ============ 配置参数 (直接修改此处) ============
    INPUT  = path_win2wsl(r"F:\input")       # 输入路径
    OUTPUT = path_win2wsl(r"F:\output")      # 输出路径
    PARAM  = 1.0                             # 处理参数 (自定义)
    QUALITY = 99                             # 保存质量 (1-100)
    KEEP_STRUCTURE = False                   # 是否保持原文件夹结构
    PREVIEW = True                           # 开启预览确认
    PREVIEW_LIMIT = 5                        # 预览显示条数
    WORKERS = None                           # 进程数 (None 为自动)
    # ===============================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        PARAM=PARAM,
        QUALITY=QUALITY,
        KEEP_STRUCTURE=KEEP_STRUCTURE,
        PREVIEW=PREVIEW,
        PREVIEW_LIMIT=PREVIEW_LIMIT,
        WORKERS=WORKERS
    )

if __name__ == "__main__":
    main()
```

## 参数配置规范

1. **集中管理**: 所有可调参数必须位于 `main()` 函数顶部的注释块内。
2. **大写命名**: 配置参数使用 `UPPER_CASE`。
3. **显式类型**: 路径参数通过 `path_win2wsl()` 处理并转换为 `Path` 对象。
4. **无需命令行**: 脚本以“即改即用”为原则，不使用 `argparse`。

## 预览与日志要求

| 项目 | 要求 | 示例 |
| :--- | :--- | :--- |
| **预览信息** | 必须显示源路径、目标路径及关键参数 | `源: /mnt/f/in/1.jpg -> 到: /mnt/f/out/000001.jpg` |
| **实时进度** | 打印当前序号/总数 | `[12/100] ✓ photo.jpg` |
| **错误容忍** | 单个文件失败不中断全局任务 | `try...except` 捕获异常 |
| **失败日志** | 自动在输出目录生成 `_process_failed.log` | `photo_01.jpg: image file is truncated` |

## 核心 Checklist

- [ ] **路径转换**: 使用 `path_win2wsl` 兼容 Windows 复制过来的路径。
- [ ] **预览模式**: 执行前必须有预览和 `input()` 确认。
- [ ] **自动创建目录**: 使用 `mkdir(parents=True, exist_ok=True)`。
- [ ] **高质量保存**: JPEG 默认 `quality=99`。
- [ ] **多进程**: CPU 密集型任务必须支持 `multiprocessing`。
- [ ] **截断处理**: 针对图片处理，考虑添加 `ImageFile.LOAD_TRUNCATED_IMAGES = True`。
