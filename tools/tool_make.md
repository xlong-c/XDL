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
- [ ] **失败记录**: 将失败的文件记录到日志，便于排查
- [ ] **截断图片**: 处理 `image file is truncated` 错误
- [ ] **进度显示**: 打印处理状态和统计
- [ ] **自动创建目录**: `mkdir(parents=True, exist_ok=True)`
- [ ] **多进程加速**: 可选的多进程并行处理

## 多进程加速

对于CPU密集型任务（图片处理、哈希计算等），可使用多进程加速。

### 使用 multiprocessing.Pool

```python
from multiprocessing import Pool, cpu_count
from functools import partial

def _process_single_wrapper(args, param):
    """包装函数，适配Pool的map接口"""
    rel, src, output_path = args
    try:
        if KEEP_STRUCTURE:
            dst = output_path / rel
        else:
            dst = output_path / f"{idx:06d}.jpg"
        _process_single(src, dst, param)
        return (src.name, True, None)
    except Exception as e:
        return (src.name, False, str(e))

def process_files(input_path: Path, output_path: Path, param: float, 
                  preview: bool = False, preview_limit: int = 5,
                  workers: int = None):  # workers=None 使用全部CPU核心
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return

    # 预览模式（保持不变）
    if preview:
        ...

    # 处理
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    worker_count = workers if workers else cpu_count()
    print(f"使用 {worker_count} 个进程并行处理...")
    
    # 多进程处理
    args_list = [(rel, src, output_path) for rel, src in files]
    process_func = partial(_process_single_wrapper, param=param)
    
    ok = fail = 0
    with Pool(processes=worker_count) as pool:
        results = pool.map(process_func, args_list)
        
    for name, success, error in results:
        if success:
            print(f"✓ {name}")
            ok += 1
        else:
            print(f"✗ {name}: {error}")
            fail += 1

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
```

### 配置参数更新

```python
def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\input")      # 输入文件夹
    OUTPUT = path_win2wsl(r"F:\output")    # 输出文件夹
    PARAM = 1.0                            # 处理参数
    PREVIEW = True                         # 预览模式
    PREVIEW_LIMIT = 5                      # 预览条数
    WORKERS = None                         # 进程数, None=自动(使用全部CPU核心)
    # ==================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        param=PARAM,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
    )
```

### 注意事项

| 场景 | 建议 |
|------|------|
| I/O密集型（文件复制） | 多线程 `ThreadPoolExecutor` 更合适 |
| CPU密集型（图片处理、计算哈希） | 多进程 `multiprocessing.Pool` |
| 小文件/少量文件 | 单进程即可，多进程反而有开销 |
| 内存敏感 | 限制 `workers` 数量，避免同时加载过多图片 |

### 多进程最佳实践

#### 1. 分批处理（大数据量必做）

当文件数量超过 1 万个时，必须使用分批处理避免内存溢出：

```python
def process_files(input_path: Path, output_path: Path, workers: int = None):
    files = list(find_images(input_path))
    
    # 只收集需要处理的文件
    files_to_process = [(rel, src) for rel, src in files 
                        if needs_processing(src)]
    
    if not files_to_process:
        print("没有需要处理的图片")
        return
    
    worker_count = workers if workers else cpu_count()
    batch_size = 1000
    total_batches = (len(files_to_process) + batch_size - 1) // batch_size
    
    with Pool(processes=worker_count) as pool:
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(files_to_process))
            batch = files_to_process[start:end]
            
            print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个文件)...")
            
            # 使用 imap_unordered 实时获取结果
            results = pool.imap_unordered(worker_func, batch)
            
            processed = 0
            for result in results:
                processed += 1
                name, status, error = result
                if status == "success":
                    print(f"✓ {name} ({processed}/{len(batch)})")
                else:
                    print(f"✗ {name}: {error} [{processed}/{len(batch)}]")
```

**关键点：**
- `pool.imap_unordered()` - 边处理边返回结果，实时显示进度
- `batch_size = 1000` - 每批处理 1000 个文件，控制内存占用
- 先筛选再处理 - 避免把不需要处理的文件传入进程池

#### 2. 实时进度显示

```python
# 不推荐：等全部处理完才显示（无法看到进度）
results = pool.map(worker_func, files)  # 阻塞等待全部完成
for result in results:
    print(result)  # 处理完后一次性显示

# 推荐：实时显示进度
results = pool.imap_unordered(worker_func, files)
for i, result in enumerate(results, 1):
    print(f"{result} ({i}/{total})")  # 实时显示
```

#### 3. 进程数设置

```python
# 使用配置指定的进程数，或自动使用全部CPU核心
worker_count = workers if workers else cpu_count()
```

#### 4. 错误处理

```python
def _process_worker(args):
    rel, src = args
    try:
        dst = output_path / rel
        process_single(src, dst)
        return (src.name, "success", None)
    except Exception as e:
        return (src.name, "error", str(e))  # 捕获异常，不中断其他进程
```

#### 5. 失败文件记录

批量处理时应记录失败的文件，便于后续排查：

```python
failed_files = []
for name, success, error, size_info in results:
    if success:
        print(f"✓ {name} ({size_info})")
        ok += 1
    else:
        print(f"✗ {name}: {error}")
        fail += 1
        failed_files.append((name, error))

print(f"\n完成: 成功 {ok}, 失败 {fail}")

if failed_files:
    log_path = output_path / "_process_failed.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# 处理失败记录\n")
        f.write(f"# 总计: {fail} 个文件失败\n\n")
        for name, error in failed_files:
            f.write(f"{name}: {error}\n")
    print(f"\n失败文件列表已保存到: {log_path}")
```

### 完整多进程模板

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

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


def _process_single(src: Path, dst: Path, param: float):
    """处理单个文件，子进程中执行"""
    # 实现单文件处理逻辑
    dst.parent.mkdir(parents=True, exist_ok=True)
    # ... 处理代码 ...


def _process_worker(args, param, keep_structure):
    """多进程工作函数"""
    idx, (rel, src), output_path = args
    try:
        if keep_structure:
            dst = output_path / rel
        else:
            dst = output_path / f"{idx:06d}.jpg"
        _process_single(src, dst, param)
        return (src.name, True, None)
    except Exception as e:
        return (src.name, False, str(e))


def process_files(input_path: Path, output_path: Path, param: float,
                  keep_structure: bool = False,
                  preview: bool = False, preview_limit: int = 5,
                  workers: int = None):
    files = list(find_images(input_path))
    if not files:
        print("未找到图片")
        return

    # 预览
    if preview:
        print("=" * 80)
        print(f"预览模式 (共 {len(files)} 个文件, 显示前 {preview_limit} 个):")
        print("=" * 80)
        for idx, (rel, src) in enumerate(files[:preview_limit], start=1):
            if keep_structure:
                dst = output_path / rel
            else:
                dst = output_path / f"{idx:06d}.jpg"
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

    # 多进程处理
    worker_count = workers if workers else cpu_count()
    print(f"使用 {worker_count} 个进程处理 {len(files)} 个文件...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    args_list = [(idx, item, output_path) for idx, item in enumerate(files, start=1)]
    worker_func = partial(_process_worker, param=param, keep_structure=keep_structure)
    
    ok = fail = 0
    failed_files = []
    with Pool(processes=worker_count) as pool:
        results = pool.map(worker_func, args_list)
        
    for name, success, error in results:
        if success:
            print(f"✓ {name}")
            ok += 1
        else:
            print(f"✗ {name}: {error}")
            fail += 1
            failed_files.append((name, error))

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    
    if failed_files:
        log_path = output_path / "_process_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# 处理失败记录\n")
            f.write(f"# 总计: {fail} 个文件失败\n\n")
            for name, error in failed_files:
                f.write(f"{name}: {error}\n")
        print(f"\n失败文件列表已保存到: {log_path}")


def main():
    # ============ 配置参数 ============
    INPUT = path_win2wsl(r"F:\input")      # 输入文件夹
    OUTPUT = path_win2wsl(r"F:\output")    # 输出文件夹
    PARAM = 1.0                            # 处理参数
    KEEP_STRUCTURE = False                 # 保持原文件夹结构
    PREVIEW = True                         # 预览模式
    PREVIEW_LIMIT = 5                      # 预览条数
    WORKERS = None                         # 进程数, None=自动
    # ==================================

    process_files(
        input_path=Path(INPUT),
        output_path=Path(OUTPUT),
        param=PARAM,
        keep_structure=KEEP_STRUCTURE,
        preview=PREVIEW,
        preview_limit=PREVIEW_LIMIT,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
```

## 参考脚本

| 脚本 | 功能 |
|------|------|
| `rename_files.py` | 批量重命名/转换 |
| `slice_reorder_image.py` | 图片切割拼接 |
| `dedup_images.py` | 重复图片检测 |
| `compare_folders.py` | 文件夹比对 |
| `delete_files.py` | 删除重复文件 |
