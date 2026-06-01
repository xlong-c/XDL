import os
import shutil
from typing import Dict, Set, Tuple


# ========== 配置区域 ==========
# 请在此处修改文件夹路径
FOLDER_A_PATH = r"F:\dataset\60HAIR\60_fill"  # 原始文件夹：{cls}_{idx1}.png
FOLDER_B_PATH = r"F:\dataset\60HAIR\60edit"  # 处理后文件夹：{cls}_{idx1}_{idx2}.png
UNPROCESSED_OUTPUT_PATH = r"F:\dataset\60HAIR\add_it"  # 未处理文件输出文件夹
DEBUG_MODE = False  # 调试模式：显示文件名示例
# =================================


def parse_filename_a(filename: str) -> Tuple[str, str] | None:
    """
    解析文件夹a中的文件名格式: {cls}_{idx1}_.png

    Args:
        filename: 文件名，如 "cat_0_.png"

    Returns:
        (cls, idx1) 元组，如果格式不匹配则返回 None
    """
    if not filename.endswith('.png'):
        return None

    name = filename[:-4]  # 移除 .png
    parts = name.split('_')

    # 格式应该是: cls_idx1_ (末尾有空字符串)
    if len(parts) >= 3 and parts[-1] == '':
        cls = '_'.join(parts[:-2])  # 处理cls中可能包含下划线的情况
        idx1 = parts[-2]
        return (cls, idx1)
    return None


def parse_filename_b(filename: str) -> Tuple[str, str, str] | None:
    """
    解析文件夹b中的文件名格式: {cls}_{idx1}__{idx2}_.png
    注意：中间是双下划线

    Args:
        filename: 文件名，如 "0_00001__00001_.png"

    Returns:
        (cls, idx1, idx2) 元组，如果格式不匹配则返回 None
    """
    if not filename.endswith('.png'):
        return None

    name = filename[:-4]  # 移除 .png
    parts = name.split('_')

    # 格式应该是: cls_idx1__idx2_ (中间有双下划线，末尾有下划线)
    # split后: ['cls', 'idx1', '', 'idx2', '']
    # 需要: len >= 5, 最后一个为空, 倒数第三个为空(双下划线导致的空字符串)
    if len(parts) >= 5 and parts[-1] == '' and parts[-3] == '':
        cls = '_'.join(parts[:-4])  # 获取cls
        idx1 = parts[-4]  # 倒数第4个是idx1
        idx2 = parts[-2]  # 倒数第2个是idx2
        return (cls, idx1, idx2)
    return None


def scan_folder_a(folder_path: str) -> Dict[Tuple[str, str], str]:
    """
    扫描文件夹a，提取所有 {cls}_{idx1} 组合

    Args:
        folder_path: 文件夹a的路径

    Returns:
        字典，键为(cls, idx1)，值为完整文件名
    """
    files = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 调试信息：显示前10个文件名
    if DEBUG_MODE:
        print(f"\n[调试] 文件夹A前10个文件 (共{len(all_files)}个文件):")
        for i, filename in enumerate(all_files[:10]):
            result = parse_filename_a(filename)
            status = "✓" if result else "✗"
            print(f"  {i+1}. {filename} -> {status}")
            if not result:
                # 显示解析失败的原因
                if not filename.endswith('.png'):
                    print(f"     原因: 不是.png文件")
                else:
                    name = filename[:-4]
                    parts = name.split('_')
                    print(f"     调试: name='{name}', parts={parts}, len={len(parts)}")
            else:
                print(f"     -> cls='{result[0]}', idx1='{result[1]}'")

    for filename in all_files:
        result = parse_filename_a(filename)
        if result:
            files[result] = filename

    return files


def scan_folder_b(folder_path: str) -> Dict[Tuple[str, str], Set[str]]:
    """
    扫描文件夹b，按 {cls}_{idx1} 分组所有的 {idx2}

    Args:
        folder_path: 文件夹b的路径

    Returns:
        字典，键为(cls, idx1)，值为idx2的集合
    """
    files = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 调试信息：显示前10个文件名
    if DEBUG_MODE:
        print(f"\n[调试] 文件夹B前10个文件:")
        for i, filename in enumerate(all_files[:10]):
            result = parse_filename_b(filename)
            status = "✓ 解析成功" if result else "✗ 解析失败"
            print(f"  {i+1}. {filename} -> {status}")
            if result:
                print(f"     结果: cls='{result[0]}', idx1='{result[1]}', idx2='{result[2]}'")

    for filename in all_files:
        result = parse_filename_b(filename)
        if result:
            cls, idx1, idx2 = result
            key = (cls, idx1)

            if key not in files:
                files[key] = set()

            files[key].add(idx2)

    return files


def compare_folders(folder_a_path: str, folder_b_path: str):
    """
    比对两个文件夹的差异

    Args:
        folder_a_path: 文件夹a路径
        folder_b_path: 文件夹b路径
    """
    print("=" * 80)
    print("文件差异比对工具")
    print("=" * 80)
    print(f"文件夹A: {folder_a_path}")
    print(f"文件夹B: {folder_b_path}")
    print("=" * 80)
    print()

    # 扫描两个文件夹
    print("正在扫描文件夹...")
    folder_a_files = scan_folder_a(folder_a_path)
    folder_b_files = scan_folder_b(folder_b_path)

    # 诊断信息
    print(f"\n[诊断] 文件夹A解析到 {len(folder_a_files)} 个文件")
    if len(folder_a_files) > 0:
        print(f"[诊断] 示例: {list(folder_a_files.values())[:3]}")
    else:
        # 尝试读取前3个文件看看为什么解析失败
        all_files = [f for f in os.listdir(folder_a_path) if os.path.isfile(os.path.join(folder_a_path, f))]
        print(f"[诊断] 前3个文件名: {all_files[:3]}")
        if all_files:
            test_file = all_files[0]
            print(f"[诊断] 测试解析 '{test_file}':")
            print(f"  - 结尾是.png: {test_file.endswith('.png')}")
            name = test_file[:-4] if test_file.endswith('.png') else test_file
            print(f"  - 去掉.png后: '{name}'")
            parts = name.split('_')
            print(f"  - split('_')结果: {parts}")
            print(f"  - 长度: {len(parts)}, 最后一个: '{parts[-1]}'")

    print(f"[诊断] 文件夹B解析到 {len(folder_b_files)} 个文件")
    if len(folder_b_files) > 0:
        print(f"[诊断] 示例keys: {list(folder_b_files.keys())[:3]}")
        # 显示文件夹B的实际文件名
        all_files_b = [f for f in os.listdir(folder_b_path) if os.path.isfile(os.path.join(folder_b_path, f))]
        print(f"[诊断] 文件夹B前3个文件名: {all_files_b[:3]}")
        if all_files_b:
            test_file_b = all_files_b[0]
            print(f"[诊断] 测试解析 '{test_file_b}':")
            name_b = test_file_b[:-4] if test_file_b.endswith('.png') else test_file_b
            print(f"  - 去掉.png后: '{name_b}'")
            parts_b = name_b.split('_')
            print(f"  - split('_')结果: {parts_b}")
            print(f"  - 长度: {len(parts_b)}")

    # 分类统计
    processed = []  # 已处理的文件
    unprocessed = []  # 未处理的文件

    for key, filename in folder_a_files.items():
        cls, idx1 = key

        if key in folder_b_files:
            # 已处理
            idx2_count = len(folder_b_files[key])
            processed.append({
                'filename': filename,
                'variants': idx2_count,
                'idx2_list': sorted(folder_b_files[key])
            })
        else:
            # 未处理
            unprocessed.append(filename)

    # 输出结果
    print("\n" + "=" * 80)
    print("比对结果")
    print("=" * 80)
    print()

    # 统计信息
    total = len(folder_a_files)
    processed_count = len(processed)
    unprocessed_count = len(unprocessed)
    percentage = (processed_count / total * 100) if total > 0 else 0


    # 已处理文件列表
    # if processed:
    #     print("-" * 80)
    #     for item in sorted(processed, key=lambda x: x['filename']):
    #         print(f"   {item['filename']} → 生成了 {item['variants']} 个变体: {item['idx2_list']}")
    #     print(f"✅ 已处理的文件 ({len(processed)} 个):")
    #     print()

    # # 未处理文件列表
    # if unprocessed:
    #     print("-" * 80)
    #     for filename in sorted(unprocessed):
    #         print(f"   {filename}")
    #     print(f"❌ 未处理的文件 ({len(unprocessed)} 个):")
    #     print()

    print(f"📊 统计摘要:")
    print(f"   总文件数: {total}")
    print(f"   已处理: {processed_count}")
    print(f"   未处理: {unprocessed_count}")
    print(f"   处理率: {percentage:.2f}%")
    
    print()
    print("=" * 80)

    # 复制未处理的文件到目标文件夹
    if unprocessed_count > 0:
        print(f"\n正在复制未处理的文件到: {UNPROCESSED_OUTPUT_PATH}")

        # 创建目标文件夹（如果不存在）
        os.makedirs(UNPROCESSED_OUTPUT_PATH, exist_ok=True)

        # 复制文件
        success_count = 0
        fail_count = 0

        for filename in sorted(unprocessed):
            src_path = os.path.join(folder_a_path, filename)
            dst_path = os.path.join(UNPROCESSED_OUTPUT_PATH, filename)

            try:
                shutil.copy2(src_path, dst_path)
                success_count += 1

                # 显示进度（每100个文件显示一次）
                if success_count % 100 == 0:
                    print(f"  已复制: {success_count}/{unprocessed_count}")
            except Exception as e:
                fail_count += 1
                print(f"  ❌ 复制失败: {filename} - {e}")

        print(f"\n✅ 复制完成:")
        print(f"   成功: {success_count} 个文件")
        if fail_count > 0:
            print(f"   失败: {fail_count} 个文件")

    print("\n" + "=" * 80)
    print("比对完成!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        compare_folders(FOLDER_A_PATH, FOLDER_B_PATH)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("\n请检查脚本顶部的 FOLDER_A_PATH 和 FOLDER_B_PATH 配置是否正确。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
