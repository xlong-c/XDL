#!/usr/bin/env python3
"""
自动 Git 提交工具
分析代码变更，生成智能提交描述，执行 commit 和 push
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple


def run_git_command(args: List[str], cwd: str = None) -> Tuple[str, str, int]:
    """运行 Git 命令"""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "命令执行超时", 1
    except Exception as e:
        return "", str(e), 1


def get_git_status(cwd: str = None) -> Dict:
    """获取 Git 状态"""
    stdout, stderr, code = run_git_command(["status", "--porcelain"], cwd)
    
    if code != 0:
        return {"error": stderr, "success": False}
    
    changes = {
        "modified": [],
        "added": [],
        "deleted": [],
        "renamed": [],
        "untracked": []
    }
    
    for line in stdout.split("\n"):
        if not line.strip():
            continue
        
        status = line[:2]
        file_path = line[3:].strip()
        
        # 处理重命名文件
        if " -> " in file_path:
            old_path, new_path = file_path.split(" -> ")
            changes["renamed"].append({"from": old_path.strip(), "to": new_path.strip()})
            continue
        
        if status.startswith("??"):
            changes["untracked"].append(file_path)
        elif status.startswith(" A"):
            changes["added"].append(file_path)
        elif status.startswith("M ") or status.startswith(" M"):
            changes["modified"].append(file_path)
        elif status.startswith(" D"):
            changes["deleted"].append(file_path)
    
    return {"changes": changes, "success": True, "has_changes": bool(stdout.strip())}


def get_diff_stats(file_path: str, cwd: str = None) -> Dict:
    """获取文件变更统计"""
    stdout, _, code = run_git_command(["diff", "--stat", file_path], cwd)
    return {"stats": stdout}


def analyze_changes(changes: Dict) -> Dict:
    """分析变更类型和模式"""
    analysis = {
        "change_types": set(),
        "file_types": set(),
        "directories": set(),
        "summary": []
    }
    
    all_files = []
    
    # 收集所有文件
    for change_type, files in changes.items():
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict):
                    # 重命名文件
                    all_files.append(f["to"])
                else:
                    all_files.append(f)
    
    # 分析文件类型和目录
    for file_path in all_files:
        # 提取文件扩展名
        if "." in file_path:
            ext = file_path.split(".")[-1].lower()
            analysis["file_types"].add(ext)
        
        # 提取目录
        if "/" in file_path:
            dir_path = file_path.rsplit("/", 1)[0]
            analysis["directories"].add(dir_path)
    
    # 识别变更类型
    if changes.get("added"):
        analysis["change_types"].add("新增")
    if changes.get("modified"):
        analysis["change_types"].add("修改")
    if changes.get("deleted"):
        analysis["change_types"].add("删除")
    if changes.get("renamed"):
        analysis["change_types"].add("重命名")
    if changes.get("untracked"):
        analysis["change_types"].add("添加新文件")
    
    return analysis


def generate_commit_message(changes: Dict, analysis: Dict) -> str:
    """生成智能提交描述"""
    
    # 根据文件类型和变更生成描述
    file_types = analysis["file_types"]
    directories = analysis["directories"]
    change_types = analysis["change_types"]
    
    # 基础描述模板
    descriptions = []
    
    # 按目录分组
    if "config" in str(directories).lower() or any(ext in file_types for ext in ["yaml", "yml", "json", "toml", "ini"]):
        descriptions.append("更新配置文件")
    
    if any(ext in file_types for ext in ["py", "js", "ts", "java", "cpp", "c", "go", "rs"]):
        if "修改" in change_types:
            descriptions.append("修改代码实现")
        if "新增" in change_types or "添加新文件" in change_types:
            descriptions.append("添加新代码")
    
    if any(ext in file_types for ext in ["md", "rst", "txt"]):
        descriptions.append("更新文档")
    
    if any(ext in file_types for ext in ["test", "spec"]):
        descriptions.append("更新测试")
    
    if "删除" in change_types:
        descriptions.append("清理删除的文件")
    
    if "重命名" in change_types:
        descriptions.append("重构文件结构")
    
    # 如果没有特定描述，使用通用描述
    if not descriptions:
        total_changes = sum(len(files) for files in changes.values() if isinstance(files, list))
        descriptions.append(f"更新 {total_changes} 个文件")
    
    # 组合主描述
    main_message = "; ".join(descriptions)
    
    # 添加详细变更列表
    details = ["\n"]
    
    if changes.get("added"):
        details.append(f"新增 ({len(changes['added'])} 个文件):")
        for f in changes["added"][:10]:  # 最多显示 10 个
            details.append(f"  + {f}")
        if len(changes["added"]) > 10:
            details.append(f"  ... 还有 {len(changes['added']) - 10} 个文件")
    
    if changes.get("modified"):
        details.append(f"\n修改 ({len(changes['modified'])} 个文件):")
        for f in changes["modified"][:10]:
            details.append(f"  ~ {f}")
        if len(changes["modified"]) > 10:
            details.append(f"  ... 还有 {len(changes['modified']) - 10} 个文件")
    
    if changes.get("deleted"):
        details.append(f"\n删除 ({len(changes['deleted'])} 个文件):")
        for f in changes["deleted"][:10]:
            details.append(f"  - {f}")
    
    if changes.get("untracked"):
        details.append(f"\n新文件 ({len(changes['untracked'])} 个):")
        for f in changes["untracked"][:10]:
            details.append(f"  ? {f}")
    
    if changes.get("renamed"):
        details.append(f"\n重命名 ({len(changes['renamed'])} 个文件):")
        for r in changes["renamed"][:5]:
            details.append(f"  {r['from']} -> {r['to']}")
    
    return main_message + "".join(details)


async def auto_commit_tool(args: dict) -> dict:
    """
    自动 Git 提交工具
    
    功能:
    1. 检查 Git 仓库状态
    2. 分析代码变更
    3. 生成智能提交描述
    4. 执行 git add, commit, push
    """
    cwd = args.get("cwd", ".")
    push = args.get("push", True)
    remote = args.get("remote", "origin")
    branch = args.get("branch", None)  # None 表示当前分支
    
    # 检查是否是 Git 仓库
    stdout, stderr, code = run_git_command(["rev-parse", "--git-dir"], cwd)
    if code != 0:
        return {
            "success": False,
            "error": "当前目录不是 Git 仓库",
            "cwd": str(Path(cwd).resolve())
        }
    
    # 获取 Git 状态
    status = get_git_status(cwd)
    if not status.get("success"):
        return {
            "success": False,
            "error": f"获取 Git 状态失败：{status.get('error', '未知错误')}"
        }
    
    if not status.get("has_changes"):
        return {
            "success": True,
            "message": "没有需要提交的变更",
            "clean": True
        }
    
    changes = status["changes"]
    
    # 分析变更
    analysis = analyze_changes(changes)
    
    # 生成提交描述
    commit_message = generate_commit_message(changes, analysis)
    
    # 执行 git add
    stdout, stderr, code = run_git_command(["add", "-A"], cwd)
    if code != 0:
        return {
            "success": False,
            "error": f"git add 失败：{stderr}",
            "stage": "add"
        }
    
    # 执行 git commit
    # 使用 -m 参数，将多行消息转换为合适的格式
    commit_args = ["commit", "-m", commit_message.split("\n")[0]]
    
    # 添加详细描述
    detail_lines = commit_message.split("\n")[1:]
    for line in detail_lines:
        if line.strip():
            commit_args.extend(["-m", line])
    
    stdout, stderr, code = run_git_command(commit_args, cwd)
    if code != 0:
        # 如果没有实际变更（可能已被忽略）
        if "nothing to commit" in stderr.lower() or "nothing to commit" in stdout.lower():
            return {
                "success": True,
                "message": "没有实际变更需要提交",
                "clean": True
            }
        return {
            "success": False,
            "error": f"git commit 失败：{stderr}",
            "stage": "commit"
        }
    
    result = {
        "success": True,
        "message": "提交成功",
        "commit_message": commit_message,
        "changes": {
            "added": len(changes.get("added", [])),
            "modified": len(changes.get("modified", [])),
            "deleted": len(changes.get("deleted", [])),
            "untracked": len(changes.get("untracked", [])),
            "renamed": len(changes.get("renamed", []))
        },
        "total_files": sum(len(files) for files in changes.values() if isinstance(files, list)),
        "stage": "committed"
    }
    
    # 执行 git push（如果需要）
    if push:
        push_args = ["push"]
        
        # 如果指定了远程和分支
        if remote:
            push_args.append(remote)
            if branch:
                push_args.append(branch)
            else:
                # 使用当前分支
                push_args.append("--set-upstream" if len(changes.get("added", [])) > 0 else "")
        
        stdout, stderr, code = run_git_command(push_args, cwd)
        if code != 0:
            result["push_success"] = False
            result["push_error"] = f"git push 失败：{stderr}"
            result["stage"] = "push_failed"
        else:
            result["push_success"] = True
            result["push_output"] = stdout
            result["stage"] = "pushed"
    
    return result


# 导出工具
__all__ = ["auto_commit_tool"]
