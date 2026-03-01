#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动 Git 提交和推送工具 - 静默模式"""

import subprocess
import sys


def run_git_command(command: list, cwd: str = None, check: bool = False) -> tuple:
    """运行 git 命令，返回 (stdout, stderr, returncode)"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            env={"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}
        )
        if check and result.returncode != 0:
            return None, result.stderr, result.returncode
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return None, str(e), 1


def main(cwd: str = None) -> None:
    """主函数 - 静默执行 git add/commit/push"""
    # 检查是否有改动
    stdout, stderr, code = run_git_command(["git", "status", "--porcelain"], cwd)
    
    if code != 0:
        print(f"Error: git status failed: {stderr}", file=sys.stderr)
        sys.exit(1)
    
    if not stdout.strip():
        # 没有改动，静默退出
        return
    
    # git add -A
    _, stderr, code = run_git_command(["git", "add", "-A"], cwd)
    if code != 0:
        print(f"Error: git add failed: {stderr}", file=sys.stderr)
        sys.exit(1)
    
    # git commit
    _, stderr, code = run_git_command(["git", "commit", "-m", "update"], cwd)
    if code != 0:
        # 可能是空提交，忽略错误
        pass
    
    # git push
    stdout, stderr, code = run_git_command(["git", "symbolic-ref", "--short", "HEAD"], cwd)
    if code != 0:
        print(f"Error: Failed to get current branch: {stderr}", file=sys.stderr)
        sys.exit(1)
    
    current_branch = stdout.strip()
    _, stderr, code = run_git_command(["git", "push", "origin", current_branch], cwd)
    if code != 0:
        print(f"Error: git push failed: {stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
