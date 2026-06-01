#!/usr/bin/env python3
"""OpenCode Skill 到 Claude Code MCP 配置转换器。

将 .opencode/skills/ 下的 OpenCode skill 配置转换为 Claude Code 的 mcp.json 格式。
直接修改 `CONFIG` 后运行，不使用命令行参数解析库。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG = {
    "opencode_skills": ".opencode/skills",
    "project_root": ".",
    "output": ".claude/mcp.json",
}


def convert_opencode_skill(
    skill_dir: Path,
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    转换单个 OpenCode skill 配置

    Args:
        skill_dir: OpenCode skill 目录
        project_root: 项目根目录

    Returns:
        Claude Code MCP 服务器配置
    """
    opencode_json = skill_dir / "opencode.json"
    if not opencode_json.exists():
        return None

    with open(opencode_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    skill_info = config.get("skill", {})
    mcp_config = config.get("mcp", {})

    if not mcp_config:
        return None

    skill_name = skill_info.get("name", skill_dir.name)

    # 转换 cwd 为绝对路径
    cwd = mcp_config.get("cwd", ".")
    if cwd.startswith("./"):
        cwd = cwd[2:]
    abs_cwd = str(project_root / cwd)

    # 转换 env 中的 PYTHONPATH
    env = mcp_config.get("env", {})
    if "PYTHONPATH" in env:
        pp = env["PYTHONPATH"]
        if pp.startswith("./"):
            pp = pp[2:]
        env["PYTHONPATH"] = str(project_root / pp)

    return {
        "command": mcp_config.get("command"),
        "args": mcp_config.get("args", []),
        "cwd": abs_cwd,
        "env": env,
    }


def convert_all(
    opencode_skills_dir: Path,
    project_root: Path,
    output_file: Path,
) -> None:
    """
    转换所有 OpenCode skills

    Args:
        opencode_skills_dir: .opencode/skills/ 目录
        project_root: 项目根目录
        output_file: 输出的 mcp.json 文件路径
    """
    mcp_servers = {}

    if not opencode_skills_dir.exists():
        print(f"警告: OpenCode skills 目录不存在: {opencode_skills_dir}")
        return

    for skill_dir in opencode_skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue

        mcp_config = convert_opencode_skill(skill_dir, project_root)
        if mcp_config:
            skill_name = skill_dir.name
            mcp_servers[skill_name] = mcp_config
            print(f"✓ 转换 skill: {skill_name}")

    result = {"mcpServers": mcp_servers}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 转换完成! 输出文件: {output_file}")
    print(f"共转换 {len(mcp_servers)} 个 MCP 服务器")


def main(config: Dict[str, str] = CONFIG) -> None:
    project_root = Path(config["project_root"]).resolve()
    opencode_skills_dir = project_root / config["opencode_skills"]
    output_file = project_root / config["output"]

    convert_all(opencode_skills_dir, project_root, output_file)


if __name__ == "__main__":
    main()
