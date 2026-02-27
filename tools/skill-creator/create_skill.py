#!/usr/bin/env python3
"""
Skill Creator for OpenCode
为 OpenCode 创建新的 MCP Skill 工具
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def create_opencode_config(
    skill_id: str,
    description: str,
    author: str,
    license_type: str,
    tools: List[dict],
) -> dict:
    """创建 opencode.json 配置文件"""
    
    config = {
        "$schema": "https://opencode.ai/config.json",
        "skill": {
            "name": skill_id,
            "version": "1.0.0",
            "description": description,
            "author": author,
            "license": license_type
        },
        "mcp": {
            "name": f"{skill_id.replace('-', '_')}-mcp",
            "type": "stdio",
            "command": "python",
            "args": ["-m", f"{skill_id.replace('-', '_')}.mcp_server"],
            "cwd": f"./{skill_id.replace('-', '_')}",
            "env": {
                "PYTHONPATH": f"./{skill_id.replace('-', '_')}"
            }
        },
        "tools": tools
    }
    
    return config


def create_readme_template(skill_name: str, description: str, tools: List[dict]) -> str:
    """创建 README.md 模板"""
    
    tools_section = ""
    if tools:
        tools_section = "\n\n## 可用工具\n\n"
        for tool in tools:
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "No description")
            tools_section += f"### `{tool_name}`\n\n{tool_desc}\n\n"
            
            if "inputSchema" in tool:
                schema = tool["inputSchema"]
                tools_section += "**输入参数**:\n\n"
                if "properties" in schema:
                    for prop_name, prop_def in schema["properties"].items():
                        prop_type = prop_def.get("type", "any")
                        prop_desc = prop_def.get("description", "")
                        tools_section += f"- `{prop_name}` ({prop_type}): {prop_desc}\n"
                    tools_section += "\n"
    
    readme = f"""# {skill_name} Skill

{description}

## 安装与配置

### 1. 依赖安装

```bash
# 在此处添加依赖安装命令
pip install -r requirements.txt
```

### 2. OpenCode 配置

Skill 已配置在 `.opencode/skills/{skill_name}/opencode.json`。

MCP 服务器通过 stdio 协议与 OpenCode 通信。
{tools_section}
## 使用示例

```json
{{
  "tool": "example_tool",
  "args": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

## 开发指南

### 添加新工具

1. 在 Python 包中实现工具函数
2. 在 `mcp_server.py` 中注册工具
3. 在 `opencode.json` 中添加工具定义

### 工具实现模板

```python
async def your_tool_name(args: dict) -> dict:
    \"\"\"工具实现\"\"\"
    # 处理逻辑
    result = {{
        "success": True,
        "data": "..."
    }}
    return result
```

## 故障排除

### 问题 1: MCP 服务器启动失败

**症状**: OpenCode 无法连接到工具

**解决**:
```bash
# 检查 Python 路径
python -m {skill_name.replace('-', '_')}.mcp_server

# 检查依赖
pip install -r requirements.txt
```

### 问题 2: 工具调用返回错误

**解决**:
- 检查输入参数格式是否匹配 inputSchema
- 查看 MCP 服务器日志
- 验证工具实现中的异常处理

## 许可证

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！
"""
    
    return readme


def create_skill_structure(
    skill_name: str,
    description: str,
    author: str = "XDL Team",
    license_type: str = "MIT",
    output_dir: str = ".",
    mcp_type: str = "stdio",
    tools: Optional[List[dict]] = None,
):
    """
    创建完整的 Skill 结构

    Args:
        skill_name: Skill 名称（使用 kebab-case，如 my-skill）
        description: Skill 描述
        author: 作者名
        license_type: 许可证类型
        output_dir: 输出目录（相对于项目根目录）
        mcp_type: MCP 类型（目前仅支持 stdio）
        tools: 工具定义列表
    """
    skill_id = skill_name.lower().replace(" ", "-")
    
    base_path = Path(output_dir)
    skills_dir = base_path / ".opencode" / "skills" / skill_id
    skills_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 创建 Skill 目录：{skills_dir}")
    
    opencode_config = create_opencode_config(
        skill_id=skill_id,
        description=description,
        author=author,
        license_type=license_type,
        tools=tools or [],
    )
    config_path = skills_dir / "opencode.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(opencode_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 创建配置文件：{config_path}")
    
    readme_content = create_readme_template(
        skill_name=skill_id,
        description=description,
        tools=tools or [],
    )
    readme_path = skills_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✓ 创建文档：{readme_path}")
    
    gitignore_content = """# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Output files
*.log
.output/
.cache/

# Environment variables
.env
"""
    gitignore_path = skills_dir / ".gitignore"
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print(f"✓ 创建 .gitignore: {gitignore_path}")
    
    if mcp_type == "stdio":
        python_hint = f"""
⚠️  注意：您需要在项目中创建 Python MCP 服务器

建议的目录结构:
├── {skill_id.replace('-', '_')}/
│   ├── __init__.py
│   ├── mcp_server.py      # MCP 服务器入口
│   └── tools/
│       ├── __init__.py
│       └── your_tool.py   # 工具实现

然后在 opencode.json 中配置:
{{
  "mcp": {{
    "command": "python",
    "args": ["-m", "{skill_id.replace('-', '_')}.mcp_server"],
    "cwd": "./{skill_id.replace('-', '_')}",
    "env": {{
      "PYTHONPATH": "./{skill_id.replace('-', '_')}"
    }}
  }}
}}
"""
        hint_path = skills_dir / "SETUP.md"
        with open(hint_path, "w", encoding="utf-8") as f:
            f.write(python_hint)
        print(f"✓ 创建设置指南：{hint_path}")
    
    if tools:
        tools_dir = base_path / skill_id.replace("-", "_") / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        
        init_path = tools_dir / "__init__.py"
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(f"# {skill_id} tools\n")
        print(f"✓ 创建工具包：{tools_dir}")
    
    print(f"\n✅ Skill '{skill_id}' 创建成功!")
    print(f"\n下一步:")
    print(f"1. 编辑 {config_path} 添加工具定义")
    print(f"2. 实现 Python MCP 服务器（参考 SETUP.md）")
    print(f"3. 测试 Skill: 在 OpenCode 中加载并调用工具")
    
    return skills_dir


def main():
    parser = argparse.ArgumentParser(
        description="为 OpenCode 创建新的 MCP Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 创建基础 Skill
  python create_skill.py --name my-skill --description "My awesome skill"
  
  # 创建带工具定义的 Skill
  python create_skill.py --name my-skill --description "..." \\
    --tool name=tool1 desc="Tool description" \\
    --tool name=tool2 desc="Another tool"
  
  # 指定输出目录
  python create_skill.py --name my-skill --description "..." --output /path/to/project
        """
    )
    
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Skill 名称（使用 kebab-case，如 my-skill）"
    )
    
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Skill 描述"
    )
    
    parser.add_argument(
        "--author", "-a",
        default="XDL Team",
        help="作者名（默认：XDL Team）"
    )
    
    parser.add_argument(
        "--license", "-l",
        default="MIT",
        help="许可证类型（默认：MIT）"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="输出目录（默认：当前目录）"
    )
    
    parser.add_argument(
        "--tool", "-t",
        action="append",
        dest="tools",
        metavar="NAME=DESCRIPTION",
        help="添加工具定义（格式：name=description），可重复使用"
    )
    
    args = parser.parse_args()
    
    tools = []
    if args.tools:
        for tool_def in args.tools:
            if "=" in tool_def:
                name, desc = tool_def.split("=", 1)
                tools.append({
                    "name": name.strip(),
                    "description": desc.strip(),
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                })
    
    create_skill_structure(
        skill_name=args.name,
        description=args.description,
        author=args.author,
        license_type=args.license,
        output_dir=args.output,
        tools=tools if tools else None,
    )


if __name__ == "__main__":
    main()
