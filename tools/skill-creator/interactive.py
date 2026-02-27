#!/usr/bin/env python3
"""
Interactive Skill Creator for OpenCode
交互式创建 MCP Skill 工具
"""

import json
import os
from pathlib import Path
from typing import List, Optional

# 导入 create_skill 函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from create_skill import create_skill_structure


def prompt(text: str, default: str = "") -> str:
    """带默认值的输入提示"""
    if default:
        user_input = input(f"{text} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{text}: ").strip()


def prompt_yes_no(text: str, default: bool = True) -> bool:
    """是/否 提示"""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{text} [{default_str}]: ").strip().lower()
    
    if not user_input:
        return default
    return user_input in ["y", "yes", "是"]


def collect_tools() -> List[dict]:
    """交互式收集工具定义"""
    tools = []
    
    print("\n" + "="*60)
    print("现在添加工具定义（至少添加一个工具）")
    print("="*60)
    
    while True:
        print(f"\n工具 #{len(tools) + 1}")
        
        tool_name = prompt("工具名称（英文，snake_case，如 resize_image）")
        if not tool_name:
            if tools:
                break
            else:
                print("❌ 至少需要添加一个工具")
                continue
        
        tool_desc = prompt("工具描述（中文，简要说明功能）")
        
        # 收集参数
        print("\n  添加工具参数（直接回车跳过）:")
        properties = {}
        required = []
        
        while True:
            param_name = prompt("  参数名（如 image_path）")
            if not param_name:
                break
            
            param_type = prompt("  参数类型", default="string")
            param_desc = prompt("  参数描述")
            is_required = prompt_yes_no("  是否必需", default=False)
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            if is_required:
                required.append(param_name)
            
            print(f"  ✓ 添加参数：{param_name}")
        
        tool = {
            "name": tool_name,
            "description": tool_desc,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        tools.append(tool)
        
        if not prompt_yes_no("\n  继续添加下一个工具", default=True):
            break
    
    return tools


def interactive_mode():
    """交互式模式"""
    print("="*60)
    print("  OpenCode Skill Creator - 交互式模式")
    print("="*60)
    print()
    
    # 收集基本信息
    print("1. 基本信息")
    print("-"*60)
    skill_name = prompt("Skill 名称（如 image-processor）")
    if not skill_name:
        print("❌ Skill 名称不能为空")
        return
    
    skill_desc = prompt("Skill 描述（如 图像处理工具集 - 支持缩放、裁剪、格式转换）")
    if not skill_desc:
        print("❌ Skill 描述不能为空")
        return
    
    author = prompt("作者名", default="XDL Team")
    license_type = prompt("许可证类型", default="MIT")
    output_dir = prompt("输出目录", default=".")
    
    # 收集工具定义
    tools = collect_tools()
    
    # 确认配置
    print("\n" + "="*60)
    print("确认配置")
    print("="*60)
    print(f"Skill 名称：{skill_name}")
    print(f"Skill 描述：{skill_desc}")
    print(f"作者：{author}")
    print(f"许可证：{license_type}")
    print(f"输出目录：{output_dir}")
    print(f"工具数量：{len(tools)}")
    
    for i, tool in enumerate(tools, 1):
        print(f"\n  工具 {i}: {tool['name']}")
        print(f"    描述：{tool['description']}")
        if tool['inputSchema']['properties']:
            print(f"    参数：{', '.join(tool['inputSchema']['properties'].keys())}")
    
    print()
    if not prompt_yes_no("确认创建", default=True):
        print("❌ 已取消")
        return
    
    # 创建 Skill
    print("\n" + "="*60)
    print("正在创建 Skill...")
    print("="*60)
    
    create_skill_structure(
        skill_name=skill_name,
        description=skill_desc,
        author=author,
        license_type=license_type,
        output_dir=output_dir,
        tools=tools,
    )
    
    print("\n" + "="*60)
    print("后续步骤")
    print("="*60)
    print(f"""
1. 实现 Python MCP 服务器:
   cd {skill_name.replace('-', '_')}
   编辑 mcp_server.py 和 tools/ 目录

2. 安装依赖:
   pip install -r requirements.txt

3. 测试 Skill:
   - 在 OpenCode 中加载 Skill
   - 调用工具验证功能

4. 完善文档:
   编辑 .opencode/skills/{skill_name}/README.md

参考文档：tools/skill-creator/README.md
""")


def main():
    """主函数"""
    print("\nOpenCode Skill Creator")
    print("="*60)
    print("选择创建模式:")
    print("1. 交互式模式（推荐新手）")
    print("2. 命令行模式（快速创建）")
    print()
    
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        print("\n使用命令行模式，请运行:")
        print("python tools/skill-creator/create_skill.py --name <skill-name> --description <description>")
        print("\n示例:")
        print("python tools/skill-creator/create_skill.py --name image-processor --description '图像处理工具集'")
    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    main()
