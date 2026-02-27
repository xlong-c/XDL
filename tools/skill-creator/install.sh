#!/bin/bash
# Skill Creator 安装脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "  OpenCode Skill Creator 安装"
echo "========================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python 3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本：$PYTHON_VERSION"

# 创建必要的目录
echo
echo "创建目录结构..."
mkdir -p "$PROJECT_ROOT/.opencode/skills"
mkdir -p "$PROJECT_ROOT/tools/skill-creator"

# 复制文件
echo "复制 Skill Creator 文件..."
cp "$SCRIPT_DIR/create_skill.py" "$PROJECT_ROOT/tools/skill-creator/"
cp "$SCRIPT_DIR/interactive.py" "$PROJECT_ROOT/tools/skill-creator/"
cp "$SCRIPT_DIR/mcp_server_template.py" "$PROJECT_ROOT/tools/skill-creator/"
cp "$SCRIPT_DIR/README.md" "$PROJECT_ROOT/tools/skill-creator/"
cp "$SCRIPT_DIR/QUICKSTART.md" "$PROJECT_ROOT/tools/skill-creator/"

chmod +x "$PROJECT_ROOT/tools/skill-creator/create_skill.py"
chmod +x "$PROJECT_ROOT/tools/skill-creator/interactive.py"
chmod +x "$PROJECT_ROOT/tools/skill-creator/mcp_server_template.py"

echo "✓ 文件已复制到 tools/skill-creator/"

# 创建 requirements.txt 模板
echo
echo "创建 requirements.txt 模板..."
cat > "$PROJECT_ROOT/tools/skill-creator/requirements.template.txt" << 'EOF'
# Skill Creator 依赖模板
# 根据实际需求修改

# 基础依赖
# (通常不需要额外依赖，MCP 服务器使用 Python 标准库)

# 如果需要 HTTP 请求
# requests>=2.28.0
# aiohttp>=3.8.0

# 如果需要数据处理
# pandas>=1.5.0
# numpy>=1.23.0

# 如果需要图像处理
# Pillow>=9.0.0

# 如果需要 YAML 支持
# pyyaml>=6.0
EOF

echo "✓ 创建 requirements.template.txt"

# 创建示例
echo
echo "创建示例 Skill..."

EXAMPLE_SKILL_NAME="example-skill"
EXAMPLE_SKILL_DIR="$PROJECT_ROOT/.opencode/skills/$EXAMPLE_SKILL_NAME"

if [ ! -d "$EXAMPLE_SKILL_DIR" ]; then
    mkdir -p "$EXAMPLE_SKILL_DIR"
    
    # 创建 opencode.json
    cat > "$EXAMPLE_SKILL_DIR/opencode.json" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "skill": {
    "name": "example-skill",
    "version": "1.0.0",
    "description": "示例 Skill - 展示如何创建 OpenCode Skill",
    "author": "XDL Team",
    "license": "MIT"
  },
  "mcp": {
    "name": "example-skill-mcp",
    "type": "stdio",
    "command": "python",
    "args": ["-m", "example_skill.mcp_server"],
    "cwd": "./example_skill",
    "env": {
      "PYTHONPATH": "./example_skill"
    }
  },
  "tools": [
    {
      "name": "echo",
      "description": "回显工具 - 返回输入的消息",
      "inputSchema": {
        "type": "object",
        "properties": {
          "message": {
            "type": "string",
            "description": "要回显的消息"
          },
          "uppercase": {
            "type": "boolean",
            "description": "是否转换为大写",
            "default": false
          }
        },
        "required": ["message"]
      }
    },
    {
      "name": "calculate",
      "description": "计算工具 - 执行简单数学运算",
      "inputSchema": {
        "type": "object",
        "properties": {
          "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
            "description": "运算类型"
          },
          "a": {
            "type": "number",
            "description": "第一个操作数"
          },
          "b": {
            "type": "number",
            "description": "第二个操作数"
          }
        },
        "required": ["operation", "a", "b"]
      }
    }
  ]
}
EOF
    
    echo "✓ 创建示例 Skill 配置"
else
    echo "✓ 示例 Skill 已存在，跳过"
fi

# 创建 Python 包示例
EXAMPLE_PYTHON_DIR="$PROJECT_ROOT/example_skill"
if [ ! -d "$EXAMPLE_PYTHON_DIR" ]; then
    mkdir -p "$EXAMPLE_PYTHON_DIR/tools"
    
    # __init__.py
    echo "# Example Skill Python Package" > "$EXAMPLE_PYTHON_DIR/__init__.py"
    
    # tools/__init__.py
    echo "# Example Skill Tools" > "$EXAMPLE_PYTHON_DIR/tools/__init__.py"
    
    # tools/echo.py
    cat > "$EXAMPLE_PYTHON_DIR/tools/echo.py" << 'EOF'
async def echo_tool(args: dict) -> dict:
    """回显工具"""
    message = args.get("message", "")
    uppercase = args.get("uppercase", False)
    
    if uppercase:
        message = message.upper()
    
    return {
        "success": True,
        "message": message,
        "metadata": {
            "original_length": len(args.get("message", "")),
            "uppercase": uppercase
        }
    }
EOF
    
    # tools/calculate.py
    cat > "$EXAMPLE_PYTHON_DIR/tools/calculate.py" << 'EOF'
async def calculate_tool(args: dict) -> dict:
    """计算工具"""
    operation = args.get("operation", "add")
    a = args.get("a", 0)
    b = args.get("b", 0)
    
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    
    if operation not in operations:
        return {
            "success": False,
            "error": f"未知运算：{operation}"
        }
    
    result = operations[operation](a, b)
    
    return {
        "success": True,
        "result": result,
        "metadata": {
            "operation": operation,
            "operands": [a, b]
        }
    }
EOF
    
    # mcp_server.py
    cat > "$EXAMPLE_PYTHON_DIR/mcp_server.py" << 'EOF'
#!/usr/bin/env python3
"""
MCP Server for Example Skill
"""

import sys
import json
import asyncio
from typing import Any, Dict
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.echo import echo_tool
from tools.calculate import calculate_tool


class ExampleSkillServer:
    """示例 Skill MCP 服务器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tools = {
            "echo": echo_tool,
            "calculate": calculate_tool,
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return await self.initialize(params)
            elif method == "tools/list":
                return await self.list_tools()
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                return await self.call_tool(tool_name, tool_args)
            elif method == "notifications/initialized":
                return {"jsonrpc": "2.0", "result": {}}
            else:
                return self._error_response(request_id, f"Unknown method: {method}")
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """初始化 MCP 连接"""
        return {
            "jsonrpc": "2.0",
            "id": params.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "example-skill-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        tools = [
            {
                "name": "echo",
                "description": "回显工具 - 返回输入的消息",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "要回显的消息"},
                        "uppercase": {"type": "boolean", "description": "是否转换为大写", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "calculate",
                "description": "计算工具 - 执行简单数学运算",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number", "description": "第一个操作数"},
                        "b": {"type": "number", "description": "第二个操作数"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        return {"jsonrpc": "2.0", "id": "list", "result": {"tools": tools}}
    
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self.tools:
            return {"jsonrpc": "2.0", "id": "call", "error": {"message": f"Tool not found: {tool_name}"}}
        
        try:
            tool_func = self.tools[tool_name]
            result = await tool_func(tool_args)
            return {"jsonrpc": "2.0", "id": "call", "result": result}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": "call", "error": {"message": str(e)}}
    
    def _error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {"jsonrpc": "2.0", "id": request_id, "error": {"message": message}}


async def main():
    """主函数"""
    server = ExampleSkillServer(verbose=True)
    
    # 从 stdin 读取
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    # 写入 stdout
    write_transport, write_protocol = await asyncio.get_event_loop().create_subprocess_exec(
        "cat",
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE
    )
    
    writer = asyncio.StreamWriter(write_transport, write_protocol, reader, None)
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            request = json.loads(line.decode())
            response = await server.handle_request(request)
            
            response_json = json.dumps(response, ensure_ascii=False)
            writer.write((response_json + "\n").encode())
            await writer.drain()
        except Exception as e:
            error_response = server._error_response(None, str(e))
            writer.write((json.dumps(error_response) + "\n").encode())
            await writer.drain()
    
    writer.close()
    await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    echo "✓ 创建示例 Python 包"
else
    echo "✓ 示例 Python 包已存在，跳过"
fi

echo
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo
echo "下一步:"
echo "1. 查看快速开始指南:"
echo "   cat tools/skill-creator/QUICKSTART.md"
echo
echo "2. 创建你的第一个 Skill:"
echo "   python tools/skill-creator/interactive.py"
echo
echo "3. 或运行示例:"
echo "   cd example_skill && python -m example_skill.mcp_server"
echo
echo "========================================"
