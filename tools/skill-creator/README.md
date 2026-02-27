# Skill Creator for OpenCode

为 OpenCode 项目快速创建新的 MCP Skill 工具。

## 快速开始

### 方式一：使用命令行工具

```bash
cd /root/workspace/xdl

# 创建基础 Skill
python tools/skill-creator/create_skill.py --name my-skill --description "My awesome skill"

# 创建带工具定义的 Skill
python tools/skill-creator/create_skill.py \
  --name my-skill \
  --description "My awesome skill with tools" \
  --tool "tool1=First tool description" \
  --tool "tool2=Second tool description"

# 指定作者和许可证
python tools/skill-creator/create_skill.py \
  --name my-skill \
  --description "..." \
  --author "Your Name" \
  --license "MIT"
```

### 方式二：使用交互式模式

```bash
cd /root/workspace/xdl
python tools/skill-creator/interactive.py
```

## 生成的目录结构

```
project/
├── .opencode/
│   └── skills/
│       └── my-skill/          # Skill 配置目录
│           ├── opencode.json  # OpenCode 配置
│           ├── README.md      # 使用文档
│           └── .gitignore     # Git 忽略文件
├── my_skill/                   # Python 包（需要自行实现）
│   ├── __init__.py
│   ├── mcp_server.py          # MCP 服务器入口
│   └── tools/
│       ├── __init__.py
│       └── tool_impl.py       # 工具实现
```

## 完整示例

### 1. 创建带工具的 Skill

```bash
python tools/skill-creator/create_skill.py \
  --name image-processor \
  --description "图像处理工具集 - 支持缩放、裁剪、格式转换" \
  --author "Your Name" \
  --tool "resize=调整图片尺寸" \
  --tool "crop=裁剪图片" \
  --tool "convert=转换图片格式"
```

### 2. 实现工具逻辑

编辑 `my_skill/tools/tool_impl.py`:

```python
from PIL import Image
from pathlib import Path

async def resize_image(args: dict) -> dict:
    """调整图片尺寸"""
    image_path = args.get("image_path")
    width = args.get("width", 800)
    height = args.get("height", 600)
    
    try:
        img = Image.open(image_path)
        img = img.resize((width, height))
        output_path = str(Path(image_path).parent / f"resized_{Path(image_path).name}")
        img.save(output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "message": f"图片已调整为 {width}x{height}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### 3. 实现 MCP 服务器

编辑 `my_skill/mcp_server.py`:

```python
"""
MCP Server for Image Processor
"""

import sys
import json
import asyncio
from typing import Any, Dict
from pathlib import Path

# 添加工具导入
sys.path.insert(0, str(Path(__file__).parent))
from tools.tool_impl import resize_image, crop_image, convert_image


class MCPServer:
    """MCP 服务器实现"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tools = {
            "resize": resize_image,
            "crop": crop_image,
            "convert": convert_image,
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
                    "name": "image-processor-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        # TODO: 根据实际工具实现此方法
        tools = [
            {
                "name": "resize",
                "description": "调整图片尺寸",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "图片路径"},
                        "width": {"type": "integer", "description": "目标宽度"},
                        "height": {"type": "integer", "description": "目标高度"}
                    },
                    "required": ["image_path"]
                }
            },
            # 添加更多工具...
        ]
        return {
            "jsonrpc": "2.0",
            "id": "list",
            "result": {"tools": tools}
        }
    
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        try:
            tool_func = self.tools[tool_name]
            result = await tool_func(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "error": {"message": str(e)}
            }
    
    def _error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"message": message}
        }


async def main():
    """主函数"""
    server = MCPServer()
    
    # 从 stdin 读取请求
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    writer_transport, writer_protocol = await asyncio.get_event_loop().create_subprocess_exec(
        "cat",
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE
    )
    
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, None)
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            request = json.loads(line.decode())
            response = await server.handle_request(request)
            
            response_json = json.dumps(response)
            writer.write((response_json + "\n").encode())
            await writer.drain()
        except Exception as e:
            error_response = server._error_response(None, str(e))
            writer.write((json.dumps(error_response) + "\n").encode())
            await writer.drain()


if __name__ == "__main__":
    asyncio.run(main())
```

### 4. 更新 opencode.json

编辑 `.opencode/skills/my-skill/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skill": {
    "name": "image-processor",
    "version": "1.0.0",
    "description": "图像处理工具集 - 支持缩放、裁剪、格式转换",
    "author": "Your Name",
    "license": "MIT"
  },
  "mcp": {
    "name": "image-processor-mcp",
    "type": "stdio",
    "command": "python",
    "args": ["-m", "my_skill.mcp_server"],
    "cwd": "./my_skill",
    "env": {
      "PYTHONPATH": "./my_skill"
    }
  },
  "tools": [
    {
      "name": "resize",
      "description": "调整图片尺寸",
      "inputSchema": {
        "type": "object",
        "properties": {
          "image_path": {
            "type": "string",
            "description": "图片路径"
          },
          "width": {
            "type": "integer",
            "description": "目标宽度"
          },
          "height": {
            "type": "integer",
            "description": "目标高度"
          }
        },
        "required": ["image_path"]
      }
    },
    {
      "name": "crop",
      "description": "裁剪图片",
      "inputSchema": {
        "type": "object",
        "properties": {
          "image_path": {"type": "string"},
          "left": {"type": "integer"},
          "top": {"type": "integer"},
          "right": {"type": "integer"},
          "bottom": {"type": "integer"}
        },
        "required": ["image_path", "left", "top", "right", "bottom"]
      }
    },
    {
      "name": "convert",
      "description": "转换图片格式",
      "inputSchema": {
        "type": "object",
        "properties": {
          "image_path": {"type": "string"},
          "format": {"type": "string", "enum": ["png", "jpg", "webp"]}
        },
        "required": ["image_path", "format"]
      }
    }
  ]
}
```

## 命令行参数

```
usage: create_skill.py [-h] --name NAME --description DESCRIPTION
                       [--author AUTHOR] [--license LICENSE] [--output OUTPUT]
                       [--tool NAME=DESCRIPTION ...]

为 OpenCode 创建新的 MCP Skill

可选参数:
  -h, --help            显示帮助信息
  --name NAME, -n NAME  Skill 名称（使用 kebab-case，如 my-skill）
  --description DESCRIPTION, -d DESCRIPTION
                        Skill 描述
  --author AUTHOR, -a AUTHOR
                        作者名（默认：XDL Team）
  --license LICENSE, -l LICENSE
                        许可证类型（默认：MIT）
  --output OUTPUT, -o OUTPUT
                        输出目录（默认：当前目录）
  --tool NAME=DESCRIPTION, -t NAME=DESCRIPTION
                        添加工具定义（格式：name=description），可重复使用
```

## 最佳实践

### 1. 命名规范

- **Skill 名称**: 使用 kebab-case（如 `image-processor`）
- **Python 包名**: 使用 snake_case（如 `image_processor`）
- **工具名称**: 使用 snake_case（如 `resize_image`）

### 2. 工具设计

- 每个工具应该是原子的，只做一件事
- 输入参数应该明确且有类型定义
- 返回值应该包含 `success` 字段
- 错误应该被捕获并返回，而不是抛出异常

### 3. 错误处理

```python
async def your_tool(args: dict) -> dict:
    try:
        # 实现逻辑
        result = process(args)
        return {
            "success": True,
            "data": result
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"文件未找到：{e}",
            "error_type": "file_not_found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "unknown"
        }
```

### 4. 文档编写

- 在 `README.md` 中提供清晰的使用示例
- 为每个工具添加输入/输出示例
- 包含故障排除部分

## 故障排除

### 问题 1: Skill 不加载

**症状**: OpenCode 无法识别新创建的 Skill

**解决**:
1. 检查 `.opencode/skills/` 目录结构是否正确
2. 验证 `opencode.json` 格式是否有效
3. 重启 OpenCode

### 问题 2: MCP 服务器启动失败

**症状**: 工具调用返回连接错误

**解决**:
```bash
# 手动测试 MCP 服务器
cd /root/workspace/xdl/my_skill
python -m my_skill.mcp_server

# 检查依赖
pip install -r requirements.txt
```

### 问题 3: 工具参数不匹配

**症状**: 工具调用返回参数错误

**解决**:
- 检查 `opencode.json` 中的 `inputSchema` 定义
- 确保实际参数类型与 Schema 匹配
- 验证必需参数是否都已提供

## 许可证

MIT License
