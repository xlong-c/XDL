# OpenCode Skill Creator

为 OpenCode 项目创建自定义 MCP Skill 的完整工具集。

## 📦 安装

Skill Creator 已经集成到项目中，无需额外安装。

**位置**: `/root/workspace/xdl/tools/skill-creator/`

## 🚀 快速开始

### 方式一：交互式创建（推荐新手）

```bash
cd /root/workspace/xdl
python tools/skill-creator/interactive.py
```

按照提示输入：
1. Skill 名称和描述
2. 作者信息和许可证
3. 工具定义（名称、描述、参数）
4. 确认后自动生成完整结构

### 方式二：命令行创建（快速）

```bash
cd /root/workspace/xdl

# 创建基础 Skill
python tools/skill-creator/create_skill.py \
  --name my-skill \
  --description "我的 Skill 描述"

# 创建带工具的 Skill
python tools/skill-creator/create_skill.py \
  --name image-processor \
  --description "图像处理工具集" \
  --author "Your Name" \
  --tool "resize=调整图片尺寸" \
  --tool "crop=裁剪图片" \
  --tool "convert=转换图片格式"
```

## 📁 生成的结构

```
project/
├── .opencode/
│   └── skills/
│       └── my-skill/
│           ├── opencode.json    # OpenCode 配置
│           ├── README.md        # 使用文档
│           ├── SETUP.md         # 设置指南
│           └── .gitignore
└── my_skill/
    ├── __init__.py
    ├── mcp_server.py           # 需要实现
    └── tools/
        ├── __init__.py
        └── tool_impl.py        # 需要实现
```

## 🛠️ 完整示例

### 示例 1: 创建计算器 Skill

```bash
python tools/skill-creator/create_skill.py \
  --name calculator \
  --description "数学计算工具集" \
  --tool "add=加法运算" \
  --tool "subtract=减法运算" \
  --tool "multiply=乘法运算" \
  --tool "divide=除法运算"
```

### 示例 2: 创建网络工具 Skill

```bash
python tools/skill-creator/create_skill.py \
  --name web-tools \
  --description "网络请求和数据处理工具" \
  --author "Your Name" \
  --tool "fetch_url=获取网页内容" \
  --tool "check_status=检查网站状态" \
  --tool "extract_links=提取网页链接"
```

## 📝 实现步骤

### 步骤 1: 创建 Skill 结构

使用上述命令创建 Skill 后，会生成配置文件和目录结构。

### 步骤 2: 实现工具逻辑

编辑 `my_skill/tools/tool_name.py`:

```python
async def echo_tool(args: dict) -> dict:
    """回显工具实现"""
    message = args.get("message", "")
    uppercase = args.get("uppercase", False)
    
    if uppercase:
        message = message.upper()
    
    return {
        "success": True,
        "message": message,
        "metadata": {
            "length": len(message),
            "uppercase": uppercase
        }
    }
```

### 步骤 3: 实现 MCP 服务器

编辑 `my_skill/mcp_server.py`:

```python
#!/usr/bin/env python3
"""MCP Server for My Skill"""

import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.echo import echo_tool

class MySkillServer:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.tools = {"echo": echo_tool}
    
    async def handle_request(self, request):
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            return await self.call_tool(
                params.get("name"),
                params.get("arguments", {})
            )
        # 处理其他方法...
    
    async def list_tools(self):
        tools = [{
            "name": "echo",
            "description": "回显工具",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "uppercase": {"type": "boolean", "default": False}
                },
                "required": ["message"]
            }
        }]
        return {"jsonrpc": "2.0", "result": {"tools": tools}}
    
    async def call_tool(self, name, args):
        if name not in self.tools:
            return {"error": f"Tool not found: {name}"}
        result = await self.tools[name](args)
        return {"jsonrpc": "2.0", "result": result}

async def main():
    server = MySkillServer()
    # 实现 stdin/stdout 循环...

if __name__ == "__main__":
    asyncio.run(main())
```

### 步骤 4: 配置 opencode.json

编辑 `.opencode/skills/my-skill/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skill": {
    "name": "my-skill",
    "version": "1.0.0",
    "description": "我的 Skill 描述",
    "author": "Your Name",
    "license": "MIT"
  },
  "mcp": {
    "name": "my-skill-mcp",
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
    }
  ]
}
```

### 步骤 5: 测试 Skill

```bash
# 测试 MCP 服务器
cd my_skill
python -m my_skill.mcp_server

# 在 OpenCode 中使用
# 输入：使用 echo 工具说"你好"
```

## 📚 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `create_skill.py` | 命令行创建工具 |
| `interactive.py` | 交互式创建工具 |
| `mcp_server_template.py` | MCP 服务器模板 |
| `README.md` | 详细文档 |
| `QUICKSTART.md` | 快速开始指南 |

### 生成文件

| 文件 | 说明 |
|------|------|
| `opencode.json` | OpenCode Skill 配置 |
| `README.md` | Skill 使用文档 |
| `SETUP.md` | Python 包设置指南 |
| `.gitignore` | Git 忽略规则 |

## 🎯 最佳实践

### 命名规范

- **Skill 名称**: kebab-case（如 `image-processor`）
- **Python 包**: snake_case（如 `image_processor`）
- **工具名**: snake_case（如 `resize_image`）

### 工具设计

1. **原子性**: 每个工具只做一件事
2. **明确参数**: 使用类型定义和描述
3. **错误处理**: 返回 `success` 字段，捕获异常
4. **文档**: 提供清晰的输入/输出示例

### 错误处理模板

```python
async def robust_tool(args: dict) -> dict:
    try:
        # 验证参数
        if "required" not in args:
            return {"success": False, "error": "缺少必需参数"}
        
        # 执行逻辑
        result = do_something(args["required"])
        
        return {"success": True, "data": result}
    except FileNotFoundError as e:
        return {"success": False, "error": f"文件未找到：{e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## 🔧 命令行参数

```
usage: create_skill.py [-h] --name NAME --description DESCRIPTION
                       [--author AUTHOR] [--license LICENSE] [--output OUTPUT]
                       [--tool NAME=DESCRIPTION ...]

为 OpenCode 创建新的 MCP Skill

可选参数:
  -h, --help            显示帮助信息
  --name NAME, -n NAME  Skill 名称（kebab-case）
  --description DESCRIPTION, -d DESCRIPTION
                        Skill 描述
  --author AUTHOR, -a AUTHOR
                        作者名（默认：XDL Team）
  --license LICENSE, -l LICENSE
                        许可证（默认：MIT）
  --output OUTPUT, -o OUTPUT
                        输出目录（默认：.）
  --tool NAME=DESCRIPTION, -t NAME=DESCRIPTION
                        添加工具定义，可重复使用
```

## 🐛 故障排除

### 问题 1: Skill 不加载

**检查清单**:
- [ ] `.opencode/skills/` 目录存在
- [ ] `opencode.json` 格式正确
- [ ] Python 包可以导入
- [ ] MCP 服务器可以启动

### 问题 2: 工具调用失败

```bash
# 手动测试 MCP 服务器
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  python -m my_skill.mcp_server

# 检查配置
cat .opencode/skills/my-skill/opencode.json | python -m json.tool
```

### 问题 3: 参数不匹配

- 检查 `inputSchema` 定义
- 验证参数类型和必需字段
- 查看 MCP 服务器日志

## 📖 参考资源

- **快速开始**: `QUICKSTART.md`
- **详细文档**: `README.md`
- **模板文件**: `mcp_server_template.py`
- **示例 Skill**: `.opencode/skills/example-skill/`
- **参考实现**: `.opencode/skills/research-tools/`

## 💡 高级技巧

### 使用异步工具

```python
import aiohttp

async def fetch_url(args: dict) -> dict:
    url = args.get("url")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            return {"success": True, "content": content}
```

### 使用缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_operation(param: str) -> dict:
    # 缓存结果
    return heavy_computation(param)

async def tool_with_cache(args: dict) -> dict:
    result = cached_operation(args["param"])
    return {"success": True, "data": result}
```

### 复杂参数验证

```python
from typing import List, Optional

async def validated_tool(args: dict) -> dict:
    # 类型检查
    items: List[str] = args.get("items", [])
    if not isinstance(items, list):
        return {"success": False, "error": "items 必须是列表"}
    
    # 值验证
    threshold: float = args.get("threshold", 0.5)
    if not 0 <= threshold <= 1:
        return {"success": False, "error": "threshold 必须在 0-1 之间"}
    
    # 处理逻辑
    return {"success": True, "result": process(items, threshold)}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 改进 Skill Creator！

## 📄 许可证

MIT License

---

**创建你的第一个 Skill 只需 5 分钟！** 🚀

```bash
python tools/skill-creator/interactive.py
```
