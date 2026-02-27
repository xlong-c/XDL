# Skill Creator 快速开始指南

## 5 分钟创建你的第一个 OpenCode Skill

### 步骤 1: 运行创建工具

```bash
cd /root/workspace/xdl

# 方式 A: 交互式创建（推荐新手）
python tools/skill-creator/interactive.py

# 方式 B: 命令行创建（快速）
python tools/skill-creator/create_skill.py \
  --name hello-skill \
  --description "我的第一个 Skill" \
  --tool "greet=打招呼工具" \
  --tool "calculate=计算工具"
```

### 步骤 2: 查看生成的结构

创建成功后，你会看到：

```
.opencode/skills/hello-skill/
├── opencode.json    # Skill 配置
├── README.md        # 使用文档
└── .gitignore

hello_skill/          # Python 包（需要实现）
├── __init__.py
├── mcp_server.py    # MCP 服务器（模板已生成）
└── tools/
    └── __init__.py
```

### 步骤 3: 实现工具逻辑

编辑 `hello_skill/tools/greet.py`:

```python
async def greet(args: dict) -> dict:
    """打招呼工具"""
    name = args.get("name", "World")
    language = args.get("language", "zh")
    
    if language == "zh":
        message = f"你好，{name}！"
    else:
        message = f"Hello, {name}!"
    
    return {
        "success": True,
        "message": message,
        "metadata": {
            "name": name,
            "language": language
        }
    }
```

### 步骤 4: 实现 MCP 服务器

编辑 `hello_skill/mcp_server.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server_template import MCPServer, run_server
from tools.greet import greet

class HelloSkillServer(MCPServer):
    def __init__(self, verbose=True):
        super().__init__(verbose)
        self.tools = {
            "greet": greet,
        }
    
    async def list_tools(self):
        tools = [{
            "name": "greet",
            "description": "打招呼工具",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "姓名"},
                    "language": {"type": "string", "description": "语言", "default": "zh"}
                }
            }
        }]
        return {"jsonrpc": "2.0", "id": "list", "result": {"tools": tools}}

async def main():
    server = HelloSkillServer(verbose=True)
    await run_server(server)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 步骤 5: 测试 Skill

```bash
# 测试 MCP 服务器能否启动
cd /root/workspace/xdl/hello_skill
python -m hello_skill.mcp_server

# 在 OpenCode 中使用
# 1. 打开 OpenCode
# 2. 输入：使用 greet 工具打招呼
# 3. Skill 应该会被自动加载并调用
```

## 常见问题

### Q1: 如何添加更多工具？

1. 在 `hello_skill/tools/` 目录创建新的 `.py` 文件
2. 实现工具函数
3. 在 `mcp_server.py` 中导入并注册
4. 在 `.opencode/skills/hello-skill/opencode.json` 中添加工具定义

### Q2: 如何添加工具参数？

编辑 `opencode.json`:

```json
{
  "name": "greet",
  "inputSchema": {
    "properties": {
      "name": {
        "type": "string",
        "description": "姓名"
      },
      "language": {
        "type": "string",
        "enum": ["zh", "en"],
        "default": "zh"
      }
    },
    "required": ["name"]
  }
}
```

### Q3: 如何调试 Skill？

```bash
# 1. 手动测试 MCP 服务器
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  python -m hello_skill.mcp_server

# 2. 查看详细日志
PYTHONPATH=./hello_skill python -c "
from hello_skill.mcp_server import HelloSkillServer
import asyncio
server = HelloSkillServer(verbose=True)
"

# 3. 检查配置
cat .opencode/skills/hello-skill/opencode.json | python -m json.tool
```

### Q4: Skill 不工作怎么办？

检查清单：
- [ ] `.opencode/skills/` 目录结构正确
- [ ] `opencode.json` 格式有效
- [ ] Python 包可以导入
- [ ] MCP 服务器可以启动
- [ ] 工具函数返回值符合格式

## 进阶技巧

### 使用异步工具

```python
async def async_tool(args: dict) -> dict:
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            data = await response.json()
            return {"success": True, "data": data}
```

### 错误处理最佳实践

```python
async def robust_tool(args: dict) -> dict:
    try:
        # 验证参数
        if "required_param" not in args:
            return {
                "success": False,
                "error": "缺少必需参数：required_param"
            }
        
        # 执行逻辑
        result = do_something(args["required_param"])
        
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

### 使用缓存

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_operation(param_hash: str) -> dict:
    # 缓存结果
    pass

async def tool_with_cache(args: dict) -> dict:
    # 创建参数哈希
    param_str = json.dumps(args, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    
    # 使用缓存
    result = cached_operation(param_hash)
    return {"success": True, "data": result}
```

## 参考资源

- 完整文档：`tools/skill-creator/README.md`
- 模板文件：`tools/skill-creator/mcp_server_template.py`
- 示例 Skill：`.opencode/skills/research-tools/`

## 需要帮助？

1. 查看 `tools/skill-creator/README.md` 详细文档
2. 参考现有 Skill 实现：`.opencode/skills/research-tools/`
3. 运行测试命令验证配置

祝你创建顺利！🚀
