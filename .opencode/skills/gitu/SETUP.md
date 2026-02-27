
⚠️  注意：您需要在项目中创建 Python MCP 服务器

建议的目录结构:
├── gitu/
│   ├── __init__.py
│   ├── mcp_server.py      # MCP 服务器入口
│   └── tools/
│       ├── __init__.py
│       └── your_tool.py   # 工具实现

然后在 opencode.json 中配置:
{
  "mcp": {
    "command": "python",
    "args": ["-m", "gitu.mcp_server"],
    "cwd": "./gitu",
    "env": {
      "PYTHONPATH": "./gitu"
    }
  }
}
