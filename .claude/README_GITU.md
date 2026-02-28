# gitU Claude Code Skill

自动 Git 提交推送工具 - Claude Code 兼容版

## 功能特性

- 🤖 自动分析代码变更
- 📝 智能生成提交描述
- 🚀 一键 add/commit/push
- 📊 详细的变更统计

## 配置说明

### 在 `.claude/mcp.json` 中已配置

```json
{
  "mcpServers": {
    "gitu": {
      "command": "python",
      "args": ["-m", "gitu.mcp_server"],
      "cwd": "/root/workspace/xdl/gitu",
      "env": {
        "PYTHONPATH": "/root/workspace/xdl/gitu"
      }
    }
  }
}
```

## 使用方法

### 在 Claude Code 对话中使用

```
使用 gitu 的 auto_commit 工具提交当前变更
```

或者指定参数：

```
使用 gitu 提交变更，推送到 origin main 分支
```

### 可用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cwd` | string | "." | Git 仓库目录 |
| `push` | boolean | true | 是否推送到远程 |
| `remote` | string | "origin" | 远程仓库名 |
| `branch` | string | (当前分支) | 目标分支 |

## 测试 MCP 服务器

```bash
cd /root/workspace/xdl/gitu

# 测试 tools/list
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m gitu.mcp_server

# 测试 initialize
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python -m gitu.mcp_server
```

## 目录结构

```
gitu/
├── __init__.py
├── mcp_server.py           # MCP 服务器（Claude 兼容版）
└── tools/
    ├── __init__.py
    └── auto_commit.py      # 自动提交工具实现
```

## 提交信息生成规则

gitU 会根据以下规则智能生成提交信息：

- 配置文件（.yaml, .json, .toml）→ "更新配置文件"
- 代码文件（.py, .js, .ts 等）→ "修改代码实现" 或 "添加新代码"
- 文档文件（.md, .rst）→ "更新文档"
- 测试文件 → "更新测试"
- 删除文件 → "清理删除的文件"
- 重命名文件 → "重构文件结构"

## 故障排除

### MCP 服务器无法启动

1. 检查 Python 路径：
```bash
cd /root/workspace/xdl/gitu
python -c "import gitu.mcp_server; print('OK')"
```

2. 验证依赖：
```bash
# gitU 只需要标准库和 git
git --version
```

### 工具调用失败

- 确保当前目录是 Git 仓库
- 检查是否有未提交的变更
- 查看 Claude Code 的 MCP 日志输出

### 推送失败

- 检查远程仓库配置：`git remote -v`
- 确认有推送权限
- 手动执行 `git push` 查看详细错误
