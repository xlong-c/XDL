# Claude Code MCP 配置

本目录包含 Claude Code 的 MCP (Model Context Protocol) 服务器配置。

## 配置文件说明

### `mcp.json`
Claude Code 的 MCP 服务器主配置文件，定义了所有可用的 MCP 服务器。

## 可用的 MCP 服务器

### 1. research-tools
科研工具集 - 零 token 消耗的论文分析、分类、摘要获取和报告生成工具。

**工具列表：**
- `dblp_analyze` - 分析 DBLP 期刊/会议页面
- `fetch_abstracts` - 批量获取论文摘要
- `classify_papers` - 论文分类
- `extract_paper_info` - 提取论文关键信息
- `generate_analysis_report` - 生成分析报告
- `run_full_workflow` - 一键执行完整科研工作流

### 2. gitu
自动 Git 提交推送工具 - 智能生成提交描述并提交推送所有代码变更。

**工具列表：**
- `auto_commit` - 自动分析代码变更并提交推送

## 从 OpenCode 迁移

本配置是从 OpenCode skill 系统迁移而来。主要变更：

| OpenCode | Claude Code |
|----------|-------------|
| `.opencode/skills/{name}/opencode.json` | `.claude/mcp.json` |
| 每个 skill 单独配置 | 统一在 `mcpServers` 中配置 |
| `skill.mcp.cwd` 相对路径 | 绝对路径或相对于项目根目录 |

## 使用方法

1. 确保 Claude Code 已安装并配置好
2. 重启 Claude Code 以加载 MCP 配置
3. 在对话中直接使用相关工具

## 测试 MCP 服务器

手动测试 MCP 服务器是否正常工作：

```bash
# 测试 research-tools
cd /root/workspace/xdl/sci_research
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m skills.mcp_server

# 测试 gitu
cd /root/workspace/xdl/gitu
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m gitu.mcp_server
```

## 故障排除

### MCP 服务器无法启动
- 检查 Python 路径和依赖
- 验证 `cwd` 路径是否正确
- 确认 `PYTHONPATH` 设置正确

### 工具调用失败
- 查看 Claude Code 的日志输出
- 验证 MCP 服务器的 stdio 通信
- 检查工具参数是否匹配 `inputSchema`
