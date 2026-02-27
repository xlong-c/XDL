# example-skill Skill

示例 Skill - 展示如何创建 OpenCode Skill

## 安装与配置

### 1. 依赖安装

```bash
# 在此处添加依赖安装命令
pip install -r requirements.txt
```

### 2. OpenCode 配置

Skill 已配置在 `.opencode/skills/example-skill/opencode.json`。

MCP 服务器通过 stdio 协议与 OpenCode 通信。


## 可用工具

### `echo`

回显工具 - 返回输入的消息

**输入参数**:


### `calculate`

计算工具 - 执行简单数学运算

**输入参数**:



## 使用示例

```json
{
  "tool": "example_tool",
  "args": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

## 开发指南

### 添加新工具

1. 在 Python 包中实现工具函数
2. 在 `mcp_server.py` 中注册工具
3. 在 `opencode.json` 中添加工具定义

### 工具实现模板

```python
async def your_tool_name(args: dict) -> dict:
    """工具实现"""
    # 处理逻辑
    result = {
        "success": True,
        "data": "..."
    }
    return result
```

## 故障排除

### 问题 1: MCP 服务器启动失败

**症状**: OpenCode 无法连接到工具

**解决**:
```bash
# 检查 Python 路径
python -m example_skill.mcp_server

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
