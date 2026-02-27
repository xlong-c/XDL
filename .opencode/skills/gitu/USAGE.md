# gitU 使用指南

## 🚀 快速开始

### 安装完成

gitU Skill 已经创建并配置完成！

**位置**: `/root/workspace/xdl/.opencode/skills/gitu/`

### 立即使用

在 OpenCode 中输入：

```
使用 gitU 提交并推送所有代码变更
```

或者更简单：

```
提交所有代码
```

---

## 📋 功能说明

### gitU 能做什么？

1. **自动检测变更** - 扫描工作区所有未提交的变更
2. **智能生成描述** - 根据文件类型和变更内容生成清晰的提交信息
3. **一键提交推送** - 自动执行 `git add` → `git commit` → `git push`

### 提交描述示例

```
修改代码实现; 更新文档; 添加新代码

新增 (2 个文件):
  + new_feature.py
  + utils/helper.py

修改 (5 个文件):
  ~ src/main.py
  ~ src/utils.py
  ~ README.md

删除 (1 个文件):
  - old_module.py
```

---

## 💡 使用方式

### 方式 1: 自然语言（推荐）

```
请帮我提交并推送所有代码变更
```

```
提交代码但不推送
```

```
提交到 upstream 远程仓库
```

### 方式 2: JSON 调用

```json
{
  "tool": "auto_commit",
  "args": {
    "push": true,
    "remote": "origin"
  }
}
```

### 方式 3: 命令行测试

```bash
# 测试 Git 状态检测
cd /root/workspace/xdl
python -c "from gitu.tools.auto_commit import get_git_status; print(get_git_status('.'))"

# 测试提交描述生成
python -c "from gitu.tools.auto_commit import analyze_changes, generate_commit_message; ..."
```

---

## 🔧 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cwd` | string | `.` | 工作目录 |
| `push` | boolean | `true` | 是否推送 |
| `remote` | string | `origin` | 远程仓库名 |
| `branch` | string | - | 分支名（默认当前分支） |

---

## 📊 返回结果

### 成功示例

```json
{
  "success": true,
  "message": "提交成功",
  "commit_message": "修改代码实现; 更新文档",
  "changes": {
    "added": 2,
    "modified": 5,
    "deleted": 1,
    "untracked": 0,
    "renamed": 0
  },
  "total_files": 8,
  "push_success": true,
  "stage": "pushed"
}
```

### 失败示例

```json
{
  "success": false,
  "error": "当前目录不是 Git 仓库",
  "cwd": "/path/to/dir"
}
```

---

## ⚙️ 前置要求

### 1. Git 配置

```bash
# 配置用户信息（必需）
git config user.name "Your Name"
git config user.email "your@email.com"

# 检查配置
git config user.name
git config user.email
```

### 2. 远程仓库

```bash
# 添加远程仓库（如果需要推送）
git remote add origin git@github.com:user/repo.git

# 检查远程仓库
git remote -v
```

### 3. 推送权限

```bash
# 测试 SSH 连接（GitHub）
ssh -T git@github.com

# 或使用 HTTPS + credentials
git config --global credential.helper store
```

---

## 🛠️ 故障排除

### 错误：当前目录不是 Git 仓库

**原因**: 不在 Git 仓库目录中

**解决**:
```bash
# 初始化新仓库
git init

# 或切换到正确的目录
cd /path/to/git/repo
```

### 错误：git push 失败

**原因**: 没有推送权限或未配置远程

**解决**:
```bash
# 检查远程配置
git remote -v

# 配置 SSH 密钥
ssh-keygen -t ed25519 -C "your@email.com"
# 添加公钥到 GitHub/GitLab

# 或使用 HTTPS
git remote set-url origin https://github.com/user/repo.git
```

### 错误：git commit 失败

**原因**: 
- 没有配置用户信息
- pre-commit 钩子失败

**解决**:
```bash
# 配置用户信息
git config user.name "Your Name"
git config user.email "your@email.com"

# 检查 pre-commit
git config --get core.hooksPath

# 临时跳过钩子（不推荐）
git commit --no-verify -m "message"
```

### 输出：没有需要提交的变更

**说明**: 工作区干净，无需提交

**检查变更**:
```bash
git status
git diff
```

---

## 📝 最佳实践

### 1. 提交前检查

```bash
# 查看变更
git status
git diff --stat

# 确认要提交的文件
git diff --name-only
```

### 2. 敏感文件

确保敏感文件已加入 `.gitignore`:

```gitignore
# 密钥和配置
.env
*.pem
*.key
credentials.json

# IDE 和缓存
.vscode/
.idea/
__pycache__/
*.pyc
```

### 3. 小步提交

- 频繁提交，每次提交一个逻辑变更
- 提交描述清晰明确
- 避免一次性提交大量不相关的变更

---

## 🎯 使用场景

### 场景 1: 日常开发提交

```
提交并推送所有代码
```

### 场景 2: 完成功能后提交

```
提交所有变更并推送到 origin
```

### 场景 3: 只提交不推送

```
只提交不推送
```

### 场景 4: 推送到其他远程

```
提交并推送到 upstream
```

---

## 🔍 测试 gitU

### 测试 Git 状态检测

```bash
cd /root/workspace/xdl
python -c "
from gitu.tools.auto_commit import get_git_status
import json
result = get_git_status('.')
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

### 测试提交描述生成

```bash
python -c "
from gitu.tools.auto_commit import analyze_changes, generate_commit_message
changes = {
    'modified': ['src/main.py', 'README.md'],
    'added': ['new.py'],
    'deleted': [],
    'untracked': [],
    'renamed': []
}
analysis = analyze_changes(changes)
message = generate_commit_message(changes, analysis)
print(message)
"
```

### 测试 MCP 服务器

```bash
cd /root/workspace/xdl/gitu
python -m gitu.mcp_server < /dev/null &
```

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `.opencode/skills/gitu/opencode.json` | OpenCode 配置 |
| `.opencode/skills/gitu/README.md` | 使用文档 |
| `gitu/mcp_server.py` | MCP 服务器 |
| `gitu/tools/auto_commit.py` | 工具实现 |

---

## 🤝 贡献

发现问题或有改进建议？欢迎提交 Issue 和 Pull Request！

---

**现在就开始使用 gitU 提交你的代码吧！** 🚀

```
使用 auto_commit 工具提交所有代码变更
```
