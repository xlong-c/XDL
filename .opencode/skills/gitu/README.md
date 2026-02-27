# gitU Skill

自动 Git 提交推送工具 - 智能生成提交描述并提交推送所有代码变更

## 功能特性

- ✅ **自动检测变更**: 智能分析工作区的所有代码变更
- ✅ **智能提交描述**: 根据变更类型和文件生成清晰合理的提交信息
- ✅ **一键提交推送**: 自动执行 `git add`、`git commit`、`git push`
- ✅ **分类统计**: 按新增、修改、删除、重命名分类统计文件
- ✅ **详细报告**: 返回完整的变更列表和提交结果

## 快速开始

### 使用方式 1: 直接提交（推荐）

```
请帮我提交并推送所有代码变更
```

gitU 会自动：
1. 检测所有未提交的变更
2. 生成智能提交描述
3. 执行 `git add -A`
4. 执行 `git commit -m "描述"`
5. 执行 `git push`

### 使用方式 2: 指定参数

```json
{
  "tool": "auto_commit",
  "args": {
    "push": true,
    "remote": "origin"
  }
}
```

## 可用工具

### `auto_commit` - 自动提交推送

自动分析代码变更，生成智能提交描述，执行 git add、commit 和 push

**输入参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cwd` | string | `.` | 工作目录路径 |
| `push` | boolean | `true` | 是否自动推送到远程仓库 |
| `remote` | string | `origin` | 远程仓库名称 |
| `branch` | string | - | 分支名称（不指定则使用当前分支） |

**返回结果**:

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

## 使用示例

### 示例 1: 基本使用

**输入**:
```
提交并推送所有代码
```

**处理流程**:
1. 检测变更：`modified: src/main.py, utils/helper.py`
2. 生成描述：`"修改代码实现"`
3. 执行提交：`git add -A && git commit -m "修改代码实现" && git push`

**输出**:
```
✅ 提交成功
- 修改了 2 个文件
- 已推送到 origin/master
```

### 示例 2: 只提交不推送

**输入**:
```json
{
  "tool": "auto_commit",
  "args": {
    "push": false
  }
}
```

**输出**:
```
✅ 提交成功
- 修改了 2 个文件
- 本地提交完成，未推送
```

### 示例 3: 指定远程仓库

**输入**:
```json
{
  "tool": "auto_commit",
  "args": {
    "remote": "upstream"
  }
}
```

## 提交描述生成规则

gitU 会根据变更的文件类型生成智能描述：

| 文件类型 | 提交描述 |
|----------|----------|
| `.py`, `.js`, `.ts`, `.java` | 修改代码实现 / 添加新代码 |
| `.md`, `.rst`, `.txt` | 更新文档 |
| `.yaml`, `.yml`, `.json`, `.toml` | 更新配置文件 |
| `test_*.py`, `*.test.js` | 更新测试 |
| 删除文件 | 清理删除的文件 |
| 重命名文件 | 重构文件结构 |

**组合示例**:
- 修改了 Python 文件和文档 → `"修改代码实现; 更新文档"`
- 新增配置文件和代码 → `"更新配置文件; 添加新代码"`

## 详细变更报告

提交成功后，gitU 会返回详细的变更列表：

```
新增 (2 个文件):
  + new_feature.py
  + README.md

修改 (3 个文件):
  ~ src/main.py
  ~ src/utils.py
  ~ config.yaml

删除 (1 个文件):
  - old_module.py

新文件 (1 个):
  ? temp.txt
```

## 故障排除

### 问题 1: 不是 Git 仓库

**错误**: `当前目录不是 Git 仓库`

**解决**: 确保在 Git 仓库目录中运行
```bash
git init  # 初始化新仓库
# 或
cd /path/to/git/repo
```

### 问题 2: 推送失败

**错误**: `git push 失败：permission denied`

**解决**:
1. 检查远程仓库配置：`git remote -v`
2. 配置 SSH 密钥或 credentials
3. 确认有推送权限

### 问题 3: 没有变更

**输出**: `没有需要提交的变更`

**说明**: 工作区干净，无需提交

### 问题 4: 提交被拒绝

**错误**: `git commit 失败`

**可能原因**:
- 预提交钩子失败（lint 检查等）
- 没有配置 git user.name 和 user.email

**解决**:
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
```

## 注意事项

⚠️ **使用前请确认**:
1. 已配置 Git 用户信息（user.name, user.email）
2. 有远程仓库的推送权限
3. 本地变更确实需要提交

⚠️ **安全提示**:
- gitU 会提交**所有**未暂存的变更
- 建议先检查变更：`git status`
- 敏感文件（密钥、配置）请加入 `.gitignore`

## 许可证

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！

**使用示例**:
```
使用 auto_commit 工具提交当前所有代码变更
```
