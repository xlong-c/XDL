---
name: gitc
description: Git 代码提交辅助工具，自动化远程同步和提交流程。使用此技能当用户请求提交代码更新时，需要先检查远程更新、比对差异、处理冲突。适用于：1) 常规代码提交流程，2) 需要确保与远程同步后再提交，3) 检测并处理合并冲突，4) 提交前查看本地变更。触发关键词："提交代码"、"gitc"、"提交更新"等。
---

# Git 代码提交辅助工具 (gitc)

## 工作流决策树

```
开始
  ↓
检查远程状态（git fetch + 检查 ahead/behind）
  ↓
是否有远程更新？
  ├─ 否 → 检查本地变更 → 提交 → 推送 → 完成
  └─ 是 → 尝试拉取（git pull）
          ↓
        拉取成功？
          ├─ 是 → 检查本地变更 → 提交 → 推送 → 完成
          └─ 否（冲突）→ 打印冲突信息 → 停止 → 等待手动处理
```

## 执行步骤

### 步骤 1：检查远程状态

执行以下命令检查远程更新情况：

```bash
# 获取远程最新信息（不合并）
git fetch --all

# 检查本地分支状态
git status -sb
```

**解读输出**：
- `## master...origin/master`：本地与远程同步
- `## master...origin/master [ahead 2]`：本地领先 2 个提交
- `## master...origin/master [behind 1]`：本地落后 1 个提交
- `## master...origin/master [ahead 1, behind 3]`：双向分歧

### 步骤 2：决策分支

#### 情况 A：本地与远程同步或本地领先（无远程更新）

继续执行[步骤 3：查看本地变更](#步骤-3查看本地变更)

#### 情况 B：本地落后远程（有远程更新）

尝试拉取远程更新：

```bash
# 尝试拉取远程更新
git pull
```

- **成功**：继续执行[步骤 3：查看本地变更](#步骤-3查看本地变更)
- **失败（冲突）**：跳转到[冲突处理](#冲突处理)

#### 情况 C：双向分歧

**自动停止并告知用户**，提供选项：

```
检测到双向分歧：
  本地领先 N 个提交，落后 M 个提交

建议操作：
  1. 如果本地提交重要：先 push，然后手动处理远程变更
  2. 如果远程更新重要：先 pull 并解决冲突，再 push
  3. 如果都要保留：需要手动 rebase

请选择操作方式或手动处理。
```

### 步骤 3：查看本地变更

在提交前查看本地变更情况：

```bash
# 查看未跟踪和修改的文件
git status

# 查看未暂存的变更
git diff

# 查看已暂存的变更
git diff --staged
```

**确认变更内容**，确保只提交预期的文件。

### 步骤 4：提交并推送

如果存在本地变更，执行提交和推送：

```bash
# 添加变更到暂存区
git add .  # 或指定文件：git add <file1> <file2>

# 提交（使用清晰的中文提交信息）
git commit -m "<类型>: <描述>"

# 推送到远程
git push
```

**提交信息规范**：

| 类型 | 用途 |
|------|------|
| 添加 | 新增功能、文件、代码 |
| 修复 | 修复 bug、错误 |
| 更新 | 更新现有功能、逻辑 |
| 优化 | 性能优化、代码改进 |
| 重构 | 代码重构、结构调整 |
| 删除 | 删除功能、文件 |
| 文档 | 文档更新 |

**示例**：
- `添加: 实现用户认证功能`
- `修复: 解决登录超时问题`
- `优化: 提升数据库查询性能`

### 步骤 5：验证结果

```bash
# 确认工作区干净
git status

# 查看最新提交
git log --oneline -1
```

## 冲突处理

当 `git pull` 产生冲突时：

### 1. 停止自动流程

**不要继续提交**，立即停止并报告冲突。

### 2. 打印冲突信息

```bash
# 查看冲突状态
git status

# 显示冲突文件列表
git diff --name-only --diff-filter=U
```

### 3. 提供详细冲突报告

向用户报告：

```
⚠️ 检测到合并冲突

冲突文件：
  - src/file1.py
  - src/file2.py

处理步骤：
  1. 查看冲突文件：git diff
  2. 手动解决冲突（编辑标记为 <<<<<<< HEAD 的文件）
  3. 标记为已解决：git add <resolved-file>
  4. 完成合并：git commit
  5. 重新执行 gitc

或者使用工具辅助：
  - VSCode: 内置冲突解决界面
  - GitKraken: 图形化冲突解决
  - git mergetool: 启动配置的合并工具
```

### 4. 查看具体冲突内容

```bash
# 显示冲突文件的具体差异
git diff <conflict-file>

# 或者只显示冲突部分
git diff --diff-filter=U
```

**冲突标记示例**：
```python
<<<<<<< HEAD
# 本地修改
result = calculate_local()
=======
# 远程修改
result = calculate_remote()
>>>>>>> origin/master
```

## 重要注意事项

### 执行前检查

1. **确认分支**：确保在正确的分支上操作
   ```bash
   git branch --show-current
   ```

2. **检查敏感文件**：避免提交 `.env`、密钥、大文件等
   ```bash
   git diff --name-only  # 查看待提交文件列表
   ```

3. **工作区状态**：确保当前工作区没有未保存的编辑

### 提交信息

- 使用**简洁的中文描述**
- 遵循 `类型: 描述` 格式
- 不包含 AI 生成标记

### 冲突处理原则

- **自动停止**：遇到冲突时不要尝试自动解决
- **明确报告**：详细列出冲突文件和处理步骤
- **用户主导**：等待用户手动解决后再继续

## 完整示例

### 无冲突的常规提交流程

```bash
# 1. 检查远程
$ git fetch --all
$ git status -sb
## master...origin/master

# 2. 无远程更新，继续查看本地变更
$ git status
Changes not staged for commit:
  modified:   src/model.py

# 3. 查看变更
$ git diff src/model.py
... (变更内容)

# 4. 提交并推送
$ git add src/model.py
$ git commit -m "修复: 修正模型初始化错误"
$ git push
```

### 有远程更新但无冲突的提交流程

```bash
# 1. 检查远程
$ git fetch --all
$ git status -sb
## master...origin/master [behind 1]

# 2. 拉取远程更新
$ git pull
Updating 6bdf9f7..d36891d
Fast-forward

# 3. 继续查看本地变更
$ git status
...

# 4. 提交并推送
$ git add .
$ git commit -m "添加: 实现新功能"
$ git push
```

### 有冲突时的处理流程

```bash
# 1. 检查远程
$ git fetch --all
$ git status -sb
## master...origin/master [behind 1]

# 2. 尝试拉取
$ git pull
CONFLICT (content): Merge conflict in src/model.py
Automatic merge failed; fix conflicts and then commit the result.

# 3. 停止并报告冲突
$ git status
both modified:   src/model.py

# ⚠️ 检测到合并冲突
# 冲突文件：src/model.py
# 请手动解决冲突后重新执行 gitc
```
