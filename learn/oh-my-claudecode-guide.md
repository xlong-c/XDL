# Oh-My-ClaudeCode (OMC) 中文详解指南

> **版本**: 4.13.5 | **作者**: XDL Team | **最后更新**: 2026-04-29

---

## 目录

1. [什么是 OMC](#1-什么是-omc)
2. [核心设计理念](#2-核心设计理念)
3. [安装与启动](#3-安装与启动)
4. [Agent 智能体体系](#4-agent-智能体体系)
5. [模型路由策略](#5-模型路由策略)
6. [Skills 技能体系](#6-skills-技能体系)
7. [团队管道 (Team Pipeline)](#7-团队管道-team-pipeline)
8. [Commit 协议](#8-commit-协议)
9. [状态与记忆系统](#9-状态与记忆系统)
10. [工具与代码智能](#10-工具与代码智能)
11. [Hook 钩子系统](#11-hook-钩子系统)
12. [执行协议与最佳实践](#12-执行协议与最佳实践)
13. [配置与定制](#13-配置与定制)
14. [常用命令速查表](#14-常用命令速查表)

---

## 1. 什么是 OMC

**Oh-My-ClaudeCode (OMC)** 是建立在 Claude Code CLI 之上的**多智能体编排层**。它不像 Claude Code 那样只依靠单个模型来完成工作，而是将复杂任务拆解为若干子任务，分派给不同专长的 Agent 协作完成。

### 核心理念：三个角色分工

```
人类做决策  →  Claude(你)做协调  →  Agent 团队做执行
```

- **你（人类）**：做决策，把握方向，给出高层次指令。
- **Claude（主会话）**：理解意图，拆解任务，分配给合适的 Agent，合成结果。
- **OMC Agent 团队**：各司其职——有人分析需求，有人设计方案，有人写代码，有人审查，持续循环直到任务完成。

### 与原生 Claude Code 的区别

| 维度 | 原生 Claude Code | OMC |
|------|-----------------|-----|
| 模型 | 单一模型处理所有任务 | 根据任务类型自动路由到最适合的模型 |
| 协调 | 自行判断和执行 | 有结构化的 Agent 分工和团队管道 |
| 记忆 | 依赖会话上下文 | 持久化的项目记忆、状态、Notepad |
| 工作流 | 即兴发挥 | 预定义的 Tier-0 工作流（autopilot、ralph 等） |
| 验证 | 自行验证 | 独立的验证 Agent，禁止自审自批 |

---

## 2. 核心设计理念

### 2.1 专门化分工

不在某个 Agent 里堆积能力，而是让每个 Agent 聚焦在它最擅长的事情上。比如：
- 搜索代码 → `explore`（Haiku，轻量快速）
- 方案设计 → `architect`（Opus，深度思考）
- 写代码 → `executor`（Sonnet，高效实现）
- 审查代码 → `code-reviewer`（Opus，谨慎审查）

### 2.2 证据驱动，不猜测

- 在声明"完成了"之前必须先验证。
- 用 `verifier` / `tracer` 收集证据。
- 坚决拒绝自审自批——编码和审查必须由不同 Agent 完成。

### 2.3 轻量优先

能 `grep` 解决的不开 Agent，能 Haiku 搞定的不用 Opus，能用内置工具的不用外部工作流。

### 2.4 分离关注点

- **编写和审查**是两个独立步骤，必须在不同时间、由不同 Agent 完成。
- **探索和规划**分开：先了解全貌，再制定方案。
- **修复和验证**形成闭环，直到验证通过。

---

## 3. 安装与启动

### 3.1 安装

在会话中输入：

```
setup omc
```

或直接调用技能：

```
/oh-my-claudecode:omc-setup
```

安装流程会自动检测当前环境（插件/NPM/本地开发），选择合适的安装路径。

### 3.2 调试和修复

如果遇到安装问题：

```
/oh-my-claudecode:omc-doctor
```

### 3.3 停用 OMC

设置环境变量：

```bash
DISABLE_OMC=1          # 完全禁用
OMC_SKIP_HOOKS=hook1,hook2  # 跳过特定 hook
```

### 3.4 取消当前执行模式

```
/oh-my-claudecode:cancel
```

取消 autopilot / ralph / ultrawork / team 等正在运行的模式。

---

## 4. Agent 智能体体系

OMC 提供了 19 个专业 Agent，每个都有明确的职责和推荐的模型。

### 4.1 查询与分析类

| Agent | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `explore` | Haiku | 代码库搜索和映射 | 需要快速找到文件、搜索模式、理解代码结构 |
| `analyst` | Opus | 需求澄清 | 需求模糊时需要厘清隐藏约束和真正目标 |
| `scientist` | Sonnet | 数据分析 | 统计分析、数据处理、科研计算 |

### 4.2 设计规划类

| Agent | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `planner` | Opus | 执行计划排序 | 多步骤任务，需要合理安排执行顺序 |
| `architect` | Opus | 系统架构设计 | 架构级变更、跨模块设计、长远权衡 |
| `designer` | Sonnet | UI/UX 设计 | 前端界面、交互设计 |
| `critic` | Opus | 计划审查 | 对已有的计划或设计提出挑战和审阅 |

### 4.3 执行实现类

| Agent | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `executor` | Sonnet | 代码实现和重构 | 多文件改动、功能实现、重构（复杂工作用 Opus） |
| `code-simplifier` | Opus | 代码简化 | 精简代码，保持行为不变的前提下提升可读性 |
| `debugger` | Sonnet | 故障诊断 | 根因分析、堆栈追踪、编译错误排查 |
| `writer` | Haiku | 文档写作 | 轻量文档、注释、README |

### 4.4 质量保障类

| Agent | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `code-reviewer` | Opus | 全面代码审查 | PR 审查、安全审查、设计审查 |
| `security-reviewer` | Sonnet | 安全漏洞检测 | 检查 OWASP Top 10、密钥泄露、不安全模式 |
| `test-engineer` | Sonnet | 测试策略 | 编写测试、回归覆盖、集成测试 |
| `qa-tester` | Sonnet | 运行时验证 | 手动测试、CLI 交互测试（用 tmux 管理会话） |
| `verifier` | Sonnet | 完成验证 | 确认改动是否真正生效，收集完成证据 |

### 4.5 专项类

| Agent | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `tracer` | Sonnet | 因果追踪 | 假设驱动的问题追踪，收集支持/反对证据 |
| `document-specialist` | Sonnet | 外部文档查询 | 查 SDK/API 文档（仓库文档优先，Context Hub 次之，Web 回退） |
| `git-master` | Sonnet | Git 操作 | 原子化提交、rebase、历史整理 |

### 4.6 使用 Agent 的方式

**直接委托**（在 prompt 中指定）：
```text
让 code-reviewer 审查 src/auth.py
用 architect 设计新的缓存层
```

**通过团队管道**：
```
/team 3:executor "实现用户认证模块"
```

---

## 5. 模型路由策略

OMC 根据任务复杂度自动选择最佳模型：

| 模型 | 适用场景 | 特点 |
|------|---------|------|
| **Haiku** | 快速查询、轻量检查、窄范围文档 | 速度最快，成本最低 |
| **Sonnet** | 标准实现、调试、审查 | 性价比最佳，日常主力 |
| **Opus** | 架构设计、深度分析、共识规划、高风险审查 | 推理能力最强 |

### 委托时的模型选择

```text
# 简单搜索用 Haiku
让 explore 搜索所有 DataLoader 实现

# 标准实现用 Sonnet（默认）
让 executor 修复 data_loader.py 的内存泄漏

# 复杂架构用 Opus
让 architect(model=opus) 设计新的分布式训练方案
```

### 直接写入权限

以下路径可以直接写入，无需委托 Agent：
- `~/.claude/**`
- `.omc/**`
- `.claude/**`
- `CLAUDE.md`
- `AGENTS.md`

---

## 6. Skills 技能体系

Skills 是预构建的工作流。调用方式：`/oh-my-claudecode:<skill-name>`

### 6.1 Tier-0 核心工作流（最常用）

| Skill | 描述 | 触发词 |
|-------|------|--------|
| **autopilot** | 从想法到代码的完全自主执行 | `autopilot` |
| **ralph** | 持久化循环，直到任务完成并通过验证 | `ralph` |
| **ultrawork** | 高吞吐量并行执行 | `ulw` |
| **team** | 协调多个 Agent 并行工作 | N/A（显式调用） |
| **ralplan** | 共识规划工作流 | `ralplan` |

### 6.2 辅助工作流

| Skill | 用途 |
|-------|------|
| `ccg` | Claude + Codex + Gemini 三模型综合 |
| `ultraqa` | QA 循环：测试 → 验证 → 修复 → 重复 |
| `trace` | 证据驱动的因果追踪 |
| `deep-interview` | 苏格拉底式需求访谈（数学级歧义门控） |
| `deepinit` | 层次化 AGENTS.md 文档生成 |
| `sciomc` | 科学/研究分析工作流 |
| `visual-verdict` | 结构化的视觉 QA 判定（截图对比参考） |
| `ai-slop-cleaner` | 清理 AI 生成代码中的冗余和安全回退 |
| `external-context` | 外部文档/研究并行查询 |
| `release` | 通用发布助手 |

### 6.3 实用工具技能

| Skill | 用途 |
|-------|------|
| `skill` | 管理本地技能：列表、添加、删除、搜索 |
| `learner` | 从当前会话提取可复用的技能 |
| `remember` | 检查可复用的项目知识 |
| `note` | 管理 Notepad 笔记 |
| `hud` | 配置 HUD 显示选项 |
| `mcp-setup` | 配置 MCP 服务器 |
| `configure-notifications` | 配置通知集成（Telegram/Discord/Slack） |
| `project-session-manager` | 基于 Worktree 的开发环境管理 |
| `writer-memory` | 写作记忆系统（角色、关系、场景） |

### 6.4 关键词触发

OMC 会从你的消息中自动识别关键词并触发对应工作流：

| 你说的话 | 触发的 Skill |
|---------|-------------|
| "autopilot" | autopilot（全自主执行） |
| "ralph" | ralph（持久化循环） |
| "ulw" | ultrawork（并行执行） |
| "ccg" | ccg（三模型综合） |
| "ralplan" | ralplan（共识规划） |
| "deep interview" | deep-interview（深度访谈） |
| "deslop" / "anti-slop" | ai-slop-cleaner（清理代码） |
| "deep-analyze" | 分析模式 |
| "tdd" | TDD 模式 |
| "deepsearch" | 代码库深度搜索 |
| "ultrathink" | 深度推理模式 |

---

## 7. 团队管道 (Team Pipeline)

当任务跨越多个关注面时，使用 Team Pipeline 进行有结构的编排。

### 7.1 管道阶段

```
team-plan → team-prd → team-exec → team-verify → team-fix (循环)
```

1. **team-plan**: Planner 分析任务，制定执行计划
2. **team-prd**: 明确需求和验收标准
3. **team-exec**: Executor 并行执行具体实现
4. **team-verify**: Verifier 独立验证所有改动
5. **team-fix**: 如验证失败，进入有限次数的修复循环

### 7.2 使用示例

```text
# N 个 Agent 并行执行同一任务
/team 3:executor "实现 data_loader.py 的流式加载"

# 启动 Ralph 风格的团队验证循环
/team ralph

# 跨模型协作
omc team 2:codex "生成测试用例"
omc team 2:gemini "审查 API 设计"

# 三模型综合
/ccg "对比三种数据加载方案"
```

### 7.3 何时使用 Team Pipeline

**适合使用**：
- 多个独立任务可以并行执行
- 任务的协调开销小于并行带来的加速
- 需要不同视角（Claude + Codex + Gemini）综合

**不适合使用**：
- 简单的单文件修改
- 强依赖的任务序列（一个的输出是另一个的输入）
- 调试和根因分析（用 debugger 更合适）

---

## 8. Commit 协议

OMC 使用结构化的 Git 提交消息格式，在每个 commit 中保留决策上下文。

### 8.1 格式

```text
<type>(<scope>): <为什么做这个改动> — 意图先行

<可选的正文，包含上下文和理由>

<结构化 trailers>
```

### 8.2 Trailers 字段

| Trailer | 含义 | 可选值 |
|---------|------|--------|
| `Constraint:` | 影响决策的活跃约束 | 自由文本 |
| `Rejected:` | 被拒绝的替代方案及理由 | 方案 \| 理由 |
| `Directive:` | 前瞻性警告或指示 | 自由文本 |
| `Confidence:` | 改动信心等级 | `high` / `medium` / `low` |
| `Scope-risk:` | 改动影响范围 | `narrow` / `moderate` / `broad` |
| `Not-tested:` | 已知的验证盲区 | 自由文本 |

### 8.3 完整示例

```text
feat(docs): 精简 CLAUDE.md 的 OMC 指令占用

将纯参考性的编排内容移入原生 Claude skill，
保持 session-start 阶段指引简洁的同时，
保证详细 OMC 参考随时可查。

Constraint: 保持基于 marker 的 CLAUDE.md 安装流程
Rejected: 在旧安装流程中同步全部内置 skill | 影响范围超出当前需求
Confidence: high
Scope-risk: narrow
Not-tested: 全新 Claude profile 下的端到端插件市场安装
```

### 8.4 与常规 Commit 的区别

传统 commit 只记录"做了什么"，OMC commit **额外记录**：
- 当时面临的约束是什么
- 考虑过但放弃的方案及原因
- 对未来维护者的注意事项
- 已知的测试覆盖盲区

这样，半年后回头看这个 commit，你能理解当时的**决策全景**，而不只是代码变更。

---

## 9. 状态与记忆系统

OMC 提供多层次的持久化机制，解决"AI 会话失忆"问题。

### 9.1 五个持久化层次

```
        持久性递增
Project Memory  ←→  Notepad  ←→  State  ←→  Session  ←→  Memory
  (永久)            (工作)        (会话)      (会话)        (跨会话)
```

### 9.2 Project Memory（项目记忆）

存储在 `.omc/project-memory.json`，跨会话持久化。适合保存：
- 项目约定和约束
- 关键决策记录
- 架构决策记录 (ADR)

操作：
```text
记住：所有 API 必须加 rate limiting
记下：周五之前冻结合并，下周三发布
```

对应的 MCP 工具：`project_memory_read`、`project_memory_write`、`project_memory_add_note`、`project_memory_add_directive`

### 9.3 Notepad（记事本）

存储在 `.omc/notepad.md`，用于会话内的优先级工作记忆。三层写入：

| 层级 | 用途 |
|------|------|
| `notepad_write_priority` | 高优先级事项 |
| `notepad_write_working` | 当前工作追踪 |
| `notepad_write_manual` | 自由记录 |

### 9.4 State（运行状态）

存储在 `.omc/state/` 和 `.omc/state/sessions/{sessionId}/`，追踪：
- 活跃的 Agent
- 当前执行模式（autopilot/ralph/ultrawork）
- Hook 状态
- HUD 配置

### 9.5 跨会话 Memory（Auto Memory）

存储在 `~/.claude/projects/` 下，用于跨项目、跨会话的用户级别记忆：
- 用户偏好和习惯
- 项目间共用的配置知识
- 长期积累的工作经验

### 9.6 持久化指令

```text
<remember>           # 普通记忆，保留 7 天
<remember priority>  # 永久记忆，不会过期
```

---

## 10. 工具与代码智能

### 10.1 外部 AI / 编排工具

```text
/team N:agent "task"          # 启动 N 个 Agent 的团队
omc team N:codex|gemini "..." # 跨模型团队
omc ask <claude|codex|gemini>  # 向特定模型提问
/ccg                          # Claude + Codex + Gemini 综合
```

### 10.2 代码智能 (LSP)

OMC 集成了完整的 LSP 客户端，提供 IDE 级别的代码分析能力：

| 工具 | 功能 |
|------|------|
| `lsp_hover` | 类型信息悬停 |
| `lsp_goto_definition` | 跳转到定义 |
| `lsp_find_references` | 查找所有引用 |
| `lsp_diagnostics` | 获取诊断信息 |
| `lsp_diagnostics_directory` | 目录级别诊断 |
| `lsp_document_symbols` | 文档符号列表 |
| `lsp_workspace_symbols` | 工作区符号搜索 |
| `lsp_code_actions` | 代码操作建议 |
| `lsp_rename` | 重命名符号 |

### 10.3 AST 工具

```text
ast_grep_search  # 结构化代码搜索（比正则更精确）
ast_grep_replace # 结构化代码替换
```

### 10.4 Python REPL

```text
python_repl  # 交互式 Python 环境，快速验证想法
```

---

## 11. Hook 钩子系统

Hook 是 OMC 的事件响应机制，可以在特定时机执行自定义操作。

### 11.1 常见 Hook 事件

- `SessionStart` — 会话开始时触发（CLAUDE.md 从此载入）
- `PreToolUse` — 工具调用前触发
- `PostToolUse` — 工具调用后触发
- `Notification` — 通知事件
- `Stop` — 会话结束时触发
- `UserPromptSubmit` — 用户提交 prompt 时触发

### 11.2 Hook 注入

Hook 通过 `<system-reminder>` 标签向会话注入信息：

```
hook success: Success     # 操作成功
[MAGIC KEYWORD: ...]      # 触发技能
The boulder never stops   # ralph/ultrawork 活跃中
```

### 11.3 关键提示

- Hook 在 `settings.json` 中配置，不是通过 Memory 或 Prompt 指令
- 权限管理（"允许 X 命令"、"添加 X 权限"）需要通过 `update-config` skill 来配置
- 如果某个 Hook 行为异常，可以用 `OMC_SKIP_HOOKS=hook_name` 跳过

---

## 12. 执行协议与最佳实践

### 12.1 黄金法则

1. **探索再规划**：面对宽泛的需求，先用 `explore` 了解现状，再用 `planner` 制定方案。
2. **独立任务并行**：2 个以上互不依赖的任务，同时启动。
3. **后台运行长时间任务**：构建、测试用 `run_in_background`，不阻塞会话。
4. **编写和审查分离**：编码用一个 Agent，审查用另一个 Agent，绝不自审自批。
5. **完成前验证**：声明完成前必须收集验证证据，测试通过。

### 12.2 委托决策树

```
需要做什么？
├── 读/搜索文件 → 直接 grep / find / explore Agent
├── 理解需求 → analyst Agent
├── 设计方案 → architect Agent（复杂）/ planner Agent（排序）
├── 写代码 →
│   ├── 单文件小改动 → 直接写
│   ├── 多文件重构 → executor Agent
│   ├── 简化现有代码 → code-simplifier Agent
│   └── 架构级改动 → architect 先设计，executor 后执行
├── 调试 → debugger Agent
├── 审查 →
│   ├── 普通审查 → code-reviewer Agent
│   └── 安全审查 → security-reviewer Agent
├── 测试 → test-engineer Agent
├── 查外部文档 → document-specialist Agent
└── 提交代码 → git-master Agent
```

### 12.3 反模式

| 反模式 | 正确做法 |
|--------|---------|
| 不探索直接写代码 | 先 `grep` / `find` 了解现状 |
| 自己写代码自己审 | 用 `code-reviewer` 做独立审查 |
| 不验证就说完成 | 用 `verifier` 收集完成证据 |
| 小任务开大 Agent | 够用就好，优先 Haiku/Sonnet |
| 串行执行独立任务 | 并行启动多个 Agent |
| 猜测路径和函数名 | `grep` 或 `find` 确认 |

### 12.4 处理失败的策略

- **验证失败**：收集失败证据 → 分析根因 → 修复 → 再次验证（循环）
- **Agent 阻塞**：检查状态、日志 → 必要时 `cancel` 重启
- **Hook 问题**：用 `OMC_SKIP_HOOKS` 隔离 → `omc-doctor` 诊断
- **安装问题**：`omc-setup` 重新安装 → `omc-doctor` 诊断报告

---

## 13. 配置与定制

### 13.1 核心配置文件

| 文件 | 位置 | 用途 |
|------|------|------|
| `CLAUDE.md` | 项目根目录 | 项目指令和 AI 行为准则 |
| `.claude/CLAUDE.md` | 项目 `.claude` 目录 | OMC 编排层指令（自动生成） |
| `.claude/settings.json` | 项目全局设置 | 权限、环境变量、Hook |
| `.claude/settings.local.json` | 项目本地设置 | 覆盖全局设置的本地配置 |
| `.claude/mcp.json` | MCP 配置 | MCP 服务器定义 |

### 13.2 配置命令

```text
# 修改设置（权限、环境变量、Hook 等）
/oh-my-claudecode:update-config

# 配置 MCP 服务器
/oh-my-claudecode:mcp-setup

# 管理本地 Skills
/oh-my-claudecode:skill

# 配置通知
/oh-my-claudecode:configure-notifications
```

### 13.3 permissions 权限配置示例

```json
{
  "permissions": {
    "allow": [
      "WebSearch",
      "Bash(git push:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(python -m pytest tests/ -v --timeout=60)"
    ],
    "deny": [
      "Bash(rm -rf:*)"
    ]
  }
}
```

### 13.4 MCP 服务器配置示例

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_package.mcp_server"],
      "cwd": "/path/to/project",
      "env": {
        "API_KEY": "xxx"
      }
    }
  }
}
```

---

## 14. 常用命令速查表

### 工作流操作

| 命令 | 说明 |
|------|------|
| `setup omc` | 安装 OMC |
| `/oh-my-claudecode:omc-setup` | 安装/更新 OMC |
| `/oh-my-claudecode:omc-doctor` | 诊断 OMC 问题 |
| `/oh-my-claudecode:cancel` | 取消当前执行模式 |
| `/oh-my-claudecode:hud` | 配置 HUD 显示 |

### 自主工作流

| 命令/触发词 | 说明 |
|------------|------|
| `autopilot` | 全自主执行，从想法到代码 |
| `ralph` | 持久化循环直到完成 |
| `ulw` | 高吞吐量并行执行 |
| `/team N:agent "task"` | 启动 N 个 Agent 的并行团队 |
| `/ccg` | Claude + Codex + Gemini 三模型综合 |

### 技能操作

| 命令 | 说明 |
|------|------|
| `/oh-my-claudecode:skill` | 管理技能列表 |
| `/oh-my-claudecode:learner` | 从会话中提取技能 |
| `/oh-my-claudecode:remember` | 检查可复用的知识 |
| `/oh-my-claudecode:note` | 管理 Notepad |

### 代码质量

| 命令/触发词 | 说明 |
|------------|------|
| `deslop` / `anti-slop` | 清理 AI 冗余代码 |
| `tdd` | TDD 测试驱动开发模式 |
| `/oh-my-claudecode:release` | 发布流程 |

### 跨模型操作

| 命令 | 说明 |
|------|------|
| `omc ask claude` | 向 Claude 提问 |
| `omc ask codex` | 向 Codex 提问 |
| `omc ask gemini` | 向 Gemini 提问 |
| `omc team N:codex "..."` | Codex 团队 |
| `omc team N:gemini "..."` | Gemini 团队 |

---

## 总结

OMC 的核心价值在于：

1. **把 AI 从"一个聪明但偶尔犯错的助手"变成了"一个有分工、有审查、有验证的工程团队"**
2. **通过 Agent 分工、模型路由、验证闭环三大机制，显著提高代码质量和开发效率**
3. **结构化的 Commit 协议和持久化记忆系统，让 AI 辅助编程不止于当前会话**

一句话概括：**人类做决策，Claude 做协调，Agent 团队做执行。各司其职，有据可查，闭环验证。**
