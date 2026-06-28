# XDL / XQT 文档三层重构计划

> 本计划只覆盖 `XDL` 与 `XQT` 相关文档. 不处理 `learn/`, `research/`, `tools/`, `examples/` 等其他目录的文档治理.

## 目标

在不立即大规模改写正文的前提下,先把 `XDL` 与 `XQT` 的文档结构理顺,为后续重构建立稳定边界. 本计划聚焦三类正文:

- 架构文档: 定义系统边界, 模块关系, 生命周期, API 稳定性和禁止项.
- 说明文档: 解释概念, 术语, 理解路径和常见误区.
- 使用文档: 提供安装, 命令, workflow, 验证方式和 handoff 说明.

同时保留一个很薄的 agent 入口文档,只负责规则和导航,不承载正文事实源.

## 范围

### 覆盖范围

- `docs/md/README.md`
- `docs/md/README_SUMMARY.md`
- `docs/md/XQT.md`
- `docs/md/XQT_SUMMARY.md`
- `docs/html/index.html`
- `docs/html/xqt.html`
- `xdl/USAGE.md`
- `xqt/README.md`
- `xqt/FRAMEWORK.md`

### 不在本计划内

- `docs/html/style-showcase.html` 的视觉系统设计本身
- `docs/html/dataset-structure.html` 的页面重绘
- `learn/`, `research/`, `tools/`, `examples/`, `infer/` 等目录的文档治理
- 子目录 `AGENTS.md` / `CLAUDE.md` 的全面重组

这些内容如果后续需要调整,应单独立项,不要并入本次 XDL / XQT 文档重构.

## 现状判断

当前文档已经具备雏形,但职责混杂:

1. `docs/md/README.md` 同时承担架构, 说明, 使用, 写作规范和 HTML 规范, 已经是过重单体事实源.
2. `docs/md/README_SUMMARY.md` 与 `docs/html/index.html` 都在做入口和阅读地图, 但一个偏 agent, 一个偏人类, 结构尚未按三层模型统一.
3. `xdl/USAGE.md` 是 agent 使用入口, 但更像 handoff / 工作约束, 还没有纳入完整的使用文档体系.
4. `docs/md/XQT.md`, `xqt/README.md`, `xqt/FRAMEWORK.md` 已经天然分成"长期事实源 + 薄入口 + 工程契约", 但命名和引用层次还不够明确.
5. HTML 页面当前表达的是"MD / HTML 两层", 后续应改为"架构 / 说明 / 使用三层, HTML 主要承载说明层阅读体验".

## 目标结构

重构后, `docs/md/` 下的 XDL / XQT 正文按三层组织:

```text
docs/md/
  index.md
  architecture/
    index.md
    xdl.md
    xqt.md
    api-boundary.md
  explanation/
    index.md
    xdl-concepts.md
    xqt-concepts.md
    dataset-structure.md
    html-style.md
  usage/
    index.md
    xdl-install-and-verify.md
    xdl-workflows.md
    xdl-agent-handoff.md
    xqt-workflows.md
```

说明:

- `architecture/` 是唯一长期事实源主体.
- `explanation/` 是解释层, 帮人和 agent 建立认知, 不单独定义契约.
- `usage/` 是可执行层, 写到可以直接照着操作和交接.
- `docs/md/index.md` 是新的总入口, 统一取代"所有正文都收拢在 README.md"这种模式.

## 三类文档职责

### 1. 架构文档

负责回答:

- 系统是什么, 不是什么
- 模块怎么分, 为什么这样分
- 生命周期和依赖方向是什么
- 哪些 API 稳定, 哪些只是 provisional
- 哪些事情明确不要做

不负责:

- 安装命令
- 完整操作手册
- 长 FAQ
- 视觉展示

### 2. 说明文档

负责回答:

- 核心概念是什么
- 名词怎么理解
- 为什么有这套设计
- 与相邻概念的区别是什么
- 常见误区是什么

不负责:

- 定义新的行为契约
- 放完整命令细节
- 承载兼容承诺

### 3. 使用文档

负责回答:

- 如何安装和验证
- 如何完成常见工作流
- 输入输出和产物在哪里
- 如何判断成功或失败
- 如何把结果交给下一个人或 agent

不负责:

- 重新解释整套架构
- 长篇设计背景
- 独立定义模块边界

## 引用关系

引用方向固定如下:

```text
AGENTS.md / docs/AGENTS.md
  -> docs/md/index.md

docs/md/index.md
  -> architecture/index.md
  -> explanation/index.md
  -> usage/index.md

architecture/*
  -> 可引用 explanation/* 解释术语
  -> 可引用 usage/* 指向实际操作
  -> 不复制具体命令和排障细节

explanation/*
  -> 引用 architecture/* 作为事实边界
  -> 引用 usage/* 作为下一步入口
  -> 不单独发明新契约

usage/*
  -> 引用 architecture/* 说明约束来源
  -> 引用 explanation/* 解释必要概念
  -> 以步骤, 命令, 验证为主
```

规则:

- 同一事实只能有一个 canonical 页面.
- 其他页面只摘要并链接过去, 不复制大段正文.
- 如果一个主题既要解释概念又要给命令, 则拆成 explanation + usage 两页, 不在单页混写.

## 现有文件映射

### XDL

| 现有文件 | 后续角色 | 处理方式 |
| --- | --- | --- |
| `docs/md/README.md` | 待拆分的旧总文档 | 拆入三层目录, 旧文件保留兼容跳转页 |
| `docs/md/README_SUMMARY.md` | 旧摘要入口 | 内容并入 `docs/md/index.md` 或 `explanation/index.md`, 旧文件保留跳转 |
| `xdl/USAGE.md` | 使用层 / handoff | 归并到 `usage/xdl-agent-handoff.md`, 保留包内薄入口语义 |
| `docs/html/index.html` | 说明层阅读入口 | 调整为三层地图的 HTML 阅读版 |

### XQT

| 现有文件 | 后续角色 | 处理方式 |
| --- | --- | --- |
| `docs/md/XQT.md` | XQT 架构事实源 | 迁到 `architecture/xqt.md` 或保留原路径作为兼容壳 |
| `docs/md/XQT_SUMMARY.md` | XQT 说明层入口 | 并入 `explanation/xqt-concepts.md` 或 `docs/md/index.md` |
| `xqt/README.md` | XQT 使用层薄入口 | 归并到 `usage/xqt-workflows.md`, 原文件保留薄入口 |
| `xqt/FRAMEWORK.md` | XQT 工程契约 / 架构补充 | 并入 `architecture/xqt.md` 的工程边界章节, 或保持独立但作为架构层引用 |
| `docs/html/xqt.html` | XQT 说明层阅读页 | 继续保留, 但其导航目标改为新三层结构 |

## 分阶段实施顺序

### Phase 0: 结构先行

先建立新目录和新入口, 不立即搬空旧文档:

1. 新建 `docs/md/index.md`
2. 新建 `docs/md/architecture/index.md`
3. 新建 `docs/md/explanation/index.md`
4. 新建 `docs/md/usage/index.md`

目标:

- 先把新信息架构稳定下来
- 让后续拆分有明确落点
- 避免一次性重写导致链接大面积失效

### Phase 1: XDL 正文拆分

从 `docs/md/README.md` 拆出:

- `architecture/xdl.md`
  - 项目定位
  - 核心分层
  - `Registry` / `CoreModel` / `Trainer`
  - 模块功能边界
- `architecture/api-boundary.md`
  - Stable / Provisional / Internal
  - 废弃策略
  - 版本规则
- `explanation/xdl-concepts.md`
  - 配置系统认知
  - dataset 模板理解
  - HTML 文档分层理解
- `usage/xdl-install-and-verify.md`
  - 安装
  - 环境要求
  - 验证命令
- `usage/xdl-workflows.md`
  - 纯代码路径
  - YAML 路径
  - 扩展模型 / 数据集 / callback / config
- `usage/xdl-agent-handoff.md`
  - 从 `xdl/USAGE.md` 迁入的 agent / handoff 约束

### Phase 2: XQT 正文拆分

以 `docs/md/XQT.md`, `xqt/README.md`, `xqt/FRAMEWORK.md` 为基础:

- `architecture/xqt.md`
  - 项目定位
  - 核心边界
  - 配置方式
  - Stage 约定
  - profiling 约定
- `explanation/xqt-concepts.md`
  - XQT 做什么, 不做什么
  - 各能力块的理解地图
- `usage/xqt-workflows.md`
  - session 入口
  - YAML workflow 入口
  - readiness / report / artifact 使用路径

### Phase 3: 兼容层与阅读层同步

完成拆分后:

1. 旧 `docs/md/README.md` 改为兼容入口页
2. 旧 `docs/md/README_SUMMARY.md` 改为兼容入口页
3. 旧 `docs/md/XQT.md` 和 `docs/md/XQT_SUMMARY.md` 根据需要改为跳转 / 摘要壳
4. 更新 `docs/html/index.html`
5. 更新 `docs/html/xqt.html`
6. 更新 `docs/AGENTS.md` 中的文档组织说明

## 文件模板

### 架构文档模板

```md
# 文档标题

## 负责什么
## 不负责什么
## 核心边界
## 模块关系
## 生命周期 / 数据流
## 扩展点
## 禁止项
## 变更检查清单
```

### 说明文档模板

```md
# 文档标题

## 这是什么
## 为什么需要
## 核心概念
## 与相近概念的区别
## 常见误区
## 继续阅读
```

### 使用文档模板

```md
# 文档标题

## 适用场景
## 前置条件
## 最短路径
## 常用命令
## 输入与输出
## 验证方式
## 常见失败
## 交付清单
```

## 验收标准

在开始实际重构前, 本计划的验收标准是:

1. 已经明确只覆盖 `XDL` 与 `XQT` 文档.
2. 已经明确三类正文的职责边界.
3. 已经明确新目录结构和文件落点.
4. 已经明确旧文件到新结构的映射关系.
5. 已经明确分阶段迁移顺序.
6. 已经明确兼容策略, 不会在第一轮就破坏现有链接.

在进入实施阶段后, 还需要额外满足:

1. 新入口文件全部创建完成.
2. 旧总文档不再继续膨胀为单体事实源.
3. HTML 阅读页完成三层导航同步.
4. `docs/AGENTS.md` 与根 `AGENTS.md` 的文档导航同步更新.
5. 涉及结构性变化时, 运行 `index_repository` 刷新知识图谱.

## 风险与控制

### 风险 1: 一次性大搬迁导致链接失效

控制:

- 先建新入口, 后迁正文
- 旧文件先保留兼容页
- 每一轮只迁一个主题块

### 风险 2: 说明层和使用层再次混写

控制:

- 每个新文件开头固定写"负责什么 / 不负责什么"
- 同一主题拆成概念页和工作流页

### 风险 3: XQT 文档被 XDL 文档风格吞没

控制:

- 保留 XQT 的"模型侧工具链"边界
- XQT 继续独立维护自己的架构与 workflow 页面, 不并回 XDL 总文档

## 下一步

本计划确认后, 下一轮实施应只做第一步:

1. 创建 `docs/md/index.md`
2. 创建 `docs/md/architecture/index.md`
3. 创建 `docs/md/explanation/index.md`
4. 创建 `docs/md/usage/index.md`
5. 更新 `docs/AGENTS.md` 的目录说明, 让它认识新结构

这一轮不做正文大迁移, 只把骨架搭起来.
