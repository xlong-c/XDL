# XDL MD 详细版事实源

先看低负担入口时,优先打开 [index.md](index.md) 或 [README_SUMMARY.md](README_SUMMARY.md). 本文保留为 `XDL` 的兼容详细版事实源,负责完整章节正文,具体字段,边界,兼容规则和检查清单.

## 第一规则

- `docs/md/` 是给 agents 和开发者写代码前看的工作文档.
- 源码和本目录中的 MD 定义事实边界;`research/` 下的 HTML 调研页只做阅读层,不定义契约.
- 行为,字段,API 或兼容承诺变化时,先改本目录对应 MD,再同步引用同一主题的 HTML 页面.

研究笔记,阶段性分析和一次性草案应放在 `research/`,不在 `docs/md/` 堆叠.

## 给 agents 写代码看的 MD

新的 Markdown 主导航已经迁到 [index.md](index.md), 并按架构 / 说明 / 使用三层组织. 本文继续保留完整章节正文,供旧链接和高密度阅读使用.

下面这些主题直接链接到本文章节.

| 顺序 | 章节 | 负责内容 |
| --- | --- | --- |
| 1 | [XDL 安装与验证](#xdl-安装与验证) | 安装,验证,当前可运行入口. |
| 2 | [XDL 项目结构与使用说明](#xdl-项目结构与使用说明) | 框架定位,核心分层,训练入口,扩展方式. |
| 3 | [XDL Config 系统说明](#xdl-config-系统说明) | `setup_from_yaml()`,schema v1,`target + params`,YAML 组织方式. |
| 4 | [XDL Dataset 模板规划](#xdl-dataset-模板规划) | 数据集语义形态,磁盘组织形态,内置模板和新增 dataset 流程. |
| 5 | [XDL API 稳定边界](#xdl-api-稳定边界) | 稳定公共 API,实验性 API,内部实现边界和废弃策略. |
| 6 | [XDL 写作标点规范](#xdl-写作标点规范) | 中文文档和注释的半角标点要求,以及自动归一化工具. |
| 7 | [XDL HTML 阅读页样式规范](#xdl-html-阅读页样式规范) | 自有 HTML 阅读页统一样式,主题 token,色彩和交互规范. |
| 8 | [XDL 模块功能边界速查](#xdl-模块功能边界速查) | 各源码子模块职责速查. |
| 9 | [XDL 当前优化方向](#xdl-当前优化方向) | 当前仍有效的后续优化方向. |

## 改动前必读

| 改动范围 | 先读 | 同步要求 |
| --- | --- | --- |
| 公开 API,导出符号,兼容策略 | [XDL API 稳定边界](#xdl-api-稳定边界) | 行为变化必须更新 Stable / Provisional / Internal 边界. |
| 训练生命周期,`CoreModel`,`Trainer`,Callback | [XDL 项目结构与使用说明](#xdl-项目结构与使用说明),[../../xdl/trainer/README.md](../../xdl/trainer/README.md) | 同步说明手动优化,batch 迁移,日志和回调顺序. |
| YAML 配置,schema,`target + params` | [XDL Config 系统说明](#xdl-config-系统说明) | 同步字段,示例和配置主链路. |
| 数据集模板,collate,manifest | [XDL Dataset 模板规划](#xdl-dataset-模板规划) | 同步模板选择表,注册名和测试要求. |
| 安装,依赖,wheel,运行入口 | [XDL 安装与验证](#xdl-安装与验证),[../../xdl/USAGE.md](../../xdl/USAGE.md) | 同步安装命令和包内用法入口. |
| 子模块职责,目录迁移 | [XDL 模块功能边界速查](#xdl-模块功能边界速查) | 同步目录职责和依赖方向. |
| 文档结构,索引,长期文档边界 | 当前文件,[../AGENTS.md](../AGENTS.md) | 保持 `docs/md/` 事实源与 `research/` HTML 阅读页分层清楚. |
| 中文文档,注释或研究草案 | [XDL 写作标点规范](#xdl-写作标点规范) | 对本次改动文件运行 `scripts/normalize_punctuation.py`,不要一次性重写大量历史文档. |
| HTML 调研页样式,公共 CSS,主题交互 | [architecture/html-style-policy.md](architecture/html-style-policy.md) | 同步 `../html/assets/` 和 `research/` 页面引用. |

## XDL 知识图谱使用约定

- 本仓库在 `codebase-memory-mcp` 中的项目名固定为 `root-workspace-xdl`.
- 使用 `search_graph`, `trace_path`, `get_code_snippet`, `query_graph`, `get_architecture`, `detect_changes`, `index_status` 等 MCP 图谱工具时, `project` 参数统一传 `root-workspace-xdl`,不要写成 `xdl`.
- 需要重建索引时,对仓库根目录 `/root/workspace/xdl` 运行 `index_repository`,生成的项目仍应视为同一个 `root-workspace-xdl`.
- 完成新增/删除模块,公开符号重命名,模块调用关系调整等结构性改动后,应重建或刷新该项目索引,保持知识图谱和代码一致.

## 文档边界

- [XDL 安装与验证](#xdl-安装与验证) 只讲环境,安装和验证.
- [XDL 项目结构与使用说明](#xdl-项目结构与使用说明) 只讲框架结构,训练入口和扩展方式.
- [XDL Config 系统说明](#xdl-config-系统说明) 只讲配置系统,不展开 dataset 全量模板规划.
- [XDL Dataset 模板规划](#xdl-dataset-模板规划) 只讲 dataset 模板规划,选择和扩展方式.
- [XDL API 稳定边界](#xdl-api-稳定边界) 只讲公共 API 兼容边界,不重复使用教程.
- [XDL HTML 阅读页样式规范](#xdl-html-阅读页样式规范) 只讲 `research/` 自有 HTML 调研页的视觉系统和样式维护规则,不定义框架行为.
- [XDL 模块功能边界速查](#xdl-模块功能边界速查) 只做模块职责速查,不重复写长篇使用指南.
- [XDL 当前优化方向](#xdl-当前优化方向) 只保留仍然有效的待办,不复述现状说明.

## XDL 写作标点规范

- 文档和注释默认使用半角英文标点,例如 `,`, `.`, `:`, `;`, `?`, `!`, `()`, `[]`, 引号和路径分隔符.
- 写中文文档时不要求手动逐字检查标点,但提交前应对本次改动文件运行归一化工具.
- 只检查不写入:

```bash
XDL_PUNCT_CHECK=1 XDL_PUNCT_PATHS=docs/md/README.md python scripts/normalize_punctuation.py
```

- 自动替换明确列出的全角标点:

```bash
XDL_PUNCT_PATHS=docs/md/README.md python scripts/normalize_punctuation.py
```

- 多文件或目录可用逗号分隔:

```bash
XDL_PUNCT_PATHS=docs/md,research python scripts/normalize_punctuation.py
```

- 工具使用固定映射,不使用 Unicode NFKC,避免误改中文正文,全角数字,单位,数学符号或模型名. 默认扫描 `docs,research`,也可通过 `XDL_PUNCT_EXTS=.md,.txt` 限制后缀.

## HTML 同步规则

- `research/` 下的 HTML 调研页只做阅读层,不是事实源.
- HTML 可以重排,提炼和图文化 MD 内容,但不要引入和 MD 或源码冲突的新事实.
- 同一主题的行为,字段或 API 发生变化时,先更新对应 MD,再同步覆盖同一主题的 HTML 页面.
- 新增或重构自有 HTML 时,body 必须且只能包含 `xdl-style-atlas` 或 `xdl-style-ledger` 两种模板之一,默认引用 `../html/assets/xdl-doc.css`;需要交互式主题切换时再引用 `../html/assets/xdl-theme.js`.
- 公共 CSS 只能收敛到 `../html/assets/xdl-doc.css`;目录专属 CSS 只能作为薄入口和局部组件扩展,不能复制公共版式,主题变量或通用阅读组件.
- 新增 CSS 文件前先读 [XDL HTML 阅读页样式规范](#xdl-html-阅读页样式规范) 的 CSS 收拢边界,并确认现有入口无法承载.


## 正文收拢说明

本文件现在是 `XDL` 的兼容详细版事实源. 新的 canonical 页面已经开始迁到三层目录:

- 安装与验证: [usage/xdl-install-and-verify.md](usage/xdl-install-and-verify.md)
- 架构与模块边界: [architecture/xdl.md](architecture/xdl.md)
- API 稳定边界: [architecture/api-boundary.md](architecture/api-boundary.md)
- 概念理解: [explanation/xdl-concepts.md](explanation/xdl-concepts.md)
- 工作流与交接: [usage/xdl-workflows.md](usage/xdl-workflows.md), [usage/xdl-agent-handoff.md](usage/xdl-agent-handoff.md)
- dataset 概念说明: [explanation/dataset-structure.md](explanation/dataset-structure.md)
- HTML 分层说明: [explanation/html-style.md](explanation/html-style.md)

本文继续保留完整章节,用于兼容旧锚点,旧链接和高密度集中阅读. 后续新增长期主题,优先直接落到三层目录,不要继续堆回本文件.

## XDL 安装与验证

本章节保留为兼容锚点. `XDL` 安装方式, 验证命令和当前运行入口的 canonical 页面已经迁到:

- [usage/xdl-install-and-verify.md](usage/xdl-install-and-verify.md)

这里继续保留章节名, 是为了兼容旧链接和旧锚点. 如果你要更新环境要求, 安装命令, 验证方法或运行入口, 优先更新上面的使用层文档, 再视需要同步本页.

## XDL 项目结构与使用说明

本章节保留为兼容锚点. `XDL` 的系统定位, 核心分层, 生命周期, 模块边界和训练工作流已经拆到:

- [architecture/xdl.md](architecture/xdl.md)
- [explanation/xdl-concepts.md](explanation/xdl-concepts.md)
- [usage/xdl-workflows.md](usage/xdl-workflows.md)

如果你要更新框架定位, `Registry` / `CoreModel` / `Trainer` 语义, 推荐接入路径或扩展方式, 优先更新这些 canonical 页面, 再视需要同步本页.

## XDL Config 系统说明

本章节保留为兼容锚点. `XDL` 配置系统的 canonical 正文已经迁到:

- [usage/xdl-config-workflows.md](usage/xdl-config-workflows.md)

如果你要更新 `setup_from_yaml()` 主链路, `schema v1`, `target + params`, dataset / dataloader / collate 的配置关系, 优先更新上面的使用层文档, 再视需要同步本页.

相关配套页面:

- [architecture/api-boundary.md](architecture/api-boundary.md): 公开配置入口与稳定边界
- [explanation/dataset-structure.md](explanation/dataset-structure.md): dataset 结构理解与模板选择背景
- [XDL Dataset 模板规划](#xdl-dataset-模板规划): 仍保留在本页的 dataset 详细规划与扩展规则

## XDL Dataset 模板规划

本章节保留为兼容锚点. `XDL` dataset 模板选择, 磁盘组织方式和新增规则的 canonical 页面已经迁到:

- [architecture/dataset-policy.md](architecture/dataset-policy.md)
- [explanation/dataset-structure.md](explanation/dataset-structure.md)
- [usage/xdl-config-workflows.md](usage/xdl-config-workflows.md)

如果你要更新 dataset 模板选择, manifest / folder / sidecar 的推荐顺序, `collate` 接入边界或新增 dataset 的落地规则, 优先更新上面的架构层, 说明层和使用层页面, 再视需要同步本页.

## XDL API 稳定边界

本章节保留为兼容锚点. `XDL` Stable / Provisional / Internal 边界, 废弃策略和版本规则的 canonical 页面已经迁到:

- [architecture/api-boundary.md](architecture/api-boundary.md)

如果你要更新公开 API 承诺, 导出入口, 兼容路径或废弃策略, 优先更新上面的架构层文档, 再视需要同步本页.

### 版本规则

建议按语义化版本管理:

- Patch:bugfix,文档,测试,非破坏性兼容修复.
- Minor:新增公共能力,扩展配置字段,增加组件.
- Major:删除或破坏 Stable API.

当前 `0.x` 阶段仍允许调整设计,但应避免无提示破坏本文档列出的 Stable API.

## XDL HTML 阅读页样式规范

本章节保留为兼容锚点. 自有 HTML 教程与调研页的长期规范已经迁到:

- [architecture/html-style-policy.md](architecture/html-style-policy.md)
- [explanation/html-style.md](explanation/html-style.md)

如果你要更新 HTML 与 Markdown 的分工, 公共 CSS / 主题脚本边界, body 模板, token, 可访问性约束或新增页面检查清单, 优先更新上面的架构层与说明层页面, 再视需要同步本页.

## XDL 模块功能边界速查

本章节保留为兼容锚点. `XDL` 子模块职责, 依赖方向和高影响结构改动的 canonical 页面已经迁到:

- [architecture/module-boundaries.md](architecture/module-boundaries.md)

如果你要更新子模块职责, 目录迁移, 依赖方向或结构性改动判断标准, 优先更新上面的架构层文档, 再视需要同步本页.

## XDL 当前优化方向

本章节保留为兼容锚点. `XDL` 当前仍有效的优化方向已经迁到:

- [architecture/roadmap.md](architecture/roadmap.md)

如果你要更新当前仍然成立的改进项, 中短期方向或文档治理后续动作, 优先更新上面的架构层文档, 再视需要同步本页.
