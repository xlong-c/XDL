# docs - 仓库文档目录

## 第一规则

- `docs/md/` 是给 agents 和开发者写代码前看的工作文档.
- `docs/html/` 是给人类用户阅读的可视化文档.
- 行为,字段,API,兼容承诺和目录边界的事实源是源码和 `docs/md/`.
- HTML 可以提炼,重排和图文化 MD 内容,但不能单独定义新契约.
- 同一主题发生变化时,先更新 `docs/md/` 中的事实源,再同步 `docs/html/` 中的阅读页.

## 目录职责

- `md/`: 长期维护的 Markdown 工作文档. 面向 agents,强调入口,边界,约束,检查清单和源码关联.
- `html/`: 长期维护的 HTML 阅读页. 面向人类读者,强调阅读路线,概念理解,视觉层次和交互体验.
- `AGENTS.md`: 当前文件,只保存目录级执行规则.
- `CLAUDE.md`: 兼容指针,不承载独立内容.

`docs/` 根目录不新增普通长期文档. 新增 MD 放入 `docs/md/`,新增 HTML 放入 `docs/html/`.

`docs/md/` 内部按三层组织 `XDL` 与 `XQT` 长期正文:

- `md/index.md`: Markdown 总入口,只负责导航和阅读顺序.
- `md/architecture/`: 架构层,定义边界,模块关系,生命周期和兼容规则.
- `md/explanation/`: 说明层,解释概念,术语和理解路径.
- `md/usage/`: 使用层,提供安装,workflow,验证和 handoff 说明.

## 当前内容

### `docs/md/`

- `index.md`: XDL / XQT Markdown 总入口,说明三层结构和当前兼容入口.
- `architecture/index.md`: 架构文档入口.
- `architecture/xdl.md`: XDL 架构正文,承接定位,分层,生命周期和模块边界.
- `architecture/dataset-policy.md`: XDL dataset 模板选择与新增规则.
- `architecture/module-boundaries.md`: XDL 子模块职责与依赖方向.
- `architecture/api-boundary.md`: XDL API 稳定边界与废弃策略.
- `architecture/html-style-policy.md`: XDL 自有 HTML 阅读页长期规范.
- `architecture/roadmap.md`: XDL 当前仍有效的优化方向.
- `architecture/xqt.md`: XQT 架构正文,承接模型侧边界,Stage 约定和 profiling 边界.
- `explanation/index.md`: 说明文档入口.
- `explanation/xdl-concepts.md`: XDL 概念说明与理解路径.
- `explanation/xqt-concepts.md`: XQT 概念说明与能力理解地图.
- `usage/index.md`: 使用文档入口.
- `usage/xdl-install-and-verify.md`: XDL 安装与验证.
- `usage/xdl-config-workflows.md`: XDL 配置工作流入口.
- `usage/xdl-workflows.md`: XDL 工作流入口.
- `usage/xdl-agent-handoff.md`: XDL agent / handoff 落点.
- `usage/xqt-workflows.md`: XQT 工作流入口.
- `README_SUMMARY.md`: XDL 的兼容摘要入口,用于渐进式披露和阅读路线引导.
- `README.md`: XDL 兼容详细版事实源,保留完整章节正文和旧链接承接.
- `XQT_SUMMARY.md`: `xqt/` 的兼容摘要入口,指向新三层结构.
- `HTML_STYLE.md`: 自有 HTML 阅读页样式规范,主题 token 和交互约束的兼容跳转页.
- `XQT.md`: `xqt/` 的兼容长期事实源,覆盖模型优化工具链完整正文并承接旧链接.

### `docs/html/`

- `index.html`: 面向人类用户的 HTML 阅读入口.
- `dataset-structure.html`: 通用 dataset 模块结构说明,不包含 hair 特殊数据集.
- `xqt.html`: XQT 压缩与部署工具链阅读页,只提炼 `docs/md/XQT.md` 和当前源码事实.
- `style-showcase.html`: HTML 阅读页两种固定模板和组件展示.
- `assets/xdl-doc.css`: 自有 HTML 阅读页统一样式入口.
- `assets/xdl-theme.js`: HTML 阅读页主题和强调色切换脚本.
- `assets/xdl-style-showcase.css`: 样式展示页的局部 CSS.

## 修改约束

- 文档内容必须与真实代码保持一致,不写脱离实现的理想化描述.
- 涉及 API,目录,生命周期,配置字段或注册名称时,先搜索源码再更新文档.
- 涉及公开 API,导出符号或兼容承诺时,必须同步检查并更新 `md/README.md#xdl-api-稳定边界`.
- 大改动时同步更新交叉引用,`md/README.md` 和 `html/index.html`.
- 重构 `XDL` / `XQT` 长期文档时,优先把正文落到 `md/architecture/`, `md/explanation/`, `md/usage/`; 不要继续把新主题堆进单个 `md/README.md`.
- 规划文档与现状说明要分开写,避免把待办写成已实现事实.
- `docs/` 只保留长期有效内容;阶段性调研优先放 `research/`.
- 新增普通 MD 只能放在 `docs/md/`;不要继续把长期文档散放在 `docs/` 根目录.
- 新增 HTML 页面默认放在 `docs/html/`,保持静态自包含,避免依赖外部 CDN.
- HTML 页面是阅读层,不是唯一事实源;同一主题的 MD 事实变化时,必须同步对应 HTML.
- 新增或重构自有 HTML 时必须先遵循 `md/architecture/html-style-policy.md`: 页面 body 必须且只能包含 `xdl-style-atlas` 或 `xdl-style-ledger` 两种模板之一,默认引用 `html/assets/xdl-doc.css`,不要复制大段内联 `<style>` 或散落 `style=`.
- 需要主题交互时使用 `html/assets/xdl-theme.js` 和 `data-theme-value` / `data-accent-value`,不要为单页另写一套主题脚本.
- 公共阅读页 CSS 只收敛到 `html/assets/xdl-doc.css`;目录专属 CSS 只能保留薄入口和命名空间明确的局部组件,不能复制公共版式,主题变量或通用阅读组件.

## 写作建议

- 每篇 MD 开头先写清楚负责什么,不负责什么.
- 入口页保持薄,只做导航和边界说明,不复述大段正文.
- 超过单篇快速扫读负担的长期 MD,优先采用 "摘要入口 + 详细事实源" 的分体结构.
- MD 优先写入口,边界,限制,检查清单和验证方式.
- HTML 优先面向阅读路径和概念理解,可以重排内容,但不要改写事实边界.
- 避免多个文件重复讲同一套架构背景.
- 文档路径写具体路径,不要用 "这里" 或 "上面" 这类容易在移动后失效的指代.
