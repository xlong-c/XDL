# XDL 文档索引

`docs/` 现在按阅读对象分为两类:

- `*.md`: Codex 和开发者写代码前看的工作文档. 目标是信息密度高,便于快速理解真实契约,入口,边界,检查清单和源码关联.
- `html/`: 面向用户阅读的可视化文档. 目标是更美观,更顺滑,更容易读懂. HTML 不单独定义新契约,事实变更时先改对应 MD 或源码说明,再同步 HTML.

研究笔记,阶段性分析和一次性草案应放在 `research/`,不在 `docs/` 堆叠.

## 给用户看的 HTML

| 文档 | 负责内容 |
| --- | --- |
| [html/index.html](html/index.html) | 阅读版总入口,按安装,理解框架,写配置,扩展数据集和 API 边界组织阅读路线. |
| [html/dataset-structure.html](html/dataset-structure.html) | 通用 dataset 模块结构图和文件职责说明,不包含 hair 特殊数据集. |

## 给 Codex 写代码看的 MD

| 顺序 | 文档 | 负责内容 |
| --- | --- | --- |
| 1 | [INSTALL.md](INSTALL.md) | 安装,验证,当前可运行入口. |
| 2 | [XDL.md](XDL.md) | 框架定位,核心分层,训练入口,扩展方式. |
| 3 | [CONFIG.md](CONFIG.md) | `setup_from_yaml()`,schema v1,`target + params`,YAML 组织方式. |
| 4 | [DATASET.md](DATASET.md) | 数据集语义形态,磁盘组织形态,内置模板和新增 dataset 流程. |
| 5 | [API.md](API.md) | 稳定公共 API,实验性 API,内部实现边界和废弃策略. |
| 6 | [xdl-functional-boundary.md](xdl-functional-boundary.md) | 各源码子模块职责速查. |
| 7 | [xdl-optimization-plan.md](xdl-optimization-plan.md) | 当前仍有效的后续优化方向. |

## 文档边界

MD 工作文档:

- `INSTALL.md` 只讲环境,安装和验证.
- `XDL.md` 只讲框架结构,训练入口和扩展方式.
- `CONFIG.md` 只讲配置系统.
- `DATASET.md` 只讲 dataset 模板规划,选择和扩展方式.
- `API.md` 只讲公共 API 兼容边界,不重复使用教程.
- `xdl-functional-boundary.md` 只做模块职责速查,不重复写长篇使用指南.
- `xdl-optimization-plan.md` 只保留仍然有效的待办,不复述现状说明.

HTML 阅读版:

- `html/index.html` 只做面向用户的阅读入口和文档地图.
- `html/dataset-structure.html` 只做 dataset 模块结构的可视化说明.
- HTML 可以重排,提炼和图文化 MD 内容,但不要引入和 MD 或源码冲突的新事实.
- 同一主题的行为,字段或 API 发生变化时,先更新对应 MD,再同步覆盖同一主题的 HTML 页面.
