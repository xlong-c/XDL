# XDL / XQT 文档入口

本文是 `XDL` 与 `XQT` 长期 Markdown 文档的新入口. 它只负责导航, 分层和阅读顺序, 不承载完整事实正文.

## 负责什么

- 说明 `docs/md/` 的三层结构.
- 给出 `XDL` 与 `XQT` 的推荐阅读路径.
- 指向当前的架构, 说明, 使用三类文档.

## 不负责什么

- 不重复完整安装命令和 workflow 细节.
- 不单独定义 API 契约或兼容承诺.
- 不替代具体主题文档的正文事实源.

## 三层结构

`docs/md/` 现在按三类正文组织:

- [architecture/index.md](architecture/index.md): 架构层. 定义边界, 模块关系, 生命周期和兼容规则.
- [explanation/index.md](explanation/index.md): 说明层. 解释概念, 术语, 理解路径和常见误区.
- [usage/index.md](usage/index.md): 使用层. 提供安装, 命令, workflow, 验证与 handoff 指南.

## 推荐阅读顺序

### XDL

1. [architecture/xdl.md](architecture/xdl.md)
2. [architecture/dataset-policy.md](architecture/dataset-policy.md)
3. [architecture/module-boundaries.md](architecture/module-boundaries.md)
4. [architecture/api-boundary.md](architecture/api-boundary.md)
5. [explanation/xdl-concepts.md](explanation/xdl-concepts.md)
6. [usage/xdl-install-and-verify.md](usage/xdl-install-and-verify.md)
7. [usage/xdl-config-workflows.md](usage/xdl-config-workflows.md)
8. [usage/xdl-workflows.md](usage/xdl-workflows.md)

### XQT

1. [architecture/xqt.md](architecture/xqt.md)
2. [architecture/xqt-kernel-wrapper-nn-boundary.md](architecture/xqt-kernel-wrapper-nn-boundary.md)
3. [architecture/xqt-realignment-guide.md](architecture/xqt-realignment-guide.md)
4. [explanation/xqt-concepts.md](explanation/xqt-concepts.md)
5. [explanation/xqt-kernel-guidance.md](explanation/xqt-kernel-guidance.md)
6. [explanation/backends/index.md](explanation/backends/index.md)
7. [explanation/flux2-klein-nvfp4-backends.md](explanation/flux2-klein-nvfp4-backends.md)
8. [explanation/operator-kernel-tuning-guide.md](explanation/operator-kernel-tuning-guide.md)
9. [explanation/inference-backends-primer.md](explanation/inference-backends-primer.md)
10. [usage/xqt-workflows.md](usage/xqt-workflows.md)

## 当前兼容入口

在正文完全拆分完成前, 现有入口仍然有效:

- [README.md](README.md): `XDL` 兼容详细版事实源.
- [README_SUMMARY.md](README_SUMMARY.md): `XDL` 兼容摘要入口.
- [XQT.md](XQT.md): `XQT` 兼容长期事实源.
- [XQT_SUMMARY.md](XQT_SUMMARY.md): `XQT` 兼容摘要入口.

## 迁移原则

- 新文档先建骨架, 再迁正文.
- 同一事实只保留一个 canonical 页面.
- 旧入口先保留兼容, 不在第一轮直接删除.
