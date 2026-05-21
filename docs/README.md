# XDL 文档索引

`docs/` 只保留与 XDL 框架本身直接相关的长期文档。研究笔记、阶段性分析和一次性草案应放在 `research/`，不在这里堆叠。

## 推荐阅读顺序

1. [INSTALL.md](INSTALL.md)
   - 安装、验证、当前可运行入口。

2. [XDL.md](XDL.md)
   - 框架定位、核心分层、两条推荐使用路径。

3. [CONFIG.md](CONFIG.md)
   - `setup_from_yaml()`、schema v1、`target + params`、YAML 组织方式。

4. [xdl-functional-boundary.md](xdl-functional-boundary.md)
   - 各源码子模块职责速查。

5. [xdl-optimization-plan.md](xdl-optimization-plan.md)
   - 当前仍有效的后续优化方向。

## 文档边界

- `INSTALL.md` 只讲环境、安装和验证。
- `XDL.md` 只讲框架结构、训练入口和扩展方式。
- `CONFIG.md` 只讲配置系统。
- `xdl-functional-boundary.md` 只做模块职责速查，不重复写长篇使用指南。
- `xdl-optimization-plan.md` 只保留仍然有效的待办，不复述现状说明。
