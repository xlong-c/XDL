# XDL Agent Handoff

本文承接 `XDL` 的 agent 使用入口和交接约束. 当前详细文本仍以 [../../../xdl/USAGE.md](../../../xdl/USAGE.md) 为准, 本页先作为使用层落点.

## 负责什么

- 说明 agent 接手 `XDL` 任务时应先确认什么.
- 说明路径, 产物, 默认值和测试范围这类 handoff 信息应如何表达.

## 不负责什么

- 不替代完整训练架构文档.
- 不定义新的仓库级规则.
- 不复制 `xdl/USAGE.md` 的全部正文.

## 当前入口

- 包内命令: `xdl-usage`
- 模块入口: `python -m xdl.usage`
- 正文事实源: [../../../xdl/USAGE.md](../../../xdl/USAGE.md)

## 当前 handoff 重点

根据现有 `xdl/USAGE.md`, agent 接手任务前应先确认:

1. 主要改动服务哪类任务
2. 实现入口是显式代码还是 YAML 驱动
3. 路径和产物是否统一放进单个 `run_dir`
4. 默认保存的是 adapter, preview, 还是全量 checkpoint
5. 哪些事情允许自动补全, 哪些必须显式确认

## 交付时至少要说明

- 实现方案
- 路径约定
- 默认值和假设
- 测试范围

## 后续计划

等 `xdl/USAGE.md` 与三层文档体系完全对齐后, 本页再吸收其稳定正文, 并把包内薄入口和仓库内使用层统一起来.
