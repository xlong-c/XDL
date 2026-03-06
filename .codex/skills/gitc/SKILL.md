---
name: gitc
description: 自动执行 Git 全量提交与推送流程（`git add -A`、`git commit`、`git push`）。当用户要求“直接推送所有更新”“从 add 到 push 一次完成”“不要确认直接提交并推送”时使用。默认提交信息保持简洁清晰，正文按点分列改动摘要。
---

# GitC

## Overview

在当前仓库直接完成 `add -> commit -> push` 全流程，不进行交互确认。提交信息自动生成，格式为简短标题 + 分点摘要，但摘要必须优先回答“这次提交新增/改进了什么能力”，而不是逐条罗列底层编辑动作。

## Workflow

1. 在仓库根目录执行脚本：
```bash
bash .codex/skills/gitc/scripts/gitc_push.sh
```
2. 脚本自动执行 `git add -A`，把所有改动加入暂存区。
3. 脚本自动生成 commit message 并提交。
4. 脚本自动 `git push` 到当前分支的上游；若未配置上游且存在 `origin`，自动使用 `git push -u origin <branch>`。

## Commit Message Rules

1. 默认标题：`<scope> <动作>`，例如 `gitc skill 更新`、`gitc skill 修复`。
   - `scope` 自动根据改动范围识别（例如单一 skill 目录会生成 `gitc skill`）。
   - `动作` 自动根据变更类型识别（常见为 `新增` / `更新` / `修复` / `重构` / `清理`）。
   - 传入脚本参数时优先使用手动标题。
2. 正文优先输出“做成了什么”的中文语义摘要，先写功能变化，再写支撑性改动，避免流水账式罗列头文件、宏、常量、import：
```text
补充 CUDA GEMM 示例与公共工具

- 新增异步拷贝版 SGEMM CUDA 示例
- 新增 cuBLASEx SGEMM 示例
- 抽取 cuBLAS/cuBLASLt GEMM 示例公共工具
- 补充 CUDA 混合精度与运行时头文件 (cuda_bf16.h、cuda_fp16.h 等)
```
3. 低层细节只作为补充信息聚合展示，例如：
   - 多个头文件合并为“补充 CUDA 混合精度与运行时头文件”
   - 多个宏合并为“定义辅助宏”
   - 多个函数名合并为“补充函数/工具”
   - 不允许把 `添加头文件 (xxx)` 连续输出成正文主体
4. 当语义摘要不足时，按文件提炼“行为变化点”摘要（如参数调整、逻辑分支、规则修订、函数新增）；仅在无法提炼时才回退到文件级摘要（新增/更新/删除/重命名/复制）。
5. 列表过长时，仅展示前若干条，并补充中文汇总信息，必须包含：
   - 各类型变更统计（新增/更新/删除/重命名/复制）
   - 主要涉及目录
   - 省略项提示（不能只写 `remaining N files`）
6. 标题中的动作词也应贴近结果：
   - 以新增文件/示例为主时，优先用 `新增`
   - 仅在明显修 bug、兼容性问题、异常处理时才用 `修复`
   - 不因代码里出现 `error` 字样就误判为 `修复`

## Non-Interactive Behavior

1. 触发本 skill 后，不再询问是否执行 `add/commit/push`。
2. 默认直接提交并推送当前仓库全部改动。
3. 若没有改动，直接退出并提示无可提交内容。

## Script

使用 [scripts/gitc_push.sh](/root/workspace/xdl/.codex/skills/gitc/scripts/gitc_push.sh) 执行固定流程。可选第一个参数覆盖标题：
```bash
bash .codex/skills/gitc/scripts/gitc_push.sh "feat: 优化训练流程"
```
