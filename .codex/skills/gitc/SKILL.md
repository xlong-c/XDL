---
name: gitc
description: 自动执行 Git 全量提交与推送流程（`git add -A`、`git commit`、`git push`）。当用户要求“直接推送所有更新”“从 add 到 push 一次完成”“不要确认直接提交并推送”时使用。默认提交信息保持简洁清晰，正文按点分列改动摘要。
---

# GitC

## Overview

在当前仓库直接完成 `add -> commit -> push` 全流程，不进行交互确认。提交信息自动生成，格式为简短标题 + 分点摘要，便于快速追踪改动。

## Workflow

1. 在仓库根目录执行脚本：
```bash
bash .codex/skills/gitc/scripts/gitc_push.sh
```
2. 脚本自动执行 `git add -A`，把所有改动加入暂存区。
3. 脚本自动生成 commit message 并提交。
4. 脚本自动 `git push` 到当前分支的上游；若未配置上游且存在 `origin`，自动使用 `git push -u origin <branch>`。

## Commit Message Rules

1. 默认标题：`chore: 同步工作区更新`
2. 正文优先输出“改了什么”的中文语义摘要（而不是只列文件路径），推荐格式：
```text
添加 SGEMM CUDA 内核实现

- 添加 CUDA 类型头文件 (cuda_bf16, cuda_fp16)
- 定义 WARP_SIZE 常量和类型转换宏
- 实现 naive SGEMM 内核 (fp32)
```
3. 当语义摘要不足时，回退到文件级摘要（新增/更新/删除/重命名/复制）。
4. 列表过长时，仅展示前若干条，并补充中文汇总信息，必须包含：
   - 各类型变更统计（新增/更新/删除/重命名/复制）
   - 主要涉及目录
   - 省略项提示（不能只写 `remaining N files`）

## Non-Interactive Behavior

1. 触发本 skill 后，不再询问是否执行 `add/commit/push`。
2. 默认直接提交并推送当前仓库全部改动。
3. 若没有改动，直接退出并提示无可提交内容。

## Script

使用 [scripts/gitc_push.sh](/root/workspace/xdl/.codex/skills/gitc/scripts/gitc_push.sh) 执行固定流程。可选第一个参数覆盖标题：
```bash
bash .codex/skills/gitc/scripts/gitc_push.sh "feat: 优化训练流程"
```
