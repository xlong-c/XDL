"""XDL 使用说明和面向 agent 的结构化引导."""

from __future__ import annotations

import json
import sys
from importlib import resources
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

USAGE_VERSION = "1"


@dataclass(frozen=True)
class UsageQuestionOption:
    label: str
    description: str


@dataclass(frozen=True)
class UsageQuestion:
    header: str
    id: str
    question: str
    options: List[UsageQuestionOption]


@dataclass(frozen=True)
class UsageSection:
    title: str
    body: str


_USAGE_SECTIONS: List[UsageSection] = [
    UsageSection(
        title="XDL 是什么",
        body=(
            "- XDL 是 PyTorch 上层训练组织层, 不是新的张量框架.\n"
            "- 常用路径有两条: `CoreModel + Trainer` 纯代码训练, 或 `setup_from_yaml()` 配置化构建.\n"
            "- `CoreModel.training_step()` 采用手动优化模式, 需要自己调用 `zero_grad()`, `manual_backward()`, `optimizer.step()`.\n"
            "- 多卡和混合精度优先通过 `Trainer(..., accelerate_config=...)` 或外部 `torchrun` 使用.\n"
            "- 横切能力优先用 Callback, 如 checkpoint, tqdm, TensorBoard, 自定义保存."
        ),
    ),
    UsageSection(
        title="给 agent 的执行前提问协议",
        body=(
            "- 先读完本文件, 再开始写项目代码.\n"
            "- 先按下面的问题顺序向项目方确认约束, 再落代码.\n"
            "- 每个问题都要给出推荐默认值, 让项目方只需要选项而不是长篇描述.\n"
            "- 如果项目方没有明确回答, 按推荐默认值继续, 但要把假设写清楚.\n"
            "- 实现时只围绕已经确认的答案, 不要在入口里偷偷引入新的中转路径."
        ),
    ),
    UsageSection(
        title="建议提问顺序",
        body=(
            "- 这次要解决的是训练入口, 配置加载, 数据准备, 还是工具入口.\n"
            "- 你希望新的实现尽量显式, 还是保留少量自动推断.\n"
            "- 路径和产物要统一到单个 run_dir, 还是由调用方显式指定.\n"
            "- 默认只保存 adapter, 还是同时保存 preview 和全量 checkpoint.\n"
            "- 这次改动只服务当前仓库, 还是要保留 wheel 安装后的跨仓库复用."
        ),
    ),
]

_USAGE_QUESTIONS: List[UsageQuestion] = [
    UsageQuestion(
        header="任务类型",
        id="task_kind",
        question="这次改动主要服务哪类任务?",
        options=[
            UsageQuestionOption(
                label="训练",
                description="面向训练入口, Trainer, dataloader, checkpoint 等核心路径.",
            ),
            UsageQuestionOption(
                label="配置",
                description="面向 YAML, structured config, schema 和参数加载.",
            ),
            UsageQuestionOption(
                label="工具",
                description="面向脚本, usage, 文档或辅助入口, 不改核心训练逻辑.",
            ),
        ],
    ),
    UsageQuestion(
        header="入口风格",
        id="entry_style",
        question="你希望 agent 采用哪种实现入口风格?",
        options=[
            UsageQuestionOption(
                label="显式代码",
                description="优先可读的 Python API, 尽量少依赖隐式环境变量.",
            ),
            UsageQuestionOption(
                label="YAML 驱动",
                description="让配置文件承载大部分选择, 入口只负责装配.",
            ),
            UsageQuestionOption(
                label="兼容现状",
                description="尽量不改现有调用习惯, 只补结构和文档.",
            ),
        ],
    ),
    UsageQuestion(
        header="路径策略",
        id="path_policy",
        question="路径和产物应该怎么管理, 才最方便后续维护?",
        options=[
            UsageQuestionOption(
                label="run_dir",
                description="统一落到单个运行目录, 其下派生 logs, checkpoints, previews, config.",
            ),
            UsageQuestionOption(
                label="显式指定",
                description="调用方显式传入每个根路径, 不做自动派生.",
            ),
            UsageQuestionOption(
                label="沿用环境",
                description="继续以环境变量和默认常量为主, 只改善文档.",
            ),
        ],
    ),
    UsageQuestion(
        header="保存范围",
        id="save_scope",
        question="训练过程中默认保存哪些产物?",
        options=[
            UsageQuestionOption(
                label="仅 adapter",
                description="只保存轻量可复用产物, 保持目录干净.",
            ),
            UsageQuestionOption(
                label="adapter + preview",
                description="同时保存 adapter 和验证预览图, 便于快速检查.",
            ),
            UsageQuestionOption(
                label="全量 checkpoint",
                description="把 optimizer, scheduler, 模型状态一并保留.",
            ),
        ],
    ),
    UsageQuestion(
        header="自动化边界",
        id="automation_boundary",
        question="哪些事情允许 agent 自动补全, 哪些必须显式确认?",
        options=[
            UsageQuestionOption(
                label="仅补默认值",
                description="agent 只补文档里已经明确的默认值, 不扩展新行为.",
            ),
            UsageQuestionOption(
                label="补辅助代码",
                description="允许补少量辅助函数或包装层, 但不能改变主流程语义.",
            ),
            UsageQuestionOption(
                label="可做重构",
                description="允许整理路径和入口, 但要先把项目偏好问清楚.",
            ),
        ],
    ),
]


def get_usage_sections() -> List[UsageSection]:
    """返回面向人类和 agent 的 usage 段落."""

    return list(_USAGE_SECTIONS)


def get_usage_questions() -> List[UsageQuestion]:
    """返回编写功能前建议先确认的问答对."""

    return list(_USAGE_QUESTIONS)


def get_usage_payload() -> dict[str, Any]:
    """返回结构化 usage 内容, 便于上层工具直接消费."""

    return {
        "usage_version": USAGE_VERSION,
        "sections": [
            {"title": section.title, "body": section.body}
            for section in _USAGE_SECTIONS
        ],
        "questions": [
            {
                "header": question.header,
                "id": question.id,
                "question": question.question,
                "options": [
                    {"label": option.label, "description": option.description}
                    for option in question.options
                ],
            }
            for question in _USAGE_QUESTIONS
        ],
    }


def get_usage_text() -> str:
    """返回随 wheel 分发的 XDL 单文件使用说明."""

    return resources.read_text("xdl", "USAGE.md", encoding="utf-8")


def print_usage() -> None:
    """打印随 wheel 分发的 XDL 单文件使用说明."""

    print(get_usage_text(), end="")


def format_usage_questions(markdown: bool = True) -> str:
    """返回面向 agent 的问答清单."""

    if markdown:
        lines: List[str] = ["按顺序提问, 先确认约束再写代码:"]
        for question in _USAGE_QUESTIONS:
            lines.append(f"- {question.header}: {question.question}")
            for option in question.options:
                lines.append(f"  - {option.label}: {option.description}")
        return "\n".join(lines)
    return json.dumps(get_usage_payload()["questions"], ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 默认打印 usage, 支持结构化输出."""

    args = list(sys.argv[1:] if argv is None else argv)
    if "--json" in args:
        print(json.dumps(get_usage_payload(), ensure_ascii=False, indent=2))
        return 0
    if "--questions" in args:
        print(format_usage_questions(markdown=True))
        return 0
    print_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
