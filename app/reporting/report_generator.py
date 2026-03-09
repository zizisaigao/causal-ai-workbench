"""Markdown report generation for causal analysis outputs."""

from __future__ import annotations

from app.causal.base import CausalEstimate


DISCLAIMER = (
    "⚠️ **免责声明**：本工具结果依赖观测数据与模型假设，"
    "不能自动保证严格因果识别。请结合业务背景、实验设计和稳健性检验综合判断。"
)


def generate_markdown_report(result: CausalEstimate) -> str:
    lines = [
        f"# Causal Analysis Report - {result.method.upper()}",
        "",
        DISCLAIMER,
        "",
        "## Core Effect",
        f"- ATE: `{result.ate}`",
        f"- 95% CI: `{result.ci_low}` to `{result.ci_high}`",
        "",
        "## Assumptions",
    ]
    lines.extend([f"- {x}" for x in result.assumptions])
    lines.append("")
    lines.append("## Limitations")
    lines.extend([f"- {x}" for x in result.limitations])
    lines.append("")
    lines.append("## Diagnostics")
    for k, v in result.diagnostics.items():
        lines.append(f"- {k}: `{v}`")

    return "\n".join(lines)
