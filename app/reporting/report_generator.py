"""Markdown report generation for causal analysis outputs."""

from __future__ import annotations

from app.causal.base import CausalEstimate


DISCLAIMER = (
    "⚠️ **免责声明**：本工具结果依赖观测数据与模型假设，"
    "不能自动保证严格因果识别。请结合业务背景、实验设计和稳健性检验综合判断。"
)


UPLIFT_DISCLAIMER = (
    "⚠️ **Uplift 说明**：以下排序用于干预优先级建议，"
    "不等于严格无偏的个体因果效应（ITE）。"
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

    if result.method == "uplift":
        lines.append("## Uplift Method Notes")
        lines.append("- 基于 transformed outcome + RandomForest 进行异质性排序。")
        lines.append("- 适合回答“哪些人更值得优先干预”。")
        lines.append(UPLIFT_DISCLAIMER)
        lines.append("")

        top_preview = result.diagnostics.get("top_uplift_preview", [])
        top_profile = result.diagnostics.get("top_segment_profile_mean", {})

        lines.append("## Suggested Priority Segment")
        if top_preview:
            lines.append("- 预测 uplift 最高的 Top 人群样本（预览）可优先纳入干预。")
        if top_profile:
            lines.append(f"- Top 人群特征均值画像：`{top_profile}`")
        lines.append("")

    lines.append("## Diagnostics")
    for k, v in result.diagnostics.items():
        lines.append(f"- {k}: `{v}`")

    return "\n".join(lines)
