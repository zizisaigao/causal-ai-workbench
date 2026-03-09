"""Markdown report generation for causal analysis outputs."""

from __future__ import annotations

from app.causal.base import CausalEstimate


DISCLAIMER = (
    "⚠️ **免责声明**：本工具结果依赖观测数据与模型假设，"
    "不能自动保证严格因果识别。请结合业务背景、实验设计和稳健性检验综合判断。"
)


RANKING_DISCLAIMER = (
    "⚠️ **排序说明**：以下分数主要用于干预优先级建议，"
    "不等于严格无偏的个体因果效应（ITE）。"
)


def _append_heterogeneity_section(lines: list[str], result: CausalEstimate) -> None:
    method_title = "Uplift" if result.method == "uplift" else "Causal Forest"
    lines.append(f"## {method_title} Method Notes")
    if result.method == "uplift":
        lines.append("- 基于 transformed outcome + RandomForest 进行异质性排序。")
    else:
        lines.append("- 优先使用 CausalForestDML 估计 CATE；不可用时降级到 transformed outcome RF。")
    lines.append("- 适合回答“哪些人更值得优先干预”。")
    lines.append(RANKING_DISCLAIMER)
    lines.append("")

    top_profile = result.diagnostics.get("top_segment_profile_mean", {})
    top_preview = result.diagnostics.get("top_uplift_preview", result.diagnostics.get("top_effect_preview", []))

    lines.append("## Suggested Priority Segment")
    if top_preview:
        lines.append("- 预测增益/效应最高的 Top 人群（预览）可优先纳入干预。")
    if top_profile:
        lines.append(f"- Top 人群特征均值画像：`{top_profile}`")
    lines.append("")


def _append_rdd_section(lines: list[str]) -> None:
    lines.append("## RDD Method Notes")
    lines.append("- 本方法实现 sharp RDD 的局部线性估计。")
    lines.append("- 结果解释为 cutoff 附近的局部效应（local effect），不代表全样本 ATE。")
    lines.append("")


def _append_iv_section(lines: list[str]) -> None:
    lines.append("## IV Method Notes")
    lines.append("- 本方法使用最小可用 2SLS 流程（两阶段回归）。")
    lines.append("- 因果解释高度依赖工具变量相关性、排除限制和外生性强假设。")
    lines.append("- 请重点关注第一阶段强度（如 first-stage F statistic）。")
    lines.append("")


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

    if result.method in {"uplift", "causal_forest"}:
        _append_heterogeneity_section(lines, result)
    if result.method == "rdd":
        _append_rdd_section(lines)
    if result.method == "iv":
        _append_iv_section(lines)

    lines.append("## Diagnostics")
    for k, v in result.diagnostics.items():
        lines.append(f"- {k}: `{v}`")

    return "\n".join(lines)
