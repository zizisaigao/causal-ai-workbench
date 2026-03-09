"""One-click auto detect + recommend + analyze orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.api.schemas import ApiError
from app.causal.base import CausalAnalysisInput, CausalEstimate
from app.services.analysis_service import run_causal_analysis
from app.services.field_detection_service import detect_fields
from app.services.recommendation_service import recommend_method_from_detected_fields


@dataclass
class AutoAnalysisResult:
    detected_fields: dict
    recommended_method: str
    recommendation_rationale: list[str]
    recommendation_limitations: list[str]
    analysis_result: CausalEstimate


def run_auto_analysis(
    df: pd.DataFrame,
    cutoff: float | None = None,
    include_llm_explanation: bool = True,
) -> AutoAnalysisResult:
    detection = detect_fields(df)
    defaults = detection.defaults

    treatment_col = defaults.get("treatment_col")
    outcome_col = defaults.get("outcome_col")
    covariates = defaults.get("covariates") or []

    if not treatment_col or not outcome_col:
        raise ApiError(
            "auto_detection_failed",
            "Unable to detect treatment/outcome fields reliably. Please provide manual column mapping.",
            status_code=422,
        )

    treatment_binary = False
    try:
        treatment_binary = df[str(treatment_col)].nunique(dropna=True) == 2
    except Exception:
        pass

    rec = recommend_method_from_detected_fields(
        has_time_col=bool(defaults.get("time_col")),
        has_group_col=bool(defaults.get("group_col")),
        has_running_col=bool(defaults.get("running_col")),
        has_instrument_col=bool(defaults.get("instrument_col")),
        treatment_binary=treatment_binary,
        wants_targeting=False,
    )

    method = rec.primary_method
    limits = [
        "Recommendation is heuristic and may be wrong for your study design.",
        "Please validate field mapping and assumptions before business decisions.",
    ]

    analysis_input = CausalAnalysisInput(
        data=df,
        treatment_col=str(treatment_col),
        outcome_col=str(outcome_col),
        covariate_cols=[str(c) for c in covariates],
        time_col=defaults.get("time_col"),
        group_col=defaults.get("group_col"),
        running_col=defaults.get("running_col"),
        cutoff=cutoff if cutoff is not None else (0.0 if method == "rdd" else None),
        instrument_col=defaults.get("instrument_col"),
        metadata={"auto_detected": True, "include_llm_explanation": include_llm_explanation},
    )

    result = run_causal_analysis(
        method=method,
        payload=analysis_input,
        psm_caliper=1.0,
        uplift_buckets=5,
        cf_n_estimators=200,
        cf_min_samples_leaf=5,
        rdd_bandwidth=1.0 if method == "rdd" else None,
    )
    result.diagnostics["recommended_method"] = method
    result.diagnostics["recommendation_rationale"] = rec.rationale
    result.diagnostics["recommendation_limitations"] = limits
    result.diagnostics["detected_field_defaults"] = detection.defaults

    return AutoAnalysisResult(
        detected_fields={
            "defaults": detection.defaults,
            "candidates": detection.candidates,
            "notes": detection.notes,
        },
        recommended_method=method,
        recommendation_rationale=rec.rationale,
        recommendation_limitations=limits,
        analysis_result=result,
    )
