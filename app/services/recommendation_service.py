"""Heuristic method recommendations based on available columns and study intent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MethodRecommendation:
    primary_method: str
    alternatives: list[str]
    rationale: list[str]


def recommend_methods(
    has_time: bool,
    has_group: bool,
    treatment_binary: bool,
    observational: bool,
    wants_targeting: bool,
) -> MethodRecommendation:
    rationale: list[str] = []

    if has_time and has_group:
        rationale.append("Detected both time and treatment-group structure, suitable for DID.")
        return MethodRecommendation("did", ["psm", "uplift"], rationale)

    if wants_targeting:
        rationale.append("User intent indicates ranking users by incremental impact.")
        return MethodRecommendation("uplift", ["causal_forest", "psm"], rationale)

    if treatment_binary and not observational:
        rationale.append("Treatment looks experiment-like; A/B style mean-difference is plausible.")
        return MethodRecommendation("ab_test", ["did", "uplift"], rationale)

    rationale.append("Observational cross-sectional structure detected; PSM is a pragmatic baseline.")
    return MethodRecommendation("psm", ["uplift"], rationale)


def recommend_method_from_detected_fields(
    has_time_col: bool,
    has_group_col: bool,
    has_running_col: bool,
    has_instrument_col: bool,
    treatment_binary: bool,
    wants_targeting: bool,
) -> MethodRecommendation:
    rationale: list[str] = []

    if has_instrument_col:
        rationale.append("Detected instrument-like field; IV is prioritized.")
        return MethodRecommendation("iv", ["did", "psm"], rationale)

    if has_time_col and has_group_col:
        rationale.append("Detected time and group structure; DID is prioritized.")
        return MethodRecommendation("did", ["psm", "uplift"], rationale)

    if wants_targeting:
        rationale.append("Targeting/heterogeneity preference detected; uplift/causal_forest is prioritized.")
        return MethodRecommendation("causal_forest", ["uplift", "psm"], rationale)

    if treatment_binary:
        rationale.append("Binary treatment in observational setup; PSM is selected as baseline.")
        return MethodRecommendation("psm", ["did", "uplift"], rationale)

    rationale.append("Fallback to uplift due to weak structural signals for quasi-experimental designs.")
    return MethodRecommendation("uplift", ["psm"], rationale)
