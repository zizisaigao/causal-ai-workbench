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
        return MethodRecommendation("uplift", ["psm"], rationale)

    if treatment_binary and not observational:
        rationale.append("Treatment looks experiment-like; A/B style mean-difference is plausible.")
        return MethodRecommendation("ab_test", ["did", "uplift"], rationale)

    rationale.append("Observational cross-sectional structure detected; PSM is a pragmatic baseline.")
    return MethodRecommendation("psm", ["uplift"], rationale)
