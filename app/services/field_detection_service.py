"""Heuristic auto field detection for causal analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FieldDetectionResult:
    defaults: dict[str, str | list[str] | None]
    candidates: dict[str, list[str]]
    notes: list[str]


def _find_candidates(columns: list[str], include_keywords: list[str], exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    out = []
    for c in columns:
        lc = c.lower()
        if c in exclude:
            continue
        if any(k in lc for k in include_keywords):
            out.append(c)
    return out


def detect_fields(df: pd.DataFrame) -> FieldDetectionResult:
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    treatment_candidates = _find_candidates(cols, ["treat", "variant", "exposed", "is_target", "group_flag"])
    outcome_candidates = _find_candidates(cols, ["outcome", "y", "sales", "revenue", "spend", "conversion", "gmv"])
    time_candidates = _find_candidates(cols, ["time", "date", "period", "week", "month", "day"])
    group_candidates = _find_candidates(cols, ["group", "segment", "cohort", "is_target"])
    instrument_candidates = _find_candidates(cols, ["instrument", "iv", "encourage", "assign", "z_"])
    running_candidates = _find_candidates(cols, ["running", "score", "distance", "rank", "age", "income", "spend"]) or numeric_cols

    treatment_default = treatment_candidates[0] if treatment_candidates else None
    if treatment_default is None:
        for c in cols:
            nunique = df[c].nunique(dropna=True)
            if nunique == 2:
                treatment_default = c
                break

    outcome_default = outcome_candidates[0] if outcome_candidates else None
    if outcome_default is None:
        # pick numeric col with highest variance, excluding treatment
        candidates = [c for c in numeric_cols if c != treatment_default]
        if candidates:
            outcome_default = max(candidates, key=lambda c: float(df[c].var(skipna=True)))

    time_default = time_candidates[0] if time_candidates else None
    group_default = group_candidates[0] if group_candidates else None
    instrument_default = instrument_candidates[0] if instrument_candidates else None
    running_default = running_candidates[0] if running_candidates else None

    excluded = {x for x in [treatment_default, outcome_default, time_default, group_default, instrument_default, running_default] if x}
    covariate_defaults = [c for c in cols if c not in excluded][:8]

    notes = [
        "Detection uses lightweight heuristics (column names, dtypes, cardinality).",
        "Please review and override detected fields before final interpretation.",
    ]

    return FieldDetectionResult(
        defaults={
            "treatment_col": treatment_default,
            "outcome_col": outcome_default,
            "time_col": time_default,
            "group_col": group_default,
            "instrument_col": instrument_default,
            "running_col": running_default,
            "covariates": covariate_defaults,
        },
        candidates={
            "treatment_col": treatment_candidates,
            "outcome_col": outcome_candidates,
            "time_col": time_candidates,
            "group_col": group_candidates,
            "instrument_col": instrument_candidates,
            "running_col": running_candidates,
            "covariates": [c for c in cols if c not in {treatment_default, outcome_default}],
        },
        notes=notes,
    )
