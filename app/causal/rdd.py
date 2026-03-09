"""Minimal sharp RDD estimator using local linear regression around cutoff."""

from __future__ import annotations

import numpy as np
import statsmodels.formula.api as smf

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class RDDEstimator(BaseCausalEstimator):
    method_name = "rdd"

    def __init__(self, bandwidth: float | None = None):
        self.bandwidth = bandwidth

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        if payload.running_col is None:
            raise ValueError("RDD requires running_col.")
        if payload.cutoff is None:
            raise ValueError("RDD requires cutoff.")

        running_col = payload.running_col
        cutoff = float(payload.cutoff)

        df = payload.data.dropna(subset=[payload.outcome_col, running_col]).copy()
        df["running_centered"] = df[running_col] - cutoff
        df["rdd_treat"] = (df[running_col] >= cutoff).astype(int)

        if self.bandwidth is None:
            bw = max(float(df["running_centered"].std()), 1e-6)
        else:
            bw = max(float(self.bandwidth), 1e-6)

        local_df = df[df["running_centered"].abs() <= bw].copy()
        if len(local_df) < 6:
            raise ValueError("RDD local sample is too small near cutoff. Try a larger bandwidth.")

        formula = f"{payload.outcome_col} ~ rdd_treat + running_centered + rdd_treat:running_centered"
        model = smf.ols(formula, data=local_df).fit(cov_type="HC1")

        ate = float(model.params["rdd_treat"])
        ci_low, ci_high = model.conf_int().loc["rdd_treat"]

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            assumptions=[
                "Potential outcomes are continuous at the cutoff in absence of treatment.",
                "No precise manipulation of the running variable around cutoff.",
                "Local linear functional form is adequate in chosen bandwidth.",
            ],
            limitations=[
                "Estimated effect is local to the cutoff (LATE at threshold), not global ATE.",
                "Bandwidth choice can materially change estimates.",
                "Results may be sensitive to misspecification near cutoff.",
            ],
            diagnostics={
                "n_obs_total": int(len(df)),
                "n_obs_local": int(len(local_df)),
                "running_col": running_col,
                "cutoff": cutoff,
                "bandwidth": bw,
                "treated_share_local": float(local_df["rdd_treat"].mean()),
            },
        )
