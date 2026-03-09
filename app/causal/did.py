"""Difference-in-differences estimator for panel/repeated-cross-section style data."""

from __future__ import annotations

import statsmodels.formula.api as smf

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class DIDEstimator(BaseCausalEstimator):
    method_name = "did"

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        if not payload.time_col:
            raise ValueError("DID requires a time_col.")
        if not payload.group_col:
            raise ValueError("DID requires a group_col identifying treated group.")

        df = payload.data.dropna(subset=[payload.outcome_col, payload.time_col, payload.group_col]).copy()
        if df[payload.time_col].nunique() < 2:
            raise ValueError("DID requires at least 2 time periods.")

        sorted_times = sorted(df[payload.time_col].unique())
        pre, post = sorted_times[0], sorted_times[-1]
        df = df[df[payload.time_col].isin([pre, post])].copy()
        df["post"] = (df[payload.time_col] == post).astype(int)

        formula = f"{payload.outcome_col} ~ {payload.group_col} + post + {payload.group_col}:post"
        model = smf.ols(formula, data=df).fit(cov_type="HC1")

        coef_key = f"{payload.group_col}:post"
        ate = float(model.params[coef_key])
        ci_low, ci_high = model.conf_int().loc[coef_key]

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            assumptions=[
                "Parallel trends between treatment and control groups.",
                "No contemporaneous shocks affecting only one group.",
                "Composition of groups remains stable over time.",
            ],
            limitations=[
                "Biased if parallel trends assumption fails.",
                "Only uses earliest and latest periods in this MVP implementation.",
                "Time-varying confounding is not explicitly modeled.",
            ],
            diagnostics={
                "n_obs": int(len(df)),
                "pre_period": str(pre),
                "post_period": str(post),
                "r_squared": float(model.rsquared),
            },
        )
