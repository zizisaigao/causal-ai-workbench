"""Minimal IV estimator using two-stage least squares (2SLS) approximation."""

from __future__ import annotations

import statsmodels.api as sm

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class IVEstimator(BaseCausalEstimator):
    method_name = "iv"

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        instrument_col = payload.instrument_col
        if not instrument_col:
            raise ValueError("IV requires instrument_col.")

        cols = [payload.treatment_col, payload.outcome_col, instrument_col] + payload.covariate_cols
        df = payload.data.dropna(subset=cols).copy()
        if df[instrument_col].nunique() < 2:
            raise ValueError("IV requires non-constant instrument.")

        # Stage 1: treatment ~ instrument + covariates
        x1 = sm.add_constant(df[[instrument_col] + payload.covariate_cols], has_constant="add")
        stage1 = sm.OLS(df[payload.treatment_col], x1).fit(cov_type="HC1")
        df["t_hat"] = stage1.predict(x1)

        # Stage 2: outcome ~ predicted treatment + covariates
        x2 = sm.add_constant(df[["t_hat"] + payload.covariate_cols], has_constant="add")
        stage2 = sm.OLS(df[payload.outcome_col], x2).fit(cov_type="HC1")

        ate = float(stage2.params["t_hat"])
        ci_low, ci_high = stage2.conf_int().loc["t_hat"]

        first_stage_coef = float(stage1.params.get(instrument_col, 0.0))
        first_stage_pvalue = float(stage1.pvalues.get(instrument_col, 1.0))
        first_stage_f = float(stage1.fvalue) if stage1.fvalue is not None else None
        weak_iv_warning = bool(first_stage_f is not None and first_stage_f < 10)

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            assumptions=[
                "Instrument relevance: instrument affects treatment assignment.",
                "Exclusion restriction: instrument affects outcome only via treatment.",
                "Instrument exogeneity: instrument is independent of unobserved confounders.",
            ],
            limitations=[
                "This MVP uses manual 2SLS approximation without full finite-sample corrections.",
                "Weak instruments can produce unstable or biased estimates.",
                "Causal interpretation heavily depends on exclusion and exogeneity assumptions.",
            ],
            diagnostics={
                "n_obs": int(len(df)),
                "instrument_col": instrument_col,
                "first_stage_coef": first_stage_coef,
                "first_stage_pvalue": first_stage_pvalue,
                "first_stage_f_stat": first_stage_f,
                "weak_instrument_warning": weak_iv_warning,
            },
        )
