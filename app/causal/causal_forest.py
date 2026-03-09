"""CATE estimation via CausalForestDML with fallback when econml is unavailable."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


def _build_heterogeneity_diagnostics(
    scored: pd.DataFrame,
    payload: CausalAnalysisInput,
    treatment_col: str,
    outcome_col: str,
    score_col: str,
    n_buckets: int,
) -> dict:
    scored = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    sample_scores = scored[["rank", score_col] + payload.covariate_cols].head(100).to_dict(orient="records")

    n_buckets = int(np.clip(n_buckets, 2, 10))
    scored["effect_bucket"] = pd.qcut(scored[score_col], q=n_buckets, labels=False, duplicates="drop")
    scored["effect_bucket"] = scored["effect_bucket"].astype("Int64") + 1

    bucket_rows: list[dict] = []
    for bucket, g in scored.groupby("effect_bucket", dropna=True):
        treated = g[g[treatment_col] == 1][outcome_col]
        control = g[g[treatment_col] == 0][outcome_col]
        treated_mean = float(treated.mean()) if len(treated) > 0 else None
        control_mean = float(control.mean()) if len(control) > 0 else None
        observed_gap = (treated_mean - control_mean) if treated_mean is not None and control_mean is not None else None
        bucket_rows.append(
            {
                "bucket": int(bucket),
                "n_samples": int(len(g)),
                "avg_predicted_effect": float(g[score_col].mean()),
                "treated_outcome_mean": treated_mean,
                "control_outcome_mean": control_mean,
                "observed_outcome_gap": observed_gap,
            }
        )

    top_summary = (
        scored.head(max(1, len(scored) // n_buckets))
        .agg({col: "mean" for col in payload.covariate_cols})
        .to_dict()
    )

    top_preview = scored.head(10)[payload.covariate_cols + [score_col]].to_dict(orient="records")

    return {
        "n_buckets": int(n_buckets),
        "sample_effect_scores_preview": sample_scores,
        "bucket_summary": sorted(bucket_rows, key=lambda r: r["bucket"]),
        "top_effect_preview": top_preview,
        "top_segment_profile_mean": top_summary,
    }


class CausalForestEstimator(BaseCausalEstimator):
    method_name = "causal_forest"

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 5, n_buckets: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_buckets = n_buckets
        self.random_state = random_state

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        df = payload.data.dropna(subset=[payload.treatment_col, payload.outcome_col] + payload.covariate_cols).copy()
        x = df[payload.covariate_cols]
        t = df[payload.treatment_col].astype(int)
        y = df[payload.outcome_col]

        implementation = "econml_causal_forest"
        used_fallback = False

        cate_scores: np.ndarray
        try:
            from econml.dml import CausalForestDML  # type: ignore

            cf = CausalForestDML(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            cf.fit(Y=y, T=t, X=x)
            cate_scores = cf.effect(x)
        except Exception:
            used_fallback = True
            implementation = "fallback_transformed_outcome_rf"
            p = np.clip(t.mean(), 1e-3, 1 - 1e-3)
            transformed = y * (t - p) / (p * (1 - p))
            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
            rf.fit(x, transformed)
            cate_scores = rf.predict(x)

        ate = float(np.mean(cate_scores))
        scored = df.copy()
        scored["cate_score"] = cate_scores

        hetero = _build_heterogeneity_diagnostics(
            scored=scored,
            payload=payload,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            score_col="cate_score",
            n_buckets=self.n_buckets,
        )

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=None,
            ci_high=None,
            assumptions=[
                "Unconfoundedness given selected covariates.",
                "Overlap/positivity across treatment assignment.",
                "CATE model captures treatment effect heterogeneity structure.",
            ],
            limitations=[
                "When econml is unavailable, falls back to transformed-outcome random forest approximation.",
                "CATE scores are useful for prioritization but not guaranteed unbiased ITE.",
                "Observational outcome gaps by bucket may still reflect residual confounding.",
            ],
            diagnostics={
                "n_obs": int(len(df)),
                "treated_rate": float(t.mean()),
                "implementation": implementation,
                "used_fallback": used_fallback,
                **hetero,
            },
        )
