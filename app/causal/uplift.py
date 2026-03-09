"""Uplift / heterogeneity analysis with sample-level scores and bucket summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class UpliftEstimator(BaseCausalEstimator):
    method_name = "uplift"

    def __init__(self, n_estimators: int = 200, random_state: int = 42, n_buckets: int = 5):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.n_buckets = n_buckets

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        df = payload.data.dropna(subset=[payload.treatment_col, payload.outcome_col] + payload.covariate_cols).copy()
        x = df[payload.covariate_cols]
        t = df[payload.treatment_col].astype(int)
        y = df[payload.outcome_col]

        p = np.clip(t.mean(), 1e-3, 1 - 1e-3)
        transformed = y * (t - p) / (p * (1 - p))

        self.model.fit(x, transformed)
        uplift_scores = self.model.predict(x)
        ate = float(np.mean(uplift_scores))

        scored = df.copy()
        scored["uplift_score"] = uplift_scores

        scored = scored.sort_values("uplift_score", ascending=False).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)

        # Sample-level score output (compact preview for API payload size)
        sample_scores = scored[["rank", "uplift_score"] + payload.covariate_cols].head(100).to_dict(orient="records")

        # Bucket summary by predicted uplift quantiles
        n_buckets = int(np.clip(self.n_buckets, 2, 10))
        scored["uplift_bucket"] = pd.qcut(scored["uplift_score"], q=n_buckets, labels=False, duplicates="drop")
        scored["uplift_bucket"] = scored["uplift_bucket"].astype("Int64") + 1

        bucket_rows: list[dict] = []
        for bucket, g in scored.groupby("uplift_bucket", dropna=True):
            treated = g[g[payload.treatment_col] == 1][payload.outcome_col]
            control = g[g[payload.treatment_col] == 0][payload.outcome_col]
            treated_mean = float(treated.mean()) if len(treated) > 0 else None
            control_mean = float(control.mean()) if len(control) > 0 else None
            observed_gap = (treated_mean - control_mean) if treated_mean is not None and control_mean is not None else None
            bucket_rows.append(
                {
                    "bucket": int(bucket),
                    "n_samples": int(len(g)),
                    "avg_predicted_uplift": float(g["uplift_score"].mean()),
                    "treated_outcome_mean": treated_mean,
                    "control_outcome_mean": control_mean,
                    "observed_outcome_gap": observed_gap,
                }
            )

        bucket_rows = sorted(bucket_rows, key=lambda r: r["bucket"])

        top_summary = (
            scored.head(max(1, len(scored) // n_buckets))
            .agg({col: "mean" for col in payload.covariate_cols})
            .to_dict()
        )

        top_uplift_preview = scored.head(10)[payload.covariate_cols + ["uplift_score"]].to_dict(orient="records")

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=None,
            ci_high=None,
            assumptions=[
                "Ignorability given selected covariates.",
                "Treatment assignment probability is positive.",
                "Model ranking approximates treatment effect heterogeneity.",
            ],
            limitations=[
                "This MVP uses transformed-outcome random forest, not a full causal forest estimator.",
                "Scores are for prioritization and not guaranteed unbiased individual treatment effects.",
                "Bucket-level observed gaps can still be confounded in observational data.",
            ],
            diagnostics={
                "n_obs": int(len(df)),
                "treated_rate": float(t.mean()),
                "n_buckets": int(n_buckets),
                "sample_uplift_scores_preview": sample_scores,
                "bucket_summary": bucket_rows,
                "top_uplift_preview": top_uplift_preview,
                "top_segment_profile_mean": top_summary,
            },
        )
