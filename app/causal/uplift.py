"""Simple uplift / heterogeneity analysis using doubly robust style transformed outcome proxy."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class UpliftEstimator(BaseCausalEstimator):
    method_name = "uplift"

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

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

        ranking = (
            df.assign(uplift=uplift_scores)
            .sort_values("uplift", ascending=False)
            .head(10)[payload.covariate_cols + ["uplift"]]
            .to_dict(orient="records")
        )

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=None,
            ci_high=None,
            assumptions=[
                "Ignorability given selected covariates.",
                "Treatment assignment probability is positive.",
                "Model captures meaningful treatment effect heterogeneity.",
            ],
            limitations=[
                "This MVP uses a transformed-outcome approximation, not a full causal forest.",
                "Uncertainty intervals for uplift are not provided yet.",
                "Results can be unstable for small sample sizes.",
            ],
            diagnostics={
                "n_obs": int(len(df)),
                "treated_rate": float(t.mean()),
                "top_uplift_preview": ranking,
            },
        )
