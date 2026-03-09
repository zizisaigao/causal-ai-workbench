"""Propensity score matching estimator for observational cross-sectional data."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .base import BaseCausalEstimator, CausalAnalysisInput, CausalEstimate


class PSMEstimator(BaseCausalEstimator):
    method_name = "psm"

    def __init__(self, caliper: float = 0.1):
        self.caliper = caliper

    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        df = payload.data.dropna(subset=[payload.treatment_col, payload.outcome_col] + payload.covariate_cols).copy()
        treated_mask = df[payload.treatment_col] == 1
        control_mask = df[payload.treatment_col] == 0

        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("PSM requires both treated and control samples.")

        x = df[payload.covariate_cols]
        t = df[payload.treatment_col]
        y = df[payload.outcome_col]

        model = LogisticRegression(max_iter=1000)
        model.fit(x, t)
        propensity = model.predict_proba(x)[:, 1]

        treated_scores = propensity[treated_mask]
        control_scores = propensity[control_mask]
        control_outcomes = y[control_mask].to_numpy()
        treated_outcomes = y[treated_mask].to_numpy()

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_scores.reshape(-1, 1))
        distances, indices = nn.kneighbors(treated_scores.reshape(-1, 1))

        within_caliper = distances.flatten() <= self.caliper
        if not within_caliper.any():
            raise ValueError("No matched pairs found within caliper. Try a larger caliper.")

        matched_treated = treated_outcomes[within_caliper]
        matched_control = control_outcomes[indices.flatten()[within_caliper]]
        effects = matched_treated - matched_control

        ate = float(np.mean(effects))
        se = np.std(effects, ddof=1) / np.sqrt(len(effects)) if len(effects) > 1 else 0.0
        z = norm.ppf(0.975)

        return CausalEstimate(
            method=self.method_name,
            ate=ate,
            ci_low=float(ate - z * se),
            ci_high=float(ate + z * se),
            assumptions=[
                "No unobserved confounding (selection on observables).",
                "Positivity holds for covariates across treatment groups.",
                "Stable unit treatment value assumption (SUTVA).",
            ],
            limitations=[
                "Sensitive to omitted confounders.",
                "Matching quality depends on propensity model specification.",
                "Can lose samples when caliper is strict.",
            ],
            diagnostics={
                "n_total": int(len(df)),
                "n_treated": int(treated_mask.sum()),
                "n_control": int(control_mask.sum()),
                "n_matched_pairs": int(len(effects)),
                "caliper": self.caliper,
            },
        )
