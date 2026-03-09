import pandas as pd

from app.causal.base import CausalAnalysisInput
from app.causal.psm import PSMEstimator
from app.causal.uplift import UpliftEstimator


def _toy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'treatment': [1, 1, 1, 0, 0, 0, 1, 0],
            'outcome': [10, 12, 11, 7, 8, 6, 13, 5],
            'x1': [1.0, 1.1, 0.9, 0.4, 0.3, 0.5, 1.2, 0.2],
            'x2': [5, 6, 5, 2, 3, 2, 6, 1],
        }
    )


def test_psm_runs():
    payload = CausalAnalysisInput(
        data=_toy_data(),
        treatment_col='treatment',
        outcome_col='outcome',
        covariate_cols=['x1', 'x2'],
    )
    result = PSMEstimator(caliper=1.0).fit(payload)
    assert result.method == 'psm'
    assert result.ate is not None


def test_uplift_runs():
    payload = CausalAnalysisInput(
        data=_toy_data(),
        treatment_col='treatment',
        outcome_col='outcome',
        covariate_cols=['x1', 'x2'],
    )
    result = UpliftEstimator(n_estimators=10).fit(payload)
    assert result.method == 'uplift'
    assert isinstance(result.diagnostics.get('top_uplift_preview'), list)
