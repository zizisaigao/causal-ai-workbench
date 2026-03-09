import pandas as pd

from app.causal.base import CausalAnalysisInput
from app.causal.did import DIDEstimator


def test_did_runs():
    df = pd.DataFrame(
        {
            'outcome': [10, 11, 12, 14, 9, 9, 10, 10],
            'period': ['pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post'],
            'is_target_group': [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    payload = CausalAnalysisInput(
        data=df,
        treatment_col='is_target_group',
        outcome_col='outcome',
        covariate_cols=[],
        time_col='period',
        group_col='is_target_group',
    )
    result = DIDEstimator().fit(payload)
    assert result.method == 'did'
    assert result.ate is not None
