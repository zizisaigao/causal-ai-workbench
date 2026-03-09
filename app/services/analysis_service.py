"""Orchestration service for causal method execution."""

from __future__ import annotations

from app.api.schemas import ApiError, MethodEnum
from app.causal.base import CausalAnalysisInput, CausalEstimate
from app.causal.causal_forest import CausalForestEstimator
from app.causal.did import DIDEstimator
from app.causal.psm import PSMEstimator, PSMNoMatchError
from app.causal.uplift import UpliftEstimator


def run_causal_analysis(
    method: str,
    payload: CausalAnalysisInput,
    psm_caliper: float = 0.5,
    uplift_buckets: int = 5,
    cf_n_estimators: int = 200,
    cf_min_samples_leaf: int = 5,
) -> CausalEstimate:
    method = method.lower().strip()
    if method not in MethodEnum.SUPPORTED:
        raise ApiError("unsupported_method", f"Unsupported method: {method}", status_code=422)

    estimator_map = {
        "psm": PSMEstimator(caliper=psm_caliper),
        "did": DIDEstimator(),
        "uplift": UpliftEstimator(n_buckets=uplift_buckets),
        "causal_forest": CausalForestEstimator(
            n_estimators=cf_n_estimators,
            min_samples_leaf=cf_min_samples_leaf,
            n_buckets=uplift_buckets,
        ),
    }

    try:
        return estimator_map[method].fit(payload)
    except PSMNoMatchError as exc:
        raise ApiError("psm_no_matches", str(exc), status_code=422) from exc
    except ValueError as exc:
        raise ApiError("invalid_analysis_input", str(exc), status_code=422) from exc
