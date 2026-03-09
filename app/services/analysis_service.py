"""Orchestration service for causal method execution."""

from __future__ import annotations

from app.api.schemas import ApiError, MethodEnum
from app.causal.base import CausalAnalysisInput, CausalEstimate
from app.causal.did import DIDEstimator
from app.causal.psm import PSMEstimator
from app.causal.uplift import UpliftEstimator


def run_causal_analysis(method: str, payload: CausalAnalysisInput) -> CausalEstimate:
    method = method.lower().strip()
    if method not in MethodEnum.SUPPORTED:
        raise ApiError("unsupported_method", f"Unsupported method: {method}", status_code=422)

    estimator_map = {
        "psm": PSMEstimator(),
        "did": DIDEstimator(),
        "uplift": UpliftEstimator(),
    }

    try:
        return estimator_map[method].fit(payload)
    except ValueError as exc:
        raise ApiError("invalid_analysis_input", str(exc), status_code=422) from exc
