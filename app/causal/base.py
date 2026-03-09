"""Shared interfaces and data models for causal estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CausalAnalysisInput:
    """Standardized input payload for all causal methods."""

    data: pd.DataFrame
    treatment_col: str
    outcome_col: str
    covariate_cols: list[str]
    time_col: str | None = None
    group_col: str | None = None
    running_col: str | None = None
    cutoff: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEstimate:
    """Common output object for all estimators."""

    method: str
    ate: float | None
    ci_low: float | None
    ci_high: float | None
    assumptions: list[str]
    limitations: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    subgroup_effects: dict[str, float] = field(default_factory=dict)


class BaseCausalEstimator(ABC):
    """Unified interface to keep methods composable and testable."""

    method_name: str

    @abstractmethod
    def fit(self, payload: CausalAnalysisInput) -> CausalEstimate:
        """Fit the method and return standardized results."""
        raise NotImplementedError
