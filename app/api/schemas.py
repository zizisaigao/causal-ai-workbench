"""Request and response schemas for API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: dict[str, Any]


class RecommendRequest(BaseModel):
    has_time: bool = False
    has_group: bool = False
    treatment_binary: bool = True
    observational: bool = True
    wants_targeting: bool = False


class MethodRecommendationResponse(BaseModel):
    primary_method: str
    alternatives: list[str]
    rationale: list[str]


class AnalyzeResultResponse(BaseModel):
    method: str
    ate: float | None
    ci_low: float | None
    ci_high: float | None
    assumptions: list[str]
    limitations: list[str]
    diagnostics: dict[str, Any]
    subgroup_effects: dict[str, float]


class AnalyzeResponse(BaseModel):
    status: Literal["success"] = "success"
    result: AnalyzeResultResponse
    report_markdown: str


class DataSummaryResponse(BaseModel):
    n_rows: int
    n_cols: int
    missing_by_column: dict[str, int]
    dtypes: dict[str, str]
    duplicated_rows: int
    numeric_outliers_iqr: dict[str, int]


class ApiError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class MethodEnum:
    SUPPORTED = {"psm", "did", "uplift", "causal_forest", "rdd", "iv"}
