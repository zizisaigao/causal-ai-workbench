"""FastAPI route definitions for the MVP causal workbench."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import AnalyzeResponse, DataSummaryResponse, MethodRecommendationResponse, RecommendRequest
from app.causal.base import CausalAnalysisInput
from app.reporting.report_generator import generate_markdown_report
from app.services.analysis_service import run_causal_analysis
from app.services.data_service import summarize_dataframe
from app.services.recommendation_service import recommend_methods

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/data/summary", response_model=DataSummaryResponse)
async def data_summary(file: UploadFile = File(...)) -> DataSummaryResponse:
    df = pd.read_csv(file.file)
    summary = summarize_dataframe(df)
    return DataSummaryResponse(**summary.__dict__)


@router.post("/methods/recommend", response_model=MethodRecommendationResponse)
def methods_recommend(payload: RecommendRequest) -> MethodRecommendationResponse:
    rec = recommend_methods(**payload.model_dump())
    return MethodRecommendationResponse(**rec.__dict__)


@router.post("/analyze/{method}", response_model=AnalyzeResponse)
async def run_analysis(
    method: str,
    file: UploadFile = File(...),
    treatment_col: str = Form(...),
    outcome_col: str = Form(...),
    covariates: str = Form(...),
    time_col: str | None = Form(default=None),
    group_col: str | None = Form(default=None),
    running_col: str | None = Form(default=None),
    cutoff: float | None = Form(default=None),
    psm_caliper: float = Form(default=0.5),
    uplift_buckets: int = Form(default=5),
) -> AnalyzeResponse:
    df = pd.read_csv(file.file)
    covariate_cols = [c.strip() for c in covariates.split(",") if c.strip()]

    analysis_input = CausalAnalysisInput(
        data=df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        time_col=time_col,
        group_col=group_col,
        running_col=running_col,
        cutoff=cutoff,
        metadata={"filename": Path(file.filename).name if file.filename else "uploaded.csv"},
    )

    result = run_causal_analysis(
        method=method,
        payload=analysis_input,
        psm_caliper=psm_caliper,
        uplift_buckets=uplift_buckets,
    )
    report = generate_markdown_report(result)
    return AnalyzeResponse(result=result.__dict__, report_markdown=report)
