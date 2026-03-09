"""FastAPI route definitions for the MVP causal workbench."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import (
    AnalyzeResponse,
    AutoAnalyzeResponse,
    DataSummaryResponse,
    FieldDetectionResponse,
    MethodRecommendationResponse,
    RecommendRequest,
)
from app.causal.base import CausalAnalysisInput
from app.reporting.report_generator import generate_markdown_report
from app.services.analysis_service import run_causal_analysis
from app.services.auto_analysis_service import run_auto_analysis
from app.services.data_service import summarize_dataframe
from app.services.field_detection_service import detect_fields
from app.services.llm_explainer import generate_llm_explanation
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


@router.post("/fields/detect", response_model=FieldDetectionResponse)
async def detect_fields_route(file: UploadFile = File(...)) -> FieldDetectionResponse:
    df = pd.read_csv(file.file)
    detected = detect_fields(df)
    return FieldDetectionResponse(defaults=detected.defaults, candidates=detected.candidates, notes=detected.notes)


@router.post("/methods/recommend", response_model=MethodRecommendationResponse)
def methods_recommend(payload: RecommendRequest) -> MethodRecommendationResponse:
    rec = recommend_methods(**payload.model_dump())
    return MethodRecommendationResponse(**rec.__dict__)


@router.post("/auto/analyze", response_model=AutoAnalyzeResponse)
async def auto_analyze(
    file: UploadFile = File(...),
    cutoff: float | None = Form(default=None),
    include_llm_explanation: bool = Form(default=True),
) -> AutoAnalyzeResponse:
    df = pd.read_csv(file.file)
    out = run_auto_analysis(df=df, cutoff=cutoff, include_llm_explanation=include_llm_explanation)

    report = generate_markdown_report(out.analysis_result)
    llm_explanation = None
    if include_llm_explanation:
        llm_explanation = generate_llm_explanation(
            method=out.recommended_method,
            result=out.analysis_result.__dict__,
            analysis_context={
                "detected_fields": out.detected_fields,
                "recommended_method": out.recommended_method,
                "recommendation_rationale": out.recommendation_rationale,
                "recommendation_limitations": out.recommendation_limitations,
            },
        )

    return AutoAnalyzeResponse(
        detected_fields=out.detected_fields,
        recommended_method=out.recommended_method,
        recommendation_rationale=out.recommendation_rationale,
        recommendation_limitations=out.recommendation_limitations,
        analysis_result=out.analysis_result.__dict__,
        report_markdown=report,
        llm_explanation=llm_explanation,
    )


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
    cf_n_estimators: int = Form(default=200),
    cf_min_samples_leaf: int = Form(default=5),
    rdd_bandwidth: float | None = Form(default=None),
    instrument_col: str | None = Form(default=None),
    include_llm_explanation: bool = Form(default=True),
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
        instrument_col=instrument_col,
        metadata={"filename": Path(file.filename).name if file.filename else "uploaded.csv"},
    )

    result = run_causal_analysis(
        method=method,
        payload=analysis_input,
        psm_caliper=psm_caliper,
        uplift_buckets=uplift_buckets,
        cf_n_estimators=cf_n_estimators,
        cf_min_samples_leaf=cf_min_samples_leaf,
        rdd_bandwidth=rdd_bandwidth,
    )
    report = generate_markdown_report(result)

    llm_explanation = None
    if include_llm_explanation:
        llm_explanation = generate_llm_explanation(
            method=method,
            result=result.__dict__,
            analysis_context={
                "treatment_col": treatment_col,
                "outcome_col": outcome_col,
                "covariates": covariate_cols,
                "time_col": time_col,
                "group_col": group_col,
                "running_col": running_col,
                "cutoff": cutoff,
                "instrument_col": instrument_col,
            },
        )

    return AnalyzeResponse(result=result.__dict__, report_markdown=report, llm_explanation=llm_explanation)
