"""FastAPI route definitions for the MVP causal workbench."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.causal.base import CausalAnalysisInput
from app.causal.did import DIDEstimator
from app.causal.psm import PSMEstimator
from app.causal.uplift import UpliftEstimator
from app.reporting.report_generator import generate_markdown_report
from app.services.data_service import summarize_dataframe
from app.services.recommendation_service import recommend_methods

router = APIRouter()


class RecommendRequest(BaseModel):
    has_time: bool = False
    has_group: bool = False
    treatment_binary: bool = True
    observational: bool = True
    wants_targeting: bool = False


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/data/summary")
async def data_summary(file: UploadFile = File(...)) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    df = pd.read_csv(tmp_path)
    summary = summarize_dataframe(df)
    return summary.__dict__


@router.post("/methods/recommend")
def methods_recommend(payload: RecommendRequest) -> dict:
    rec = recommend_methods(**payload.model_dump())
    return rec.__dict__


@router.post("/analyze/{method}")
async def run_analysis(
    method: str,
    file: UploadFile = File(...),
    treatment_col: str = Form(...),
    outcome_col: str = Form(...),
    covariates: str = Form(...),
    time_col: str | None = Form(default=None),
    group_col: str | None = Form(default=None),
) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    df = pd.read_csv(tmp_path)
    covariate_cols = [c.strip() for c in covariates.split(",") if c.strip()]

    analysis_input = CausalAnalysisInput(
        data=df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        time_col=time_col,
        group_col=group_col,
    )

    estimator_map = {
        "psm": PSMEstimator(),
        "did": DIDEstimator(),
        "uplift": UpliftEstimator(),
    }
    if method not in estimator_map:
        return {"error": f"Unsupported method: {method}"}

    result = estimator_map[method].fit(analysis_input)
    report = generate_markdown_report(result)
    return {"result": result.__dict__, "report_markdown": report}
