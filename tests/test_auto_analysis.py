import io

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.field_detection_service import detect_fields
from app.services.recommendation_service import recommend_method_from_detected_fields

client = TestClient(app)


def test_field_detection_returns_defaults_and_candidates():
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1],
            "outcome": [10, 20, 15, 25],
            "period": ["pre", "pre", "post", "post"],
            "is_target_group": [0, 1, 0, 1],
            "age": [20, 30, 25, 35],
        }
    )
    out = detect_fields(df)
    assert out.defaults["treatment_col"] == "treatment"
    assert out.defaults["outcome_col"] == "outcome"
    assert isinstance(out.candidates["covariates"], list)


def test_recommend_method_from_detected_fields_prefers_did():
    rec = recommend_method_from_detected_fields(
        has_time_col=True,
        has_group_col=True,
        has_running_col=False,
        has_instrument_col=False,
        treatment_binary=True,
        wants_targeting=False,
    )
    assert rec.primary_method == "did"


def test_auto_analyze_success():
    with open("data/sample/hillstrom_style_sample.csv", "rb") as f:
        resp = client.post("/api/auto/analyze", files={"file": ("sample.csv", f, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert "detected_fields" in body
    assert "recommended_method" in body
    assert "analysis_result" in body
    assert "report_markdown" in body
    assert "llm_explanation" in body


def test_auto_analyze_graceful_degradation_when_detection_fails():
    bad_csv = io.BytesIO(b"foo\n1\n2\n3\n")
    resp = client.post("/api/auto/analyze", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] in {"auto_detection_failed", "invalid_analysis_input"}
