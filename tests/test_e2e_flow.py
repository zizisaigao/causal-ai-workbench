from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
SAMPLE = Path("data/sample/hillstrom_style_sample.csv")


def test_end_to_end_psm_flow():
    with SAMPLE.open("rb") as f:
        summary_resp = client.post("/api/data/summary", files={"file": (SAMPLE.name, f, "text/csv")})
    assert summary_resp.status_code == 200
    assert summary_resp.json()["n_rows"] > 0

    rec_resp = client.post(
        "/api/methods/recommend",
        json={
            "has_time": False,
            "has_group": False,
            "treatment_binary": True,
            "observational": True,
            "wants_targeting": False,
        },
    )
    assert rec_resp.status_code == 200
    assert rec_resp.json()["primary_method"] == "psm"

    with SAMPLE.open("rb") as f:
        analyze_resp = client.post(
            "/api/analyze/psm",
            files={"file": (SAMPLE.name, f, "text/csv")},
            data={
                "treatment_col": "treatment",
                "outcome_col": "outcome",
                "covariates": "age,income,prior_spend",
                "psm_caliper": "1.0",
            },
        )
    assert analyze_resp.status_code == 200
    body = analyze_resp.json()
    assert body["status"] == "success"
    assert body["result"]["method"] == "psm"
    assert "免责声明" in body["report_markdown"]


def test_unsupported_method_returns_unified_error():
    with SAMPLE.open("rb") as f:
        resp = client.post(
            "/api/analyze/unknown",
            files={"file": (SAMPLE.name, f, "text/csv")},
            data={
                "treatment_col": "treatment",
                "outcome_col": "outcome",
                "covariates": "age,income,prior_spend",
                "psm_caliper": "1.0",
            },
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unsupported_method"


def test_psm_no_match_returns_business_error():
    with SAMPLE.open("rb") as f:
        resp = client.post(
            "/api/analyze/psm",
            files={"file": (SAMPLE.name, f, "text/csv")},
            data={
                "treatment_col": "treatment",
                "outcome_col": "outcome",
                "covariates": "age,income,prior_spend",
                "psm_caliper": "0.000001",
            },
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "psm_no_matches"
    assert "sample size is too small" in body["error"]["message"]
