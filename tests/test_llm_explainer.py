import os

from app.services.llm_explainer import generate_llm_explanation


def test_template_explanation_when_no_llm_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    result = {
        "ate": 1.23,
        "ci_low": 0.5,
        "ci_high": 2.0,
        "diagnostics": {"n_obs": 100},
    }
    exp = generate_llm_explanation("did", result, {"treatment_col": "t"})

    assert exp["mode"] == "template"
    assert "executive_summary" in exp
    assert isinstance(exp["key_findings"], list)


def test_template_fallback_when_llm_call_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")

    result = {"ate": 0.8, "ci_low": None, "ci_high": None, "diagnostics": {}}
    exp = generate_llm_explanation("rdd", result, {"running_col": "x", "cutoff": 0})

    assert exp["mode"] == "template"
    assert "LLM unavailable" in exp["fallback_reason"]
