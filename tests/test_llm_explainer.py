import os

from app.services import llm_explainer
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
    assert exp["fallback_reason"].startswith("connection_failed:")


def test_timeout_is_configurable_and_classified(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "180")

    def _raise_timeout(*_args, **_kwargs):
        raise llm_explainer.LLMError("timeout", "request timed out after 180.0s")

    monkeypatch.setattr(llm_explainer, "_http_post_json", _raise_timeout)
    exp = generate_llm_explanation("did", {"ate": 1.0, "diagnostics": {}}, {"group_col": "g"})

    assert exp["mode"] == "template"
    assert exp["fallback_reason"].startswith("timeout:")


def test_model_response_error_is_classified(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    def _bad_payload(*_args, **_kwargs):
        return {"done": True}

    monkeypatch.setattr(llm_explainer, "_http_post_json", _bad_payload)
    exp = generate_llm_explanation("uplift", {"ate": 0.1, "diagnostics": {}}, {"treatment_col": "t"})

    assert exp["mode"] == "template"
    assert exp["fallback_reason"].startswith("model_response_error:")
