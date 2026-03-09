"""LLM-based (or template fallback) explanation layer for causal results.

This module never computes causal effects. It only explains existing outputs.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any


DEFAULT_LLM_TIMEOUT_SECONDS = 120.0


class LLMError(RuntimeError):
    """Structured LLM integration error for deterministic fallback reasons."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _llm_timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)).strip()
    try:
        value = float(raw)
        return value if value > 0 else DEFAULT_LLM_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS


def _method_specific_caveat(method: str) -> str:
    caveats = {
        "did": "DID relies on parallel trends; violations can bias effect estimates.",
        "psm": "PSM assumes no unobserved confounding and adequate overlap.",
        "uplift": "Uplift scores are prioritization signals, not unbiased individual causal effects.",
        "causal_forest": "CATE rankings may be unstable under weak overlap or limited sample size.",
        "rdd": "RDD identifies a local effect near the cutoff, not a population-wide average effect.",
        "iv": "IV requires strong relevance, exclusion restriction, and instrument exogeneity assumptions.",
    }
    return caveats.get(method, "Interpret results with method assumptions and data limitations in mind.")


def _template_explanation(method: str, result: dict[str, Any], analysis_context: dict[str, Any], reason: str) -> dict[str, Any]:
    ate = result.get("ate")
    ci_low, ci_high = result.get("ci_low"), result.get("ci_high")
    diagnostics = result.get("diagnostics", {})

    return {
        "mode": "template",
        "fallback_reason": reason,
        "executive_summary": (
            f"Method `{method}` completed. Estimated effect is {ate}"
            + (f" with 95% CI [{ci_low}, {ci_high}]" if ci_low is not None and ci_high is not None else ".")
        ),
        "method_why": f"`{method}` was used based on selected columns/analysis design: {analysis_context}.",
        "key_findings": [
            f"Core estimate: {ate}",
            f"Diagnostics snapshot: {str(diagnostics)[:300]}",
        ],
        "caveats": [
            _method_specific_caveat(method),
            "LLM assistant does not compute causal effects; it only explains model outputs.",
        ],
        "business_takeaways": [
            "Use this estimate as decision support, not as a standalone proof.",
            "Validate with robustness checks and domain knowledge before action.",
        ],
    }


def _build_prompt(method: str, result: dict[str, Any], analysis_context: dict[str, Any]) -> str:
    return (
        "You are a cautious causal analysis explainer. "
        "You must not recompute statistics; only explain provided outputs. "
        "Do not overstate causality. Distinguish average effects, local effects (RDD), and ranking scores (uplift/CATE). "
        "Return STRICT JSON with keys: executive_summary, method_why, key_findings (array), caveats (array), business_takeaways (array).\n\n"
        f"method={method}\n"
        f"analysis_context={json.dumps(analysis_context, ensure_ascii=False)}\n"
        f"result={json.dumps(result, ensure_ascii=False)}\n"
        f"method_specific_limit={_method_specific_caveat(method)}"
    )


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_llm_timeout_seconds()) as resp:  # nosec - controlled URL via env
            return json.loads(resp.read().decode("utf-8"))
    except TimeoutError as exc:
        raise LLMError("timeout", f"request timed out after {_llm_timeout_seconds()}s") from exc
    except socket.timeout as exc:
        raise LLMError("timeout", f"request timed out after {_llm_timeout_seconds()}s") from exc
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:300]
        except Exception:  # pragma: no cover
            body = ""
        raise LLMError("http_error", f"status={exc.code}, body_preview={body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError | socket.timeout):
            raise LLMError("timeout", f"request timed out after {_llm_timeout_seconds()}s") from exc
        raise LLMError("connection_failed", f"unable to connect to LLM endpoint: {reason}") from exc


def _call_openai(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("config_missing", "OPENAI_API_KEY not set")

    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMError("config_missing", "openai SDK not installed; run `pip install openai`") from exc

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    timeout = _llm_timeout_seconds()

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are a careful causal reporting assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except APITimeoutError as exc:
        raise LLMError("timeout", f"request timed out after {timeout}s") from exc
    except APIConnectionError as exc:
        raise LLMError("connection_failed", f"failed to reach OpenAI endpoint: {exc}") from exc
    except APIStatusError as exc:
        body_preview = ""
        if getattr(exc, "response", None) is not None:
            try:
                body_preview = str(exc.response.text)[:200]
            except Exception:  # pragma: no cover
                body_preview = ""
        raise LLMError("http_error", f"status={exc.status_code}, body_preview={body_preview or str(exc)}") from exc
    except Exception as exc:
        raise LLMError("response_parse_error", f"OpenAI request failed before parse: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMError("response_structure_error", "OpenAI response missing output_text")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        preview = output_text[:200].replace("\n", " ")
        raise LLMError("response_parse_error", f"model output is not valid JSON; preview={preview}; error={exc}") from exc


def _call_ollama(prompt: str) -> dict[str, Any]:
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise LLMError("config_missing", "OLLAMA_MODEL not set")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    body = _http_post_json(
        f"{base_url.rstrip('/')}/api/generate",
        payload={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        headers={},
    )
    content = body.get("response")
    if content is None:
        raise LLMError("response_structure_error", "missing `response` field in Ollama output")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        preview = content[:200].replace("\n", " ")
        raise LLMError("response_parse_error", f"invalid JSON payload from Ollama; preview={preview}; error={exc}") from exc


def generate_llm_explanation(method: str, result: dict[str, Any], analysis_context: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(method=method, result=result, analysis_context=analysis_context)

    try:
        if os.getenv("OPENAI_API_KEY"):
            parsed = _call_openai(prompt)
            parsed["mode"] = "llm_openai"
            return parsed
        if os.getenv("OLLAMA_MODEL"):
            parsed = _call_ollama(prompt)
            parsed["mode"] = "llm_ollama"
            return parsed
        return _template_explanation(method, result, analysis_context, reason="config_missing: no LLM credentials/config found")
    except LLMError as exc:
        return _template_explanation(method, result, analysis_context, reason=f"{exc.code}: {exc.message}")
    except Exception as exc:  # pragma: no cover
        return _template_explanation(method, result, analysis_context, reason=f"response_structure_error: unexpected error: {exc}")
