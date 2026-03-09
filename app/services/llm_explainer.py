"""LLM-based (or template fallback) explanation layer for causal results.

This module never computes causal effects. It only explains existing outputs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


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
        with urllib.request.urlopen(req, timeout=45) as resp:  # nosec - controlled URL via env
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _call_openai(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    body = _http_post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful causal reporting assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def _call_ollama(prompt: str) -> dict[str, Any]:
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise RuntimeError("OLLAMA_MODEL not set")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    body = _http_post_json(
        f"{base_url.rstrip('/')}/api/generate",
        payload={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        headers={},
    )
    content = body.get("response", "{}")
    return json.loads(content)


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
        return _template_explanation(method, result, analysis_context, reason="No LLM credentials/config found")
    except Exception as exc:
        return _template_explanation(method, result, analysis_context, reason=f"LLM unavailable: {exc}")
