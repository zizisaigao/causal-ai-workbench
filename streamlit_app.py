"""Enhanced Streamlit demo for causal-ai-workbench MVP with auto-detect/recommend flow."""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Causal AI Workbench Demo", layout="wide")

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"

st.title("Causal AI Workbench · MVP Demo")
st.caption("上传 CSV 后可自动识别字段、推荐方法并一键分析，也可手动覆盖后运行。")

api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

if "analysis_payload" not in st.session_state:
    st.session_state.analysis_payload = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None
if "detected_fields" not in st.session_state:
    st.session_state.detected_fields = None
if "auto_summary" not in st.session_state:
    st.session_state.auto_summary = None

if not uploaded:
    st.info("请先上传 CSV 文件。")
    st.stop()

file_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(file_bytes))
columns = df.columns.tolist()

st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)

# Auto detect fields
if st.session_state.detected_fields is None:
    try:
        files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
        r = requests.post(f"{api_base}/fields/detect", files=files, timeout=60)
        if r.status_code == 200:
            st.session_state.detected_fields = r.json()
        else:
            st.session_state.detected_fields = {"defaults": {}, "candidates": {}, "notes": ["字段自动识别失败，已回退为手动选择。"]}
    except Exception:
        st.session_state.detected_fields = {"defaults": {}, "candidates": {}, "notes": ["字段自动识别不可用，已回退为手动选择。"]}

defaults = (st.session_state.detected_fields or {}).get("defaults", {})

st.markdown("### 自动识别与推荐")
if st.session_state.detected_fields:
    st.json(st.session_state.detected_fields)

if st.button("Run Recommended Analysis", type="secondary"):
    try:
        files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
        r = requests.post(f"{api_base}/auto/analyze", files=files, data={"include_llm_explanation": "true"}, timeout=120)
        if r.status_code == 200:
            st.session_state.analysis_payload = {
                "result": r.json().get("analysis_result"),
                "report_markdown": r.json().get("report_markdown"),
                "llm_explanation": r.json().get("llm_explanation"),
            }
            st.session_state.auto_summary = {
                "recommended_method": r.json().get("recommended_method"),
                "recommendation_rationale": r.json().get("recommendation_rationale"),
                "recommendation_limitations": r.json().get("recommendation_limitations"),
            }
            st.session_state.analysis_error = None
        else:
            st.session_state.analysis_payload = None
            st.session_state.analysis_error = r.json()
    except requests.RequestException as exc:
        st.session_state.analysis_payload = None
        st.session_state.analysis_error = {"error": {"code": "request_exception", "message": str(exc)}}

if st.session_state.auto_summary:
    st.success(f"Recommended method: {st.session_state.auto_summary['recommended_method']}")
    st.write("Rationale:", st.session_state.auto_summary["recommendation_rationale"])
    st.write("Limitations:", st.session_state.auto_summary["recommendation_limitations"])

st.markdown("### 手动分析（可覆盖自动识别）")
method = st.selectbox("分析方法", ["did", "psm", "uplift", "causal_forest", "rdd", "iv"])

def _index_of(col_name: str | None) -> int:
    if col_name and col_name in columns:
        return columns.index(col_name)
    return 0

col1, col2 = st.columns(2)
with col1:
    treatment_col = st.selectbox("treatment_col", columns, index=_index_of(defaults.get("treatment_col")))
with col2:
    outcome_col = st.selectbox("outcome_col", columns, index=_index_of(defaults.get("outcome_col")))

cov_defaults = [c for c in defaults.get("covariates", []) if c in columns]
covariates = st.multiselect("covariates（多选）", options=columns, default=cov_defaults)

time_col = ""
group_col = ""
psm_caliper = 1.0
uplift_buckets = 5
cf_n_estimators = 200
cf_min_samples_leaf = 5
running_col = ""
cutoff = 0.0
rdd_bandwidth = 1.0
instrument_col = ""

if method == "did":
    p1, p2 = st.columns(2)
    with p1:
        time_col = st.selectbox("time_col", [""] + columns, index=([""] + columns).index(defaults.get("time_col")) if defaults.get("time_col") in columns else 0)
    with p2:
        group_col = st.selectbox("group_col", [""] + columns, index=([""] + columns).index(defaults.get("group_col")) if defaults.get("group_col") in columns else 0)
elif method == "psm":
    psm_caliper = st.number_input("psm_caliper", min_value=0.000001, value=1.0, step=0.1, format="%.6f")
elif method == "uplift":
    uplift_buckets = st.selectbox("uplift_buckets", [5, 10], index=0)
elif method == "causal_forest":
    c1, c2, c3 = st.columns(3)
    with c1:
        uplift_buckets = st.selectbox("cf_buckets", [5, 10], index=0)
    with c2:
        cf_n_estimators = st.number_input("cf_n_estimators", min_value=50, value=200, step=50)
    with c3:
        cf_min_samples_leaf = st.number_input("cf_min_samples_leaf", min_value=1, value=5, step=1)
elif method == "rdd":
    c1, c2, c3 = st.columns(3)
    with c1:
        running_col = st.selectbox("running_col", [""] + columns, index=([""] + columns).index(defaults.get("running_col")) if defaults.get("running_col") in columns else 0)
    with c2:
        cutoff = st.number_input("cutoff", value=0.0, step=0.1)
    with c3:
        rdd_bandwidth = st.number_input("rdd_bandwidth", min_value=0.0001, value=1.0, step=0.1)
else:
    instrument_col = st.selectbox("instrument_col", [""] + columns, index=([""] + columns).index(defaults.get("instrument_col")) if defaults.get("instrument_col") in columns else 0)

if st.button("运行分析", type="primary"):
    files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
    data = {
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "covariates": ",".join(covariates),
        "include_llm_explanation": "true",
    }
    if method == "did":
        data["time_col"] = time_col
        data["group_col"] = group_col
    elif method == "psm":
        data["psm_caliper"] = str(psm_caliper)
    elif method == "uplift":
        data["uplift_buckets"] = str(uplift_buckets)
    elif method == "causal_forest":
        data["uplift_buckets"] = str(uplift_buckets)
        data["cf_n_estimators"] = str(cf_n_estimators)
        data["cf_min_samples_leaf"] = str(cf_min_samples_leaf)
    elif method == "rdd":
        data["running_col"] = running_col
        data["cutoff"] = str(cutoff)
        data["rdd_bandwidth"] = str(rdd_bandwidth)
    else:
        data["instrument_col"] = instrument_col

    try:
        r = requests.post(f"{api_base}/analyze/{method}", files=files, data=data, timeout=120)
        if r.status_code == 200:
            st.session_state.analysis_payload = r.json()
            st.session_state.analysis_error = None
        else:
            st.session_state.analysis_payload = None
            st.session_state.analysis_error = r.json()
    except requests.RequestException as exc:
        st.session_state.analysis_payload = None
        st.session_state.analysis_error = {"error": {"code": "request_exception", "message": str(exc)}}

st.markdown("---")
st.subheader("分析结果")

left, right = st.columns([1, 1])
with left:
    st.markdown("#### 核心结果（JSON）")
    if st.session_state.analysis_payload:
        st.json(st.session_state.analysis_payload.get("result", {}))
    else:
        st.info("尚无成功结果。")
with right:
    st.markdown("#### 错误提示")
    if st.session_state.analysis_error:
        err = st.session_state.analysis_error.get("error", {})
        st.error(f"{err.get('code', 'unknown_error')}: {err.get('message', '未知错误')}")
        st.json(st.session_state.analysis_error)
    else:
        st.success("当前无错误。")

st.markdown("#### Markdown 报告")
if st.session_state.analysis_payload:
    report_markdown = st.session_state.analysis_payload.get("report_markdown", "")
    st.markdown(report_markdown)
    st.download_button("下载 Markdown 报告", data=report_markdown, file_name=f"causal_report_{method}.md", mime="text/markdown")
else:
    st.info("运行成功后将在此展示报告，并可下载 Markdown 文件。")

st.markdown("#### AI Explanation")
if st.session_state.analysis_payload:
    llm_exp = st.session_state.analysis_payload.get("llm_explanation")
    if llm_exp:
        mode = llm_exp.get("mode", "unknown")
        if mode == "template":
            st.info(f"当前使用模板化解释（{llm_exp.get('fallback_reason', 'no llm')}）。")
        else:
            st.success(f"当前使用 LLM 解释模式：{mode}")
        st.markdown(f"**Executive Summary**\n\n{llm_exp.get('executive_summary', '')}")
        st.markdown(f"**Method Why**\n\n{llm_exp.get('method_why', '')}")
        st.markdown("**Key Findings**")
        for item in llm_exp.get("key_findings", []):
            st.markdown(f"- {item}")
        st.markdown("**Caveats**")
        for item in llm_exp.get("caveats", []):
            st.markdown(f"- {item}")
        st.markdown("**Business Takeaways**")
        for item in llm_exp.get("business_takeaways", []):
            st.markdown(f"- {item}")
    else:
        st.info("本次未返回 AI Explanation。")
else:
    st.info("运行成功后将在此展示 AI Explanation。")
