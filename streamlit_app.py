"""Enhanced minimal Streamlit demo for causal-ai-workbench MVP."""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Causal AI Workbench Demo", layout="wide")

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"

st.title("Causal AI Workbench · MVP Demo")
st.caption("上传 CSV，选择字段并调用现有 FastAPI 接口运行 DID / PSM / Uplift / CausalForest / RDD / IV 分析。")

api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

if "analysis_payload" not in st.session_state:
    st.session_state.analysis_payload = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

if not uploaded:
    st.info("请先上传 CSV 文件。")
    st.stop()

file_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(file_bytes))

st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)

columns = df.columns.tolist()
method = st.selectbox("分析方法", ["did", "psm", "uplift", "causal_forest", "rdd", "iv"])

col1, col2 = st.columns(2)
with col1:
    treatment_col = st.selectbox("treatment_col", columns, index=columns.index("treatment") if "treatment" in columns else 0)
with col2:
    outcome_col = st.selectbox("outcome_col", columns, index=columns.index("outcome") if "outcome" in columns else 0)

covariates_default = [c for c in ["age", "income", "prior_spend"] if c in columns]
covariates = st.multiselect("covariates（多选）", options=columns, default=covariates_default)

st.markdown("### 方法参数")
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
        time_col = st.selectbox("time_col", [""] + columns, index=([""] + columns).index("period") if "period" in columns else 0)
    with p2:
        group_col = st.selectbox(
            "group_col",
            [""] + columns,
            index=([""] + columns).index("is_target_group") if "is_target_group" in columns else 0,
        )
    st.caption("DID 需要 time_col 与 group_col。")
elif method == "psm":
    psm_caliper = st.number_input("psm_caliper", min_value=0.000001, value=1.0, step=0.1, format="%.6f")
    st.caption("PSM 可能因样本重叠不足失败；可适当提高 caliper。")
elif method == "uplift":
    uplift_buckets = st.selectbox("uplift_buckets", [5, 10], index=0)
    st.caption("Uplift 会输出样本排序分数与分桶统计，用于干预优先级建议。")
elif method == "causal_forest":
    c1, c2, c3 = st.columns(3)
    with c1:
        uplift_buckets = st.selectbox("cf_buckets", [5, 10], index=0)
    with c2:
        cf_n_estimators = st.number_input("cf_n_estimators", min_value=50, value=200, step=50)
    with c3:
        cf_min_samples_leaf = st.number_input("cf_min_samples_leaf", min_value=1, value=5, step=1)
    st.caption("Causal Forest 优先使用 econml；若不可用会自动降级到近似方法。")
elif method == "rdd":
    c1, c2, c3 = st.columns(3)
    with c1:
        running_col = st.selectbox("running_col", [""] + columns)
    with c2:
        cutoff = st.number_input("cutoff", value=0.0, step=0.1)
    with c3:
        rdd_bandwidth = st.number_input("rdd_bandwidth", min_value=0.0001, value=1.0, step=0.1)
    st.caption("RDD 估计的是 cutoff 附近的局部效应。")
else:
    instrument_col = st.selectbox("instrument_col", [""] + columns)
    st.caption("IV 使用两阶段最小二乘(2SLS)，请确保工具变量相关且满足排除限制。")

run_clicked = st.button("运行分析", type="primary")

if run_clicked:
    if not covariates:
        st.session_state.analysis_error = {"error": {"code": "invalid_input", "message": "请至少选择一个 covariate。"}}
        st.session_state.analysis_payload = None
    elif method == "did" and (not time_col or not group_col):
        st.session_state.analysis_error = {
            "error": {"code": "invalid_input", "message": "DID 需要选择 time_col 和 group_col。"}
        }
        st.session_state.analysis_payload = None
    elif method == "rdd" and not running_col:
        st.session_state.analysis_error = {"error": {"code": "invalid_input", "message": "RDD 需要 running_col。"}}
        st.session_state.analysis_payload = None
    elif method == "iv" and not instrument_col:
        st.session_state.analysis_error = {"error": {"code": "invalid_input", "message": "IV 需要 instrument_col。"}}
        st.session_state.analysis_payload = None
    else:
        files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
        data = {
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "covariates": ",".join(covariates),
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
            with st.spinner("调用后端分析中..."):
                response = requests.post(f"{api_base}/analyze/{method}", files=files, data=data, timeout=120)
            if response.status_code == 200:
                st.session_state.analysis_payload = response.json()
                st.session_state.analysis_error = None
            else:
                st.session_state.analysis_payload = None
                st.session_state.analysis_error = response.json()
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
    st.download_button(
        "下载 Markdown 报告",
        data=report_markdown,
        file_name=f"causal_report_{method}.md",
        mime="text/markdown",
    )
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

if st.session_state.analysis_payload and method in {"uplift", "causal_forest"}:
    diagnostics = st.session_state.analysis_payload.get("result", {}).get("diagnostics", {})

    st.markdown("#### 异质性样本排序预览")
    preview_rows = diagnostics.get("sample_uplift_scores_preview", diagnostics.get("sample_effect_scores_preview", []))
    if preview_rows:
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    st.markdown("#### 分桶结果")
    bucket_rows = diagnostics.get("bucket_summary", [])
    if bucket_rows:
        st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True)

    st.markdown("#### Top 人群摘要")
    st.json(diagnostics.get("top_segment_profile_mean", {}))
