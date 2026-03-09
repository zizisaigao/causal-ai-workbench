"""Minimal Streamlit demo for causal-ai-workbench MVP."""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Causal AI Workbench Demo", layout="wide")

DEFAULT_API_BASE = "http://127.0.0.1:8000/api"

st.title("Causal AI Workbench · MVP Demo")
st.caption("上传 CSV，选择字段并调用现有 FastAPI 接口运行 DID / PSM 分析。")

api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

if not uploaded:
    st.info("请先上传 CSV 文件。")
    st.stop()

file_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(file_bytes))

st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)

columns = df.columns.tolist()

col1, col2, col3 = st.columns(3)
with col1:
    treatment_col = st.selectbox("treatment_col", columns, index=columns.index("treatment") if "treatment" in columns else 0)
with col2:
    outcome_col = st.selectbox("outcome_col", columns, index=columns.index("outcome") if "outcome" in columns else 0)
with col3:
    method = st.selectbox("分析方法", ["did", "psm"])

covariates_default = [c for c in ["age", "income", "prior_spend"] if c in columns]
covariates = st.multiselect("covariates", options=columns, default=covariates_default)

optional1, optional2, optional3 = st.columns(3)
with optional1:
    time_col = st.selectbox("time_col (可选)", [""] + columns, index=( [""] + columns ).index("period") if "period" in columns else 0)
with optional2:
    group_col = st.selectbox("group_col (可选)", [""] + columns, index=( [""] + columns ).index("is_target_group") if "is_target_group" in columns else 0)
with optional3:
    psm_caliper = st.number_input("psm_caliper (仅 PSM)", min_value=0.000001, value=1.0, step=0.1, format="%.6f")

if st.button("运行分析", type="primary"):
    if not covariates:
        st.error("请至少选择一个 covariate。")
        st.stop()

    files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
    data = {
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "covariates": ",".join(covariates),
        "psm_caliper": str(psm_caliper),
    }
    if time_col:
        data["time_col"] = time_col
    if group_col:
        data["group_col"] = group_col

    try:
        with st.spinner("调用后端分析中..."):
            response = requests.post(f"{api_base}/analyze/{method}", files=files, data=data, timeout=120)

        if response.status_code != 200:
            payload = response.json()
            st.error(f"分析失败：{payload.get('error', {}).get('code', 'unknown_error')}")
            st.json(payload)
            st.stop()

        payload = response.json()
        st.success("分析完成")

        st.subheader("结果 JSON")
        st.json(payload.get("result", {}))

        st.subheader("Markdown 报告")
        st.markdown(payload.get("report_markdown", ""))

    except requests.RequestException as exc:
        st.error(f"请求后端失败：{exc}")
        st.info("请确认 FastAPI 服务已启动，并且 API Base URL 可访问。")
