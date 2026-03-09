# causal-ai-workbench (MVP)

一个面向数据分析师/业务分析场景的因果推断工具。当前阶段目标是形成**可本地运行的最小闭环**：

`CSV 上传 -> 数据摘要 -> 方法推荐 -> PSM 分析 -> Markdown 报告`

## 项目目标
- 上传 CSV 数据并自动做数据质量检查与摘要。
- 根据数据结构自动推荐因果方法。
- 提供可运行的基础方法：PSM、DID、Uplift、Causal Forest、RDD、IV。
- 输出包含假设与限制说明的 Markdown 报告，避免“相关性=因果性”的误解。

## MVP 范围
### 已包含
1. **数据导入与校验**
   - CSV 上传
   - 缺失值、字段类型、重复值、IQR 异常值检查
2. **因果分析方法**
   - PSM：倾向得分 + 最近邻匹配（caliper）
   - DID：处理组 × 后期交互项（OLS + robust SE）
   - Uplift：transformed outcome + RandomForest 异质性近似
3. **自动方法推荐**
   - 有时间+分组 → DID
   - 有 targeting 诉求 → Uplift
   - 实验风格 treatment → A/B 风格提示
   - 观测横截面 → PSM
4. **结果输出**
   - ATE / 置信区间（方法允许时）
   - 诊断信息
   - Markdown 报告（含免责声明、假设、局限）
5. **示例数据**
   - `data/sample/hillstrom_style_sample.csv`

### 暂未完整实现
- RDD/IV 仅预留接口方向（未实现正式估计器）
- Uplift 尚未接入完整 Causal Forest 推断与不确定性区间

## 项目结构
```text
app/
  api/
  causal/
  reporting/
  services/
data/sample/
notebooks/
tests/
docs/
```

## 本地启动步骤
1. 创建虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 启动服务
```bash
uvicorn app.main:app --reload
```

4. 健康检查
```bash
curl http://127.0.0.1:8000/api/health
```

## 本地验证状态（与你反馈一致）
- ✅ 服务可启动
- ✅ `python -m pytest -q` 通过
- ✅ `POST /api/data/summary` 成功
- ✅ `POST /api/methods/recommend` 成功
- ✅ `POST /api/analyze/did` 成功
- ⚠️ `POST /api/analyze/psm` 在小样本下可能无匹配对（可通过 `psm_caliper` 调整，失败时返回结构化业务错误）

## API 最小调用示例（闭环）
> 使用示例数据：`data/sample/hillstrom_style_sample.csv`
>
> **推荐最小闭环方法：DID**（该 sample 数据对 DID 更稳定；PSM 受匹配重叠与 caliper 影响更大）。

1) 数据摘要
```bash
curl -X POST http://127.0.0.1:8000/api/data/summary \
  -F "file=@data/sample/hillstrom_style_sample.csv"
```

2) 方法推荐（按 DID 场景请求）
```bash
curl -X POST http://127.0.0.1:8000/api/methods/recommend \
  -H "Content-Type: application/json" \
  -d '{"has_time": true, "has_group": true, "treatment_binary": true, "observational": true, "wants_targeting": false}'
```

3) 执行 DID 并生成报告（推荐最小闭环）
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/did \
  -F "file=@data/sample/hillstrom_style_sample.csv" \
  -F "treatment_col=treatment" \
  -F "outcome_col=outcome" \
  -F "covariates=age,income,prior_spend" \
  -F "time_col=period" \
  -F "group_col=is_target_group"
```

4) （可选）执行 PSM 并生成报告
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/psm \
  -F "file=@data/sample/hillstrom_style_sample.csv" \
  -F "treatment_col=treatment" \
  -F "outcome_col=outcome" \
  -F "covariates=age,income,prior_spend" \
  -F "psm_caliper=1.0"
```

## Streamlit Demo 页面（最小版）
1) 启动 FastAPI（后端）
```bash
uvicorn app.main:app --reload
```

2) 新开终端启动 Streamlit（前端）
```bash
streamlit run streamlit_app.py
```

3) 打开页面（默认）：`http://localhost:8501`

当前页面支持方法：
- `did`
- `psm`（支持 `psm_caliper` 参数）
- `uplift`（支持 `uplift_buckets`，输出分桶与 top 人群摘要）
- `causal_forest`（支持 `cf_n_estimators`、`cf_min_samples_leaf`，返回 CATE 排序与分桶）
- `rdd`（支持 `running_col`、`cutoff`、`rdd_bandwidth`）
- `iv`（支持 `instrument_col`）

页面交互能力：
- 上传 CSV 后自动读取列名，以下拉框选择 `treatment_col`、`outcome_col`、`time_col`、`group_col`
- `covariates` 使用多选框
- 动态参数区：
  - `did` 显示 `time_col`、`group_col`
  - `psm` 显示 `psm_caliper`
  - `uplift` 显示 `uplift_buckets`
  - `causal_forest` 显示 `cf_buckets`、`cf_n_estimators`、`cf_min_samples_leaf`
  - `rdd` 显示 `running_col`、`cutoff`、`rdd_bandwidth`
  - `iv` 显示 `instrument_col`
- 结果区分开展示：核心结果（JSON）、markdown 报告、错误提示
- uplift / causal_forest 场景额外展示：样本排序预览、分桶统计表、Top 人群摘要
- 支持下载 markdown 报告

## LLM 解释助手（仅解释，不参与估计）
项目新增了 LLM explanation 层：在分析完成后，基于已有结构化结果生成更自然的解释。

职责边界：
- ✅ LLM 只做解释、总结、方法说明和业务建议
- ❌ LLM 不参与 DID/PSM/Uplift/CausalForest/RDD/IV 的核心统计估计

环境变量配置（二选一）：
1) OpenAI 兼容接口
```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
# 可选
export OPENAI_BASE_URL=https://api.openai.com/v1
```

2) 本地 Ollama
```bash
export OLLAMA_MODEL=qwen2.5:7b
# 可选
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

未配置时的降级行为：
- 自动返回模板化解释（`mode=template`），主分析流程不受影响。
- Streamlit 页面会提示“当前使用模板化解释”。

LLM 返回结构（示例字段）：
- `executive_summary`
- `method_why`
- `key_findings`
- `caveats`
- `business_takeaways`

## 运行测试
在项目根目录执行：
```bash
pytest -q
```

> 已通过 `pytest.ini` 固定 `pythonpath=.`，用于兼容 Windows 下直接使用 `pytest -q` 时的导入路径问题（避免 `ModuleNotFoundError: app`）。

## 方法假设与限制
- PSM 依赖“可观测混杂充分控制”；漏掉关键混杂变量会导致偏差。
- DID 依赖“平行趋势”假设；若不成立，结果可能失真。
- Uplift 模型用于“优先干预人群排序”，并不自动等于可解释的严格因果机制。
- `causal_forest` 优先调用 `econml.CausalForestDML`；若环境未安装 econml，会自动降级为近似排序实现。
- `rdd` 估计的是 cutoff 附近的局部效应（local effect），不代表全样本 ATE。
- `iv` 依赖工具变量相关性、排除限制和外生性等强假设，请重点检查第一阶段强度。
- 所有自动化结果仅供决策参考，需结合业务背景与额外稳健性检验。

## Roadmap
- 加入多期 DID、事件研究和固定效应。
- 接入 EconML CausalForestDML 与置信区间估计。
- 增加可视化（uplift 曲线、平衡性图、趋势图）。
- 补充 RDD/IV 框架。
- 增加 Streamlit demo 页面与报告下载。

详见：`docs/development_plan.md`
