# causal-ai-workbench (MVP)

一个面向数据分析师/业务分析场景的因果推断工具。当前阶段目标是形成**可本地运行的最小闭环**：

`CSV 上传 -> 数据摘要 -> 方法推荐 -> PSM 分析 -> Markdown 报告`

## 项目目标
- 上传 CSV 数据并自动做数据质量检查与摘要。
- 根据数据结构自动推荐因果方法。
- 提供可运行的基础方法：PSM、DID、Uplift（异质性分析）。
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

## API 最小调用示例（闭环）
> 使用示例数据：`data/sample/hillstrom_style_sample.csv`

1) 数据摘要
```bash
curl -X POST http://127.0.0.1:8000/api/data/summary \
  -F "file=@data/sample/hillstrom_style_sample.csv"
```

2) 方法推荐
```bash
curl -X POST http://127.0.0.1:8000/api/methods/recommend \
  -H "Content-Type: application/json" \
  -d '{"has_time": false, "has_group": false, "treatment_binary": true, "observational": true, "wants_targeting": false}'
```

3) 执行 PSM 并生成报告
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/psm \
  -F "file=@data/sample/hillstrom_style_sample.csv" \
  -F "treatment_col=treatment" \
  -F "outcome_col=outcome" \
  -F "covariates=age,income,prior_spend"
```

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
- 所有自动化结果仅供决策参考，需结合业务背景与额外稳健性检验。

## Roadmap
- 加入多期 DID、事件研究和固定效应。
- 接入 EconML CausalForestDML 与置信区间估计。
- 增加可视化（uplift 曲线、平衡性图、趋势图）。
- 补充 RDD/IV 框架。
- 增加 Streamlit demo 页面与报告下载。

详见：`docs/development_plan.md`
