# causal-ai-workbench (MVP)

一个面向数据分析师/业务分析场景的因果推断工具。MVP 目标是：用户上传表格数据后，快速完成**基础因果分析**、**异质性分析**和**结果解读报告**。

## 1) 项目目标
- 上传 CSV 数据并自动做数据质量检查与摘要。
- 根据数据结构自动推荐因果方法。
- 提供可运行的基础方法：PSM、DID、Uplift（异质性分析）。
- 输出包含假设与限制说明的 Markdown 报告，避免“相关性=因果性”的误解。

## 2) MVP 范围
### 已包含
1. **数据导入与校验**
   - CSV 上传
   - 缺失值、字段类型、重复值、IQR 异常值检查
2. **因果分析方法**
   - PSM：倾向得分 + 最近邻匹配（caliper）
   - DID：处理组 × 后期交互项（OLS + robust SE）
   - Uplift：基于 transformed outcome + RandomForest 的异质性近似
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
   - `data/sample/hillstrom_style_sample.csv` 最小示例

### MVP 暂未完整实现
- RDD/IV 仅预留接口方向（未实现正式估计器）
- Uplift 尚未接入完整 Causal Forest 推断与不确定性区间

## 3) 项目结构

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

## 4) 安装方式
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5) 运行方式
启动 API：
```bash
uvicorn app.main:app --reload
```

访问：
- 健康检查：`GET /api/health`
- 数据摘要：`POST /api/data/summary`
- 方法推荐：`POST /api/methods/recommend`
- 运行分析：`POST /api/analyze/{method}`，`method in [psm, did, uplift]`

## 6) 测试
```bash
pytest
```

## 7) 方法假设与限制（关键提醒）
- PSM 依赖“可观测混杂充分控制”；漏掉关键混杂变量会导致偏差。
- DID 依赖“平行趋势”假设；若不成立，结果可能失真。
- Uplift 模型用于“优先干预人群排序”，并不自动等于可解释的严格因果机制。
- 所有自动化结果仅供决策参考，需结合业务背景与额外稳健性检验。

## 8) Roadmap
- 加入多期 DID、事件研究和固定效应。
- 接入 EconML CausalForestDML 与置信区间估计。
- 增加可视化（uplift 曲线、平衡性图、趋势图）。
- 补充 RDD/IV 框架。
- 增加 Streamlit demo 页面与报告下载。

详见：`docs/development_plan.md`
