# MVP Development Plan (GitHub Issue Style)

## P0 (Must-have)

### Issue #1: Data upload and profiling API hardening
- **Goal**: Support stable CSV upload/validation for analyst workflows.
- **Tasks**:
  - Add strict schema validation and better type inference.
  - Add column-role suggestion (`treatment`, `outcome`, `time`, `group`).
  - Return richer quality checks (missing patterns, skewness, cardinality).
- **Acceptance Criteria**:
  - `/api/data/summary` can process medium CSV files and returns actionable warnings.

### Issue #2: End-to-end causal pipeline orchestration
- **Goal**: One API request can run recommendation + method execution + report generation.
- **Tasks**:
  - Add orchestration service layer.
  - Add unified request/response models with run metadata.
  - Persist artifacts (run config + report markdown).
- **Acceptance Criteria**:
  - Single endpoint returns method recommendation and results with traceable config.

### Issue #3: PSM and DID robustness improvements
- **Goal**: Improve reliability of baseline estimators.
- **Tasks**:
  - Add covariate balance diagnostics for PSM.
  - Support multi-period DID with fixed effects.
  - Add hypothesis tests and clearer warnings.
- **Acceptance Criteria**:
  - Results include diagnostics and warnings when assumptions are likely violated.

## P1 (Should-have)

### Issue #4: Uplift/heterogeneity upgrades
- **Goal**: Improve targeting recommendations quality.
- **Tasks**:
  - Replace transformed-outcome proxy with EconML CausalForestDML fallback.
  - Add uplift curve/Qini metrics and subgroup summaries.
  - Add model stability checks across random seeds.
- **Acceptance Criteria**:
  - Uplift output includes ranking, quality metrics, and stability indicator.

### Issue #5: Minimal Streamlit demo page
- **Goal**: Provide analyst-friendly interactive demo.
- **Tasks**:
  - Build upload + column mapping + run analysis workflow.
  - Render key charts and markdown report.
  - Add download buttons for report and results.
- **Acceptance Criteria**:
  - Non-technical user can complete one full analysis from UI.

## P2 (Could-have)

### Issue #6: RDD/IV interface scaffolding
- **Goal**: Prepare extension points for quasi-experimental methods.
- **Tasks**:
  - Define common contracts for running variable/cutoff and instruments.
  - Add placeholder implementations with validation-only mode.
- **Acceptance Criteria**:
  - API exposes consistent parameters and clear not-yet-implemented guidance.
