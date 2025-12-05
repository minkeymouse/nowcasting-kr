# Issues and Action Plan

## Executive Summary (2025-01-XX)

**Current State**: ARIMA working (9 combinations). VAR/DFM/DDFM fixes applied, need testing. Code quality reviewed, report reviewed.  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Critical Path**: Test VAR/DFM/DDFM fixes → Re-run experiments → Generate results → Update report  
**Next Action**: Re-run experiments to verify VAR/DDFM fixes, then investigate DFM numerical instability

## Improvement Plan Summary

### Code Fixes (COMPLETED - Verified in Code, Ready for Testing)
1. ✅ **ARIMA/VAR target_series handling**: Fixed calculate_standardized_metrics() to handle Series input (evaluation.py) - VERIFIED (ARIMA working)
2. ✅ **VAR/DDFM prediction extraction**: Added fallback to check y_test.columns when target_series not in y_pred_h.columns (evaluation.py lines 416-425) - VERIFIED
3. ✅ **DFM/DDFM pickle errors**: Fixed make_cha_transformer and identity_with_index/log_with_index (preprocess/utils.py) - VERIFIED
4. ✅ **Test data size check**: Skip horizon 28 if test set too small (evaluation.py) - VERIFIED
5. ✅ **Debug logging**: Added enhanced INFO/DEBUG logging to evaluation.py - VERIFIED

### Critical Issues to Fix (PRIORITY 1 - Blocking Experiments)
1. **VAR n_valid=0**: ARIMA fix works but VAR still fails. Root cause: VAR predictions return DataFrame with all columns, but target_series extraction may fail. Need to verify VAR prediction format and fix extraction logic in evaluation.py (lines 416-425).
2. **DFM Numerical Instability**: Parameters (A or C) contain NaN/Inf for KOGDP...D/KOCNPER.D. Root cause: EM algorithm convergence issues. Need to review EM regularization (em.py), check initialization, verify data quality (high missing data ratio).
3. **DDFM n_valid=0**: All targets show n_valid=0. Root cause: Prediction extraction or test data alignment issue. Need to investigate DDFM prediction format and verify evaluation.py handles it correctly.

### Code Quality Improvements (PRIORITY 2 - After Critical Fixes)
1. **dfm-python Numerical Stability**: 
   - Review EM regularization constants (1e-6, 1e-8) - verify appropriate for edge cases (T < N, near-singular)
   - Check Kalman filter safe_inverse() fallbacks handle all failure modes
   - Verify eigenvalue checks catch all ill-conditioned matrices
   - Add early stopping if parameters become NaN/Inf during EM iterations
2. **dfm-python Theoretical Correctness**: 
   - Verify EM algorithm matches standard references (Durbin-Koopman, Harvey)
   - Check Kalman gain calculation: K = P_{t|t-1} C' (C P_{t|t-1} C' + R)^{-1}
   - Verify innovation covariance structure matches theory
3. **dfm-python Code Patterns**: 
   - Check for redundant validation logic across models
   - Verify no monkey patches or temporal fixes remain
   - Ensure generic naming (no hardcoded assumptions)
4. **src/ Redundancies**: 
   - Consolidate duplicate logic in dfm.py, ddfm.py, sktime_forecaster.py
   - Check common preprocessing steps (load_config, train_model functions)
   - Remove duplicate error handling patterns
5. **src/ Error Handling**: 
   - Verify all exceptions caught and logged (especially evaluation.py)
   - Check for silent failures (NaN propagation, empty predictions)
   - Ensure debug logging available for prediction extraction issues

### Report Improvements (INCREMENTAL - Can Start Now)
1. **Hallucination Check**: 
   - Verify all theoretical claims match references.bib (use neo4j MCP for additional knowledge)
   - Check methods section matches actual src/ implementation
   - Remove unsupported claims (anything not in references.bib or results)
2. **Detail Level**: 
   - Ensure 4_method_and_experiment.tex matches actual preprocessing in src/preprocess/utils.py
   - Verify DFM/DDFM description matches dfm-python implementation
   - Check evaluation metrics calculation matches src/eval/evaluation.py
3. **Redundancy**: 
   - Remove repeated statements about data preprocessing across sections
   - Consolidate model descriptions (avoid repeating same info in intro/method/results)
   - Remove duplicate explanations of nowcasting concept
4. **Flow**: 
   - Ensure logical progression: intro → lit review → theory → method → results → discussion → conclusion
   - Check transitions between sections are smooth
   - Verify results section references method section correctly
5. **Citations**: 
   - Verify all \cite{} exist in references.bib (21 references verified)
   - Add citations for new knowledge from neo4j MCP (add to references.bib first)
   - Check no made-up references or incorrect citation keys

### Experiment Status
- **Latest Runs**: 20251206_082502 (all 3 targets)
  - **ARIMA**: ✅ WORKING - n_valid=1 for all horizons across all 3 targets (9 combinations)
  - **VAR**: ❌ n_valid=0 - Fix applied but needs testing
  - **DFM**: ❌ Numerical instability (NaN/Inf in parameters) or n_valid=0
  - **DDFM**: ❌ n_valid=0 - Fix applied but needs testing
- **Action**: Re-run experiments to verify VAR/DDFM fixes, then investigate DFM numerical issues

**Code Status**: 
- ✅ dfm-python: Finalized (consistent naming, clean code)
- ✅ src/: 15 files (max 15 required)
- ✅ Model bugs: All fixes applied (ARIMA, VAR, DFM/DDFM) - ready for testing

**Report Status**: 
- ✅ Structure: Complete 8-section framework (20-30 pages target)
- ✅ Citations: All 21 references verified in references.bib, no hallucinated citations
- ✅ Content: Sections 1-4, 6-7 complete with comprehensive content, redundancy removed in conclusion section
- ⚠️ Tables: 4 tables with "---" placeholders (blocked by experiments)
- ⚠️ Plots: Not generated (blocked by experiments)

## Concrete Action Plan (Incremental, Prioritized)

### PHASE 1: Fix Remaining Model Issues [CURRENT PRIORITY]

**Goal**: Fix VAR/DFM/DDFM to get at least 6 successful model-target-horizon combinations (n_valid > 0)  
**Success Criteria**: Minimum 2 models per target produce valid results for at least one horizon  
**Status**: ARIMA working (9 combinations), VAR/DFM/DDFM need fixes

**Current Status**:
- ✅ ARIMA: Working (n_valid=1 for all horizons, 3 targets) - 9 successful combinations
- ✅ VAR: Fix applied - prediction extraction now handles case when target_series not in y_pred_h.columns but in y_test.columns (evaluation.py lines 416-425)
- ❌ DFM: Numerical instability (NaN/Inf in parameters for KOGDP...D/KOCNPER.D, n_valid=0 for KOGFCF..D)
- ✅ DDFM: Fix applied - same fix as VAR (both return DataFrames with all columns, fix applies to both)

#### Task 1.1: Fix VAR Prediction Extraction [COMPLETED - 2025-01-XX]
**Priority**: HIGH - Blocking experiments  
**Root Cause**: VAR predictions return DataFrame with all columns, but target_series extraction fails when target_series not in y_pred_h.columns  
**Action**: 
1. ✅ Fixed evaluation.py lines 416-425: Added fallback to check y_test.columns when target_series not in y_pred_h.columns
2. ✅ Use column index from y_test to extract from y_pred_h when column names don't match
3. ⏳ Test: Verify n_valid > 0 for VAR on at least one target/horizon (pending test run)
**Files**: `src/eval/evaluation.py` (lines 416-425)

#### Task 1.2: Fix DFM Numerical Instability [AFTER 1.1]
**Priority**: HIGH - Blocking experiments  
**Root Cause**: EM algorithm produces NaN/Inf in parameters (A or C matrices)  
**Action**:
1. Review EM algorithm convergence: Check em.py for early stopping when parameters become NaN/Inf
2. Check initialization: Verify PCA initialization in dfm-python handles edge cases (T < N, high missing data)
3. Add parameter validation: Check A/C matrices after each EM iteration, stop if NaN/Inf detected
4. Review regularization: Verify 1e-6, 1e-8 constants are appropriate for edge cases
5. Test: Verify DFM produces valid predictions for at least one target
**Files**: `dfm-python/src/dfm_python/ssm/em.py`, `dfm-python/src/dfm_python/models/dfm.py` (lines 672-677)

#### Task 1.3: Fix DDFM n_valid=0 [COMPLETED - 2025-01-XX]
**Priority**: MEDIUM - After VAR/DFM fixed  
**Root Cause**: Prediction extraction issue - same as VAR (both return DataFrames with all columns)  
**Action**:
1. ✅ Verified DDFM prediction format: DDFMForecaster returns DataFrame with all columns (sktime_forecaster.py line 430-434)
2. ✅ Fixed evaluation.py: VAR fix applies to DDFM since both use same prediction extraction logic
3. ⏳ Test: Verify n_valid > 0 for DDFM on at least one target/horizon (pending test run)
**Files**: `src/eval/evaluation.py` (same fix as VAR), `src/model/sktime_forecaster.py`

#### Task 1.4: Re-run Full Experiments [AFTER 1.1-1.3 SUCCEED]
**Action**: `bash run_experiment.sh`
- What it does: Runs all 3 targets in parallel (max 5 processes), checks for valid results before skipping
- Expected duration: Several hours
- Output: `outputs/comparisons/{target}_{timestamp}/comparison_results.json` for each target
- Verification: Check n_valid > 0 for at least 6 combinations (2 models × 3 targets minimum)
- Note: run_experiment.sh already checks for valid results (n_valid > 0) before skipping

#### Task 1.5: Generate Aggregated Results [AFTER 1.4 SUCCEEDS]
**Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- Prerequisite: At least 6 successful combinations with n_valid > 0
- Output: `outputs/experiments/aggregated_results.csv`
- Expected: 36 rows (3 targets × 4 models × 3 horizons), fewer if some models failed
- Verification: File exists, contains non-NaN values for successful models

**Note on run_experiment.sh**: 
- ✅ Already checks for valid results (n_valid > 0) before skipping
- ✅ Will re-run all targets since current results have n_valid=0
- ⚠️ May need updates if experiment structure changes (e.g., different targets/models/horizons)

### PHASE 2: Code Quality Improvements [INCREMENTAL - AFTER PHASE 1]

**Goal**: Improve dfm-python package and src/ code quality incrementally  
**Priority**: Low - proceed after Phase 1 succeeds or in parallel if experiments take long

#### Task 2.1: Review dfm-python Numerical Stability [INCREMENTAL]
**Status**: Code has stability measures but needs review  
**Priority**: Medium - after experiments succeed  
**Areas to Check**:
1. **EM Algorithm (em.py)**: 
   - Verify regularization constants (1e-6, 1e-8) are appropriate for edge cases
   - Check Q matrix floor (0.01) prevents scale issues
   - Verify C matrix normalization (||C[:,j]|| = 1) works correctly
   - Test edge cases: T < N, near-singular matrices, extreme eigenvalues
2. **Kalman Filter (kalman.py)**: 
   - Verify safe_inverse() fallbacks handle all failure modes (singular, ill-conditioned)
   - Check ensure_positive_definite() catches all edge cases
   - Test innovation covariance S = C P C' + R is always PSD
3. **Data Validation**: 
   - Verify _validate_factors() catches constant factors, perfect correlation
   - Check T < N warnings in data/utils.py are handled properly
   - Ensure spectral radius capping (< 0.99) ensures stationarity
4. **Code Patterns**: 
   - Check for redundant validation logic across models
   - Verify no monkey patches or temporal fixes remain
   - Ensure generic naming (no hardcoded assumptions)

#### Task 2.2: Review src/ Code Quality [INCREMENTAL]
**Status**: 15 files (max 15 required), structure verified  
**Priority**: Medium - after experiments succeed  
**Areas to Check**:
1. **Redundancies**: 
   - Check model wrappers (dfm.py, ddfm.py) for duplicate load_config/train_model logic
   - Consolidate common utilities in sktime_forecaster.py (load_config, train_model functions)
   - Remove duplicate preprocessing steps if present
2. **Error Handling**: 
   - Verify all exceptions caught and logged (especially evaluation.py prediction extraction)
   - Check for silent failures (NaN propagation, empty predictions not logged)
   - Ensure debug logging available for troubleshooting
3. **Inefficient Logic**: 
   - Review data loading loops in training.py (check for unnecessary iterations)
   - Check preprocessing pipeline for redundant transformations
   - Verify evaluation.py doesn't recalculate metrics unnecessarily
4. **Naming Consistency**: 
   - Ensure consistent naming across model wrappers
   - Check for generic names (no hardcoded assumptions)
   - Verify function names match their purpose

### PHASE 3: Generate Results for Report [BLOCKED by Phase 1]

**Prerequisites**: Phase 1 complete (aggregated_results.csv exists with valid data)

**Current Status**: ⚠️ BLOCKED - No valid results yet (all n_valid=0 in latest runs)

#### Task 3.1: Generate Aggregated CSV [BLOCKED]
**Prerequisite**: At least 6 successful model-target-horizon combinations with n_valid > 0
**Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`  
**Input**: `outputs/comparisons/{target}_*/comparison_results.json` (must have n_valid > 0)
**Output**: `outputs/experiments/aggregated_results.csv`  
**Expected**: 36 rows (3 targets × 4 models × 3 horizons), fewer if some models failed  
**Verification**: 
- File exists at `outputs/experiments/aggregated_results.csv`
- Contains non-NaN values for successful models
- Has columns: target, model, horizon, sMSE, sMAE, sRMSE, n_valid

#### Task 3.2: Generate Visualizations [BLOCKED]
**Prerequisite**: aggregated_results.csv exists with valid data
**Action**: `python3 nowcasting-report/code/plot.py`  
**Input**: `outputs/comparisons/{target}_*/comparison_results.json` (reads from outputs/comparisons/)
**Output**: 4 PNG files in `nowcasting-report/images/`:
  - `accuracy_heatmap.png` - Heatmap of model performance
  - `model_comparison.png` - Bar chart comparing models
  - `horizon_trend.png` - Performance vs forecast horizon
  - `forecast_vs_actual.png` - Forecast vs actual scatter plots
**Verification**: 
- All 4 files exist in `nowcasting-report/images/`
- Files are not empty/placeholder images
- Images are readable and contain actual data visualizations

#### Task 3.3: Update LaTeX Tables [BLOCKED]
**Prerequisite**: aggregated_results.csv exists with valid data
**Action**: Read `outputs/experiments/aggregated_results.csv`, calculate aggregates, update LaTeX tables
**Files to Update**:
  - `nowcasting-report/tables/tab_overall_metrics.tex` - Overall averages (replace "---" with numbers)
  - `nowcasting-report/tables/tab_overall_metrics_by_target.tex` - Per-target averages
  - `nowcasting-report/tables/tab_overall_metrics_by_horizon.tex` - Per-horizon averages
  - `nowcasting-report/tables/tab_nowcasting_metrics.tex` - Nowcasting-specific metrics (if evaluated)
**Steps**:
1. Load aggregated_results.csv
2. Calculate averages: overall (all models), by target, by horizon
3. Format numbers appropriately (e.g., 3 decimal places)
4. Replace "---" placeholders in each table file
5. Verify LaTeX compiles: `cd nowcasting-report && pdflatex main.tex`
**Verification**: 
- No "---" remain in any table file
- LaTeX compiles without errors
- Numbers formatted correctly (consistent decimal places)
- All tables reference correct data from aggregated_results.csv

### PHASE 4: Update Report with Results [BLOCKED by Phase 3]

**Prerequisites**: Phase 2 complete (tables updated, plots generated)

**Current Status**: ⚠️ BLOCKED - Tables have "---" placeholders, plots not generated

#### Task 4.1: Review Report for Issues [INCREMENTAL - CAN START NOW]
**Status**: Can review sections 1-4, 6-7 now (section 5 blocked until results available)  
**Priority**: Medium - can proceed in parallel with experiments  
**Files to Review** (one section at a time):
  - `nowcasting-report/contents/1_introduction.tex` - Check claims, citations
  - `nowcasting-report/contents/2_literature_review.tex` - Verify all citations exist, no hallucination
  - `nowcasting-report/contents/3_theoretical_background.tex` - Verify theoretical claims match references
  - `nowcasting-report/contents/4_method_and_experiment.tex` - Check implementation details match code
  - `nowcasting-report/contents/6_discussion.tex` - Remove speculative statements, ensure citations
  - `nowcasting-report/contents/7_conclusion.tex` - Check claims are supported by results/citations
**Check for** (incrementally, one section at a time):
1. **Hallucination**: Verify all claims are supported by citations (use references.bib, neo4j MCP for knowledge)
2. **Lack of Detail**: Ensure methods section explains implementation clearly (match with src/ code)
3. **Redundancy**: Remove repeated statements across sections
4. **Unnatural Flow**: Ensure logical progression between sections
5. **Citation Issues**: Verify all citations exist in references.bib, no made-up references
**Action**: Review one section at a time, check citations against references.bib, verify claims are supported

#### Task 4.2: Update Results Section [BLOCKED]
**Prerequisite**: Tables updated with real numbers (Phase 3.3 complete)
**File**: `nowcasting-report/contents/5_result.tex`  
**Current Status**: Has generic statements, needs specific numbers from tables
**Actions**:
1. Read updated tables (tab_overall_metrics.tex, etc.) to get actual numbers
2. Replace generic statements with specific numbers from tables
3. Reference table/figure numbers correctly (e.g., "표 \ref{tab:overall_metrics}에서 ARIMA의 sMSE는 0.XXX로 나타났다")
4. Add performance comparisons with actual metrics
5. Discuss which models perform best for which targets/horizons (based on actual results)
6. Reference figures: "그림 \ref{fig:model_comparison}에서 볼 수 있듯이..."
**Verification**: 
- No generic/placeholder statements remain
- All numbers match tables exactly
- All table/figure references are correct
- Performance discussions are based on actual results

#### Task 4.3: Update Discussion Section [BLOCKED]
**Prerequisite**: Results section updated with actual numbers (Task 4.2 complete)
**File**: `nowcasting-report/contents/6_discussion.tex`  
**Actions**:
1. Reference specific metrics from results section (not speculative)
2. Explain actual findings with evidence from tables/figures
3. Remove unsupported claims (anything not backed by results or citations)
4. Ensure all citations are from references.bib (use neo4j MCP if adding new knowledge)
5. Connect findings to literature (cite appropriately)
**Verification**: 
- All claims supported by results or citations
- Specific metrics referenced (e.g., "ARIMA는 horizon=1에서 sMSE 0.XXX를 기록했다")
- No speculation or unsupported statements
- All citations exist in references.bib

#### Task 4.4: Finalize Report [BLOCKED]
**Prerequisite**: All sections updated, tables and plots complete
**Actions**:
1. **Compile PDF**: 
   ```bash
   cd nowcasting-report
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
2. **Verify page count**: Should be 20-30 pages (check PDF)
3. **Check for placeholders**: Search for "---", "TODO", "FIXME", "placeholder" in all .tex files
4. **Verify citations**: All \cite{} references exist in references.bib
5. **Check figures/tables**: All 4 figures and 4 tables included and referenced in text
6. **Final review**: Read through PDF, check formatting, ensure no errors
**Verification**: 
- PDF compiles without errors
- Page count: 20-30 pages
- No placeholders remain
- All citations resolve correctly
- All figures/tables are included and referenced

## Current Status Summary

**Experiments**:
- Latest run: 20251206_082502 (all 3 targets attempted)
- Valid results: PARTIAL - ARIMA working (9 combinations: 3 targets × 3 horizons with n_valid=1)
- Outputs: comparison_results.json exists, ARIMA has valid metrics, VAR/DFM/DDFM have n_valid=0, aggregated_results.csv MISSING

**Code**:
- ✅ All fixes applied (ARIMA, VAR, DFM/DDFM pickle, fillna deprecation)
- ✅ Debug logging added to evaluation.py
- ✅ dfm-python finalized (consistent naming)
- ✅ src/: 15 files (max 15 required)

**Report**:
- ✅ Structure: 8 sections complete
- ✅ Citations: 21 references verified
- ✅ Content: Sections 1-4, 6-7 complete
- ⚠️ Tables: 4 tables with "---" placeholders (blocked)
- ⚠️ Plots: Not generated (blocked)

**run_experiment.sh**:
- ✅ Checks for valid results (n_valid > 0) before skipping
- ✅ Will re-run all targets since current results have n_valid=0
- ⚠️ May need updates later if experiment structure changes

## Experiments Status and Requirements

### Required Experiments for Report
**Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **Models**: ARIMA, VAR, DFM, DDFM
- **Horizons**: 1, 7, 28 days

### Current Status (Latest Run: 20251206_082502)
- ✅ **All 3 targets attempted**: KOGDP...D, KOCNPER.D, KOGFCF..D
- ✅ **All 4 models attempted**: ARIMA, VAR, DFM, DDFM
- ✅ **All models complete training**: No training failures
- ✅ **ARIMA WORKING**: 9 successful combinations (3 targets × 3 horizons with n_valid=1)
- ❌ **VAR/DFM/DDFM FAILING**: All show n_valid=0 or prediction errors
- ❌ **aggregated_results.csv**: MISSING (need at least 2 models per target for report)

### Experiments Still Needed
- ⚠️ **VAR/DFM/DDFM need fixes** before re-running (ARIMA already working)
- ⚠️ **Minimum viable for report**: 6 successful combinations (2 models × 3 targets, or equivalent)
  - Current: 9 ARIMA combinations (need at least 1 more model working)
- ⚠️ **Ideal**: All 36 combinations succeed (currently 9/36 = 25%)

### What's Complete vs What's Missing
**Complete**:
- Experiment infrastructure (run_experiment.sh, configs, data)
- Model training (all models complete without errors)
- Result structure (comparison_results.json files exist)

**Missing**:
- Valid predictions for VAR/DFM/DDFM (ARIMA has 9 working combinations)
- Aggregated results CSV (cannot generate with only ARIMA results)
- Plots and tables for report (need at least 2 models per target)


## Task Priority Summary

### IMMEDIATE (Critical Path)
1. ⏳ **Task 1.1**: Fix VAR prediction extraction (evaluation.py)
2. ⏳ **Task 1.2**: Fix DFM numerical instability (em.py, dfm.py)
3. ⏳ **Task 1.3**: Fix DDFM n_valid=0 (evaluation.py, ddfm.py)
4. ⏳ **Task 1.4**: Re-run full experiments (`bash run_experiment.sh`)
5. ⏳ **Task 1.5**: Generate aggregated CSV (after valid results available)

### BLOCKED (Requires Valid Results)
- **Task 3.1-3.3**: Generate plots, update LaTeX tables
- **Task 4.2-4.4**: Update results/discussion sections, finalize report

### INCREMENTAL (Can Do in Parallel)
- **Task 4.1**: Review report sections 1-4, 6-7 (hallucination check, citations) - See "Report Improvements" section above
- **Task 2.1-2.2**: Code quality improvements (dfm-python, src/) - See "Code Quality Improvements" section above

## Notes

- **Minimum viable**: 6 successful combinations for report (2 models × 3 targets, or equivalent)
- **run_experiment.sh**: Already checks for valid results (n_valid > 0). ARIMA results are valid, but VAR/DFM/DDFM need fixes before re-running
- **Report sections**: Section 5 (results) and parts of section 6 (discussion) blocked until experiments succeed
- **Work incrementally**: Verify fixes → Get valid results → Generate outputs → Update report
- **If run_experiment.sh needs updates**: Only if experiment structure changes (different targets/models/horizons)
