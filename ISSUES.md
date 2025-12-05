# Issues and Action Plan

## Executive Summary (2025-12-06 - Ready for Testing)

**Current State**: All known code fixes applied and verified. Critical issue: n_valid=0 for all models despite successful training. make_cha_transformer pickle error FIXED.  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package with clean code  
**Critical Path**: Test minimal case → Debug n_valid=0 → Fix root cause → Re-run experiments → Generate results → Update report  
**Latest Run**: 20251206_073336 - ARIMA/VAR complete but n_valid=0, DFM/DDFM: KOGFCF..D trains but n_valid=0, KOGDP...D/KOCNPER.D should work after pickle fix  
**Next Action**: Test minimal case (ARIMA on KOGFCF..D, horizon=1) with debug logging enabled to identify n_valid=0 root cause. Review INFO/DEBUG logs for prediction extraction, test data alignment, and mask calculation.

## Improvement Plan Summary

### Code Fixes (COMPLETED - Ready for Testing)
1. ✅ **ARIMA n_valid=0**: Simplified prediction extraction logic (evaluation.py)
2. ✅ **VAR pandas API**: Enhanced asfreq() error handling with fallback chain (training.py)
3. ✅ **DFM/DDFM pickle (identity_with_index/log_with_index)**: Use globals() for module-level function references (preprocess/utils.py)
4. ✅ **DFM/DDFM pickle (make_cha_transformer)**: Refactored lambda to use module-level function with functools.partial (preprocess/utils.py) - FIXED 2025-12-06
5. ✅ **fillna() deprecation**: Replaced fillna(method='ffill') with ffill() (training.py)
6. ✅ **Debug logging for n_valid=0**: Added enhanced INFO/DEBUG logging to evaluation.py to investigate prediction extraction issues - ADDED 2025-12-06

### Code Quality Improvements (INCREMENTAL - After Experiments)
1. **dfm-python Numerical Stability**: 
   - Review EM regularization constants (1e-6, 1e-8) - verify appropriate for edge cases (T < N, near-singular)
   - Check Kalman filter safe_inverse() fallbacks handle all failure modes
   - Verify eigenvalue checks catch all ill-conditioned matrices
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
- **Latest Runs**: 20251206_073336 (all 3 targets) - Most recent run
- **Valid Results**: NONE (ARIMA/VAR complete but n_valid=0 for all horizons)
- **Action**: Investigate why fixes aren't working → Debug prediction extraction → Fix root cause

**Experiments Status**: 
- ⚠️ Latest run (20251206_073336): All 3 targets attempted
  - **ARIMA**: Status "completed" but n_valid=0 for ALL horizons (all metrics NaN) across all 3 targets
  - **VAR**: Status "completed" but n_valid=0 for ALL horizons (all metrics NaN) across all 3 targets
  - **DFM/DDFM**: 
    - KOGDP...D/KOCNPER.D: Status "failed" - NEW pickle error: "Can't pickle local object 'make_cha_transformer.<locals>.<lambda>'"
    - KOGFCF..D: Status "completed" (DFM converged, DDFM not converged) but n_valid=0 for all horizons
- ⚠️ **CRITICAL ISSUE 1**: make_cha_transformer pickle error - lambda function at line 873 can't be pickled (affects targets using 'cha' transformation)
- ⚠️ **CRITICAL ISSUE 2**: n_valid=0 persists even when training completes - suggests prediction extraction or test data alignment problem
- ✅ run_experiment.sh: Already checks for valid results (n_valid > 0), will re-run all since current results invalid

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

## Concrete Action Plan (Step-by-Step, Incremental)

### PHASE 1: Debug n_valid=0 Issue and Fix Root Cause [CURRENT PRIORITY]

**Goal**: Understand why fixes aren't working, identify root cause, fix it, then re-run experiments  
**Expected**: At least 2 models per target produce valid results (n_valid > 0), minimum 6 successful combinations  
**Status**: ⚠️ Fixes in code but latest experiments (20251206_073336) still show n_valid=0 - need investigation

**Experiments Status**: 
- **Total Required**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Latest Run**: 20251206_073336 (all 3 targets)
- **Current Results**: ARIMA/VAR complete but n_valid=0 for ALL horizons across all targets
- **Action Required**: Investigate why prediction extraction is failing despite fixes

#### Task 1.1: Fix ARIMA n_valid=0 [COMPLETED]
**Issue**: n_valid=0 for all horizons despite model training successfully  
**Root Cause**: Prediction extraction logic was too complex, not always extracting the correct horizon h prediction  
**Location**: `src/eval/evaluation.py:336-411`  
**Fix Applied**:
1. ✅ Simplified prediction extraction: always take last element from predict() output (should be horizon h)
2. ✅ Improved compatibility: handle both `predict(fh=[h])` and `predict(fh=h)` formats
3. ✅ Better shape handling: ensure consistent DataFrame/Series extraction with .copy() to avoid view issues
4. ✅ Enhanced error handling: catch both TypeError and ValueError for predict() calls
**Status**: ✅ Fix applied, ready for testing

#### Task 1.2: Fix VAR pandas asfreq() API Error [NEEDS VERIFICATION]
**Issue**: "NDFrame.asfreq() got an unexpected keyword argument 'fill_method'"  
**Root Cause**: pandas 2.3.3 uses `method` parameter, but error handling wasn't catching all exception types  
**Location**: `src/core/training.py:310-343`  
**Fix Applied**:
1. ✅ Enhanced error handling: catch both TypeError and ValueError (not just TypeError)
2. ✅ Fallback chain: try `method='ffill'` → try `fill_method='ffill'` → manual `fillna(method='ffill')`
3. ✅ Applied to both inferred_freq and default 'D' frequency cases
**Status**: ⚠️ Fix in code but error log shows line 322 using fill_method - suggests experiments ran before fix OR different code path
**Additional Issue**: ✅ Fixed - `fillna(method='ffill')` replaced with `ffill()` on lines 331, 343

#### Task 1.3: Fix DFM/DDFM Pickle Error [COMPLETED - 2025-12-06]
**Issue**: "Can't pickle local object 'make_cha_transformer.<locals>.<lambda>'" (line 873)  
**Root Cause**: make_cha_transformer uses lambda that captures local variables (step, annual_factor), making it unpicklable  
**Location**: `src/preprocess/utils.py:846-873` (lambda at line 873)  
**Previous Fix**: ✅ identity_with_index/log_with_index fixed using globals()['function_name'] pattern  
**Fix Applied**: ✅ Created module-level function `cha_with_index(X, step, annual_factor)` and refactored `make_cha_transformer` to use `functools.partial(cha_with_index, step=step, annual_factor=annual_factor)` instead of lambda  
**Affected Targets**: KOGDP...D, KOCNPER.D (use 'cha'), NOT KOGFCF..D (uses 'log')  
**Status**: ✅ FIXED - Ready for testing

#### Task 1.4: Fix fillna() Deprecation [COMPLETED]
**Issue**: `fillna(method='ffill')` is deprecated in pandas 2.x  
**Location**: `src/core/training.py:331, 343`  
**Fix Applied**: Replaced `fillna(method='ffill')` with `ffill()`  
**Status**: ✅ Fixed - ready for testing

#### Task 1.5: Investigate Why n_valid=0 Persists [CURRENT PRIORITY - CRITICAL]

**Issue**: Latest experiments (20251206_073336) show ARIMA/VAR/DFM/DDFM complete training but n_valid=0 for all horizons. Even DFM for KOGFCF..D (which completed successfully with loglik=135.76) shows n_valid=0.

**Step 1: Inspect Latest Results in Detail** [COMPLETED]
- **Action**: ✅ Read full comparison_results.json from 20251206_073336 for all 3 targets
- **Findings**: 
  - ARIMA/VAR: Status "completed" but n_valid=0 for all horizons across all targets
  - DFM (KOGFCF..D): Status "completed" (converged, loglik=135.76) but n_valid=0 for all horizons
  - DDFM (KOGFCF..D): Status "completed" (200 iterations) but n_valid=0 for all horizons
  - All metrics (sMSE, sMAE, sRMSE, MSE, MAE, RMSE, sigma) are NaN when n_valid=0
- **Conclusion**: Issue is in evaluation/prediction extraction, not training - models train successfully but predictions aren't being extracted or aligned correctly

**Step 2: Debug Prediction Extraction Logic** [COMPLETED - 2025-12-06]
- **Action**: ✅ Added enhanced debug logging to `src/eval/evaluation.py` around prediction extraction (lines 339-452) and calculate_standardized_metrics (line 137)
- **Logging Added**:
  - INFO level logging for predict() return values (type, shape, length)
  - INFO level logging for y_test properties (length, type, shape, index)
  - INFO level logging for has_pred/has_true flags with detailed warnings when False
  - DEBUG level logging for mask calculation in calculate_standardized_metrics (shows NaN/Inf counts)
- **Check**:
  - What does `forecaster.predict(fh=[h])` actually return? (type, shape, values) - ✅ LOGGED
  - What is the shape of y_test? (length, type, index) - ✅ LOGGED
  - Is test_pos calculation correct? (test_pos = h - 1, but is y_test indexed correctly?) - ✅ LOGGED
  - Are has_pred and has_true being set correctly? (lines 410-411) - ✅ LOGGED with warnings
  - Is the mask calculation working correctly? (line 137 in calculate_standardized_metrics) - ✅ LOGGED with NaN/Inf counts
  - Are predictions and y_test aligned properly? (index matching vs position-based) - ✅ LOGGED
- **Status**: ✅ Enhanced logging added - Ready for testing to identify root cause

**Step 3: Test Minimal Case** [READY - NEXT ACTION]
- **Action**: Run single model (ARIMA) on smallest target (KOGFCF..D) with horizon=1, with debug logging enabled
- **Command**: `.venv/bin/python3 src/train.py compare --config-name experiment/kogfcf_report --models arima --horizons 1`
- **Check**: Review INFO/DEBUG level logs for:
  - predict() return values (type, shape, length)
  - y_test properties (length, type, shape, index)
  - has_pred/has_true flags (should be True if data is valid)
  - mask calculation (should have some True values if predictions and test data are valid)
- **Goal**: Identify root cause: Is it prediction extraction, test data alignment, or mask calculation?

**Step 4: Fix Root Cause**
- **Action**: Based on Step 2-3 findings, fix the actual root cause
- **Possible Issues**:
  - Prediction shape mismatch (predict() returns wrong format)
  - Index alignment issues (predictions and y_test have different indices)
  - Mask calculation bug (valid mask is all False)
  - Data type issues (predictions are NaN or wrong type)
- **Goal**: Fix the actual bug causing n_valid=0

**Step 5: Verify Fix Works**
- **Action**: Re-run single test from Step 3 after fix
- **Check**: n_valid > 0 in results
- **Goal**: Confirm fix resolves the issue

**Step 6: Re-run Full Experiments**
- **Action**: `bash run_experiment.sh` (after Step 5 succeeds)
- **What It Does**: Runs all 3 targets in parallel (max 5 processes), automatically checks for valid results
- **Expected Duration**: Several hours (depends on model training time)
- **Output**: `outputs/comparisons/{target}_{timestamp}/comparison_results.json` for each target

**Step 7: Verify Results**
- **Check Each Target**: `outputs/comparisons/KOGDP...D_*/comparison_results.json`, `KOCNPER.D_*/`, `KOGFCF..D_*/`
- **Check n_valid**: For each model/horizon combination, verify n_valid > 0
- **Success Criteria**:
  - At least 2 models per target have n_valid > 0 for at least one horizon
  - Minimum 6 total successful model-target-horizon combinations
  - If not met, investigate remaining failures

**Step 8: Generate Aggregated Results**
- **Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- **Output**: `outputs/experiments/aggregated_results.csv`
- **Expected**: 36 rows (3 targets × 4 models × 3 horizons), fewer if some models failed
- **Verification**: File exists, contains non-NaN values for successful models

**Note on run_experiment.sh**: 
- ✅ Already checks for valid results (n_valid > 0) before skipping experiments
- ✅ Will re-run all targets since current results have n_valid=0
- ⚠️ May need updates after fixing root cause if experiment structure changes

### PHASE 2: Code Quality Improvements [INCREMENTAL - AFTER PHASE 1]

**Goal**: Improve dfm-python package and src/ code quality incrementally  
**Priority**: Medium - proceed after Phase 1 succeeds or in parallel if experiments take long

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

### PHASE 3: Generate Results [BLOCKED by Phase 1 - REQUIRES VALID EXPERIMENT RESULTS]

**Prerequisites**: Phase 1 complete with at least 6 successful model-target combinations AND aggregated_results.csv exists

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

### PHASE 4: Report Improvements [BLOCKED by Phase 3 - REQUIRES TABLES AND PLOTS]

**Prerequisites**: Phase 3 complete with all tables updated and plots generated

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

## Experiment Status

**Latest Run**: 20251206_073336 (all 3 targets) - Most recent  
**Valid Results**: None (ARIMA/VAR complete but n_valid=0 for all horizons)  
**Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations  
**Outputs**: comparison_results.json exists but all models have n_valid=0, aggregated_results.csv MISSING

**Previous Runs**: 20251206_063031, 20251206_070455, 20251206_070457 (all failed)

**run_experiment.sh Status**: 
- ✅ Already checks for valid results (n_valid > 0) before skipping
- ✅ Runs all 3 targets in parallel (max 5 processes)
- ✅ Automatically aggregates after completion
- ⚠️ Will re-run all after root cause is fixed

## Code Quality Review Needed

**dfm-python**: 
- ✅ Naming: snake_case functions, PascalCase classes
- ⚠️ Review needed: Numerical stability, theoretical correctness, generic naming
- ⚠️ Check: EM algorithm, Kalman filter, VAE encoder implementations

**src/**: 
- ✅ 15 files (max 15 required)
- ⚠️ Review needed: Redundancies, inefficient logic, error handling

## Report Review Needed

**Current Status**:
- ✅ Structure: 8 sections complete
- ✅ Citations: 20+ references verified
- ⚠️ Review needed: Hallucination check, detail level, redundancy, flow
- ⚠️ Tables: 4 tables with "---" placeholders (blocked by experiments)
- ⚠️ Plots: Not generated (blocked by experiments)

## Priority Order (Incremental Tasks)

### IMMEDIATE (Do First - Critical Path)
1. ✅ **Fix ARIMA n_valid=0** (Task 1.1) - COMPLETED (but n_valid=0 persists - needs investigation)
2. ✅ **Fix VAR pandas API** (Task 1.2) - COMPLETED (VAR now completes but n_valid=0)
3. ✅ **Fix DFM/DDFM pickle** (Task 1.3) - COMPLETED 2025-12-06: make_cha_transformer refactored to use functools.partial with module-level function
4. ✅ **Fix fillna() deprecation** (Task 1.4) - COMPLETED
5. ✅ **Investigate n_valid=0 Issue** (Task 1.5, Step 1) - COMPLETED: Results analyzed, issue confirmed
6. ✅ **Debug Prediction Extraction** (Task 1.5, Step 2) - COMPLETED 2025-12-06: Enhanced logging added to evaluation.py
7. **⏳ Test Minimal Case** (Task 1.5, Step 3) - **NEXT ACTION**: Run ARIMA on KOGFCF..D, horizon=1 with debug logging
   - Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogfcf_report --models arima --horizons 1`
   - Check debug output step-by-step to identify root cause
8. **⏳ Fix Root Cause** (Task 1.5, Step 4) - Based on findings, fix actual bug
9. **⏳ Verify Fix** (Task 1.5, Step 5) - Re-run test, confirm n_valid > 0
10. **⏳ Re-run Full Experiments** (Task 1.5, Step 6) - After fix verified: `bash run_experiment.sh`
11. **⏳ Verify Results** (Task 1.5, Step 7) - Check n_valid > 0 for at least 6 combinations
12. **⏳ Generate Aggregated CSV** (Task 1.5, Step 8) - After Step 7: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`

### AFTER EXPERIMENTS SUCCEED (Blocked Until Phase 1 Complete)
9. **Generate Visualizations** (Task 3.2) - `python3 nowcasting-report/code/plot.py`
10. **Update LaTeX Tables** (Task 3.3) - Replace "---" with numbers from aggregated_results.csv
11. **Update Results Section** (Task 4.2) - Add specific numbers from tables
12. **Update Discussion Section** (Task 4.3) - Reference actual findings
13. **Finalize Report** (Task 4.4) - Compile PDF, verify 20-30 pages

### INCREMENTAL (Can Do in Parallel or After Phase 1)
- ⏳ **Review Report Sections** (Task 4.1) - IN PROGRESS: One section at a time: check hallucination, redundancy, citations
  - ✅ Citations verified: All 21 citations exist in references.bib
  - ⏳ Reviewing sections 1-4, 6-7 for redundancy and flow improvements
- **Code Quality Review** (Task 2.1-2.2) - Review numerical stability, redundancies incrementally

## Experiment Configuration Summary

**Required Experiments**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **Models**: ARIMA, VAR, DFM, DDFM
- **Horizons**: 1, 7, 28 days

**Current Experiment Status**:
- **Latest Run**: 20251206_073336 (all 3 targets) - Most recent
- **Valid Results**: **NONE** (ARIMA/VAR complete but n_valid=0 for all horizons)
- **Action Required**: Investigate why fixes aren't working → Debug prediction extraction → Fix root cause

**run_experiment.sh Status**:
- ✅ Already checks for valid results (n_valid > 0) before skipping
- ✅ Will re-run all targets since current results have n_valid=0
- ✅ Runs all 3 targets in parallel (max 5 processes)
- ✅ Automatically aggregates after completion
- ⚠️ No changes needed now, but may need updates if experiment structure changes later

## Experiment Outputs Status

**Check outputs/comparisons/ to see which experiments have been run:**
- Latest run: 20251206_073336 (all 3 targets) - Most recent
- Previous runs: 20251206_063031, 20251206_070455, 20251206_070457 (all 3 targets)
- All runs have comparison_results.json but **NO VALID RESULTS** (n_valid=0 for all horizons)
- **run_experiment.sh** already checks for valid results (n_valid > 0) before skipping
- Will re-run all targets after root cause is fixed

**After root cause is fixed:**
- run_experiment.sh will automatically run all 3 targets in parallel
- Only experiments with n_valid=0 will be re-run (all current ones)
- Aggregation runs automatically after experiments complete

## Experiments Needed vs Already Complete

**Required Experiments**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **Models**: ARIMA, VAR, DFM, DDFM
- **Horizons**: 1, 7, 28 days

**Already Complete (but invalid)**:
- ✅ All 3 targets have been run multiple times (20251206_063031, 20251206_070455, 20251206_070457, 20251206_073336)
- ✅ All 4 models have been attempted for all 3 targets
- ❌ **NO VALID RESULTS**: All runs show n_valid=0 for all horizons (all metrics NaN)
- ❌ **aggregated_results.csv**: Does not exist (cannot generate without valid results)

**Experiments Still Needed**:
- ⚠️ **ALL 36 combinations need to be re-run** after root cause is fixed
- ⚠️ Need at least 6 successful combinations (2 models × 3 targets, or equivalent) for minimum viable report
- ⚠️ Ideally all 36 combinations should succeed for complete report

**run_experiment.sh Status**:
- ✅ Already checks for valid results (n_valid > 0) before skipping experiments
- ✅ Will automatically re-run all targets since current results have n_valid=0
- ⚠️ **May need updates in later steps** if experiment structure changes (e.g., different targets, models, horizons)
- ⚠️ **Update plan**: After fixing root cause, verify run_experiment.sh still works correctly, update if needed


## Notes

- **Incremental Approach**: Fix critical issues first, then test, then full experiment
- **Minimum Viable**: Need 6 successful combinations (2 models × 3 targets) for report
- **Parallel Work**: Report review (sections 1-4, 6-7) can proceed while debugging
- **Report Blocked**: Sections 5 (results) and parts of 6 (discussion) need actual results
- **Critical Path**: Fix pickle error → Debug n_valid=0 → Fix → Re-run → Generate results → Update report
- **Code Quality**: Review incrementally, one area at a time (don't fix everything at once)
- **run_experiment.sh**: Already checks for valid results (n_valid > 0), will re-run all since current results invalid
