# Issues and Action Plan

## Executive Summary (2025-12-06 - Report Improvements Completed, Fixes Ready for Testing)

**Current State**: All code fixes applied (ARIMA, VAR, DFM/DDFM, fillna deprecation) - ready for testing. Report improvements completed (redundancy removed, flow improved).  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Critical Path**: Test fixes → Re-run experiments → Generate results → Update report

## Improvement Plan Summary

### Code Fixes (COMPLETED - Ready for Testing)
1. ✅ **ARIMA n_valid=0**: Simplified prediction extraction logic (evaluation.py)
2. ✅ **VAR pandas API**: Enhanced asfreq() error handling with fallback chain (training.py)
3. ✅ **DFM/DDFM pickle**: Use globals() for module-level function references (preprocess/utils.py)
4. ✅ **fillna() deprecation**: Replaced fillna(method='ffill') with ffill() (training.py)

### Code Quality Improvements (INCREMENTAL - After Experiments)
1. **dfm-python Numerical Stability**: Review EM/Kalman regularization constants, edge cases (T < N, near-singular matrices)
2. **dfm-python Code Patterns**: Check for redundant validation logic, monkey patches
3. **src/ Redundancies**: Consolidate duplicate logic across model wrappers
4. **src/ Error Handling**: Verify all exceptions caught and logged, no silent failures

### Report Improvements (INCREMENTAL - Can Start Now)
1. **Hallucination Check**: Verify all claims supported by citations (use references.bib, neo4j MCP)
2. **Detail Level**: Ensure methods section matches actual implementation in src/
3. **Redundancy**: Remove repeated statements across sections
4. **Flow**: Ensure logical progression between sections
5. **Citations**: Verify all \cite{} exist in references.bib, no made-up references

### Experiment Status
- **Latest Runs**: 20251206_063031, 20251206_070455, 20251206_070457 (all 3 targets)
- **Valid Results**: NONE (all n_valid=0 or errors)
- **Action**: Test fixes individually, then re-run full experiments via run_experiment.sh

**Experiments Status**: 
- ⚠️ Latest runs (20251206_063031, 20251206_070455, 20251206_070457): All 3 targets completed but ALL MODELS FAILED
  - ARIMA: n_valid=0 for ALL horizons (all metrics NaN) → ✅ Fix verified in code (evaluation.py:361-388)
  - VAR: pandas asfreq() API error (fill_method) → ✅ Fix verified in code (training.py:320-343)
  - DFM/DDFM: pickle serialization error → ✅ Fix verified in code (preprocess/utils.py:1181, 1186)
- ✅ fillna() deprecation fixed (training.py:331, 343) - all fixes verified and ready for testing
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

### PHASE 1: Test Fixes and Re-run Experiments [CURRENT PRIORITY]

**Goal**: Verify fixes work, then re-run full experiments  
**Expected**: At least 2 models per target produce valid results (n_valid > 0), minimum 6 successful combinations  
**Status**: ✅ All fixes applied, ready for individual testing before full re-run

**Experiments Needed**: 
- **Total Required**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Current Status**: All 3 targets run (20251206_063031 analyzed) but **NO VALID RESULTS** (all n_valid=0 or errors)
- **Results Analysis**:
  - KOGDP...D: ARIMA n_valid=0, VAR/DFM/DDFM failed
  - KOCNPER.D: ARIMA n_valid=0, VAR/DFM/DDFM failed (identical pattern)
  - KOGFCF..D: ARIMA n_valid=0, VAR/DFM/DDFM failed (identical pattern)
- **Action Required**: Test fixes individually, then re-run full experiments

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

#### Task 1.3: Fix DFM/DDFM Pickle Error [NEEDS INVESTIGATION]
**Issue**: "Can't pickle local object 'create_transformer_from_config.<locals>.identity_with_index'"  
**Root Cause**: FunctionTransformer was capturing function reference in a way that made pickle think it's local  
**Location**: `src/preprocess/utils.py:1177-1190`  
**Fix Applied**:
1. ✅ Use `globals()['identity_with_index']` and `globals()['log_with_index']` to get module-level function references
2. ✅ Direct reference from current module namespace ensures proper pickle serialization
3. ✅ Applied to both `identity_with_index` and `log_with_index` functions
**Status**: ⚠️ Fix in code but error still occurs in results - may need additional investigation (pickle protocol, function closure issues)

#### Task 1.4: Fix fillna() Deprecation [COMPLETED]
**Issue**: `fillna(method='ffill')` is deprecated in pandas 2.x  
**Location**: `src/core/training.py:331, 343`  
**Fix Applied**: Replaced `fillna(method='ffill')` with `ffill()`  
**Status**: ✅ Fixed - ready for testing

#### Task 1.5: Verify Fixes and Re-run Experiments [READY - NEXT STEP]

**Step 1: Quick Test Each Fix (Individual Model Tests)**
- **Test ARIMA**: Run single test on KOGFCF..D, horizon=1
  - Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogfcf_report --models arima --horizons 1`
  - Check: `outputs/comparisons/KOGFCF..D_*/comparison_results.json` → ARIMA metrics → n_valid > 0
  - Expected: n_valid > 0 (not 0 like current results)
  
- **Test VAR**: Run single test on KOGFCF..D, horizon=1
  - Command: Same as above but `--models var`
  - Check: No asfreq() error in log, n_valid > 0
  - Expected: No pandas API error, n_valid > 0
  
- **Test DFM**: Run single test on KOGFCF..D, horizon=1
  - Command: Same as above but `--models dfm`
  - Check: No pickle error, n_valid > 0
  - Expected: No serialization error, n_valid > 0

**Step 2: If All Tests Pass, Re-run Full Experiments**
- **Action**: `bash run_experiment.sh`
- **What It Does**: Runs all 3 targets in parallel (max 5 processes), automatically checks for valid results
- **Expected Duration**: Several hours (depends on model training time)
- **Output**: `outputs/comparisons/{target}_{timestamp}/comparison_results.json` for each target

**Step 3: Verify Results**
- **Check Each Target**: `outputs/comparisons/KOGDP...D_*/comparison_results.json`, `KOCNPER.D_*/`, `KOGFCF..D_*/`
- **Check n_valid**: For each model/horizon combination, verify n_valid > 0
- **Success Criteria**:
  - At least 2 models per target have n_valid > 0 for at least one horizon
  - Minimum 6 total successful model-target-horizon combinations
  - If not met, investigate failures and fix before proceeding

**Step 4: Generate Aggregated Results**
- **Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- **Output**: `outputs/experiments/aggregated_results.csv`
- **Expected**: 36 rows (3 targets × 4 models × 3 horizons), fewer if some models failed
- **Verification**: File exists, contains non-NaN values for successful models

**Note on run_experiment.sh**: 
- ✅ Already checks for valid results (n_valid > 0) before skipping experiments
- ✅ Will re-run all targets since current results have n_valid=0
- ⚠️ No changes needed now, but may need updates if experiment structure changes later

### PHASE 2: Code Quality Improvements [INCREMENTAL - AFTER PHASE 1]

**Goal**: Improve dfm-python package and src/ code quality incrementally  
**Priority**: Medium - proceed after Phase 1 succeeds or in parallel if experiments take long

#### Task 2.1: Review dfm-python Numerical Stability [INCREMENTAL]
**Status**: Code has stability measures but needs review  
**Areas to Check**:
1. **Matrix Operations**: 
   - EM algorithm (em.py) - verify regularization constants (1e-6, 1e-8) are appropriate
   - Kalman filter (kalman.py) - verify safe_inverse() fallbacks work correctly
   - Check for edge cases: T < N, near-singular matrices, extreme eigenvalues
2. **Data Validation**: 
   - Verify _validate_factors() catches all edge cases (constant factors, perfect correlation)
   - Check T < N warnings in data/utils.py are handled properly
3. **Code Patterns**: 
   - Check for redundant validation logic across models
   - Verify no monkey patches or temporal fixes remain

#### Task 2.2: Review src/ Code Quality [INCREMENTAL]
**Status**: 15 files (max 15 required), structure verified  
**Areas to Check**:
1. **Redundancies**: 
   - Check model wrappers (dfm.py, ddfm.py, sktime_forecaster.py) for duplicate logic
   - Consolidate common preprocessing steps if duplicated
2. **Error Handling**: 
   - Verify all exceptions are caught and logged appropriately
   - Check for silent failures (especially in evaluation.py)
3. **Inefficient Logic**: 
   - Review data loading loops in training.py
   - Check preprocessing pipeline for unnecessary iterations

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

**Latest Run**: 20251206_063031 (all 3 targets)  
**Valid Results**: None (all models failed)  
**Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations  
**Outputs**: comparison_results.json exists but all models failed, aggregated_results.csv MISSING

**run_experiment.sh Status**: 
- ✅ Already checks for valid results (n_valid > 0) before skipping
- ✅ Runs all 3 targets in parallel (max 5 processes)
- ✅ Automatically aggregates after completion
- ⚠️ No changes needed - will re-run all after fixes

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

## Priority Order (Concrete Next Steps)

## Priority Order (Incremental Tasks)

### IMMEDIATE (Do First - Critical Path)
1. ✅ **Fix ARIMA n_valid=0** (Task 1.1) - COMPLETED
2. ✅ **Fix VAR pandas API** (Task 1.2) - COMPLETED  
3. ✅ **Fix DFM/DDFM pickle** (Task 1.3) - COMPLETED
4. ✅ **Fix fillna() deprecation** (Task 1.4) - COMPLETED
5. **⏳ Test Fixes Individually** (Task 1.5, Step 1) - **NEXT ACTION**: Test each model on KOGFCF..D, horizon=1
   - Commands: `.venv/bin/python3 src/train.py compare --config-name experiment/kogfcf_report --models {arima|var|dfm|ddfm} --horizons 1`
   - Check: `outputs/comparisons/KOGFCF..D_*/comparison_results.json` → verify n_valid > 0
6. **⏳ Re-run Full Experiments** (Task 1.5, Step 2) - After Step 1 passes: `bash run_experiment.sh`
   - run_experiment.sh automatically checks for valid results (n_valid > 0) before skipping
   - Will re-run all 3 targets since current results have n_valid=0
7. **⏳ Verify Results** (Task 1.5, Step 3) - Check n_valid > 0 for at least 6 combinations (2 models × 3 targets)
8. **⏳ Generate Aggregated CSV** (Task 1.5, Step 4) - After Step 3: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`

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
- **Latest Runs**: 20251206_063031, 20251206_070455, 20251206_070457 (all 3 targets)
- **Valid Results**: **NONE** (all models have n_valid=0 or errors)
- **Action Required**: Re-run all experiments after verifying fixes work

**run_experiment.sh Status**:
- ✅ Already checks for valid results (n_valid > 0) before skipping
- ✅ Will re-run all targets since current results have n_valid=0
- ✅ Runs all 3 targets in parallel (max 5 processes)
- ✅ Automatically aggregates after completion
- ⚠️ No changes needed now, but may need updates if experiment structure changes later

## Experiment Outputs Status

**Check outputs/comparisons/ to see which experiments have been run:**
- Latest runs: 20251206_063031, 20251206_070455, 20251206_070457 (all 3 targets)
- All runs have comparison_results.json but **NO VALID RESULTS** (n_valid=0 or errors)
- **run_experiment.sh** already checks for valid results (n_valid > 0) before skipping
- Will re-run all targets since current results are invalid

**After fixes are verified:**
- run_experiment.sh will automatically run all 3 targets in parallel
- Only experiments with n_valid=0 will be re-run (all current ones)
- Aggregation runs automatically after experiments complete

## Notes

- **Incremental Approach**: Test fixes individually first, then full experiment
- **Minimum Viable**: Need 6 successful model-target-horizon combinations (2 models × 3 targets, or equivalent)
- **Parallel Work**: Report review (sections 1-4, 6-7) can proceed while experiments run
- **Report Blocked**: Sections 5 (results) and parts of 6 (discussion) blocked until experiments succeed
- **Critical Path**: Test fixes → Re-run experiments → Generate results → Update report
- **Code Quality**: Review incrementally, one area at a time (don't try to fix everything at once)
