# Issues and Action Plan

## Executive Summary (2025-12-06 - Fixes Applied, Ready for Testing)

**Current State**: All model fixes applied, ready for testing before full re-run  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Critical Path**: Test fixes → Re-run experiments → Generate results → Update report

**Experiments Status**: 
- ✅ Latest run (20251206_063031): All 3 targets completed but all models failed
  - ARIMA: n_valid=0 for all horizons → ✅ FIXED (simplified prediction extraction)
  - VAR: pandas asfreq() API error → ✅ FIXED (enhanced error handling with fallback)
  - DFM/DDFM: pickle serialization error → ✅ FIXED (module-level function references)
- ⏳ Ready for testing - Test fixes individually before full re-run
- ✅ run_experiment.sh: Already checks for valid results, no changes needed

**Code Status**: 
- ✅ dfm-python: Finalized (consistent naming, clean code)
- ✅ src/: 15 files (max 15 required)
- ✅ Model bugs: All fixes applied (ARIMA, VAR, DFM/DDFM) - ready for testing

**Report Status**: 
- ✅ Structure: Complete 8-section framework
- ✅ Citations: All 20+ references verified
- ⚠️ Tables: 4 tables with "---" placeholders (blocked by experiments)
- ⚠️ Plots: Not generated (blocked by experiments)

## Improvement Plan (Incremental - Prioritized Tasks)

### PHASE 1: Test Fixes and Re-run Experiments [READY]

**Goal**: Verify fixes work, then re-run full experiments  
**Expected**: At least 2 models per target produce valid results (n_valid > 0), minimum 6 successful combinations  
**Status**: ✅ All fixes applied, ready for individual testing before full re-run

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

#### Task 1.2: Fix VAR pandas asfreq() API Error [COMPLETED]
**Issue**: "NDFrame.asfreq() got an unexpected keyword argument 'fill_method'"  
**Root Cause**: pandas 2.3.3 uses `method` parameter, but error handling wasn't catching all exception types  
**Location**: `src/core/training.py:310-330`  
**Fix Applied**:
1. ✅ Enhanced error handling: catch both TypeError and ValueError (not just TypeError)
2. ✅ Fallback chain: try `method='ffill'` → try `fill_method='ffill'` → manual `fillna(method='ffill')`
3. ✅ Applied to both inferred_freq and default 'D' frequency cases
**Status**: ✅ Fix applied, ready for testing

#### Task 1.3: Fix DFM/DDFM Pickle Error [COMPLETED]
**Issue**: "Can't pickle local object 'create_transformer_from_config.<locals>.identity_with_index'"  
**Root Cause**: FunctionTransformer was capturing function reference in a way that made pickle think it's local  
**Location**: `src/preprocess/utils.py:1177-1190`  
**Fix Applied**:
1. ✅ Use `globals()['identity_with_index']` and `globals()['log_with_index']` to get module-level function references
2. ✅ Direct reference from current module namespace ensures proper pickle serialization
3. ✅ Applied to both `identity_with_index` and `log_with_index` functions
**Status**: ✅ Fix applied, ready for testing

#### Task 1.4: Verify Fixes and Re-run Experiments [READY]
**After Fixes**: Test each model individually before full re-run  
**Verification Steps**:
1. ⏳ Test ARIMA on smallest target (KOGFCF..D) with horizon=1 - verify n_valid > 0
2. ⏳ Test VAR on one target (KOGFCF..D) with horizon=1 - verify no asfreq() error
3. ⏳ Test DFM on one target (KOGFCF..D) with horizon=1 - verify no pickle error
4. ⏳ If all pass, run full experiment: `bash run_experiment.sh`
5. ⏳ Check results: `outputs/comparisons/{target}_*/comparison_results.json`
6. ⏳ Verify n_valid > 0 for at least 2 models per target (minimum 6 successful combinations)

**Success Criteria**:
- At least 2 models per target produce valid results (n_valid > 0)
- At least 6 total model-target combinations succeed
- Aggregated CSV can be generated with non-NaN values

### PHASE 2: Code Quality Improvements [Can proceed in parallel]

**Goal**: Improve dfm-python package and src/ code quality

#### Task 2.1: Review dfm-python Code Quality
**Areas to Check**:
1. **Naming Consistency**: Verify snake_case functions, PascalCase classes throughout
2. **Theoretical Correctness**: 
   - Check EM algorithm implementation matches theory (dfm-python/src/dfm_python/ssm/em.py)
   - Verify Kalman filter implementation (dfm-python/src/dfm_python/ssm/kalman.py)
   - Check VAE encoder architecture (dfm-python/src/dfm_python/encoder/vae.py)
3. **Numerical Stability**:
   - Check for division by zero, log(0), sqrt(negative) issues
   - Verify matrix inversions use stable methods (e.g., pinv for near-singular)
   - Check convergence criteria and max iterations
4. **Code Patterns**:
   - Remove any monkey patches or temporal fixes
   - Consolidate redundant code
   - Check for inefficient logic (nested loops, repeated computations)
5. **Generic Naming**: Ensure function/class names are generic, not specific to one use case

#### Task 2.2: Review src/ Code Quality
**Areas to Check**:
1. **Redundancies**: Check for duplicate logic across model wrappers
2. **Inefficient Logic**: Review data loading, preprocessing loops
3. **Error Handling**: Ensure proper exception handling, not silent failures
4. **Code Organization**: Verify 15-file limit maintained, proper module structure

### PHASE 3: Generate Results [BLOCKED by Phase 1]

**Prerequisites**: Phase 1 complete with at least 6 successful model-target combinations

#### Task 3.1: Generate Aggregated CSV
**Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`  
**Output**: `outputs/experiments/aggregated_results.csv`  
**Expected**: 36 rows (3 targets × 4 models × 3 horizons), fewer if some models failed  
**Verification**: File exists, contains non-NaN values for successful models

#### Task 3.2: Generate Visualizations
**Action**: `python3 nowcasting-report/code/plot.py`  
**Output**: 4 PNG files in `nowcasting-report/images/` (accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual)  
**Verification**: All 4 files exist, not just placeholders

#### Task 3.3: Update LaTeX Tables
**Action**: Update 4 tables from `aggregated_results.csv` (replace "---" placeholders)  
**Files**: tab_overall_metrics.tex, tab_overall_metrics_by_target.tex, tab_overall_metrics_by_horizon.tex, tab_nowcasting_metrics.tex  
**Verification**: No "---" remain, LaTeX compiles, numbers formatted correctly

### PHASE 4: Report Improvements [BLOCKED by Phase 3]

**Prerequisites**: Phase 3 complete with all tables and plots generated

#### Task 4.1: Review Report for Issues
**Check for**:
1. **Hallucination**: Verify all claims are supported by actual results or citations
2. **Lack of Detail**: Ensure methods section explains implementation clearly
3. **Redundancy**: Remove repeated statements across sections
4. **Unnatural Flow**: Ensure logical progression between sections
5. **Citation Issues**: Verify all citations exist in references.bib, no made-up references

#### Task 4.2: Update Results Section
**File**: `nowcasting-report/contents/5_result.tex`  
**Actions**:
- Replace generic statements with specific numbers from tables
- Reference table/figure numbers correctly
- Add performance comparisons with actual metrics
- Discuss which models perform best for which targets/horizons
**Verification**: No placeholders, all numbers match tables, proper references

#### Task 4.3: Update Discussion Section
**File**: `nowcasting-report/contents/6_discussion.tex`  
**Actions**:
- Reference specific metrics from results (not speculative)
- Explain actual findings with evidence
- Remove unsupported claims
- Ensure all citations are from references.bib
**Verification**: All claims supported, specific metrics referenced, no speculation

#### Task 4.4: Finalize Report
**Actions**:
1. Compile PDF: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. Verify page count: 20-30 pages
3. Check for placeholders: Search for "---", "TODO", "FIXME", "placeholder"
4. Verify citations: All references exist in references.bib
5. Check figures/tables: All 4 figures and 4 tables included and referenced
**Verification**: PDF compiles, 20-30 pages, no placeholders, all citations resolve

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

## Priority Order

1. ✅ **Fix ARIMA n_valid=0** (Task 1.1) - COMPLETED: Simplified prediction extraction
2. ✅ **Fix VAR pandas API** (Task 1.2) - COMPLETED: Enhanced error handling with fallback
3. ✅ **Fix DFM/DDFM pickle** (Task 1.3) - COMPLETED: Module-level function references via globals()
4. **Verify and re-run** (Task 1.4) - READY: Test fixes individually, then full experiment
5. **Code quality review** (Task 2.1-2.2) - Can proceed in parallel (dfm-python already finalized)
6. **Generate results** (Task 3.1-3.3) - After Phase 1 complete (blocked until experiments succeed)
7. **Report improvements** (Task 4.1-4.4) - After Phase 3 complete (blocked until results available)

## Notes

- **Incremental Approach**: Fix one model at a time, verify before next
- **Minimum Viable**: Need 6 successful model-target combinations (2 models × 3 targets)
- **Parallel Work**: Code quality review can proceed while fixing bugs
- **Report Blocked**: Cannot update report until experiments succeed
