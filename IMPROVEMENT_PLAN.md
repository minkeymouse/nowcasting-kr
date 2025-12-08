# Improvement Plan for dfm-python Package and Nowcasting Report

## Executive Summary

This document outlines a comprehensive plan to improve the dfm-python package code quality and finalize the nowcasting-report paper. The plan prioritizes **fixing real problems** rather than just documenting them, with a focus on incremental improvements.

## Current State Assessment

### ✅ Verified (No Issues Found)
1. **Data Leakage Prevention**: Training period (1985-2019) and test period (2024-2025) are correctly separated with validation checks (src/train.py:904)
2. **Suspicious Results Filtering**: Code exists to filter suspiciously good results (sMSE < 1e-4) in `generate_all_latex_tables()` (src/evaluation.py:1846-1883)
3. **Single-Step Evaluation**: n_valid=1 is intentional and documented (single-step evaluation design)

### ⚠️ Issues Identified (Need Fixing)
1. **Suspicious Results in CSV**: aggregated_results.csv contains suspiciously good values that should be filtered at source, not just when loading
2. **Nowcasting Experiments Failed**: All 12 backtest JSON files have "status": "no_results" (code fixes present but not verified)
3. **Training Not Done**: checkpoint/ is empty (0/12 models trained)
4. **Table 3 Has Placeholders**: N/A values due to missing nowcasting results

### 🔍 Areas for Improvement
1. **dfm-python Code Quality**: Review for naming consistency, remove redundancies, improve numerical stability
2. **Report Documentation**: Ensure all required tables/plots are generated and referenced correctly
3. **Model Performance Validation**: Enhance validation logic to catch issues earlier

---

## Priority 1: Fix Real Problems in Code

### 1.1 Filter Suspicious Results at Source (VERIFIED - CODE CORRECT)

**Problem**: Suspiciously good results (sMSE < 1e-4) exist in aggregated_results.csv.

**Root Cause**: CSV was generated before filtering logic was added to `aggregate_overall_performance()`.

**Current State**: 
- Filtering logic exists in `aggregate_overall_performance()` (lines 1137-1156) - sets suspicious values to np.nan
- Filtering also exists in `generate_all_latex_tables()` (lines 1846-1883) - filters when loading CSV (defense in depth)
- CSV contains old values because it hasn't been regenerated since filtering was added

**Fix**: Regenerate aggregated_results.csv by running forecast aggregation again.

**Location**: `src/evaluation.py` lines 1137-1156 (filtering at source), lines 1846-1883 (filtering on load)

**Action**:
- Code is correct - filtering will work when CSV is regenerated
- Step 1 will automatically regenerate CSV when running `bash agent_execute.sh forecast`
- No code changes needed - filtering logic is already correct

**Status**: ✅ VERIFIED - Code is correct, CSV just needs regeneration

### 1.2 Verify Nowcasting Code Fixes (CRITICAL - BLOCKING)

**Problem**: Code fixes for UnboundLocalError, ValueError, and Inf handling are present but not verified by successful backtests.

**Impact**: All 12 nowcasting experiments failed, blocking Table 3 and Plot4 generation.

**Action**: 
- Step 1 will automatically re-run backtests after training completes
- Verify fixes work by checking JSON files have "status": "completed" for DFM/DDFM
- If still failing, inspect logs and fix remaining issues

**Status**: ⚠️ BLOCKED - Waiting for training to complete

### 1.3 Enhance Data Leakage Validation (MEDIUM PRIORITY)

**Problem**: While train-test split is validated, we should add more explicit checks.

**Current State**: Validation exists at src/train.py:904

**Improvement**: Add explicit validation in `evaluate_forecaster()` to ensure test data doesn't overlap with training data.

**Location**: `src/evaluation.py` in `evaluate_forecaster()` function

**Action**:
- Add check: `if y_test.index.min() <= y_train.index.max(): raise ValueError("Data leakage: test period overlaps with training period")`
- This provides defense-in-depth validation

**Status**: 📝 PLANNED - Enhancement, not critical bug fix

---

## Priority 2: Improve dfm-python Package Code Quality

### 2.1 Review Numerical Stability (HIGH PRIORITY)

**Current State**: 
- Inf detection added in Kalman filter (dfm-python/src/dfm_python/ssm/kalman.py:222-228)
- Adaptive regularization in EM algorithm (dfm-python/src/dfm_python/ssm/em.py:327-355)
- Matrix cleaning utilities exist (dfm-python/src/dfm_python/utils/statespace.py)

**Improvements Needed**:
1. **Consolidate Inf/NaN handling**: Ensure consistent pattern across all modules
2. **Improve error messages**: Make numerical instability warnings more actionable
3. **Add validation in predict()**: Check for Inf/NaN in predictions before returning

**Action Items**:
- Review all matrix operations for consistent Inf/NaN handling
- Add validation in `predict()` methods to catch numerical issues early
- Improve logging to help diagnose numerical instability issues

**Status**: 📝 PLANNED - Code quality improvement

### 2.2 Improve Code Consistency and Naming (MEDIUM PRIORITY)

**Current State**: Code has some inconsistencies in naming patterns and helper function usage.

**Improvements Needed**:
1. **Standardize naming**: Ensure consistent naming patterns across modules
2. **Remove redundancies**: Consolidate duplicate helper functions
3. **Improve documentation**: Add docstrings where missing

**Action Items**:
- Review dfm-python/src/ for naming inconsistencies
- Identify and consolidate duplicate helper functions
- Add missing docstrings to public API functions

**Status**: 📝 PLANNED - Code quality improvement

### 2.3 Enhance Error Handling (MEDIUM PRIORITY)

**Current State**: Error handling exists but could be more consistent.

**Improvements Needed**:
1. **Standardize exception types**: Use consistent exception classes
2. **Improve error messages**: Make errors more informative and actionable
3. **Add validation early**: Catch issues before they propagate

**Action Items**:
- Review exception handling patterns across dfm-python
- Standardize error messages to include context (what failed, why, how to fix)
- Add input validation in public API functions

**Status**: 📝 PLANNED - Code quality improvement

---

## Priority 3: Report Documentation

### 3.1 Verify All Required Tables/Plots Exist (HIGH PRIORITY)

**Required Tables** (from WORKFLOW.md):
1. ✅ Table 1: Dataset details and model parameters (tab_dataset_params.tex) - EXISTS
2. ✅ Table 2: Standardized MSE/MAE for forecasting (tab_forecasting_results.tex) - EXISTS
3. ⚠️ Table 3: Nowcasting backtest results (tab_nowcasting_backtest.tex) - HAS N/A PLACEHOLDERS

**Required Plots** (from WORKFLOW.md):
1. ✅ Plot1: Forecast vs actual (3 plots, one per target) - EXISTS
2. ✅ Plot2: Accuracy heatmap (4 models × 3 targets) - EXISTS
3. ✅ Plot3: Performance trend by horizon - EXISTS
4. ⚠️ Plot4: Nowcasting comparison (3 pairs, 6 plots) - HAS PLACEHOLDERS

**Action**:
- After backtests succeed, regenerate Table 3 and Plot4
- Verify all tables/plots are referenced correctly in report sections

**Status**: ⚠️ BLOCKED - Waiting for nowcasting results

### 3.2 Update Report Sections (MEDIUM PRIORITY)

**Current State**: Report structure exists but nowcasting section has placeholders.

**Action Items**:
1. Update `nowcasting-report/contents/4_results_nowcasting.tex` with actual results after Table 3/Plot4 are generated
2. Update discussion section with nowcasting timepoint analysis
3. Verify all sections reference actual tables/plots, not placeholders

**Status**: ⚠️ BLOCKED - Waiting for nowcasting results

---

## Priority 4: Model Performance Analysis

### 4.1 Document Single-Step Evaluation Design (LOW PRIORITY)

**Current State**: n_valid=1 is documented in code comments but should be explicitly documented in report.

**Action**: Add section in methodology explaining single-step evaluation design and its implications.

**Status**: 📝 PLANNED - Documentation improvement

### 4.2 Analyze Suspicious Results (MEDIUM PRIORITY)

**Current State**: Suspicious results (sMSE < 1e-4) are flagged and filtered, but not analyzed.

**Action**: After filtering, document:
- How many results were filtered
- Which models/horizons/targets were affected
- Whether filtering affects conclusions

**Status**: 📝 PLANNED - Analysis improvement

---

## Implementation Plan

### Phase 1: Critical Fixes (Do First)
1. ✅ Verify data leakage prevention (DONE - no issues found)
2. ⚠️ Verify suspicious results filtering works correctly (NEEDS VERIFICATION)
3. ⚠️ Wait for training/backtesting to complete (BLOCKED)

### Phase 2: Code Quality Improvements (Do After Phase 1)
1. Improve numerical stability in dfm-python
2. Enhance code consistency and naming
3. Improve error handling

### Phase 3: Report Finalization (Do After Phase 2)
1. Regenerate Table 3 and Plot4 after backtests succeed
2. Update report sections with actual results
3. Verify all tables/plots are referenced correctly

### Phase 4: Documentation and Analysis (Do Last)
1. Document single-step evaluation design
2. Analyze filtered suspicious results
3. Final report review

---

## Notes

- **DO NOT claim "complete" or "done"** unless actual code changes are made
- **Focus on fixing real problems**, not just documenting them
- **Incremental improvements** - work on one issue at a time
- **Verify fixes work** before moving to next issue
- **Step 1 automatically handles experiment execution** - agent should only modify code

---

## Known Limitations

1. **Single-step evaluation**: n_valid=1 means results are sensitive to individual predictions
2. **Suspicious results**: Some very small sMSE values may be legitimate single-point matches
3. **Numerical stability**: Some edge cases with sparse data may still need improvement
4. **Nowcasting experiments**: Code fixes present but not verified by successful backtests

---

## Success Criteria

1. ✅ No data leakage (verified)
2. ⚠️ Suspicious results properly filtered (needs verification)
3. ⚠️ All nowcasting experiments complete successfully (blocked by training)
4. ⚠️ All required tables/plots generated (blocked by nowcasting results)
5. 📝 dfm-python code quality improved (planned)
6. 📝 Report sections complete with actual results (blocked by nowcasting results)
