# Issues and Action Plan

## Executive Summary (2025-01-XX)

**Current State**: No valid experiment results exist. Code fixes complete, ready to run experiments.  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package with clean code  
**Critical Path**: Re-run experiments → Generate results → Update report → Finalize

**Experiments**: 0/3 targets executed - No valid results exist, ready to run  
**Code**: ✅ All critical bugs fixed (ARIMA index matching, VAR frequency, DFM/DDFM weekly series)  
**Report**: ✅ Structure complete (20-30 pages), ⚠️ Tables have placeholders, ⚠️ Results section generic  
**Package**: ✅ dfm-python naming consistent, ⚠️ Need to verify theoretical correctness  
**src/**: 16 files (transformations.py deprecated but kept for backward compatibility)  
**Action Required**: Run experiments → Generate results → Update report content

## Inspection Results (2025-01-XX)

**Experiment Results Status:**
- ❌ No valid result directories exist - No `comparison_results.json` files found
- ❌ All previous invalid results (n_valid=0) have been cleaned up
- **Code Status**: All critical bugs fixed and ready for execution
- **Conclusion**: Ready to run experiments with fixed code. No valid results exist yet.

**Aggregated Results Status:**
- ❌ `outputs/experiments/aggregated_results.csv` does NOT exist
- **Blocked by**: Invalid experiment results (Step 6)

**Report Tables Status:**
- ❌ All 4 tables contain "---" placeholders:
  - `tab_overall_metrics.tex`: All models show "---"
  - `tab_overall_metrics_by_target.tex`: All models show "---"
  - `tab_overall_metrics_by_horizon.tex`: All models show "---"
  - `tab_nowcasting_metrics.tex`: DFM/DDFM show "---"
- **Blocked by**: Missing aggregated_results.csv (Step 7)

**Plots Status:**
- ⚠️ Need to verify if plots exist in `nowcasting-report/images/`
- **Blocked by**: Missing aggregated_results.csv (Step 7)

## Priority Action Plan (Incremental)

### PHASE 1: Critical Code Fixes (BLOCKING) [COMPLETED]

#### Task 1.1: Fix ARIMA Prediction Index Matching [COMPLETED]
**Priority**: CRITICAL  
**Status**: ✅ COMPLETE  
**Issue**: `n_valid=0` - prediction indices don't match test data indices  
**Location**: `src/eval/evaluation.py:320-400`  
**Fix Applied**: 
  - Simplified to use position-based matching (horizon h = position h-1 in test data)
  - More reliable than index matching since test data is created by splitting
  - Properly extracts prediction values for both Series and DataFrame

#### Task 1.2: Fix VAR Frequency Error [COMPLETED]
**Priority**: CRITICAL  
**Status**: ✅ COMPLETE  
**Issue**: `TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` in VAR prediction  
**Location**: `src/core/training.py:264-273` (VAR frequency setting)  
**Fix Applied**:
  - Set frequency on DatetimeIndex using `asfreq()` method
  - Infers frequency from index if possible, defaults to 'D' (daily) otherwise
  - Ensures VAR has frequency information for prediction

#### Task 1.3: Fix DFM/DDFM Weekly Series in Monthly Block [COMPLETED]
**Priority**: CRITICAL  
**Status**: ✅ COMPLETE  
**Issue**: Weekly series (KORELEC, KORCNST) included in monthly block (Block_Global clock='m')  
**Location**: `src/core/training.py:681-720` (series filtering)  
**Fix Applied**:
  - Added frequency hierarchy check: series can only be in blocks with clock frequency <= series frequency
  - Filters out weekly series from monthly blocks automatically
  - Warns when series are skipped due to incompatibility

**Completion Criteria**: ✅ All 3 fixes implemented, invalid results deleted

---

### PHASE 2: Experiment Execution

#### Task 2.1: Delete Invalid Results [COMPLETED]
**Priority**: HIGH  
**Status**: ✅ COMPLETE  
**Action**: Deleted 3 invalid result directories (completed)

#### Task 2.2: Update run_experiment.sh [COMPLETED]
**Priority**: HIGH  
**Status**: ✅ COMPLETE  
**Action**: Verified script logic - correctly checks for `comparison_results.json` to skip completed experiments

#### Task 2.3: Re-run Experiments [PENDING]
**Priority**: HIGH  
**Status**: ✅ READY - All code fixes complete, no blocking issues  
**Action**: Run `bash run_experiment.sh` to execute all 3 targets  
**Verification**: Check `n_valid > 0` for at least 2 models per target, verify `comparison_results.json` exists

---

### PHASE 3: Results Generation

#### Task 3.1: Generate Aggregated CSV [PENDING]
**Priority**: HIGH  
**Status**: ⚠️ BLOCKED by Phase 2  
**Action**: Run `from src.eval import main_aggregator; main_aggregator()`  
**Output**: `outputs/experiments/aggregated_results.csv`

#### Task 3.2: Generate Visualizations [PENDING]
**Priority**: HIGH  
**Status**: ⚠️ BLOCKED by Task 3.1  
**Action**: Run `python3 nowcasting-report/code/plot.py`  
**Output**: 4 PNG files in `nowcasting-report/images/`

#### Task 3.3: Update LaTeX Tables [PENDING]
**Priority**: HIGH  
**Status**: ⚠️ BLOCKED by Task 3.1  
**Action**: Update 4 tables from aggregated CSV (replace "---" placeholders)

---

### PHASE 4: Report Improvements

#### Task 4.1: Update Results Section [PENDING]
**Priority**: MEDIUM  
**Status**: ⚠️ BLOCKED by Phase 3  
**Issues**:
  - Generic descriptions without actual numbers
  - Placeholder tables referenced
  - No specific model performance analysis
**Action**: 
  - Add actual metrics from tables
  - Analyze performance differences between models
  - Reference specific numbers from results

#### Task 4.2: Improve Discussion Section [PENDING]
**Priority**: MEDIUM  
**Status**: ⚠️ BLOCKED by Task 4.1  
**Issues**:
  - Generic statements without supporting numbers
  - No reference to actual results
  - Some claims may be unsupported by data
**Action**:
  - Reference specific metrics from tables
  - Discuss actual model performance differences
  - Remove unsupported claims

#### Task 4.3: Verify Citations and Remove Hallucinations [PENDING]
**Priority**: MEDIUM  
**Status**: ⚠️ Can start now  
**Action**:
  - Verify all citations exist in references.bib
  - Check claims against actual references
  - Remove any unsupported statements
  - Use knowledgebase MCP for additional citations if needed

---

### PHASE 5: Code Quality Improvements

#### Task 5.1: Verify dfm-python Theoretical Correctness [PENDING]
**Priority**: LOW  
**Status**: ⚠️ Can start now  
**Action**:
  - Review EM algorithm implementation against standard DFM theory
  - Verify Kalman filter implementation correctness
  - Check numerical stability measures (already documented)
  - Compare with knowledgebase/references for theoretical validation

#### Task 5.2: Improve Code Naming Consistency [PENDING]
**Priority**: LOW  
**Status**: ⚠️ Can start now  
**Current**: dfm-python uses snake_case functions, PascalCase classes (consistent)  
**Action**:
  - Review src/ module for naming consistency
  - Ensure all functions use snake_case, classes use PascalCase
  - Check for non-generic names (e.g., `train`, `predict` should be more specific if needed)

#### Task 5.3: Remove Redundancies and Improve Structure [PENDING]
**Priority**: LOW  
**Status**: ⚠️ Can start now  
**Action**:
  - Consolidate duplicate logic in src/
  - Note: transformations.py is deprecated (re-exports from utils.py) but kept for backward compatibility
  - Remove any temporal fixes or monkey patches
  - Improve error handling consistency

---

### PHASE 6: Finalization

#### Task 6.1: Finalize Report [PENDING]
**Priority**: HIGH  
**Status**: ⚠️ BLOCKED by Phase 4  
**Action**: Compile PDF, verify 20-30 pages, check all references

---

## Code Quality Status

**dfm-python Package**:
- ✅ Naming: Consistent (snake_case functions, PascalCase classes)
- ✅ TODOs: None found
- ⚠️ Theoretical: Need to verify EM/Kalman implementations
- ⚠️ Numerical: Stability measures documented, need verification

**src/ Module**:
- ✅ Structure: 16 files (transformations.py deprecated but kept for compatibility)
- ✅ Naming: Verified - consistent snake_case functions, PascalCase classes
- ⚠️ Redundancies: transformations.py re-exports from utils.py (deprecated but kept)
- ✅ Bugs: All critical issues fixed (Phase 1)

---

## Report Quality Status

**Structure**: ✅ Complete 20-30 page framework  
**Content Issues**:
- ⚠️ Results section: Generic, no actual numbers
- ⚠️ Discussion: Generic statements, no supporting data
- ⚠️ Tables: All have "---" placeholders
- ✅ Citations: All verified in references.bib
- ⚠️ Hallucinations: Need to verify claims against results

---

## Experiment Status

**Current**: 0/3 targets executed - No valid results exist, ready to run  
**Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations  
**Code Status**: ✅ All critical bugs fixed, ready for execution

**run_experiment.sh**: ✅ Auto-skip logic works - will run all 3 targets

## Expected Outputs

**Per Target:**
- `outputs/comparisons/{target}_{timestamp}/comparison_results.json` - Full results with metrics
- `outputs/comparisons/{target}_{timestamp}/comparison_table.csv` - Summary table
- `outputs/models/{target}_{model}/model.pkl` - 4 trained models per target (12 total)

**Aggregated:**
- `outputs/experiments/aggregated_results.csv` - 36 rows (all combinations)

**Report Dependencies:**
- **4 Plots**: accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual
- **4 Tables**: overall_metrics, overall_metrics_by_target, overall_metrics_by_horizon, nowcasting_metrics

## Notes

**Experiment Execution:**
- Script: `run_experiment.sh` auto-skips completed targets (checks for `comparison_results.json`)
- Parallel Execution: Max 5 concurrent processes to avoid OOM
- Python Path: Uses `.venv/bin/python3` explicitly
- Timeout: 24 hours per experiment
- Logs: `outputs/comparisons/{target}_{timestamp}.log`

**Incremental Approach:**
- Complete one step fully before moving to the next
- Verify completion criteria before proceeding
- If experiments fail: Check logs → Fix code → Re-run only failed targets
- Do NOT update `run_experiment.sh` for code bugs (fix in `src/` instead)

## Next Iteration Priority

**IMMEDIATE (Phase 2 - Ready to Execute)**:
1. ✅ Task 1.1-1.3: Fix ARIMA/VAR/DFM bugs - COMPLETED
2. ✅ Task 2.1-2.2: Delete invalid results, verify script - COMPLETED
3. **Task 2.3: Re-run Experiments** → `bash run_experiment.sh` (ready to run with fixed code)

**AFTER PHASE 2**:
4. Task 3.1-3.3: Generate aggregated CSV, plots, update tables
5. Task 4.1-4.3: Update report with actual results, verify citations
6. Task 5.1-5.3: Code quality improvements (can start in parallel)
7. Task 6.1: Finalize report

**Notes**:
- ✅ Phase 1 complete - all critical bugs fixed
- Phase 2 ready - experiments can run with fixed code
- Phases 3-4 are sequential (experiments → results → report)
- Phase 5 can start immediately (low priority, non-blocking)
- Keep ISSUES.md under 1000 lines - remove completed tasks
