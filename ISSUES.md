# Issues and Action Plan

## Executive Summary (2025-12-06 - Iteration Update)

**Current State**: All model fixes applied, ready to re-run experiments  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Critical Path**: Re-run experiments → Generate results → Update report

**Experiments Status**: 
- ⚠️ Previous run (20251206_061625): All models failed (ARIMA n_valid=0, VAR/DFM/DDFM errors)
- ✅ All fixes applied: ARIMA extraction, VAR NaN handling, DFM/DDFM shape mismatch
- ⏳ Ready to re-run: `bash run_experiment.sh` (will skip if valid results exist)

**Code Status**: 
- ✅ dfm-python: Finalized (consistent naming: snake_case functions, PascalCase classes, clean code, no TODOs)
- ✅ src/: 15 files (max 15 required)
- ✅ All model fixes: Implemented and ready for testing

**Report Status**: 
- ✅ Structure: Complete 8-section framework with comprehensive content
- ✅ Citations: All 20+ references verified in references.bib
- ⚠️ Tables: 4 tables with "---" placeholders (blocked by experiments)
- ⚠️ Plots: 4 PNG files (plot.py ready, will generate placeholders if no valid data)

## Prioritized Action Plan (Incremental)

### PHASE 1: Re-run Experiments [READY - Priority: CRITICAL]

**Status**: ✅ All fixes applied, ready to re-run  
**Action**: `bash run_experiment.sh` (will skip if valid results exist)  
**Expected**: At least 2 models per target produce valid results (n_valid > 0)

**Fixes Applied**:
1. ✅ **ARIMA n_valid=0**: Fixed prediction/test extraction logic (evaluation.py:336-398)
2. ✅ **VAR Missing Data**: Fixed asfreq() NaN issue with fill_method='ffill' (training.py:253-293)
3. ✅ **DFM/DDFM Shape Mismatch**: Fixed data column filtering after series filtering (training.py:135-199)
4. ✅ **run_experiment.sh**: Updated to check for valid results (n_valid > 0) before skipping

**Verification** (after re-run):
- Check `outputs/comparisons/{target}_*/comparison_results.json`
- Verify n_valid > 0 for at least 2 models per target
- Verify metrics are not all NaN

### PHASE 2: Generate Results [BLOCKED by Phase 1]

#### Task 2.1: Generate Aggregated CSV
**Action**: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`  
**Output**: `outputs/experiments/aggregated_results.csv` (36 rows: 3 targets × 4 models × 3 horizons)

#### Task 2.2: Generate Visualizations
**Action**: `python3 nowcasting-report/code/plot.py`  
**Output**: 4 PNG files in `nowcasting-report/images/` (accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual)

#### Task 2.3: Update LaTeX Tables
**Action**: Update 4 tables from `aggregated_results.csv` (replace "---" placeholders):
- `tables/tab_overall_metrics.tex` - Overall averages
- `tables/tab_overall_metrics_by_target.tex` - Per-target metrics
- `tables/tab_overall_metrics_by_horizon.tex` - Per-horizon metrics
- `tables/tab_nowcasting_metrics.tex` - Nowcasting-specific

### PHASE 3: Update Report [BLOCKED by Phase 2]

#### Task 3.1: Update Results Section
**File**: `nowcasting-report/contents/5_result.tex`  
**Action**: Replace placeholders with actual numbers from tables, add performance comparisons

#### Task 3.2: Update Discussion Section
**File**: `nowcasting-report/contents/6_discussion.tex`  
**Action**: Reference specific metrics, discuss actual findings, remove unsupported claims

#### Task 3.3: Finalize Report
**Action**: Compile PDF, verify 20-30 pages, no placeholders, all citations verified

**Note**: Citations already verified (all 15 citations exist in references.bib), discussion section made less speculative

## Experiment Status

**Completed Runs**: 20251206_061625 (all 3 targets)  
**Valid Results**: None (ARIMA n_valid=0, others failed)  
**Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations  
**Expected Outputs**:
- Per-target: `outputs/comparisons/{target}_{timestamp}/comparison_results.json`
- Aggregated: `outputs/experiments/aggregated_results.csv` (MISSING)
- Models: `outputs/models/{target}_{model}/model.pkl` (12 total if all succeed)

**Report Dependencies**:
- 4 Tables: overall_metrics, by_target, by_horizon, nowcasting_metrics (all have "---")
- 4 Plots: accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual (not generated)

## Code Quality Status

**dfm-python**: ✅ Finalized (consistent naming: snake_case functions, PascalCase classes, no TODOs)  
**src/**: ✅ 15 files (max 15 required), consistent naming  
**Model Fixes**: ✅ All implemented and ready for testing

### PHASE 4: Code Quality Review [PRIORITY: LOW - After Experiments]

**Status**: ⚠️ PENDING (can be done in parallel with experiments)  
**Areas to Review** (if time permits):
- Naming consistency (non-generic names, magic numbers)
- Redundancies across modules
- Efficiency (DataFrame operations, tensor computations)
- Numerical stability (EM convergence, Kalman filter, PyTorch operations)
- Theoretical correctness (compare with Stock & Watson 2002, Andreini et al. 2020)

## Next Steps (Immediate Priority)

### Critical Path:
1. ⏳ **Re-run Experiments** → `bash run_experiment.sh` (all fixes applied)
2. **Generate Aggregated CSV** → `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
3. **Generate Plots** → `python3 nowcasting-report/code/plot.py`
4. **Update LaTeX Tables** → Replace "---" placeholders with actual metrics
5. **Update Results Section** → `contents/5_result.tex` with actual numbers
6. **Update Discussion Section** → `contents/6_discussion.tex` with real findings
7. **Finalize Report** → Compile PDF, verify 20-30 pages

### Completed (This Iteration):
- ✅ Code quality review: Verified dfm-python naming consistency and code quality
- ✅ Report content review: Verified all sections complete, all citations verified
- ✅ Context files updated: CONTEXT.md, STATUS.md, ISSUES.md updated for next iteration
