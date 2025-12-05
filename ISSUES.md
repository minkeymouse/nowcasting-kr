# Issues and Action Plan

## Executive Summary (2025-01-XX)

**Current State**: 0/3 experiments complete, code ready, report structure complete and improved  
**Goal**: Complete 20-30 page report with actual results  
**Critical Path**: Run experiments → Generate plots → Update tables → Update content → Finalize

**Experiments**: 0/3 targets complete (KOGDP...D, KOCNPER.D, KOGFCF..D)  
**Code**: ✅ All fixes applied, ready for execution  
**Report**: ✅ Structure complete, improved flow, ⚠️ Results are placeholders  
**Action Required**: Run `bash run_experiment.sh` to execute all 3 targets

## Priority Tasks

1. ✅ **Code fixes complete** (circular import, type hints, all imports work)
2. ✅ **Report structure complete** (20-30 page framework, improved flow, citations verified)
3. ⚠️ **Run experiments** (0/3 complete) - BLOCKING for all downstream tasks
4. ⚠️ **Generate visualizations** - BLOCKED by #3
5. ⚠️ **Update report content** - BLOCKED by #3-4

## Experiment Status

**Current**: 0/3 targets complete
- KOGDP...D: ❌ No `comparison_results.json`
- KOCNPER.D: ❌ No `comparison_results.json`
- KOGFCF..D: ❌ No `comparison_results.json`

**Configuration:**
- **3 Targets**: KOGDP...D (55 series), KOCNPER.D (50 series), KOGFCF..D (19 series)
- **4 Models**: arima, var, dfm, ddfm
- **3 Horizons**: 1, 7, 28 days
- **Total**: 3 × 4 × 3 = 36 combinations

**run_experiment.sh**: ✅ Correctly configured, auto-skip logic verified
- Will run all 3 targets (none complete)
- After completion: Will auto-skip completed targets

## Resolved Issues (All Fixed)

**Code Fixes:**
- ✅ Import errors: `app.utils` → `src.utils.config_parser` (RESOLVED)
- ✅ Type hints: PyTorch/pandas type hints use string literals (RESOLVED)
- ✅ Circular import: Moved DFM/DDFM imports inside methods (lazy imports) (RESOLVED)
- ✅ Citations: All verified in references.bib (RESOLVED)

**Previous Experiment Failures (2025-12-06):**
- ✅ 45 failed attempts analyzed, all root causes fixed
- ✅ Most recent: Circular import (050328) - FIXED
- ✅ Mid: Pandas NameError (044509) - FIXED
- ✅ Earlier: Import errors (001731-040746) - FIXED

**Report Improvements:**
- ✅ Structure complete, all sections present
- ✅ Redundant warnings removed, flow improved
- ✅ All citations verified, terminology consistent
- ✅ No hallucinations, all claims backed by references

## Action Plan (After Experiments Complete)

### Phase 1: Generate Visualizations
**Status**: ⚠️ Blocked (no data)  
**Dependencies**: Experiments complete (all 3 targets have results)

**Tasks:**
1. Verify data availability: Check `outputs/experiments/aggregated_results.csv` exists (36 rows expected)
2. Generate plots: Run `python3 nowcasting-report/code/plot.py`
3. Verify outputs: 4 PNG files exist (accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual)
4. Verify quality: Plots are not placeholders, contain actual data

**Completion Criteria**: All 4 PNG files exist, are not placeholders, contain actual data visualizations.

### Phase 2: Update Tables
**Status**: ⚠️ Blocked (no data)  
**Dependencies**: Experiments complete (aggregated CSV exists)

**Tasks:**
1. Verify source data: Check `outputs/experiments/aggregated_results.csv` structure
2. Update overall metrics table: `tables/tab_overall_metrics.tex` (overall averages)
3. Update per-target table: `tables/tab_overall_metrics_by_target.tex` (3 targets)
4. Update per-horizon table: `tables/tab_overall_metrics_by_horizon.tex` (3 horizons)
5. Update nowcasting table: `tables/tab_nowcasting_metrics.tex` (if evaluated)

**Completion Criteria**: All 4 tables updated with actual values (or marked as N/A if not available). No "---" placeholders remain.

### Phase 3: Update Report Content
**Status**: ⚠️ Blocked (no results/visualizations)  
**Dependencies**: Phase 1-2 complete (plots and tables updated)

**Tasks:**
1. Update results section: `contents/5_result.tex`
   - Remove placeholder warnings
   - Add actual metrics from tables
   - Add analysis for all 3 targets
   - Reference figures and tables properly
2. Update discussion section: `contents/6_discussion.tex`
   - Remove placeholder text
   - Discuss model performance differences with real numbers
   - Reference specific results from tables
3. Update abstract (if needed): `main.tex` - Reflect actual findings

**Completion Criteria**: 
- All placeholder warnings removed
- All 3 targets have analysis in results section
- Discussion section references actual numbers from tables
- Abstract accurately reflects study findings

### Phase 4: Finalize Report
**Status**: ⚠️ Blocked (no content updates)  
**Dependencies**: Phase 3 complete (all content updated)

**Tasks:**
1. Compile report: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. Verify completeness: Page count (20-30), all figures/tables exist, no placeholders
3. Final quality check: Citations verified, terminology consistent, numbers match tables

**Completion Criteria**: 
- PDF compiles without errors
- Page count is 20-30 pages
- All figures/tables referenced in text exist
- No placeholder text remains
- All citations verified
- Numbers in text match tables

## Code Quality Status

**dfm-python Package:**
- ✅ Naming: PascalCase classes, snake_case functions (mostly consistent)
- ✅ No TODO/FIXME: Code is clean
- ✅ Numerical stability: Regularization (1e-6) in EM, safe_determinant implemented
- ⚠️ Minor naming improvements possible (e.g., `rem_nans_spline` → `remove_nans_spline`) - Low priority, widely used

**src/ Module:**
- ✅ Architecture: 17 files (15 effective, within limit)
- ✅ Imports: All fixed, circular import resolved
- ✅ Ready for execution: All code fixes applied

## Report Quality Status

**Structure**: ✅ Complete 20-30 page framework, improved flow  
**Citations**: ✅ All verified in references.bib (20+ references)  
**Terminology**: ✅ Consistent (DFM/동적 요인 모형)  
**Flow**: ✅ Redundancies removed, warnings consolidated  
**Content**: ⚠️ Placeholders remain (blocked until experiments complete)

## Experiment Requirements

**Expected Outputs (Per Target):**
- `outputs/comparisons/{target}_{timestamp}/comparison_results.json` - Full results with metrics
- `outputs/comparisons/{target}_{timestamp}/comparison_table.csv` - Summary table
- `outputs/models/{target}_{model}/model.pkl` - 4 trained models per target (12 total)

**Expected Outputs (Aggregated):**
- `outputs/experiments/aggregated_results.csv` - 36 rows (all combinations)

**Report Dependencies:**
- **4 Plots**: accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual
- **4 Tables**: overall_metrics, overall_metrics_by_target, overall_metrics_by_horizon, nowcasting_metrics

## Notes

**Experiment Execution:**
- Script: `run_experiment.sh` auto-skips completed targets (checks for `comparison_results.json`)
- Parallel Execution: Max 5 concurrent processes to avoid OOM
- Python Path: Uses `.venv/bin/python3` explicitly (no venv activation needed)
- Timeout: 24 hours per experiment
- Logs: All output goes to `outputs/comparisons/{target}_{timestamp}.log`

**Incremental Approach:**
- Complete one phase fully before moving to the next
- Verify completion criteria before proceeding
- If experiments fail: Check logs → Fix code/config → Re-run only failed targets
- Do NOT update `run_experiment.sh` for code bugs (fix in `src/` instead)

## Next Steps (Immediate)

1. **Run experiments** → `bash run_experiment.sh` (BLOCKING - must complete first)
2. **Verify completion** → Check for 3 `comparison_results.json` files, 12 model files, aggregated CSV
3. **Generate plots** → `python3 nowcasting-report/code/plot.py`
4. **Update tables** → From aggregated CSV
5. **Update report** → Replace placeholders with actual results
6. **Finalize** → Compile PDF, verify completeness
