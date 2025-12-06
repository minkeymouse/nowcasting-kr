# Issues and Action Plan

## Executive Summary (2025-12-06 - Updated Configuration)

**Current State**: 
- ✅ **Configuration**: All 3 target configs created, series configs updated (block: null), data path fixed
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized and verified
- ✅ **Report Structure**: 6 sections ready (Introduction, Methodology, Production, Investment, Consumption, Conclusion) - condensed to under 15 pages
- ✅ **Code Refactoring**: src/ directory cleaned up, legacy patterns removed, consistent imports
- ⏳ **Experiments**: Need to run for 3 targets (36 combinations: 3 × 4 × 3)
- ⏳ **Report Content**: Need to populate with actual experiment results

**Goal**: Complete under 15 page LaTeX report comparing 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 3 Korean macroeconomic targets (Production: KOIPALL.G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE)  
**Status**: ⏳ Configuration ready, experiments pending  
**Progress**: 0/36 = 0% (experiments not yet run)

**Next Steps**:
1. ⏳ **Task 1.1**: Verify setup with test script [NEXT] - Run `./run_test_experiment.sh` to verify all targets and models work
2. ⏳ **Task 1.2**: Run full experiments - Run `./run_experiment.sh` for all 3 targets × 4 models × 3 horizons
3. ⏳ **Task 1.3**: Generate results and plots - Aggregate results, generate plots for report
4. ⏳ **Task 1.4**: Update report - Populate tables and sections with actual results
5. ⏳ **Task 1.5**: Compile PDF - Verify under 15 pages, all content present

## Experiment Status (2025-12-06)

**Current Configuration**:
- **Targets**: 3 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G)
- **Models**: 4 (ARIMA, VAR, DFM, DDFM)
- **Horizons**: 3 (1, 7, 28 days)
- **Total**: 36 combinations (3 × 4 × 3)

**Status**: Experiments not yet run. Use `run_test_experiment.sh` for verification before full run.

**Configuration Details**:
- All series configs: `block: null` (only global block for DFM/DDFM)
- Data file: `data/data.csv`
- Config files: `config/experiment/{koequipte,kowrccnse,koipallg}_report.yaml`

## ✅ Resolved Issues (This Iteration)

**Bug Fixes Completed**:
- ✅ **RESOLVED**: Syntax error in `src/model/sktime_forecaster.py` (2025-12-06)
  - **Issue**: SyntaxError at line 153 - incorrect indentation in `except` block
  - **Impact**: All experiment runs failed with SyntaxError
  - **Fix**: Corrected indentation of except block content
  - **Verification**: File compiles without errors (`python3 -m py_compile` passes)
  - **Status**: ✅ RESOLVED - Experiments can now run

**Configuration Updates Completed**:
- ✅ **Target Configs**: All 3 target configs created (KOEQUIPTE, KOWRCCNSE, KOIPALL.G)
- ✅ **Series Configs**: All series configs updated with `block: null`
- ✅ **Data Path**: All configs use `data/data.csv`
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized

**Report Updates Completed**:
- ✅ **Translation**: All 6 report sections translated to English
- ✅ **Structure**: Report updated for 3 targets, 4 models, 3 horizons
- ✅ **Tables**: Table generation code ready (auto-generate when results exist)
- ✅ **Plots**: Plot generation code ready

## Known Limitations

1. **Horizon 28**: May be unavailable for some targets if test set <28 points (80/20 split)
2. **DFM Numerical Instability**: Some targets may show numerical instability (EM algorithm fails) - will be documented as encountered
3. **Report Length**: Target is under 15 pages (condensed from previous 20-30 page target)

## Action Plan (Incremental Tasks) - Updated 2025-12-06

**Current Status**: Syntax error fixed and verified. Experiments not yet run. 0/36 combinations complete.
- **Outputs Check**: Only log files found in `outputs/comparisons/` (KOEQUIPTE_*.log, KOWRCCNSE_*.log, KOIPALL.G_*.log) - all show old SyntaxError from previous failed run
- **Root Cause**: Syntax error in `sktime_forecaster.py` - ✅ FIXED and verified
- **Missing**: No `comparison_results.json` files, no `outputs/experiments/aggregated_results.csv`
- **Action Required**: Run experiments now that syntax error is fixed

**Phase 1: Experiment Execution** ⏳ BLOCKING
**Status**: Configuration ready, experiments pending
**Priority**: Critical (blocks all report generation tasks)

**Prerequisites** (must complete before Phase 2):
- ✅ **Task 0.1**: Fix syntax error in `sktime_forecaster.py` - ✅ COMPLETED and verified
- ⏳ **Task 1.1**: Verify setup with test script (`./run_test_experiment.sh`) - 12 tests (3 targets × 4 models) [NEXT PRIORITY]
- ⏳ **Task 1.2**: Run full experiments (`./run_experiment.sh`) - 36 combinations (3 targets × 4 models × 3 horizons)
- ⏳ **Task 1.3**: Verify results exist - Check for `outputs/comparisons/{target}_{timestamp}/comparison_results.json` files
- ⏳ **Task 1.4**: Generate aggregated CSV - `main_aggregator()` will auto-generate when results exist

**Phase 2: Report Content Generation** ⏳ WAITING FOR PHASE 1
**Status**: Waiting for experiment results
**Priority**: High (report completion depends on this)

**Task 2.1: Generate Required Tables** [PRIORITY 1]
**Priority**: High | **Time**: 1-2 hours | **Status**: ✅ Code Ready (waiting for experiments)
**Dependencies**: Phase 1 complete (aggregated_results.csv and comparison_results.json exist)
**Note**: LaTeX table generation functions added to `src/eval/evaluation.py`. Tables will be auto-generated when `main_aggregator()` runs after experiments complete.

**Actions**:
1. **Table 1 - Dataset and Model Parameters**:
   - Extract dataset details from config files (targets, series count, date ranges)
   - Extract model parameters from config/model/ (ARIMA: order, VAR: lags, DFM: factors/iterations, DDFM: architecture/epochs)
   - Create LaTeX table with dataset info and model training parameters
   - Save to new table file or update existing structure

2. **Table 2 - Standardized MSE/MAE (36 rows)**:
   - Load `outputs/experiments/aggregated_results.csv` (should have 36 rows: 3 targets × 4 models × 3 horizons)
   - Extract columns: target, model, horizon, sMSE, sMAE
   - Format as LaTeX table with proper structure (target × model × horizon)
   - Update `nowcasting-report/tables/tab_overall_metrics.tex` or create new table file
   - Handle missing values (N/A for unavailable horizons like 28-day if test set too small)

3. **Table 3 - DFM/DDFM Backtest Monthly (2024-2025)**:
   - Check if monthly backtest results exist in comparison_results.json (may need to extract from forecast data)
   - If not available, note in plan that this requires additional processing from forecast outputs
   - Extract monthly sMSE and sMAE for DFM and DDFM (Jan 2024 - Oct 2025 or latest available)
   - Create LaTeX table with months as rows, models as columns
   - Save to new table file (e.g., `tab_backtest_monthly.tex`)

**Success Criteria**: 
- All 3 tables created/updated with actual data (or placeholders with clear notes if data unavailable)
- Tables properly formatted in LaTeX
- Missing data clearly marked (N/A, --, or footnotes)

**Task 2.2: Generate Required Plots** [PRIORITY 2]
**Priority**: High | **Time**: 1 hour | **Status**: ✅ Code Ready (waiting for experiments)
**Dependencies**: Phase 1 complete, comparison_results.json files exist
**Note**: Plot generation code ready in `nowcasting-report/code/plot.py`. Run `python3 nowcasting-report/code/plot.py` after experiments complete.

**Actions**:
1. **Plot 1 - Forecast vs Actual (3 plots, one per target)**:
   - Update `nowcasting-report/code/plot.py` function `plot_forecast_vs_actual()` to load actual forecast data
   - For each target (KOEQUIPTE, KOWRCCNSE, KOIPALL.G):
     - Load 30 months historical (original series, single line)
     - Load 30 months forecasts (5 lines: original, ARIMA, VAR, DFM, DDFM)
     - X-axis: monthly timestamps (60 months total)
     - Y-axis: target series values
     - Save to `nowcasting-report/images/forecast_vs_actual_{target}.png`
   - Note: May need to extract forecast data from model outputs or comparison_results.json structure

2. **Plot 2 - Accuracy Heatmap (4 models × 3 targets)**:
   - Verify `plot_accuracy_heatmap()` in plot.py works with current data structure
   - Should show standardized RMSE (sRMSE) as heatmap
   - Models: ARIMA, VAR, DFM, DDFM (rows)
   - Targets: KOEQUIPTE, KOWRCCNSE, KOIPALL.G (columns)
   - Save to `nowcasting-report/images/accuracy_heatmap.png`

3. **Plot 3 - Performance Trend by Horizon**:
   - Verify `plot_horizon_trend()` in plot.py works with current data
   - Should show sRMSE vs horizon (1, 7, 28 days) for each model
   - Save to `nowcasting-report/images/horizon_trend.png`

**Success Criteria**:
- All 3 plot types generated (4 total images: 3 forecast plots + 1 heatmap + 1 trend)
- Images saved to `nowcasting-report/images/` as PNG files
- Plots properly formatted with labels, legends, and readable text

**Task 2.3: Update LaTeX Tables** [PRIORITY 3]
**Priority**: High | **Time**: 30-60 min | **Status**: ✅ Auto-Generated (when experiments complete)
**Dependencies**: Task 2.1 complete (tables generated)
**Note**: Tables are automatically generated by `main_aggregator()` which calls `generate_all_latex_tables()`. Manual updates only needed if structure changes.

**Actions**:
1. **Update tab_overall_metrics.tex**:
   - Replace placeholders (--) with actual averaged metrics across all targets/horizons
   - Update caption/footnote to reflect 3 targets (remove KOMPRI30G reference)
   - Format: Model | sMSE | sMAE | sRMSE

2. **Update tab_overall_metrics_by_target.tex**:
   - Replace placeholders with actual sRMSE values per target
   - Update to show only 3 targets (remove KOMPRI30G column)
   - Format: Model | KOEQUIPTE | KOWRCCNSE | KOIPALL.G

3. **Update tab_overall_metrics_by_horizon.tex**:
   - Replace existing data (if outdated) with actual values from aggregated_results.csv
   - Update footnotes to reflect current experiment configuration
   - Format: Model | 1 day | 7 days | 28 days (N/A if unavailable)

4. **Create/Update tab_backtest_monthly.tex** (if Task 2.1 Table 3 completed):
   - Populate with monthly backtest results for DFM and DDFM
   - Format: Month | DFM sMSE | DFM sMAE | DDFM sMSE | DDFM sMAE

**Success Criteria**:
- All existing tables updated with actual data (no placeholders)
- Tables reference correct number of targets (3, not 4)
- Missing data clearly marked with N/A or footnotes

**Task 2.4: Build/Update Report Sections** [PRIORITY 4]
**Priority**: High | **Time**: 2-3 hours | **Status**: ⏳ Pending (waiting for Tasks 2.1-2.3)
**Dependencies**: Tasks 2.1-2.3 complete (tables and plots ready)

**Actions**:
1. **Update Methodology Section** (`nowcasting-report/contents/2_methodology.tex`):
   - Verify model descriptions are accurate
   - Add/update parameter details from config files
   - Ensure all 4 models (ARIMA, VAR, DFM, DDFM) are properly described

2. **Update Production Model Section** (`nowcasting-report/contents/3_production_model.tex`):
   - Add actual results for KOIPALL.G
   - Reference tables and plots with actual numbers
   - Discuss model performance comparisons with specific metrics
   - Include forecast vs actual plot reference

3. **Update Investment Model Section** (`nowcasting-report/contents/4_investment_model.tex`):
   - Add actual results for KOEQUIPTE
   - Reference tables and plots with actual numbers
   - Discuss model performance comparisons with specific metrics
   - Include forecast vs actual plot reference

4. **Update Consumption Model Section** (`nowcasting-report/contents/5_consumption_model.tex`):
   - Add actual results for KOWRCCNSE
   - Reference tables and plots with actual numbers
   - Discuss model performance comparisons with specific metrics
   - Include forecast vs actual plot reference

5. **Update Conclusion Section** (`nowcasting-report/contents/6_conclusion.tex`):
   - Summarize findings across all 3 targets
   - Highlight best-performing models per target/horizon
   - Discuss limitations (numerical instability, horizon 28 availability)
   - Future research directions

**Success Criteria**:
- All sections reference actual numbers from tables/plots
- No placeholders or "TBD" text remaining
- Consistent formatting and citation style
- Report flows logically from methodology → results → conclusion

**Task 2.5: Finalize Report** [PRIORITY 5]
**Priority**: High | **Time**: 1 hour | **Status**: ⏳ Pending (waiting for Task 2.4)
**Dependencies**: Task 2.4 complete

**Actions**:
1. Compile LaTeX PDF and verify under 15 pages
2. Check all table/plot references resolve correctly
3. Verify all citations in references.bib are used correctly
4. Fix any formatting issues
5. Ensure no Korean text issues (if report is in English, verify all text is English)

**Success Criteria**: PDF generated, under 15 pages, all content present, all references resolve, no formatting errors

**Execution Strategy**: 
- Phase 1 (Experiments) must complete before Phase 2 can start
- Phase 2 tasks are sequential: 2.1 → 2.2 → 2.3 → 2.4 → 2.5
- Each task should be completed fully before moving to next
- If data is missing at any step, document clearly and note in plan

## Experiment Status Summary

**Total**: 36 combinations (3 targets × 4 models × 3 horizons)
**Complete**: 0/36 (0%) - Experiments not yet run
**Pending**: 36/36 (100%) - All experiments need to be run

**Targets**:
- KOEQUIPTE (Equipment Investment Index) - Investment
- KOWRCCNSE (Wholesale and Retail Trade Sales) - Consumption
- KOIPALL.G (Industrial Production Index, All Industries) - Production

**Models**: ARIMA, VAR, DFM, DDFM
**Horizons**: 1, 7, 28 days

**Status**: ⏳ Ready to run experiments. Use `run_test_experiment.sh` for verification, then `run_experiment.sh` for full run.

## Summary

**Status**: ⏳ Syntax error fixed and verified. Report translation complete. Experiments ready to run. Code consolidation needed (22 files, max 15).

**Experiment Status**:
- ✅ **Syntax Error**: Fixed and verified (`python3 -m py_compile` passes) - experiments can now run
- ⏳ **Pending**: 36/36 (100%) - All experiments need to be run (0/36 complete)
- ⏳ **Results Available**: 0/3 targets - No comparison_results.json files (experiments not run yet)

**Code Quality Issues**:
- ⚠️ **File Count**: src/ has 22 Python files, max is 15 (including __init__.py) - NEEDS CONSOLIDATION
  - **Strategy**: Merge nowcast/ modules (6 files → 2), merge preprocess/utils.py, merge utils/config_parser.py
  - **Priority**: Can be done incrementally, not blocking experiments
- ✅ **Syntax**: Verified fixed (py_compile passes)
- ⏳ **Numerical Stability**: Documented in dfm-python, may need improvements if experiments fail

**Report Status**:
- ✅ **Language**: All sections translated to English - COMPLETED
- ⏳ **Placeholders**: All tables have "--" placeholders, waiting for experiment results
- ⏳ **Content**: Sections have structure ready, need actual results from experiments

**Critical Path**: 
1. ✅ **Phase 0 (COMPLETED)**: Fix syntax error - ✅ DONE and verified
2. ✅ **Phase 0.5 (COMPLETED)**: Report translation - ✅ DONE (all sections in English)
3. ⏳ **Phase 1 (BLOCKING)**: Run experiments (Tasks 1.1-1.4) - [NEXT PRIORITY]
4. ⏳ **Phase 2 (WAITING)**: Generate report content (Tasks 2.1-2.5) - Sequential after Phase 1
5. ⚠️ **Phase 0.6 (OPTIONAL)**: Consolidate src/ files (22 → 15) - Can be done incrementally

**Next Immediate Action**: 
1. ✅ Syntax error fixed and verified - code compiles correctly
2. ✅ Report translation complete - all sections in English
3. ✅ Table/plot generation code ready - will auto-generate when results exist
4. ⏳ **NEXT**: Run test script (`./run_test_experiment.sh`) to verify setup
5. ⏳ Run full experiments (`./run_experiment.sh`) - 36 combinations
6. ⏳ Tables and plots will auto-generate when experiments complete
7. ⏳ Update report sections with actual results, compile PDF (under 15 pages)

## Improvement Plan (2025-12-06)

### Phase 0.6: Code Consolidation [PRIORITY: MEDIUM - Can be done incrementally]
**Status**: ⏳ Pending | **Priority**: Medium | **Time**: 1-2 hours

**Issue**: src/ directory has 22 Python files, but requirement is max 15 files (including __init__.py)

**Current File Count**: 22 files
- Core: `__init__.py`, `train.py`, `infer.py`, `core/__init__.py`, `core/training.py`
- Eval: `eval/__init__.py`, `eval/evaluation.py`
- Model: `model/__init__.py`, `model/dfm.py`, `model/ddfm.py`, `model/sktime_forecaster.py`
- Preprocess: `preprocess/__init__.py`, `preprocess/utils.py`
- Utils: `utils/__init__.py`, `utils/config_parser.py`
- Nowcast: `nowcast/__init__.py`, `nowcast/dataview.py`, `nowcast/helpers.py`, `nowcast/nowcast.py`, `nowcast/splitters.py`, `nowcast/transformers.py`, `nowcast/utils.py`

**Consolidation Strategy**:
1. **Merge nowcast/ modules** (6 files → 2 files): Merge helpers, utils, splitters, transformers into nowcast.py
2. **Merge preprocess/utils.py** into `core/training.py` or `preprocess/__init__.py`
3. **Merge utils/config_parser.py** into `core/training.py` or `train.py`

**Target**: 22 files → ≤15 files (including all __init__.py)

**Note**: Not blocking experiments. Can be done incrementally after experiments complete.

### Phase 2: Report Content Updates [PRIORITY: HIGH - After Experiments]
**Status**: ⏳ Pending | **Priority**: High | **Dependencies**: Phase 1 complete (experiments run)

**Issues**:
1. ✅ **Language**: All report sections translated to English - COMPLETED
2. ⏳ **Placeholders**: Tables have "--" placeholders - Waiting for experiments
3. ⏳ **Content**: Sections have structure ready, need actual results from experiments

**Remaining Actions** (after experiments complete):
- Populate sections with actual results from experiments
- Replace placeholders with specific metrics and numbers
- Add discussion based on actual findings
- Generate plots and update table references

**Success Criteria**:
- ✅ All sections in English - COMPLETED
- ⏳ All placeholders replaced with actual results - PENDING (waiting for experiments)
- ✅ No hallucinated claims - COMPLETED (placeholders used)
- ✅ All citations from references.bib only - COMPLETED

### Code Quality Improvements (dfm-python)
**Status**: ⏳ Optional | **Priority**: Medium | **Time**: 2-4 hours (if needed)

**Current State**: dfm-python is finalized with clean patterns, but numerical stability issues are documented.

**Potential Improvements** (only if experiments reveal issues):
1. **Numerical Stability** (if EM algorithm fails):
   - Review regularization_scale defaults (currently 1e-5, may need 1e-4 for some targets)
   - Check Q matrix floor (0.01) - may need adjustment
   - Verify spectral radius capping (< 0.99) is working correctly
   - Add adaptive regularization based on condition number

2. **Convergence Issues** (if DDFM training fails):
   - Review learning rate scheduling (decay_learning_rate=True)
   - Check gradient clipping (if not present, add)
   - Verify batch normalization is working correctly
   - Add early stopping based on validation loss

3. **Theoretical Correctness** (verify against knowledgebase):
   - Verify EM algorithm implementation matches theoretical formulation
   - Check Kalman filter/smoother implementation
   - Verify DDFM encoder/decoder architecture matches Andreini et al. (2020)
   - Check tent kernel aggregation for mixed-frequency (Mariano & Murasawa, 2003)

**Action**: Only address if experiments reveal specific issues. Document any improvements needed.

### Experiment Configuration Updates
**Status**: ✅ Complete | **Priority**: N/A

**Current State**: 
- run_experiment.sh already has logic to skip completed experiments (is_experiment_complete function)
- Script checks for comparison_results.json and validates results before skipping
- No changes needed - script will only run missing experiments

**Verification**: Script checks outputs/comparisons/ for existing results and skips if complete.
