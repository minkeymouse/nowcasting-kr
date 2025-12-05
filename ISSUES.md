# Issues and Action Plan

## Current Status (Iteration Summary - 2025-01-XX)

### Critical Blocker
- ⚠️ **Experiments Not Run**: Phase 2 (Run Experiments) has not been executed
  - **Status**: 0/3 targets complete (KOGDP...D, KOCNPER.D, KOGFCF..D)
  - **Previous Issues**: All resolved (relative imports, missing src, missing dependencies)
  - **Current State**: Code ready, dependencies marked as installed (Phase 1 complete)
  - **Action**: Run `bash run_experiment.sh` to execute Phase 2

### Experiment Status
- ❌ **0/3 targets complete** (experiments not run yet)
- ❌ **No result files**: Experiments have not been executed
- ❌ **No trained models**: Experiments have not been executed
- ❌ **No aggregated results**: Experiments have not been executed
- ✅ **Script verified**: `run_experiment.sh` has correct skip logic
- ✅ **Code ready**: All import/path issues resolved, ready for execution

### Report Status
- ✅ Structure complete, hallucinations removed, citations verified
- ✅ Redundancy removed: Improved flow, removed repeated statements
- ⚠️ All content has placeholders (blocked until experiments complete)
- ⚠️ Tables show "---", plots are placeholders

### Code Status
- ✅ src/ has 17 files (2 deprecated wrappers, effective code in 15 files)
- ✅ dfm-python finalized with consistent naming

---

## Experiments Required for Report

**Configuration:** 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **Models**: arima, var, dfm, ddfm
- **Horizons**: 1, 7, 28 days

**Status**: 0/3 targets complete (all failed due to missing dependencies)

**Experiments Needed**: All 3 targets must be run
- `run_experiment.sh` will auto-skip completed targets (checks for `comparison_results.json`)
- Currently all 3 targets need to run after Phase 1

**Expected Outputs**:
- 3 result directories: `outputs/comparisons/{target}_{timestamp}/`
- 3 JSON files: `comparison_results.json` (full metrics per model/horizon)
- 3 CSV files: `comparison_table.csv` (summary tables)
- 12 model files: `outputs/models/{target}_{model}/model.pkl`
- 1 aggregated CSV: `outputs/experiments/aggregated_results.csv` (36 rows)

**Report Requirements**:
- 4 plots: `accuracy_heatmap.png`, `forecast_vs_actual.png`, `horizon_trend.png`, `model_comparison.png`
- 4 tables: `tab_overall_metrics.tex`, `tab_overall_metrics_by_target.tex`, `tab_overall_metrics_by_horizon.tex`, `tab_nowcasting_metrics.tex`
- Results section: Actual metrics for all 3 targets (currently placeholders)
- Discussion section: Actual findings and cross-target comparisons

---

## Action Plan (Incremental Tasks - Priority Order)

### Phase 1: Install Dependencies [CRITICAL - BLOCKS ALL]
**Status**: ✅ COMPLETE (marked in previous iteration) | **Time**: Completed | **Unblocks**: Phase 2

**Tasks**:
1. ✅ Verified venv: `.venv` exists
2. ✅ Activated venv: `source .venv/bin/activate`
3. ✅ Installed core: `hydra-core>=1.3.2`, `omegaconf>=2.3.0`
4. ✅ Installed sktime: `sktime[forecasting]>=0.40.1`
5. ✅ Verified imports: All dependencies import successfully
6. ✅ Verified data/config: 1 CSV + 3 YAML configs exist
7. ✅ Created outputs: All output directories created

**Success**: ✅ Dependencies marked as installed - Phase 2 ready to execute
**Note**: If experiments fail with import errors, re-verify Phase 1 completion

### Phase 2: Run Experiments [READY: Phase 1 complete]
**Status**: 0/3 targets | **Time**: Several hours | **Blocks**: Phases 3-6

**Tasks**:
1. Verify Phase 1 complete (dependencies, files, directories)
2. Run script: `bash run_experiment.sh` (auto-skips completed targets, runs 3 targets in parallel)
3. Verify results: `find outputs/comparisons -name "comparison_results.json" | wc -l` (expect 3)
4. Verify models: `find outputs/models -name "model.pkl" | wc -l` (expect 12)
5. Aggregate: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
6. Verify aggregated: `wc -l outputs/experiments/aggregated_results.csv` (expect 37 lines: 1 header + 36 rows)

**Success**: 3 result dirs, 3 JSON files, 3 CSV files, 12 models, 1 aggregated CSV (36 rows)

### Phase 3: Generate Visualizations [BLOCKED: Phase 2]
**Status**: Waiting | **Time**: 5-10 min | **Blocks**: Phase 4

**Tasks**:
1. Verify inputs: `outputs/experiments/aggregated_results.csv` and `outputs/comparisons/*/comparison_results.json` exist
2. Generate plots: `python3 nowcasting-report/code/plot.py`
3. Verify 4 PNG files exist in `nowcasting-report/images/`:
   - `accuracy_heatmap.png`, `forecast_vs_actual.png`, `horizon_trend.png`, `model_comparison.png`
4. Visual check: Images contain actual data (not placeholders), labels readable

**Success**: 4 PNG files with actual data, no placeholders

### Phase 4: Update Report Tables [BLOCKED: Phase 2]
**Status**: Waiting | **Time**: 30-60 min | **Blocks**: Phase 5

**Tasks**:
1. Load aggregated results: `outputs/experiments/aggregated_results.csv` (verify 36 rows)
2. Update `tab_overall_metrics.tex`: 4 models × 3 metrics (avg across all targets/horizons)
3. Update `tab_overall_metrics_by_target.tex`: 3 targets × 4 models × 3 metrics
4. Update `tab_overall_metrics_by_horizon.tex`: 3 horizons × 4 models × 3 metrics
5. Update `tab_nowcasting_metrics.tex`: If nowcasting evaluation was run (may not exist)
6. Verify LaTeX: All tables valid, format consistent (4 decimal places, `---` for missing)

**Success**: All tables updated with actual metrics, valid LaTeX, consistent format

### Phase 5: Update Report Content [BLOCKED: Phases 2-4]
**Status**: Waiting | **Time**: 1-2 hours | **Blocks**: Phase 6

**Tasks**:
1. Review aggregated results: Identify best models per target/horizon, patterns
2. Update `5_result.tex`: Replace placeholders for KOCNPER.D, KOGFCF..D, verify GDP results
3. Update `6_discussion.tex`: Add cross-target comparisons, horizon patterns, actual insights
4. Update `main.tex` abstract: Summarize all 3 targets, key findings, best models
5. Update `1_introduction.tex`: Reflect completed experiments (all 3 targets)
6. Verify numbers: All report numbers match tables and aggregated CSV
7. Verify citations: All `\cite{}` exist in `references.bib`, no hallucinations

**Success**: Content updated with actual results, no placeholders, all claims supported, citations valid

### Phase 6: Finalize Report [BLOCKED: Phase 5]
**Status**: Waiting | **Time**: 30-60 min | **Blocks**: None

**Tasks**:
1. Compile LaTeX (pass 1): `cd nowcasting-report && pdflatex main.tex`
2. Run BibTeX: `cd nowcasting-report && bibtex main`
3. Compile LaTeX (pass 2): `cd nowcasting-report && pdflatex main.tex`
4. Compile LaTeX (pass 3): `cd nowcasting-report && pdflatex main.tex`
5. Verify page count: PDF has 20-30 pages
6. Verify compilation: No errors, no missing references, no "??" markers
7. Verify placeholders: Search for "---", "TBD", "placeholder" (should be none)
8. Verify numbers: Cross-check report vs tables vs aggregated CSV
9. Verify references: All figures/tables referenced exist, sequential numbering
10. Verify formatting: Consistent font/spacing/margins, professional appearance

**Success**: Complete 20-30 page PDF, no placeholders, all claims verified, professional quality

---

## Execution Notes

**run_experiment.sh**:
- Auto-skips completed targets (checks `comparison_results.json`)
- Runs 3 targets in parallel (max 5 processes)
- Currently 0/3 complete → all 3 will run after Phase 1
- Can re-run safely (will skip completed targets)

**Strategy**:
- Complete Phase 1 fully before Phase 2
- Phase 2 may take hours → monitor via `outputs/comparisons/*.log`
- Phases 3-6 proceed sequentially after Phase 2
- Verify each step before proceeding

**If Experiments Fail**:
- Check logs: `outputs/comparisons/*.log`
- Verify Phase 1 complete (dependencies, files)
- Fix issues before proceeding

---

## Resolved Issues

### Code Issues (RESOLVED)
- ✅ **Relative import errors**: Fixed import paths in src/ module
- ✅ **Missing src module**: Fixed path setup in `utils/path_setup.py`
- ✅ **Import structure**: All code import issues resolved, ready for execution

### Report Issues (RESOLVED)
- ✅ **Hallucination check**: All hallucinations removed, placeholders added
- ✅ **Citations**: All report citations verified in `references.bib`
- ✅ **Structure**: Complete LaTeX framework ready for results

## Code Quality Improvements (Priority: MEDIUM - After Report Complete)

### Task 1: Verify Temporary File Fallback (src/model/sktime_forecaster.py)
**Status**: Pending | **Priority**: Medium
- **Issue**: Lines 155-169, 363-382 have fallback to temporary files when ImportError occurs
- **Action**: After dependencies installed (Phase 1), verify if fallback is ever triggered
- **If triggered**: Investigate why in-memory data_module fails, fix root cause
- **If not triggered**: Remove fallback code to simplify (after confirming stability)
- **Note**: Currently documented as fallback, non-blocking for report

### Task 2: Resolve TODO in infer.py (src/infer.py line 397)
**Status**: Pending | **Priority**: Medium
- **Issue**: Nowcasting evaluation not fully implemented (returns 'not_implemented')
- **Action**: Implement full nowcasting evaluation or document as known limitation
- **Requirements**: Load trained models, run simulate_nowcasting_evaluation(), calculate metrics
- **Note**: Non-blocking for report (nowcasting metrics optional), but should be resolved

### Task 3: Numerical Stability Review (dfm-python/src/dfm_python/ssm/)
**Status**: Review needed | **Priority**: Low
- **Current**: Regularization already implemented (1e-6, 1e-5, 1e-8) in EM/Kalman filter
- **Action**: Review if regularization values are appropriate, check for edge cases
- **Files**: `em.py`, `kalman.py`, `utils.py` (ensure_positive_definite, ensure_symmetric)
- **Note**: Code already has good numerical stability measures, review for optimization

### Task 4: Theoretical Correctness Verification (dfm-python/)
**Status**: Review needed | **Priority**: Low
- **Action**: Verify DFM implementation matches Stock & Watson (2002) methodology
- **Action**: Verify DDFM implementation matches Andreini et al. (2020) methodology
- **Files**: `models/dfm.py`, `models/ddfm.py`, `ssm/em.py`, `encoder/vae.py`
- **Note**: Compare with literature, ensure clock-based mixed-frequency handling is correct

### Task 5: Naming Consistency Review (dfm-python/)
**Status**: ✅ COMPLETE (2025-01-XX) | **Priority**: Low
- **Current**: PascalCase classes, snake_case functions (consistent across all modules)
- **Review**: Verified encoder/, ssm/, trainer/, models/ - all follow consistent naming
- **Files**: All files in `dfm-python/src/dfm_python/` reviewed
- **Result**: Naming is consistent, no changes needed

### Task 6: Code Efficiency (If Performance Issues)
**Status**: On-demand | **Priority**: Low
- **Action**: Profile if experiments are slow, optimize bottlenecks
- **Focus**: EM algorithm iterations, Kalman filter operations, data loading
- **Note**: Only if performance becomes an issue

## Report Quality Improvements (Priority: HIGH - After Experiments)

### Task 7: Replace Placeholders with Actual Results (Phases 4-5)
**Status**: Blocked until Phase 2 | **Priority**: High
- **Action**: Update all tables, plots, and text with actual experiment results
- **Files**: `contents/5_result.tex`, `contents/6_discussion.tex`, `tables/*.tex`, `main.tex`
- **Note**: Covered in Phases 4-5 of action plan above

### Task 8: Improve Report Flow and Remove Redundancy
**Status**: ✅ COMPLETE (2025-01-XX) | **Priority**: Medium
- **Action**: Review report for redundant statements, improve transitions between sections
- **Files**: All `contents/*.tex` files
- **Focus**: Remove repeated "실험 완료 후 제시할 예정" statements, improve narrative flow
- **Completed**: Redundant statements removed, flow improved, placeholders consolidated

### Task 9: Verify No Hallucinations
**Status**: Ongoing | **Priority**: High
- **Action**: Cross-check all numerical claims against aggregated_results.csv
- **Action**: Verify all citations exist in references.bib
- **Action**: Ensure no made-up numbers or unsupported claims
- **Note**: Critical - verify after results are added

---

## Experiment Status Check (2025-01-XX)

**Completed Experiments**: 0/3 targets
- Checked: `find outputs/comparisons -name "comparison_results.json"` → 0 files
- **All 3 targets need to run**: KOGDP...D, KOCNPER.D, KOGFCF..D
- **run_experiment.sh**: Correctly configured to skip completed targets (checks for comparison_results.json)
- **Status**: Experiments have not been run yet - ready to execute Phase 2
- **Action**: Run `bash run_experiment.sh` to execute all 3 targets

## Key Guidelines
- **Incremental**: Complete one phase before next
- **Verify**: Check outputs at each step
- **No hallucinations**: Use only `references.bib`, verify all claims
- **No new files**: Only modify existing code
- **Experiments first**: Report blocked until Phase 2 completes
- **Manageable tasks**: Keep ISSUES.md under 1000 lines, prioritize critical items
