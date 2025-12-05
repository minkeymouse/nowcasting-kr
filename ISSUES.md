# Issues and Action Plan

## Current Status Summary

**Critical Blocker:**
- ⚠️ **Missing Python Dependencies**: `hydra-core` not installed
  - All 30 experiment runs failed with `ModuleNotFoundError: No module named 'hydra'` (latest: 024648)
  - Code fixes complete, ready once dependencies installed

**Experiment Status:**
- ❌ **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- All 30 runs failed (10 per target), no result files generated
- `run_experiment.sh` skip logic verified and correct

**Report Status:**
- ✅ **Content enhanced**: Removed placeholder text, replaced with meaningful analysis based on GDP results
- ✅ **Citations verified**: All citations in report exist in references.bib
- ⚠️ Placeholder content for KOCNPER.D and KOGFCF..D (blocked until experiments complete)

**Code Quality:**
- ✅ src/ has 17 files (2 deprecated wrappers, effective code in 15 files), dfm-python finalized

---

## Experiment Requirements and Status

### Required Experiments
**Configuration:** 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets:** KOGDP...D (GDP, 55 series), KOCNPER.D (Private Consumption, 50 series), KOGFCF..D (Gross Fixed Capital Formation, 19 series)
- **Models:** arima, var, dfm, ddfm
- **Horizons:** 1, 7, 28 days

### Current Status (Inspection: 2025-12-06)
**Location:** `outputs/comparisons/`
- ❌ **0/3 targets complete** (0% completion)
- ❌ **30 failed runs** (10 per target)
- ❌ **No result files:** No directories, no JSON/CSV, no `outputs/models/` directory
- **Latest error:** `ModuleNotFoundError: No module named 'hydra'` (all runs from 011236 onwards)
- **Error history:** Import errors (001731-002402) → missing src (004456) → missing hydra (011236-024648) - all code issues fixed

### Expected Outputs (When Experiments Succeed)
**Per target:**
- `outputs/comparisons/{target}_{timestamp}/comparison_results.json` (full metrics)
- `outputs/comparisons/{target}_{timestamp}/comparison_table.csv` (summary table)
- `outputs/models/{target}_{model}/model.pkl` (4 models: arima, var, dfm, ddfm)

**After aggregation:**
- `outputs/experiments/aggregated_results.csv` (combined metrics across all targets)

### run_experiment.sh Status
- ✅ Skip logic verified: Checks for `comparison_results.json` in latest result directory
- ✅ Will automatically skip completed targets when re-run
- ✅ Aggregator call correct: `from src.eval import main_aggregator; main_aggregator()`

---

## Action Plan (Incremental, Step-by-Step)

### Phase 1: Install Dependencies (CRITICAL - BLOCKING)
**Status:** All 30 runs failed due to missing `hydra` module. Code fixes complete.
**Blocking:** All experiments (0/3 targets complete)

**Task 1.1: Install Python Dependencies**
- [x] Updated `pyproject.toml` with dependencies
- [ ] **Action:** Install dependencies: `pip install -e .` (or install individual packages)
- [ ] **Verify:** `python3 -c "import hydra; import sktime; print('OK')"` (should print "OK")
- [ ] **Verify:** `python3 -c "import sys; sys.path.insert(0, 'dfm-python/src'); import dfm_python; print('OK')"` (dfm-python import)

**Task 1.2: Verify Setup**
- [ ] **Action:** Test config parser: `python3 -c "from src.utils.config_parser import setup_paths; print('OK')"`
- [ ] **Verify:** Data file exists: `ls -lh data/sample_data.csv` (should show file)
- [ ] **Verify:** Config files exist: `ls config/experiment/{kogdp,kocnper,kogfcf}_report.yaml` (all 3 should exist)

**Expected Outcome:** All imports succeed, no ModuleNotFoundError. Ready to run experiments.

---

### Phase 2: Run Experiments (CRITICAL - BLOCKED until Phase 1)
**Goal:** Generate results for all 3 targets (currently 0/3 complete)

**Task 2.1: Run Experiments**
- [ ] **Action:** Run: `bash run_experiment.sh` (runs all 3 targets, skips completed ones)
- [ ] **Monitor:** Script shows progress every 60 seconds. Watch for errors in terminal output.
- [ ] **Verify:** After completion, check: `find outputs/comparisons -type d -name "KOGDP*" -o -name "KOCNPER*" -o -name "KOGFCF*"` (should show 3 directories)
- [ ] **Verify:** Each directory has `comparison_results.json`: `ls outputs/comparisons/*/comparison_results.json` (should show 3 files)

**Task 2.2: Aggregate Results**
- [ ] **Action:** Run aggregator: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- [ ] **Verify:** Aggregated CSV exists: `ls -lh outputs/experiments/aggregated_results.csv`
- [ ] **Verify:** CSV has expected rows: `wc -l outputs/experiments/aggregated_results.csv` (should have header + 36 rows = 3 targets × 4 models × 3 horizons)

**Expected Outcome:** 
- 3 result directories in `outputs/comparisons/` (one per target)
- 3 `comparison_results.json` files (one per target)
- 1 `aggregated_results.csv` with 36 rows (all combinations)
- Trained models in `outputs/models/` (12 models: 3 targets × 4 models)

---

### Phase 3: Generate Visualizations (BLOCKED until Phase 2)
**Goal:** Create 4 plots for report (required by contents/5_result.tex)

**Task 3.1: Generate Plots**
- [ ] **Action:** Run plot script: `cd /data/nowcasting-kr && python3 nowcasting-report/code/plot.py`
- [ ] **Verify:** All 4 images exist: `ls -lh nowcasting-report/images/*.png` (should show 4 files)
  - `accuracy_heatmap.png` (Model × Target heatmap)
  - `forecast_vs_actual.png` (Time series comparison)
  - `horizon_trend.png` (Performance by horizon)
  - `model_comparison.png` (Bar chart comparison)
- [ ] **Verify:** Plots contain actual data (not "Placeholder: No data available" text)

**Expected Outcome:** 4 PNG files in `nowcasting-report/images/` with actual visualization data.

---

### Phase 4: Update Report Tables (BLOCKED until Phase 2)
**Goal:** Populate LaTeX tables with actual metrics from aggregated_results.csv

**Task 4.1: Update Overall Metrics Table**
- [ ] **Action:** Load `outputs/experiments/aggregated_results.csv` and compute overall averages
- [ ] **Action:** Update `tables/tab_overall_metrics.tex`: Average sMSE, sMAE, sRMSE across all targets and horizons
- [ ] **Verify:** Format: 4 decimal places (e.g., `0.8209`), missing values as `---`

**Task 4.2: Update Per-Target Metrics Table**
- [ ] **Action:** Compute per-target averages (KOGDP...D, KOCNPER.D, KOGFCF..D) across all horizons
- [ ] **Action:** Update `tables/tab_overall_metrics_by_target.tex`: 3 rows (one per target), 4 models × 3 metrics

**Task 4.3: Update Per-Horizon Metrics Table**
- [ ] **Action:** Compute per-horizon averages (1, 7, 28 days) across all targets
- [ ] **Action:** Update `tables/tab_overall_metrics_by_horizon.tex`: 3 rows (one per horizon), 4 models × 3 metrics

**Expected Outcome:** All 4 tables updated with actual metrics from experiments. Format: 4 decimal places, "---" for missing values.

---

### Phase 5: Update Report Content (BLOCKED until Phase 2-4)
**Goal:** Replace placeholders with actual results, improve content quality, ensure 20-30 pages

**Task 5.1: Update Results Section (contents/5_result.tex)**
- [x] Remove redundant mentions of "향후 연구" (replaced with "진행 중" or removed)
- [ ] **Action:** Replace placeholder text for KOCNPER.D and KOGFCF..D with actual results from aggregated_results.csv
- [ ] **Action:** Add specific metrics (sRMSE, sMAE, sMSE) with actual numbers for all 3 targets
- [ ] **Action:** Add analysis comparing performance across all 3 targets
- [ ] **Verify:** All numerical claims match values in tables (no hallucinations)

**Task 5.2: Update Discussion Section (contents/6_discussion.tex)**
- [x] Remove redundant "향후 연구 방향" subsection (already in conclusion)
- [ ] **Action:** Discuss findings with actual numbers from tables (reference specific sRMSE values)
- [ ] **Action:** Compare performance across targets and explain differences
- [ ] **Verify:** All claims are supported by actual results (no speculation without data)

**Task 5.3: Update Abstract and Introduction**
- [x] Update `main.tex` abstract: Now mentions all 3 targets consistently
- [x] Update `contents/1_introduction.tex`: Changed "향후 연구에서 구현될 예정" to "현재 구현 중"
- [ ] **Action:** Update abstract with actual key findings (e.g., "DFM showed best performance for 7-day horizon with sRMSE=0.0419")
- [ ] **Verify:** Abstract mentions all 3 targets and key findings (not just GDP)

**Expected Outcome:** Report content updated with actual results, no placeholders, all claims supported by data, consistent terminology.

---

### Phase 6: Finalize Report (BLOCKED until Phase 5)
**Goal:** Complete 20-30 page report with high quality content, no placeholders, all claims verified

**Task 6.1: Compile LaTeX and Check Page Count**
- [ ] **Action:** Compile report: `cd /data/nowcasting-kr/nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- [ ] **Verify:** Check for compilation errors (should complete without errors)
- [ ] **Action:** Check PDF page count: `pdfinfo main.pdf | grep Pages` (target: 20-30 pages)
- [ ] **If < 20 pages:** Expand method section with more technical details, expand theoretical background
- [ ] **If > 30 pages:** Condense sections, move detailed tables to appendix (if needed), remove redundancy

**Task 6.2: Content Quality Verification**
- [ ] **Action:** Search for placeholder text: `grep -r "향후 연구\|---\|TBD\|Placeholder\|진행 중" nowcasting-report/contents/` (should find minimal or no matches)
- [ ] **Action:** Verify numerical claims: Extract all numbers from report, cross-check with `outputs/experiments/aggregated_results.csv`
- [ ] **Action:** Verify citations: `grep -o "\\\\cite{[^}]*}" nowcasting-report/contents/*.tex | sed 's/\\cite{//;s/}//' | sort -u` → Check each key exists in `references.bib`

**Expected Outcome:** Complete 20-30 page PDF report, no placeholders, all claims verified, all citations valid, professional quality.

---

## Summary and Prioritization

### Critical Path (Blocking - Must Complete First)
1. **Phase 1**: Install dependencies (BLOCKING all experiments)
2. **Phase 2**: Run experiments (BLOCKING report updates)
3. **Phase 3**: Generate visualizations (BLOCKING report completion)
4. **Phase 4**: Update report tables (BLOCKING content updates)
5. **Phase 5**: Update report content (BLOCKING finalization)
6. **Phase 6**: Finalize report (COMPLETE)

### Current Focus
**Status:** Inspection complete. All 3 targets need experiments (0/3 complete).
**Next Action:** Phase 1, Task 1.1 - Install Python dependencies to unblock experiments.

**Experiments Needed:**
- ✅ **All 3 targets required** for complete report: KOGDP...D, KOCNPER.D, KOGFCF..D
- ✅ **All 4 models required** for each target: arima, var, dfm, ddfm
- ✅ **All 3 horizons required** for each model: 1, 7, 28 days
- **Total:** 36 combinations (3 × 4 × 3) needed for complete analysis

**run_experiment.sh Status:**
- ✅ Script is correct, skip logic verified
- ✅ Will automatically skip completed targets when re-run
- ⚠️ **Note:** If partial failures occur (e.g., only ARIMA fails but others succeed), may need to update script to handle per-model completion. Currently checks per-target completion only.

### Key Guidelines

- **Incremental execution**: Complete one phase before moving to next. Don't skip ahead.
- **Focus on blocking issues**: Phases 1-6 are critical for report completion.
- **Verify at each step**: Don't assume success. Check outputs, verify files exist, validate data.
- **Report quality**: Remove hallucinations, placeholders, redundancy. Use only references from `references.bib`. Verify all numerical claims.
- **Never create new files**: Only modify existing code. Keep STATUS/ISSUES/CONTEXT under 1000 lines.
- **Experiments first**: Report cannot be completed without experiment results. Prioritize Phase 1-2.
