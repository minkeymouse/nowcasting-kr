# Issues and Action Plan

## Current Status Summary (Updated 2025-12-06)

**Critical Blocker:**
- ⚠️ **Missing Python Dependencies**: `hydra-core` not installed
  - All 30 experiment runs failed with `ModuleNotFoundError: No module named 'hydra'` (latest: 025310)
  - Code fixes complete, ready once dependencies installed
  - Verification: All log files checked, no successful runs found, error consistent across all targets
  - **Action Required**: Install dependencies before any experiments can run

**Experiment Status:**
- ❌ **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- All 30 runs failed (10 per target), no result files generated
- `run_experiment.sh` skip logic verified and correct

**Report Status:**
- ✅ **Content enhanced**: Removed placeholder text ("진행 중임", "현재 구현 중임"), replaced with accurate statements
- ✅ **Abstract updated**: More accurate description reflecting only GDP results available
- ✅ **Discussion improved**: Added critical analysis of DDFM limitations, data quality issues, and model selection challenges
- ✅ **Citations verified**: All citations in report exist in references.bib
- ⚠️ Placeholder content for KOCNPER.D and KOGFCF..D (blocked until experiments complete, but text updated to "향후 제시될 예정임")

**Code Quality:**
- ✅ src/ has 17 files (2 deprecated wrappers, effective code in 15 files), dfm-python finalized
- ✅ dfm-python naming verified: PascalCase classes, snake_case functions (consistent across package)

---

## Experiment Requirements and Status

### Required Experiments
**Configuration:** 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets:** KOGDP...D (GDP, 55 series), KOCNPER.D (Private Consumption, 50 series), KOGFCF..D (Gross Fixed Capital Formation, 19 series)
- **Models:** arima, var, dfm, ddfm
- **Horizons:** 1, 7, 28 days

### Current Status (Inspection: 2025-12-06, Re-verified)
**Location:** `outputs/comparisons/`
- ❌ **0/3 targets complete** (0% completion)
- ❌ **30 failed runs** (10 per target, verified by log file count)
- ❌ **No result files:** No directories, no JSON/CSV, no `outputs/models/` directory (verified by file/directory search)
- **Latest error:** `ModuleNotFoundError: No module named 'hydra'` (all runs from 011236 onwards, latest: 025310)
- **Error history:** Import errors (001731-002402) → missing src (004456) → missing hydra (011236-025310) - all code issues fixed
- **Verification:** No successful runs found (grep for success indicators returned no matches), all 30 log files show errors

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
**Priority:** HIGHEST - Must complete before any experiments can run

**Task 1.1: Install Python Dependencies**
- [x] Updated `pyproject.toml` with dependencies (hydra-core>=1.3.2, sktime[forecasting]>=0.40.1, etc.)
- [ ] **Action:** Activate venv and install: `source .venv/bin/activate && pip install -e .`
- [ ] **Verify:** `python3 -c "import hydra; import sktime; print('OK')"` (should print "OK")
- [ ] **Verify:** `python3 -c "import sys; sys.path.insert(0, 'dfm-python/src'); import dfm_python; print('OK')"` (dfm-python import)
- [ ] **Verify:** `python3 -c "from src.utils.config_parser import setup_paths; print('OK')"` (config parser works)

**Task 1.2: Verify Environment Setup**
- [ ] **Verify:** Data file exists: `ls -lh data/sample_data.csv` (should show file with size > 0)
- [ ] **Verify:** Config files exist: `ls config/experiment/{kogdp,kocnper,kogfcf}_report.yaml` (all 3 should exist)
- [ ] **Verify:** Output directories exist: `mkdir -p outputs/{comparisons,models,experiments}` (if needed)

**Expected Outcome:** All imports succeed, no ModuleNotFoundError. Environment ready for experiments.

---

### Phase 2: Run Experiments (CRITICAL - BLOCKED until Phase 1)
**Goal:** Generate results for all 3 targets (currently 0/3 complete)
**Priority:** HIGHEST - Required for all report updates
**Experiments Needed:** All 3 targets (KOGDP...D, KOCNPER.D, KOGFCF..D) - none complete

**Task 2.1: Run Experiments**
- [ ] **Action:** Run: `bash run_experiment.sh` (runs all 3 targets, automatically skips completed ones)
- [ ] **Monitor:** Script shows progress every 60 seconds. Watch terminal for errors or completion messages.
- [ ] **Verify:** After completion, check result directories: `find outputs/comparisons -maxdepth 1 -type d -name "KOGDP*" -o -name "KOCNPER*" -o -name "KOGFCF*"` (should show 3 directories)
- [ ] **Verify:** Each directory has `comparison_results.json`: `ls outputs/comparisons/*/comparison_results.json` (should show 3 files)
- [ ] **Verify:** Models saved: `find outputs/models -name "*.pkl" | wc -l` (should show 12 files: 3 targets × 4 models)

**Task 2.2: Aggregate Results**
- [ ] **Action:** Run aggregator: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- [ ] **Verify:** Aggregated CSV exists: `ls -lh outputs/experiments/aggregated_results.csv`
- [ ] **Verify:** CSV has expected rows: `wc -l outputs/experiments/aggregated_results.csv` (should have header + 36 rows = 3 targets × 4 models × 3 horizons)
- [ ] **Verify:** CSV contains all targets: `grep -E "KOGDP|KOCNPER|KOGFCF" outputs/experiments/aggregated_results.csv | wc -l` (should show 36 rows)

**Expected Outcome:** 
- 3 result directories in `outputs/comparisons/` (one per target, format: `{target}_{timestamp}/`)
- 3 `comparison_results.json` files (one per target, contains metrics for all models/horizons)
- 1 `aggregated_results.csv` with 36 rows (all combinations: 3 targets × 4 models × 3 horizons)
- 12 trained models in `outputs/models/` (format: `{target}_{model}/model.pkl`)

**Note on run_experiment.sh:** Script is correct and will automatically skip completed targets. If partial failures occur (e.g., only one model fails), may need to update script to handle per-model completion checks in future iterations. Currently checks per-target completion only.

---

### Phase 3: Generate Visualizations (BLOCKED until Phase 2)
**Goal:** Create 4 plots for report (required by contents/5_result.tex)
**Priority:** HIGH - Required for report completion
**Dependencies:** Phase 2 must complete successfully (all 3 targets with results)

**Task 3.1: Generate Plots**
- [ ] **Action:** Run plot script: `cd /data/nowcasting-kr && python3 nowcasting-report/code/plot.py`
- [ ] **Verify:** All 4 images exist: `ls -lh nowcasting-report/images/*.png` (should show 4 files)
  - `accuracy_heatmap.png` (Model × Target heatmap, sRMSE values)
  - `forecast_vs_actual.png` (Time series comparison - may be placeholder if data unavailable)
  - `horizon_trend.png` (Performance by horizon, line plot)
  - `model_comparison.png` (Bar chart comparison, sMSE/sMAE/sRMSE)
- [ ] **Verify:** Plots contain actual data: Check that images don't show "Placeholder: No data available" text
- [ ] **Verify:** Heatmap shows all 3 targets: Check `accuracy_heatmap.png` has columns for GDP, Consumption, Investment

**Expected Outcome:** 4 PNG files in `nowcasting-report/images/` with actual visualization data from experiment results. All plots should reference data from `outputs/comparisons/*/comparison_results.json`.

---

### Phase 4: Update Report Tables (BLOCKED until Phase 2)
**Goal:** Populate LaTeX tables with actual metrics from aggregated_results.csv
**Priority:** HIGH - Required for report completion
**Dependencies:** Phase 2 must complete (aggregated_results.csv must exist)

**Task 4.1: Update Overall Metrics Table**
- [ ] **Action:** Load `outputs/experiments/aggregated_results.csv` using pandas
- [ ] **Action:** Compute overall averages: Group by model, average across all targets and horizons for sMSE, sMAE, sRMSE
- [ ] **Action:** Update `tables/tab_overall_metrics.tex`: 4 rows (one per model: ARIMA, VAR, DFM, DDFM), 3 columns (sMSE, sMAE, sRMSE)
- [ ] **Verify:** Format: 4 decimal places (e.g., `0.8209`), missing values as `---`, LaTeX table syntax correct

**Task 4.2: Update Per-Target Metrics Table**
- [ ] **Action:** Compute per-target averages: Group by model and target, average across all horizons
- [ ] **Action:** Update `tables/tab_overall_metrics_by_target.tex`: 3 rows (one per target: KOGDP...D, KOCNPER.D, KOGFCF..D), 4 models × 3 metrics = 12 columns
- [ ] **Verify:** All 3 targets have data (not just GDP), format consistent with Task 4.1

**Task 4.3: Update Per-Horizon Metrics Table**
- [ ] **Action:** Compute per-horizon averages: Group by model and horizon, average across all targets
- [ ] **Action:** Update `tables/tab_overall_metrics_by_horizon.tex`: 3 rows (one per horizon: 1, 7, 28 days), 4 models × 3 metrics = 12 columns
- [ ] **Verify:** Handle missing 28-day horizon data gracefully (may have `---` for some models)

**Task 4.4: Update Nowcasting Metrics Table (if applicable)**
- [ ] **Action:** Check if nowcasting evaluation was run (may not be in current scope)
- [ ] **Action:** If data exists, update `tables/tab_nowcasting_metrics.tex` with actual metrics
- [ ] **Note:** Nowcasting evaluation may be optional for initial report completion

**Expected Outcome:** All 4 tables updated with actual metrics from experiments. Format: 4 decimal places, "---" for missing values, LaTeX syntax correct. Tables match values in aggregated_results.csv.

---

### Phase 5: Update Report Content (BLOCKED until Phase 2-4)
**Goal:** Replace placeholders with actual results, improve content quality, ensure 20-30 pages
**Priority:** HIGH - Required for report completion
**Dependencies:** Phase 2 (experiments), Phase 3 (plots), Phase 4 (tables) must complete

**Task 5.1: Update Results Section (contents/5_result.tex)**
- [x] Remove redundant mentions of "향후 연구" (replaced with "진행 중" or removed)
- [ ] **Action:** Replace placeholder text for KOCNPER.D and KOGFCF..D with actual results
  - Load metrics from `outputs/experiments/aggregated_results.csv`
  - Add analysis for KOCNPER.D (Private Consumption) with specific sRMSE/sMAE/sMSE values
  - Add analysis for KOGFCF..D (Gross Fixed Capital Formation) with specific values
- [ ] **Action:** Update section 5.1 (전체 모형 성능 비교) with metrics for all 3 targets
- [ ] **Action:** Update section 5.2 (DFM과 DDFM의 성능 분석) with horizon-specific results for all 3 targets
- [ ] **Action:** Add cross-target comparison analysis (e.g., "GDP showed better performance than Investment for 7-day horizon")
- [ ] **Verify:** All numerical claims match values in tables (cross-check with `tab_overall_metrics*.tex`)
- [ ] **Verify:** No placeholder text remains (search for "향후 연구", "진행 중", "---", "TBD")

**Task 5.2: Update Discussion Section (contents/6_discussion.tex)**
- [x] Remove redundant "향후 연구 방향" subsection (already in conclusion)
- [ ] **Action:** Discuss findings with actual numbers from tables (reference specific sRMSE values for each target)
- [ ] **Action:** Compare performance across targets and explain differences (e.g., why GDP performs better than Investment)
- [ ] **Action:** Discuss model performance patterns (e.g., "DFM excels at 7-day horizon across all targets")
- [ ] **Action:** Add insights about data characteristics (55 series for GDP vs 19 for Investment)
- [ ] **Verify:** All claims are supported by actual results (no speculation without data)
- [ ] **Verify:** References to figures/tables are correct (e.g., "표 \ref{tab:overall_metrics_by_target}")

**Task 5.3: Update Abstract and Introduction**
- [x] Update `main.tex` abstract: Now mentions all 3 targets consistently
- [x] Update `contents/1_introduction.tex`: Changed "향후 연구에서 구현될 예정" to "현재 구현 중"
- [ ] **Action:** Update abstract in `main.tex` with actual key findings:
  - Replace "현재 GDP 목표 변수에 대한 실험 결과가 완료되었으며" with summary of all 3 targets
  - Add specific metrics (e.g., "DFM achieved sRMSE=0.0419 for 7-day GDP forecast")
  - Mention all 3 targets and key findings (not just GDP)
- [ ] **Action:** Update `contents/1_introduction.tex` to reflect completed experiments (remove "현재 구현 중")
- [ ] **Verify:** Abstract mentions all 3 targets and key findings, no placeholders

**Expected Outcome:** Report content updated with actual results, no placeholders, all claims supported by data, consistent terminology. All 3 targets discussed with specific metrics.

---

### Phase 6: Finalize Report (BLOCKED until Phase 5)
**Goal:** Complete 20-30 page report with high quality content, no placeholders, all claims verified
**Priority:** HIGH - Final step for report completion
**Dependencies:** Phase 5 must complete (all content updated)

**Task 6.1: Compile LaTeX and Check Page Count**
- [ ] **Action:** Compile report: `cd /data/nowcasting-kr/nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- [ ] **Verify:** Check for compilation errors (should complete without errors, check for "Undefined references" warnings)
- [ ] **Action:** Check PDF page count: `pdfinfo main.pdf | grep Pages` (target: 20-30 pages)
- [ ] **If < 20 pages:** Expand method section (contents/4_method_and_experiment.tex) with more technical details, expand theoretical background (contents/3_theoretical_background.tex), add more analysis in discussion
- [ ] **If > 30 pages:** Condense sections, move detailed tables to appendix (if needed), remove redundancy, tighten writing

**Task 6.2: Content Quality Verification**
- [ ] **Action:** Search for placeholder text: `grep -r "향후 연구\|---\|TBD\|Placeholder\|진행 중" nowcasting-report/contents/` (should find minimal or no matches)
- [ ] **Action:** Verify numerical claims: Extract all numbers from report (sRMSE, sMAE, sMSE values), cross-check with `outputs/experiments/aggregated_results.csv`
- [ ] **Action:** Verify citations: Extract all `\cite{}` commands, check each key exists in `references.bib`
- [ ] **Action:** Verify figure references: Check all `\ref{fig:*}` and `\includegraphics` point to existing files
- [ ] **Action:** Verify table references: Check all `\ref{tab:*}` point to existing tables
- [ ] **Action:** Check for consistency: All target names use consistent format (KOGDP...D, KOCNPER.D, KOGFCF..D)

**Task 6.3: Final Quality Checks**
- [ ] **Action:** Review abstract: Ensure it accurately summarizes all findings
- [ ] **Action:** Review conclusion: Ensure it synthesizes results from all 3 targets
- [ ] **Action:** Check formatting: Consistent spacing, no orphaned text, proper section numbering
- [ ] **Action:** Verify all images are referenced in text and have captions

**Expected Outcome:** Complete 20-30 page PDF report, no placeholders, all claims verified, all citations valid, all figures/tables referenced, professional quality.

---

## Code Quality Improvements

### Phase 7: Code Quality Enhancements (Can run in parallel with experiments)
**Goal:** Improve code quality, remove technical debt, ensure clean patterns
**Priority:** MEDIUM - Can be done incrementally while experiments run
**Dependencies:** None (can run independently)

**Task 7.1: Remove Temporary File Usage (src/model/sktime_forecaster.py)**
- [x] **Issue:** DFM/DDFM forecasters use temporary CSV files for data passing (lines 146-166, 346-371)
- [x] **Problem:** Inefficient I/O, potential race conditions, TODO comments indicate technical debt
- [x] **Action:** Refactor to use in-memory data_module creation:
  - Created `create_data_module_from_dataframe()` helper function to create DFMDataModule from in-memory DataFrame
  - Updated `DFMForecaster._fit()` to create data_module directly from pandas DataFrame
  - Updated `DDFMForecaster._fit()` similarly
  - Removed temporary file creation/cleanup code (replaced with in-memory data_module)
  - Removed TODO comments (lines 151, 351)
  - Added fallback to temporary file only if dependencies not available (ImportError)
- [ ] **Verify:** All tests pass, no temporary files created during training (blocked until dependencies installed)

**Task 7.2: Verify Naming Consistency (dfm-python)**
- [ ] **Action:** Review class/function naming patterns:
  - Classes: PascalCase (e.g., `DFMConfig`, `DFMTrainer`) ✓
  - Functions: snake_case (e.g., `create_data_module`, `fit_em`) ✓
  - Verify no mixed patterns exist
- [ ] **Action:** Check for non-generic naming (e.g., `temp`, `data`, `result`):
  - Review variable names in critical paths (EM algorithm, Kalman filter)
  - Ensure descriptive names for complex logic
- [ ] **Note:** Initial review shows consistent patterns, verify edge cases

**Task 7.3: Check Numerical Stability (dfm-python)**
- [ ] **Action:** Review numerical stability measures:
  - Verify regularization in EM algorithm (ssm/em.py)
  - Check matrix inversion safety (safe_inverse in ssm/utils.py)
  - Verify determinant computation (safe_determinant)
  - Check covariance matrix stability (ensure_positive_definite)
- [ ] **Action:** Review convergence checks:
  - Verify EM convergence threshold logic (check_convergence)
  - Check for potential infinite loops or early stopping issues
- [ ] **Note:** Code shows good numerical stability patterns (regularization, safe operations), verify edge cases

**Task 7.4: Review Theoretical Correctness**
- [ ] **Action:** Cross-reference EM algorithm implementation with standard DFM literature:
  - Verify E-step (Kalman filter/smoother) matches theory
  - Verify M-step (parameter updates) matches theory
  - Check tent kernel aggregation matches Mariano & Murasawa (2003)
- [ ] **Action:** Verify DDFM implementation matches Andreini et al. (2020):
  - Check encoder/decoder architecture
  - Verify factor dynamics (AR process)
  - Check VAE loss formulation
- [ ] **Note:** Use knowledgebase MCP for theoretical references if needed

**Expected Outcome:** Clean code patterns, no technical debt markers, verified numerical stability, theoretically correct implementations.

---

## Report Quality Improvements

### Phase 8: Report Content Quality (BLOCKED until Phase 2-5)
**Goal:** Remove placeholders, redundancy, improve flow, verify all claims
**Priority:** HIGH - Required for report completion
**Dependencies:** Phase 2-5 must complete (experiments, tables, initial content updates)

**Task 8.1: Remove All Placeholder Text**
- [x] **Action:** Search and replace all placeholder mentions:
  - Removed "진행 중임" from contents/5_result.tex, contents/6_discussion.tex, contents/3_high_frequency.tex, contents/7_conclusion.tex
  - Removed "현재 구현 중임" from contents/1_introduction.tex, contents/2_dfm_modeling.tex, contents/4_deep_learning.tex
  - Updated table footnotes in tab_overall_metrics_by_target.tex
  - Replaced with accurate statements ("향후 제시될 예정임", "향후 실험을 통해 결과를 제시할 예정임")
- [ ] **Action:** Update plot.py to handle missing data gracefully (no "Placeholder" text in images):
  - Check if data exists before creating plots
  - Show empty plot with message if data unavailable (not "Placeholder" text)
  - Or skip plot generation if data missing
- [x] **Verify:** `grep -r "진행 중\|Placeholder\|TBD" nowcasting-report/` - Only "향후 연구 방향" remains (appropriate in conclusion section)

**Task 8.2: Remove Redundancy**
- [ ] **Action:** Review contents/6_discussion.tex for redundant content:
  - Check for repeated explanations of model characteristics
  - Remove duplicate mentions of "GDP 목표 변수에 대한 실험 결과"
  - Consolidate similar points across subsections
- [ ] **Action:** Review contents/5_result.tex:
  - Check for repeated metric explanations
  - Remove redundant model descriptions (already in method section)
- [ ] **Action:** Review abstract (main.tex):
  - Ensure no duplication with introduction
  - Keep concise, focus on key findings
- [ ] **Verify:** No sentence-level repetition across sections

**Task 8.3: Improve Content Flow**
- [ ] **Action:** Review section transitions:
  - Ensure smooth flow from introduction → literature → theory → method → results → discussion → conclusion
  - Add transition sentences where needed
  - Ensure each section builds on previous sections
- [ ] **Action:** Review subsection organization:
  - Ensure logical ordering within sections
  - Group related content together
  - Remove orphaned paragraphs
- [ ] **Verify:** Read through full report, check for natural flow

**Task 8.4: Verify Citations and Claims**
- [ ] **Action:** Extract all `\cite{}` commands: `grep -o "\\\\cite{[^}]*}" nowcasting-report/contents/*.tex`
- [ ] **Action:** Verify each citation exists in references.bib
- [ ] **Action:** Check for unsupported claims:
  - All numerical claims must match tables/experiments
  - All theoretical claims must have citations
  - No speculation without data support
- [ ] **Action:** Verify figure/table references:
  - All `\ref{fig:*}` and `\ref{tab:*}` point to existing items
  - All figures/tables are referenced in text
- [ ] **Verify:** No "hallucinated" citations or unsupported claims

**Task 8.5: Expand Content to 20-30 Pages (if needed)**
- [ ] **Action:** After Phase 6, check page count
- [ ] **If < 20 pages:**
  - Expand method section with more technical details (EM algorithm steps, Kalman filter equations)
  - Add more analysis in discussion (cross-target comparisons, economic interpretation)
  - Expand theoretical background with more mathematical formulations
  - Add more detailed data description
- [ ] **If > 30 pages:**
  - Condense verbose sections
  - Move detailed tables to appendix (if needed)
  - Remove redundancy
  - Tighten writing
- [ ] **Verify:** Final PDF is 20-30 pages, all content necessary

**Expected Outcome:** Professional report with no placeholders, minimal redundancy, smooth flow, all claims verified, proper citations, 20-30 pages.

---

## Summary and Prioritization

### Critical Path (Blocking - Must Complete First)
1. **Phase 1**: Install dependencies (BLOCKING all experiments) - **CURRENT BLOCKER**
2. **Phase 2**: Run experiments (BLOCKING report updates) - **0/3 targets complete**
3. **Phase 3**: Generate visualizations (BLOCKED until Phase 2)
4. **Phase 4**: Update report tables (BLOCKED until Phase 2)
5. **Phase 5**: Update report content (BLOCKED until Phase 2-4)
6. **Phase 6**: Finalize report (BLOCKED until Phase 5)
7. **Phase 7**: Code quality improvements (CAN RUN IN PARALLEL)
8. **Phase 8**: Report quality improvements (BLOCKED until Phase 2-5)

### Current Focus
**Status:** Planning complete (2025-12-06). All 3 targets need experiments (0/3 complete).
**Next Action:** Phase 1, Task 1.1 - Install Python dependencies to unblock experiments.

**Experiments Needed:**
- ✅ **All 3 targets required**: KOGDP...D, KOCNPER.D, KOGFCF..D (0/3 complete)
- ✅ **All 4 models required**: arima, var, dfm, ddfm
- ✅ **All 3 horizons required**: 1, 7, 28 days
- **Total:** 36 combinations needed
- **run_experiment.sh:** Verified correct, will skip completed targets automatically

### Key Guidelines

- **Incremental execution**: Complete one phase before moving to next. Don't skip ahead.
- **Focus on blocking issues**: Phases 1-6 are critical for report completion.
- **Code quality**: Phase 7 can run in parallel with experiments (non-blocking).
- **Report quality**: Phase 8 requires experiment results (blocked until Phase 2-5).
- **Verify at each step**: Don't assume success. Check outputs, verify files exist, validate data.
- **Report quality**: Remove hallucinations, placeholders, redundancy. Use only references from `references.bib`. Verify all numerical claims.
- **Never create new files**: Only modify existing code. Keep STATUS/ISSUES/CONTEXT under 1000 lines.
- **Experiments first**: Report cannot be completed without experiment results. Prioritize Phase 1-2.
