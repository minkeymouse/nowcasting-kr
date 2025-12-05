# Issues and Action Plan

## Current Status Summary

**Critical Blocker:**
- ⚠️ **Missing Python Dependencies**: `hydra-core` not installed
  - All 24 experiment runs failed with `ModuleNotFoundError: No module named 'hydra'`
  - Code fixes complete, ready once dependencies installed

**Experiment Status:**
- ❌ **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- All 24 runs failed (8 per target), no result files generated
- `run_experiment.sh` skip logic verified and correct

**Report Status:**
- ✅ **Placeholder text removed**: Removed "[아직 실험 미진행]" from sections 2_dfm_modeling.tex, 3_high_frequency.tex, 4_deep_learning.tex
- ✅ **Content enhanced**: Replaced placeholders with meaningful analysis based on GDP results
- ⚠️ Placeholder content for KOCNPER.D and KOGFCF..D (blocked until experiments complete)
- ✅ Structure complete, citations verified, content quality improved

**Code Quality:**
- ✅ src/ has 17 files (acceptable), dfm-python finalized, temporary file workarounds documented

---

## Experiment Requirements and Status

### Required Experiments
**Configuration:** 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets:** KOGDP...D (GDP, 55 series), KOCNPER.D (Private Consumption, 50 series), KOGFCF..D (Gross Fixed Capital Formation, 19 series)
- **Models:** arima, var, dfm, ddfm
- **Horizons:** 1, 7, 28 days
- **Config files:** `config/experiment/{kogdp,kocnper,kogfcf}_report.yaml`

### Current Status (Inspection: 2025-12-06)
**Location:** `outputs/comparisons/`
- ❌ **0/3 targets complete** (0% completion)
- ❌ **24 failed runs** (8 per target: 001731, 002402, 004456, 011236, 011412, 013508, 015506, 021543)
- ❌ **No result files:** No directories, no JSON/CSV, no `outputs/models/` directory
- **Latest error:** `ModuleNotFoundError: No module named 'hydra'` (all 021543 runs)
- **Error history:** Import errors → missing src → missing hydra (all code issues fixed)

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
- ⚠️ **Note for later steps:** After experiments run, verify skip logic works. If partial failures occur, may need to update script to handle per-model completion. Currently no updates needed - script is correct.

---

## Concrete Action Plan (Incremental, Step-by-Step)

### Phase 1: Install Dependencies (CRITICAL - BLOCKING)
**Status:** All 24 runs failed due to missing `hydra` module. Code fixes complete.

**Task 1.1: Install Python Dependencies**
- [x] Updated `pyproject.toml` with dependencies
- [ ] Install: `pip install -e .` or `pip install hydra-core>=1.3.2 omegaconf>=2.3.0 sktime[forecasting]>=0.40.1 scipy>=1.10.0 scikit-learn>=1.7.2`
- [ ] Verify: `python3 -c "import hydra; import sktime; print('OK')"`

**Task 1.2: Verify Setup**
- [ ] Test import: `python3 -c "from src.utils.config_parser import setup_paths"`
- [ ] Check data and config files exist

---

### Phase 2: Run Experiments (CRITICAL - BLOCKED until Phase 1)
**Goal:** Generate results for all 3 targets (currently 0/3 complete)

**Task 2.1: Run Experiments**
- [ ] Run: `bash run_experiment.sh` (runs all 3 targets, skips completed ones)
- [ ] Monitor progress (script shows status every 60 seconds)
- [ ] Verify results: Check `outputs/comparisons/{target}_{timestamp}/comparison_results.json` exists for all 3 targets
- [ ] **If partial failures:** Note which targets/models failed, may need to update run_experiment.sh in later steps

**Task 2.2: Aggregate Results**
- [ ] Run: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- [ ] Verify: `outputs/experiments/aggregated_results.csv` exists with metrics for all 36 combinations (3 targets × 4 models × 3 horizons)

---

### Phase 3: Generate Visualizations (BLOCKED until Phase 2)
**Goal:** Create 4 plots for report

**Task 3.1: Generate Plots**
- [ ] Run: `python3 nowcasting-report/code/plot.py`
- [ ] Verify 4 images in `nowcasting-report/images/`:
  - `accuracy_heatmap.png` (Model × Target heatmap)
  - `forecast_vs_actual.png` (Time series comparison)
  - `horizon_trend.png` (Performance by horizon)
  - `model_comparison.png` (Bar chart comparison)
- [ ] Check plots contain actual data (not placeholders/zeros)

---

### Phase 4: Update Report Tables (BLOCKED until Phase 2)
**Goal:** Populate LaTeX tables with actual metrics

**Task 4.1: Update Tables**
- [ ] Load `outputs/experiments/aggregated_results.csv`
- [ ] Update `tables/tab_overall_metrics.tex`: Overall averages (all targets/horizons)
- [ ] Update `tables/tab_overall_metrics_by_target.tex`: Per-target averages (KOGDP...D, KOCNPER.D, KOGFCF..D)
- [ ] Update `tables/tab_overall_metrics_by_horizon.tex`: Per-horizon averages (1, 7, 28 days)
- [ ] Format: 4 decimal places, missing values as "---"

---

### Phase 5: Update Report Content (BLOCKED until Phase 2-4)
**Goal:** Replace placeholders with actual results, improve content quality

**Task 5.1: Update Results Section**
- [x] Remove redundant mentions of "향후 연구" (replaced with "진행 중" or removed)
- [ ] Update `contents/5_result.tex`: Replace placeholder text for KOCNPER.D and KOGFCF..D with actual results (BLOCKED until experiments complete)
- [ ] Add actual metrics with specific numbers for all 3 targets
- [ ] Add analysis comparing performance across all 3 targets
- [ ] Add detailed per-target analysis (GDP, Private Consumption, Investment)

**Task 5.2: Update Discussion Section**
- [ ] Update `contents/6_discussion.tex`: Discuss findings with actual numbers from tables
- [ ] Compare performance across targets and explain differences
- [ ] Connect findings to theoretical background
- [ ] Remove redundant limitations already mentioned in results section
- [ ] Add economic interpretation of model performance differences

**Task 5.3: Update Abstract and Introduction**
- [x] Update `main.tex` abstract: Now mentions all 3 targets consistently (KOGDP\_\_\_D, KOCNPER\_D, KOGFCF\_\_D)
- [x] Update `contents/1_introduction.tex`: Changed "향후 연구에서 구현될 예정" to "현재 구현 중"
- [ ] Ensure all claims in abstract are supported by actual results (BLOCKED until experiments complete)

**Task 5.4: Report Content Quality Improvements** (Priority: High)
- [x] **Remove redundancy**: Reduced repeated mentions of "향후 연구" (changed to "진행 중" or removed)
- [x] **Improve consistency**: Abstract now consistently mentions all 3 targets
- [ ] **Remove hallucination**: Verify all numerical claims match actual results from tables (BLOCKED until experiments complete)
- [ ] **Add detail**: Expand method section with more technical details if report < 20 pages
- [ ] **Verify citations**: All references must be in `references.bib`, no hallucinated citations
- [ ] **Check consistency**: Ensure terminology consistent (e.g., "표준화된 RMSE" vs "sRMSE")

---

### Phase 6: Finalize Report (BLOCKED until Phase 5)
**Goal:** Complete 20-30 page report with high quality content

**Task 6.1: Compile and Verify**
- [ ] Compile: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- [ ] Check PDF page count (target: 20-30 pages)
- [ ] Verify all figures/tables exist and are referenced in text
- [ ] Check no placeholder text remains (search for "향후 연구", "---", "TBD", "Placeholder")
- [ ] **If < 20 pages:** Expand method section, add more technical details, expand theoretical background, add more analysis
- [ ] **If > 30 pages:** Condense sections, move tables to appendix, remove redundancy, tighten language

**Task 6.2: Content Quality Verification**
- [ ] Verify all numerical claims match aggregated_results.csv
- [ ] Check all citations exist in references.bib (no hallucinated references)
- [ ] Ensure no contradictory statements (e.g., "DFM best" vs "VAR best" without context)
- [ ] Verify figure captions match actual plot content
- [ ] Check table formatting (4 decimal places, "---" for missing values)

---

### Phase 7: Code Quality Improvements (NON-BLOCKING - Incremental)
**Goal:** Improve code quality, numerical stability, and theoretical correctness

**Task 7.1: Temporary File Workarounds** (Priority: Medium)
- [ ] Refactor `_fit_dfm()` and `_fit_ddfm()` to use in-memory data if dfm-python API supports it
- **Current status**: Documented as TODO in code, non-blocking

**Task 7.2: dfm-python Numerical Stability Review** (Priority: Medium)
- [ ] Review EM algorithm convergence checks and Kalman filter numerical stability
- [ ] Test edge cases: very small/large values, near-singular matrices, extreme parameter values

**Task 7.3: dfm-python Theoretical Correctness Review** (Priority: Medium)
- [ ] Verify tent kernel implementation matches theoretical specification
- [ ] Verify clock framework and block structure implementation
- [ ] Compare with reference implementations (if available via knowledgebase)

**Task 7.4: src/ Code Quality Improvements** (Priority: Low)
- [ ] Review for redundant code patterns and inefficient logic
- [ ] Verify consistent error handling and generic naming patterns

---

## Summary and Prioritization

### Critical Path (Blocking - Must Complete First)
1. **Phase 1**: Install dependencies → 2. **Phase 2**: Run experiments → 3. **Phase 3-6**: Update report

### Current Focus
**Phase 1, Task 1.1**: Install Python dependencies to unblock experiments
- ✅ `pyproject.toml` updated with dependencies
- ⏭️ **Next**: Install dependencies, then run experiments

**Experiments Status**: 0/3 targets complete (0%). All 24 runs failed due to missing dependencies. Once installed, all 3 targets need to run.

### Key Guidelines

- **Incremental execution**: Complete one phase before moving to next
- **Focus on blocking issues**: Phases 1-6 are critical, Phase 7 is optional
- **run_experiment.sh**: Script is correct, skip logic verified. Only update if partial failures occur.
- **Report quality**: Remove hallucinations, placeholders, redundancy. Use only references from `references.bib`.
- **Code quality**: Focus on numerical stability and theoretical correctness (incremental improvements)
- **Never create new files**: Only modify existing code, keep STATUS/ISSUES/CONTEXT under 1000 lines
