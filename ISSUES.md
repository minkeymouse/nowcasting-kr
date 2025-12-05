# Issues and Action Plan

## Current Status Summary

**Critical Blocker:**
- ⚠️ **Missing Python Dependencies**: `hydra-core` not installed
  - All 36 experiment runs failed with `ImportError: Required dependencies not available: No module named 'hydra'`
  - Code fixes complete, ready once dependencies installed
  - **Action Required**: Install dependencies before any experiments can run

**Experiment Status:**
- ❌ **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- ❌ **No result files:** Only log files exist (36 log files, no result directories)
- ❌ **No aggregated results:** `outputs/experiments/` directory exists but is empty
- ❌ **No trained models:** `outputs/models/` directory doesn't exist
- ✅ `run_experiment.sh` verified: Skip logic correct

**Report Status:**
- ✅ Content enhanced, citations verified, structure complete
- ⚠️ Placeholder content for KOCNPER.D and KOGFCF..D (blocked until experiments)
- ⚠️ Tables/plots have placeholder values (need actual results)

**Code Quality:**
- ✅ src/ has 17 files (2 deprecated wrappers, effective code in 15 files)
- ✅ dfm-python finalized: Consistent naming patterns

---

## Experiment Requirements

**Configuration:** 3 targets × 4 models × 3 horizons = 36 combinations
- **Targets:** KOGDP...D (GDP, 55 series), KOCNPER.D (Private Consumption, 50 series), KOGFCF..D (Gross Fixed Capital Formation, 19 series)
- **Models:** arima, var, dfm, ddfm
- **Horizons:** 1, 7, 28 days

**Current Status:** 0/3 targets complete (all failed due to missing dependencies)

**Expected Outputs (per target):**
- `outputs/comparisons/{target}_{timestamp}/comparison_results.json` (full metrics)
- `outputs/comparisons/{target}_{timestamp}/comparison_table.csv` (summary)
- `outputs/models/{target}_{model}/model.pkl` (4 models per target = 12 total)
- `outputs/experiments/aggregated_results.csv` (after aggregation, 36 rows)

---

## Action Plan (Priority Order)

### Phase 1: Install Dependencies (CRITICAL - BLOCKING)
**Priority:** HIGHEST - Blocks all experiments
**Status:** Code fixes complete, dependencies missing

**Tasks:**
1. Activate venv: `source .venv/bin/activate`
2. Install: `pip install -e .` or `pip install hydra-core>=1.3.2 omegaconf>=2.3.0 sktime[forecasting]>=0.40.1`
3. Verify: `python3 -c "import hydra; import sktime; print('OK')"`
4. Verify data/configs exist: `ls -lh data/sample_data.csv config/experiment/*_report.yaml`
5. Create output dirs: `mkdir -p outputs/{comparisons,models,experiments}`

**Success Criteria:** All imports succeed, environment ready.

---

### Phase 2: Run Experiments (BLOCKED until Phase 1)
**Priority:** HIGHEST - Required for all report updates
**Status:** 0/3 targets complete

**Tasks:**
1. Execute: `bash run_experiment.sh` (runs all 3 targets, auto-skips completed)
2. Verify: 3 result directories with `comparison_results.json` files
3. Verify: 12 trained models in `outputs/models/` (3 targets × 4 models)
4. Aggregate: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
5. Verify: `outputs/experiments/aggregated_results.csv` with 36 rows

**Success Criteria:** 3 result directories, 1 aggregated CSV, 12 trained models.

---

### Phase 3: Generate Visualizations (BLOCKED until Phase 2)
**Priority:** HIGH - Required for report completion

**Tasks:**
1. Run: `python3 nowcasting-report/code/plot.py`
2. Verify: 4 PNG files in `nowcasting-report/images/` (accuracy_heatmap, forecast_vs_actual, horizon_trend, model_comparison)
3. Verify: No placeholder text in images

**Success Criteria:** 4 PNG files with actual data.

---

### Phase 4: Update Report Tables (BLOCKED until Phase 2)
**Priority:** HIGH - Required for report completion

**Tasks:**
1. Load: `outputs/experiments/aggregated_results.csv`
2. Update: `tables/tab_overall_metrics.tex` (4 models × 3 metrics)
3. Update: `tables/tab_overall_metrics_by_target.tex` (3 targets × 4 models × 3 metrics)
4. Update: `tables/tab_overall_metrics_by_horizon.tex` (3 horizons × 4 models × 3 metrics)
5. Update: `tables/tab_nowcasting_metrics.tex` (if nowcasting evaluation was run)

**Success Criteria:** All tables updated with actual metrics, format consistent (4 decimal places, `---` for missing).

---

### Phase 5: Update Report Content (BLOCKED until Phase 2-4)
**Priority:** HIGH - Required for report completion

**Tasks:**
1. Update `contents/5_result.tex`: Replace placeholders for KOCNPER.D and KOGFCF..D with actual results
2. Update `contents/6_discussion.tex`: Discuss findings with actual numbers, cross-target comparisons
3. Update `main.tex` abstract: Summarize all 3 targets with key findings
4. Update `contents/1_introduction.tex`: Reflect completed experiments
5. Verify: All numerical claims match tables, no placeholder text, all citations valid

**Success Criteria:** Report content updated with actual results, no placeholders, all claims supported by data.

---

### Phase 6: Finalize Report (BLOCKED until Phase 5)
**Priority:** HIGH - Final step for report completion

**Tasks:**
1. Compile PDF: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. Verify: Check page count (target: 20-30 pages), no compilation errors
3. Verify: All placeholders removed, all numbers match results, all citations valid
4. Verify: All figures/tables referenced correctly, consistent formatting

**Success Criteria:** Complete 20-30 page PDF, no placeholders, all claims verified, professional quality.

---

## Optional Improvements (Can run in parallel)

### Code Quality (Optional)
- ✅ Temporary file removal: Completed (in-memory data_module)
- ✅ Naming consistency: Verified (PascalCase classes, snake_case functions)
- [ ] Numerical stability: Review regularization, safe matrix operations
- [ ] Theoretical correctness: Cross-reference with literature

### Report Quality (Integrated into Phase 5-6)
- Remove placeholders, verify citations, ensure 20-30 pages
- All quality checks covered in Phase 5-6 tasks

## Key Guidelines
- **Incremental execution**: Complete one phase before moving to next
- **Focus on blocking issues**: Phases 1-6 are critical for report completion
- **Verify at each step**: Check outputs, verify files exist, validate data
- **Report quality**: Remove hallucinations, use only references from `references.bib`
- **Never create new files**: Only modify existing code
- **Experiments first**: Report cannot be completed without experiment results
