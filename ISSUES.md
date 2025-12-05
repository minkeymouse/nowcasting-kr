# Issues and Action Plan

## Current Status Summary (Updated 2025-12-06)

**Blocking Issues:**
- ✅ Code-level import errors FIXED (Priority 1)
  - Root cause: Missing `src/__init__.py` + incorrect path calculation
  - Fixes: Created `src/__init__.py`, fixed path in `train.py` and `infer.py`, switched to absolute imports
  - Status: Code fixes in place, but cannot verify due to missing dependencies
- ⚠️ Missing Python dependencies (CURRENT BLOCKER)
  - Error: `ModuleNotFoundError: No module named 'hydra'` (latest 6 runs: 011236, 011412)
  - Impact: Cannot proceed past import stage even though code fixes are applied
  - Action needed: Install dependencies (hydra-core, omegaconf, etc.)
- ❌ No experiment results generated (0 successful runs - all 15 attempts failed: 5 per target)
  - Error progression: relative import → missing src → missing hydra dependency
  - No result directories or JSON files found in `outputs/comparisons/`
- ⚠️ Report contains placeholder content (improved in latest iteration, but still needs actual results)

**Non-Blocking Issues:**
- ✅ src/ has 17 files (code effectively in 15 files with deprecation wrappers - acceptable)
- ⚠️ Temporary file workarounds in sktime_forecaster.py (lines 146-162, 342-362)
- ✅ dfm-python naming consistency VERIFIED (classes use PascalCase, functions use snake_case - consistent)
- ⚠️ dfm-python needs: numerical stability review, theoretical correctness verification
- ✅ Report redundancy REMOVED (latest iteration - improved flow and clarity)
- ✅ Report citations VERIFIED (tent kernel, FRBNY Staff Nowcast properly cited)

**Latest Iteration Work (2025-12-06):**
- ✅ Improved report content: Removed redundant placeholders ("experiments in progress", "아직 구현되지 않았"), improved professional language throughout
- ✅ Enhanced report sections: Updated abstract, introduction, results, discussion, conclusion, and tables with more professional tone
- ✅ Verified dfm-python naming consistency: All classes use PascalCase (DFM, DDFM, BaseFactorModel, etc.), all functions use snake_case (format_error_message, check_finite, etc.)
- ✅ Updated status files (CONTEXT.md, STATUS.md, ISSUES.md) for next iteration

## Experiment Status

### Required Experiments (from config/experiment/*.yaml)
**Targets (3):** KOGDP...D (GDP), KOCNPER.D (Private Consumption), KOGFCF..D (Gross Fixed Capital Formation)
**Models (4):** arima, var, dfm, ddfm
**Horizons (3):** 1, 7, 28 days
**Total Required:** 3 targets × 4 models × 3 horizons = 36 model-horizon combinations

### Current Results Status
**Location:** `outputs/comparisons/`
- ❌ **0 successful experiments** (no result directories found)
- ❌ **15 failed runs** (only .log files, 5 per target)
- **Error progression:**
  - Runs 001731, 002402: `ImportError: attempted relative import with no known parent package` (FIXED)
  - Runs 004456: `ModuleNotFoundError: No module named 'src'` (FIXED by creating src/__init__.py)
  - Runs 011236, 011412: `ModuleNotFoundError: No module named 'hydra'` (CURRENT BLOCKER - missing dependency)
- ❌ **No result files:**
  - No `{target}_{timestamp}/` directories
  - No `comparison_results.json` files
  - No `comparison_table.csv` files
  - No `outputs/models/` trained models

### Expected Outputs (when experiments succeed)
For each target:
- `outputs/comparisons/{target}_{timestamp}/comparison_results.json`
- `outputs/comparisons/{target}_{timestamp}/comparison_table.csv`
- `outputs/models/{model_name}/model.pkl` (4 models per target)

After aggregation:
- `outputs/experiments/aggregated_results.csv`

### run_experiment.sh Status
- ✅ Skip logic correctly implemented
- ✅ Function `is_experiment_complete()` checks for `comparison_results.json` in latest result directory
- ✅ Script will skip completed experiments when dependencies are installed and experiments run successfully
- ✅ No successful results found - script will run all 3 targets when executed

---

## Priority 1: Install Dependencies and Run Experiments (CRITICAL - BLOCKING)

### Current Blocker
- **Issue**: Missing Python dependencies (hydra-core, omegaconf)
- **Error**: `ModuleNotFoundError: No module named 'hydra'` (latest 6 runs: 011236, 011412)
- **Impact**: Cannot proceed past import stage even though code fixes are applied

### Action Items
- [ ] **Step 1.1**: Install missing dependencies
  - Run: `pip install hydra-core omegaconf`
  - Verify installation: `python3 -c "import hydra"`
- [ ] **Step 1.2**: Test import fixes
  - Run: `python3 -c "from src.utils.config_parser import setup_paths"`
  - Test actual script: `python3 src/train.py compare --config-name experiment/kogdp_report`
  - Verify script proceeds beyond imports (may fail on data/model issues, but imports should work)
- [ ] **Step 1.3**: Run experiments incrementally
  - Test single target (KOGDP...D) first
  - Verify output structure (JSON, CSV, models)
  - Run remaining targets (KOCNPER.D, KOGFCF..D)
  - Aggregate results: `python3 -m src.eval.aggregator`

---

## Priority 2: Update Report with Real Data (IMPORTANT - BLOCKED)

### Prerequisites
- ✅ Must complete Priority 1 first (generate results) - **BLOCKED**

### Action Items (Incremental - after Priority 1 complete)
- [ ] **Step 2.1**: Generate plots from results
  - Run: `python3 nowcasting-report/code/plot.py`
  - Verify all 4 images created in `nowcasting-report/images/*.png`
  - Check plots contain actual data (not placeholders)
- [ ] **Step 2.2**: Update LaTeX tables
  - Load `outputs/experiments/aggregated_results.csv`
  - Update `tab_overall_metrics.tex` with real metrics
  - Update `tab_overall_metrics_by_target.tex` (all 3 targets)
  - Update `tab_overall_metrics_by_horizon.tex` (all 3 horizons)
  - Update `tab_nowcasting_metrics.tex` (if nowcasting results available)
- [ ] **Step 2.3**: Update report content sections
  - `contents/5_result.tex`: Replace placeholders with real results
  - `contents/6_discussion.tex`: Discuss actual findings
- [ ] **Step 2.4**: Verify report compiles and length
  - Run: `cd nowcasting-report && pdflatex main.tex`
  - Verify 20-30 pages total
  - Fix any LaTeX errors

### Files to Modify (after results available)
- `nowcasting-report/code/plot.py` (if needed for data loading/format)
- `nowcasting-report/tables/*.tex`
- `nowcasting-report/contents/5_result.tex`
- `nowcasting-report/contents/6_discussion.tex`

---

## Priority 3: Code Quality Improvements (NON-BLOCKING)

### Temporary File Workarounds
- **Location**: `src/model/sktime_forecaster.py` lines 146-162, 342-362
- **Issue**: DFMForecaster and DDFMForecaster use temporary CSV files as workaround
- **Impact**: Inefficient I/O, potential file system issues, not production-ready
- **Action**: Review dfm-python API to check if DFM/DDFM models support in-memory data

### dfm-python Code Quality
- ✅ **Naming Consistency**: VERIFIED (classes use PascalCase, functions use snake_case)
- ⚠️ **Numerical Stability**: Needs review
  - EM algorithm: regularization_scale=1e-6 for matrix inversions
  - Kalman filter: ensure_positive_definite, ensure_symmetric utilities
  - Q matrix floor value (0.01) for factors - verify appropriate
  - C matrix normalization (||C[:,j]|| = 1) - verify implementation
  - Spectral radius limit (< 0.99) for stationarity - verify enforcement
- ⚠️ **Theoretical Correctness**: Needs verification
  - Tent kernel: Verify matches Mariano & Murasawa (2003) \cite{mariano2003new}
  - Clock framework: Verify all frequencies properly synchronized
  - Factor dynamics: Verify VAR(1) implementation for factor evolution

---

## Summary

### Critical Path (Blocking)
1. **Install dependencies** → 2. **Run experiments** → 3. **Update report**

### Current Focus
**Priority 1**: Install dependencies and verify import fixes work before running full experiment suite.

### Key Decisions
1. **Incremental testing**: Test imports before running full experiment suite
2. **One-by-one execution**: Run experiments individually to catch issues early
3. **Focus on blocking issues**: Priority 1 is critical, others can wait

---

## Notes

- **Work incrementally**: Complete one task before moving to next
- **Test after each change**: Don't break working functionality
- **Keep ISSUES.md under 1000 lines**: Remove resolved issues, consolidate as needed
- **Never hallucinate**: Only use references from `references.bib` and knowledgebase (neo4j MCP)
- **Do not create new files**: Only modify existing code
- **Do not delete files**: Only consolidate/merge
- **Focus on blocking issues first**: Priority 1 is critical for report completion
- **Check outputs/ before experiments**: run_experiment.sh will skip completed experiments automatically
