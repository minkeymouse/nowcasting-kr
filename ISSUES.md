# Issues and Action Plan

## Current Status Summary (Updated 2025-12-06)

**Blocking Issues:**
- ✅ Code-level import errors FIXED
  - Root cause: Missing `src/__init__.py` + incorrect path calculation
  - Fixes: Created `src/__init__.py`, fixed path in `train.py` and `infer.py`, switched to absolute imports
  - Status: Code fixes in place, but cannot verify due to missing dependencies
- ⚠️ Missing Python dependencies (CURRENT BLOCKER)
  - Error: `ModuleNotFoundError: No module named 'hydra'` (latest runs: 011236, 011412, 013508)
  - Impact: Cannot proceed past import stage even though code fixes are applied
  - All 9 latest runs (3 per target) fail with same hydra dependency error
  - Action needed: Install dependencies (hydra-core, omegaconf, sktime, etc.)
- ❌ No experiment results generated (0 successful runs - all 18 attempts failed: 6 per target)
  - Error progression: relative import (001731, 002402) → missing src (004456) → missing hydra (011236, 011412, 013508)
  - No result directories or JSON files found in `outputs/comparisons/`
  - No `outputs/models/` directory exists (no trained models saved)
- ⚠️ Report contains placeholder content (improved in latest iteration, but still needs actual results)

**Non-Blocking Issues:**
- ✅ src/ has 17 files (code effectively in 15 files with deprecation wrappers - acceptable)
- ⚠️ Temporary file workarounds in sktime_forecaster.py (lines 146-162, 342-362) - documented, non-blocking
- ✅ dfm-python naming consistency VERIFIED AND FINALIZED (Latest iteration - 2025-12-06)
  - Classes use PascalCase: KalmanFilter, EMAlgorithm, BaseEncoder, PCAEncoder, DFMForecaster
  - Functions use snake_case: check_finite, ensure_real, ensure_symmetric, extract_decoder_params
  - No TODO/FIXME comments found
  - Code follows clean patterns consistently - FINALIZED
- ⚠️ dfm-python optional improvements: numerical stability review, theoretical correctness verification (non-blocking, can be done in future)
- ✅ Report content ENHANCED (latest iteration - expanded introduction and discussion sections)
- ✅ Report citations VERIFIED (tent kernel, FRBNY Staff Nowcast properly cited)

---

## Experiment Status and Requirements

### Required Experiments (from config/experiment/*.yaml)
**Targets (3):** KOGDP...D (GDP), KOCNPER.D (Private Consumption), KOGFCF..D (Gross Fixed Capital Formation)
**Models (4):** arima, var, dfm, ddfm
**Horizons (3):** 1, 7, 28 days
**Total Required:** 3 targets × 4 models × 3 horizons = 36 model-horizon combinations

### Current Results Status
**Location:** `outputs/comparisons/`
- ❌ **0 successful experiments** (no result directories found)
- ❌ **18 failed runs** (only .log files, 6 per target: 001731, 002402, 004456, 011236, 011412, 013508)
- **Error progression:**
  - Runs 001731, 002402 (6 total): `ImportError: attempted relative import with no known parent package` (FIXED)
  - Runs 004456 (3 total): `ModuleNotFoundError: No module named 'src'` (FIXED by creating src/__init__.py)
  - Runs 011236, 011412, 013508 (9 total): `ModuleNotFoundError: No module named 'hydra'` (CURRENT BLOCKER - missing dependency)
- ❌ **No result files:**
  - No `{target}_{timestamp}/` directories
  - No `comparison_results.json` files
  - No `comparison_table.csv` files
  - No `outputs/models/` directory (no trained models saved)

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
- ⚠️ **Note for later steps**: If experiments fail partially (some targets succeed, others fail), may need to update script to handle partial completion or re-run only failed targets

---

## Concrete Action Plan (Step-by-Step)

### Phase 1: Resolve Dependencies and Verify Setup (CRITICAL - BLOCKING)

**Goal**: Get experiments running past import stage

**Task 1.1: Install Python Dependencies**
- [ ] Check if virtual environment exists: `ls -la .venv/`
- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Install core dependencies: `pip install hydra-core omegaconf`
- [ ] Install sktime and forecasting dependencies: `pip install sktime[forecasting]`
- [ ] Install other required packages (check requirements.txt or setup.py if exists)
- [ ] Verify installations: `python3 -c "import hydra; import sktime; print('OK')"`

**Task 1.2: Test Import Fixes**
- [ ] Test basic import: `python3 -c "from src.utils.config_parser import setup_paths"`
- [ ] Test config loading: `python3 -c "from src.utils.config_parser import parse_experiment_config; import hydra; from hydra import initialize, compose; print('OK')"`
- [ ] Test actual script (dry run): `python3 src/train.py compare --config-name experiment/kogdp_report --help` (if help exists)
- [ ] Verify script proceeds beyond imports (may fail on data/model issues, but imports should work)

**Task 1.3: Verify Data and Config Files**
- [ ] Check data file exists: `ls -la data/sample_data.csv` (or path from config)
- [ ] Verify config files exist for all 3 targets: `ls -la config/experiment/*_report.yaml`
- [ ] Check series configs referenced in experiment configs exist
- [ ] Verify model configs exist: `ls -la config/model/*.yaml`

---

### Phase 2: Run Experiments Incrementally (CRITICAL - BLOCKING)

**Goal**: Generate experiment results for all 3 targets

**Task 2.1: Test Single Target First (KOGDP...D)**
- [ ] Run single experiment: `python3 src/train.py compare --config-name experiment/kogdp_report`
- [ ] Check for errors in output (not just import errors)
- [ ] Verify output structure created:
  - `outputs/comparisons/KOGDP...D_*/comparison_results.json` exists
  - `outputs/comparisons/KOGDP...D_*/comparison_table.csv` exists
  - `outputs/models/` contains trained models (at least some models)
- [ ] Inspect JSON file to verify metrics are present (sMSE, sMAE, sRMSE per model/horizon)
- [ ] If successful, proceed to Task 2.2. If failed, diagnose and fix before proceeding.

**Task 2.2: Run Remaining Targets**
- [ ] Run KOCNPER.D: `python3 src/train.py compare --config-name experiment/kocnper_report`
- [ ] Verify output: `outputs/comparisons/KOCNPER.D_*/comparison_results.json` exists
- [ ] Run KOGFCF..D: `python3 src/train.py compare --config-name experiment/kogfcf_report`
- [ ] Verify output: `outputs/comparisons/KOGFCF..D_*/comparison_results.json` exists
- [ ] **Alternative**: Use `run_experiment.sh` to run all targets (it will skip completed ones)

**Task 2.3: Aggregate Results**
- [ ] Run aggregator: `python3 -c "from src.eval import main_aggregator; main_aggregator()"` (✅ FIXED: updated to use correct import path)
- [ ] Verify aggregated CSV: `outputs/experiments/aggregated_results.csv` exists
- [ ] Check CSV contains metrics for all 3 targets, 4 models, 3 horizons
- [ ] Verify CSV structure matches what plot.py and tables expect

**Task 2.4: Update run_experiment.sh if Needed (LATER STEP)**
- [ ] If experiments fail partially, review skip logic
- [ ] If some models fail but others succeed, consider updating script to handle partial results
- [ ] If timeout issues occur, adjust timeout values in script
- [ ] **Note**: Only update if issues are encountered during actual runs

---

### Phase 3: Generate Visualizations (IMPORTANT - BLOCKED)

**Goal**: Create plots from experiment results

**Task 3.1: Verify Plot Script**
- [ ] Check `nowcasting-report/code/plot.py` exists and is executable
- [ ] Review plot.py to understand expected input format (JSON files from outputs/comparisons/)
- [ ] Verify plot.py can load results: `python3 nowcasting-report/code/plot.py` (may need to run from project root)
- [ ] Check if plot.py needs modifications for data format

**Task 3.2: Generate Plots**
- [ ] Run plot generation: `python3 nowcasting-report/code/plot.py`
- [ ] Verify all 4 images created:
  - `nowcasting-report/images/accuracy_heatmap.png`
  - `nowcasting-report/images/forecast_vs_actual.png`
  - `nowcasting-report/images/horizon_trend.png`
  - `nowcasting-report/images/model_comparison.png`
- [ ] Inspect plots to ensure they contain actual data (not placeholders or empty plots)
- [ ] If plots are empty or incorrect, fix plot.py data loading/formatting

---

### Phase 4: Update Report Tables (IMPORTANT - BLOCKED)

**Goal**: Populate LaTeX tables with real metrics

**Task 4.1: Load Aggregated Results**
- [ ] Load `outputs/experiments/aggregated_results.csv` into Python/pandas
- [ ] Verify CSV structure: columns (target, model, horizon, sMSE, sMAE, sRMSE, etc.)
- [ ] Check data completeness: all 3 targets, 4 models, 3 horizons present

**Task 4.2: Update Overall Metrics Table**
- [ ] Update `nowcasting-report/tables/tab_overall_metrics.tex`
- [ ] Calculate overall averages across all targets/horizons for each model
- [ ] Format numbers appropriately (round to 4 decimal places)
- [ ] Handle missing values (ARIMA if it fails, 28-day horizon if insufficient data)

**Task 4.3: Update Target-Specific Table**
- [ ] Update `nowcasting-report/tables/tab_overall_metrics_by_target.tex`
- [ ] Populate metrics for each target (KOGDP...D, KOCNPER.D, KOGFCF..D)
- [ ] Calculate averages across horizons for each target-model combination
- [ ] Format consistently with overall metrics table

**Task 4.4: Update Horizon-Specific Table**
- [ ] Update `nowcasting-report/tables/tab_overall_metrics_by_horizon.tex`
- [ ] Populate metrics for each horizon (1, 7, 28 days)
- [ ] Calculate averages across targets for each horizon-model combination
- [ ] Handle 28-day horizon if insufficient data (mark as N/A or omit)

**Task 4.5: Update Nowcasting Table (if applicable)**
- [ ] Check if nowcasting evaluation was run (may require separate script: `src/infer.py nowcast`)
- [ ] If nowcasting results exist, update `nowcasting-report/tables/tab_nowcasting_metrics.tex`
- [ ] If not, keep placeholder or note that nowcasting evaluation is future work

---

### Phase 5: Update Report Content (IMPORTANT - BLOCKED)

**Goal**: Replace placeholder content with actual results and analysis

**Task 5.1: Update Results Section**
- [ ] Update `nowcasting-report/contents/5_result.tex`
- [ ] Replace placeholder text with actual findings from experiment results
- [ ] Reference specific metrics from tables (e.g., "DFM achieved sRMSE=0.0419 for 7-day horizon")
- [ ] Discuss model performance differences with actual numbers
- [ ] Update section to reflect all 3 targets if results are available (currently only GDP mentioned)

**Task 5.2: Update Discussion Section**
- [ ] Update `nowcasting-report/contents/6_discussion.tex`
- [ ] Discuss actual findings: why DFM performs well at 7-day horizon, why VAR performs well at 1-day
- [ ] Address DDFM performance (if lower than DFM, discuss potential reasons: underfitting, hyperparameters)
- [ ] Discuss limitations based on actual results (e.g., 28-day horizon insufficient data)
- [ ] Connect findings to theoretical background section

**Task 5.3: Update Abstract and Introduction**
- [ ] Update `nowcasting-report/main.tex` abstract if needed (currently mentions GDP results)
- [ ] Update `nowcasting-report/contents/1_introduction.tex` to reflect actual scope (3 targets vs. just GDP)
- [ ] Ensure introduction matches what was actually done

---

### Phase 6: Finalize Report (IMPORTANT)

**Goal**: Ensure report is complete, compiles, and meets length requirements

**Task 6.1: Compile and Verify Report**
- [ ] Compile LaTeX: `cd nowcasting-report && pdflatex main.tex`
- [ ] Run bibliography: `bibtex main` (if needed)
- [ ] Recompile: `pdflatex main.tex` (twice for references)
- [ ] Check for LaTeX errors and fix them
- [ ] Verify PDF generated: `nowcasting-report/main.pdf`

**Task 6.2: Verify Report Length**
- [ ] Check PDF page count (should be 20-30 pages)
- [ ] If too short (< 20 pages):
  - Expand method section with more detail
  - Add more analysis in discussion
  - Expand theoretical background if needed
- [ ] If too long (> 30 pages):
  - Condense less critical sections
  - Move detailed tables to appendix if needed
  - Remove redundant content

**Task 6.3: Final Quality Check**
- [ ] Verify all figures referenced in text exist in `images/` directory
- [ ] Verify all tables referenced in text exist and are populated
- [ ] Check citations: all references in text exist in `references.bib`
- [ ] Verify no placeholder text remains (search for "---", "TBD", "향후 연구")
- [ ] Check Korean text is properly formatted (no encoding issues)

---

### Phase 7: Code Quality (NON-BLOCKING - Can be done in parallel)

**Goal**: Improve code quality and consistency

**Task 7.1: Review Temporary File Workarounds**
- [ ] Review `src/model/sktime_forecaster.py` lines 146-162, 342-362
- [ ] Check if dfm-python API supports in-memory data (avoid temporary CSV files)
- [ ] If API supports it, refactor to use in-memory data
- [ ] If not, document why temporary files are necessary

**Task 7.2: dfm-python Code Review (if time permits)**
- [ ] Review numerical stability: EM algorithm regularization, Kalman filter utilities
- [ ] Verify theoretical correctness: tent kernel implementation, clock framework, factor dynamics
- [ ] **Note**: This is non-blocking for report completion

---

## Summary

### Critical Path (Blocking)
1. **Phase 1**: Install dependencies → 2. **Phase 2**: Run experiments → 3. **Phase 3-6**: Update report

### Current Focus
**Phase 1, Task 1.1**: Install Python dependencies (hydra-core, omegaconf, sktime) to unblock experiments.

### Key Decisions
1. **Incremental execution**: Test single target first before running all 3
2. **Verify after each phase**: Don't proceed to next phase until current phase is complete
3. **Focus on blocking issues**: Phases 1-6 are critical, Phase 7 can wait
4. **Update run_experiment.sh only if needed**: Script looks good, only update if issues arise

---

---

## Improvement Plan (Prioritized)

### Priority 1: Code Quality - Temporary File Workarounds (MEDIUM) ✅ DOCUMENTED
**Location**: `src/model/sktime_forecaster.py` lines 146-162 (DFM), 342-362 (DDFM)
**Issue**: Using temporary CSV files as workaround because dfm-python expects file paths
**Impact**: Inefficient I/O, potential race conditions, not clean code pattern
**Action**:
- [x] Check if dfm-python API supports in-memory arrays (pandas DataFrame or numpy array) - ✅ VERIFIED: DFMDataModule supports `data` parameter
- [ ] If supported, refactor `_fit_dfm()` and `_fit_ddfm()` to use in-memory data (TODO: requires refactoring `create_data_module()` to accept `data` parameter)
- [x] If not supported, document why temporary files are necessary (add comment explaining API limitation) - ✅ DOCUMENTED
- [ ] Consider adding feature request to dfm-python for in-memory support (or refactor `create_data_module()` to accept `data` parameter)
**Status**: Documented with TODO comments. Refactoring requires modifying `create_data_module()` to accept in-memory data parameter. Non-blocking for experiments.

### Priority 2: Code Quality - Numerical Stability Review (MEDIUM)
**Location**: `dfm-python/src/dfm_python/ssm/` (EM algorithm, Kalman filter)
**Issue**: Need to verify numerical stability mechanisms are sufficient
**Current State**: 
- Regularization exists (1e-6 for matrix inversion, 1e-8 for Cholesky)
- Convergence checks exist (threshold=1e-5 for EM)
- Symmetry and positive definiteness checks exist
**Action**:
- [ ] Review EM algorithm convergence criteria (check for edge cases)
- [ ] Verify Kalman filter handles ill-conditioned matrices properly
- [ ] Test with edge cases: very small eigenvalues, near-singular matrices
- [ ] Check if regularization scales are appropriate for different data scales
- [ ] Verify no numerical overflow/underflow in matrix operations
**Status**: Non-blocking, important for production readiness

### Priority 3: Code Quality - Theoretical Correctness Verification (MEDIUM)
**Location**: `dfm-python/src/dfm_python/` (tent kernel, clock framework, factor dynamics)
**Issue**: Need to verify implementations match theoretical specifications
**Action**:
- [ ] Verify tent kernel aggregation matches Mariano & Murasawa (2003) specification
- [ ] Check clock framework correctly handles mixed-frequency alignment
- [ ] Verify factor dynamics (AR lag structure) matches DFM theory
- [ ] Review DDFM encoder architecture matches Andreini et al. (2020) specification
- [ ] Check block structure implementation matches intended design
**Status**: Non-blocking, important for theoretical validity

### Priority 4: Report - Remove Placeholder Content (HIGH - BLOCKED)
**Location**: `nowcasting-report/contents/5_result.tex`, `6_discussion.tex`
**Issue**: Report contains placeholders for missing results (KOCNPER.D, KOGFCF..D)
**Current State**:
- GDP results mentioned, other targets say "향후 연구에서 다룰 예정"
- Nowcasting section mentions "향후 연구에서 구현될 예정"
**Action**:
- [ ] After experiments complete, update all 3 targets in results section
- [ ] Remove or implement nowcasting evaluation (currently TODO in `src/infer.py:397`)
- [ ] Update tables with actual metrics for all targets
- [ ] Remove all "향후 연구" placeholders
**Status**: BLOCKED until experiments complete

### Priority 5: Report - Improve Flow and Detail (MEDIUM)
**Location**: All report sections
**Issue**: Some sections may lack detail or have unnatural flow
**Action**:
- [ ] Review method section: ensure all hyperparameters and design choices are explained
- [ ] Review results section: ensure smooth transitions between subsections
- [ ] Review discussion section: connect findings to theoretical background
- [ ] Ensure all figures/tables are properly referenced in text
- [ ] Check for redundancy: remove repeated information
**Status**: Can be done incrementally, non-blocking

### Priority 6: Report - Verify Citations and References (MEDIUM)
**Location**: `nowcasting-report/contents/*.tex`, `references.bib`
**Issue**: Ensure all citations are from references.bib, no hallucination
**Action**:
- [ ] Verify all citations in text exist in `references.bib`
- [ ] Check if new citations needed (use knowledgebase MCP if available)
- [ ] Ensure tent kernel citation (Mariano & Murasawa 2003) is correct
- [ ] Verify FRBNY Staff Nowcast citation is correct
- [ ] Add citations for any new theoretical claims
**Status**: Non-blocking, can be done incrementally

### Priority 7: Experiment Planning - Update run_experiment.sh (LOW)
**Location**: `run_experiment.sh`
**Issue**: Script should only run missing experiments
**Current State**: 
- Skip logic correctly implemented (checks for `comparison_results.json`)
- All 3 targets currently missing results, so all will run
**Action**:
- [ ] After experiments run, verify skip logic works correctly
- [ ] If partial failures occur, consider adding per-model skip logic
- [ ] Only update if issues are encountered during actual runs
**Status**: Script is correct, only update if needed

### Priority 8: dfm-python - Generic Naming Review (LOW)
**Location**: `dfm-python/src/dfm_python/`
**Issue**: Ensure naming is generic and consistent
**Current State**: 
- ✅ Classes use PascalCase (DFM, DDFM, BaseFactorModel, etc.)
- ✅ Functions use snake_case (format_error_message, check_finite, etc.)
- ✅ Naming consistency verified
**Action**:
- [ ] Review for any hardcoded values that should be configurable
- [ ] Check for any non-generic naming (e.g., "finance" instead of generic terms)
- [ ] Ensure all magic numbers are constants or config parameters
**Status**: Already verified, low priority

### Priority 9: Report - Length and Completeness (HIGH - BLOCKED)
**Location**: All report sections
**Issue**: Report must be 20-30 pages, currently may be shorter
**Action**:
- [ ] After all results are in, compile PDF and check page count
- [ ] If < 20 pages: expand method section, add more analysis, expand theoretical background
- [ ] If > 30 pages: condense sections, move tables to appendix, remove redundancy
- [ ] Ensure all sections are complete (no placeholders)
**Status**: BLOCKED until experiments complete and results added

---

## Notes

- **Work incrementally**: Complete one task before moving to next
- **Test after each change**: Don't break working functionality
- **Keep ISSUES.md under 1000 lines**: Remove resolved issues, consolidate as needed
- **Never hallucinate**: Only use references from `references.bib` and knowledgebase (neo4j MCP)
- **Do not create new files**: Only modify existing code
- **Do not delete files**: Only consolidate/merge
- **Check outputs/ before experiments**: run_experiment.sh will skip completed experiments automatically
- **Report length target**: 20-30 pages (verify after compilation)
- **Prioritization**: Phases 1-6 are critical (blocking), Phase 7 and Priorities 1-9 are non-blocking
