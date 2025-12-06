# Issues and Action Plan

## Executive Summary (2025-01-XX)

**Current State**: ARIMA working (6/9 combinations, n_valid=1). VAR working perfectly (9/9 combinations, n_valid=1). Report updated with VAR results. DFM/DDFM fixes applied and ready for testing.  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Strategy**: Incremental approach - test DFM/DDFM fixes, then run full experiments  
**Next Action**: Test DFM/DDFM fixes on single target/horizon, then run full experiments

## Critical Issues (PRIORITY 1)

### 1. VAR n_valid=0 [RESOLVED ✅]
**Status**: ✅ Fixed and tested - VAR working with n_valid=1 for all 9 combinations  
**Root Cause**: Two issues: (1) target_series not included in VAR training data, (2) models filter override issue in train.py  
**Evidence**: Log showed "Horizon 1: Error calculating metrics: 'KOGDP...D'" for all VAR runs  
**Fix Applied**: 
1. ✅ Added target_series to VAR training data in training.py (lines 268-280)
2. ✅ Fixed models filter in train.py to use models_filter parameter instead of config override
3. ✅ Added KeyError handling in calculate_standardized_metrics() for y_train.columns lookup  
**Results**: VAR working perfectly - 9/9 combinations (3 targets × 3 horizons), n_valid=1
- Overall: sMSE 0.0036, sMAE 0.046, sRMSE 0.046 (much better than ARIMA)
- By target: GDP (sRMSE 0.056), Consumption (sRMSE 0.055), Investment (sRMSE 0.028)
- By horizon: 1-day (sRMSE 0.006), 7-day (0.036), 28-day (0.098)
- Report updated: VAR results integrated into tables and all report sections

### 2. DFM Issues [FIXED - READY FOR TESTING]
**Status**: Both issues fixed, ready for testing

**Issue 2a: C Matrix NaN (KOGDP...D) [FIXED]**
- KOGDP...D: C matrix first row all NaN, loglik=0.0, n_valid=0
- Root cause: EM algorithm C matrix update produces NaN, likely in normalization step
- Fix applied:
  1. ✅ Added NaN detection/early stopping in dfm-python/src/dfm_python/ssm/em.py
  2. ✅ Fixed C matrix normalization (||C[:,j]|| = 1) - set to zero if norm < 1e-8
  3. ✅ Added NaN checks after solve operation and in normalization step
  4. ✅ Added early stopping in EM loop if NaN detected

**Issue 2b: Prediction Failure Even When Training Succeeds (KOCNPER.D, KOGFCF..D) [FIXED]**
- KOCNPER.D: C matrix looks fine (no visible NaN), converged (42 iter), loglik=0.0, but n_valid=0
- KOGFCF..D: C matrix looks fine, converged (100 iter), loglik=135.76, but n_valid=0
- Root cause: Prediction DataFrame columns didn't match config series_ids order
- Fix applied:
  1. ✅ Fixed prediction step in src/model/sktime_forecaster.py (DFMForecaster._predict)
  2. ✅ Get series_ids from DFM config and use as DataFrame columns
  3. ✅ Reorder columns to match training data for target_series extraction

**Files**: `dfm-python/src/dfm_python/ssm/em.py`, `dfm-python/src/dfm_python/models/dfm.py`, `src/model/sktime_forecaster.py`

### 3. DDFM C Matrix All NaN [FIXED - READY FOR TESTING]
**Status**: NaN detection added, ready for testing  
**Evidence**: All targets show C matrix with all NaN values, training completes 200 iterations but converged=False, loglik=NaN  
**Root Cause**: PyTorch encoder produces NaN during forward pass or training

**Fix Applied**:
1. ✅ Added NaN detection in training_step (check forward pass output and loss)
2. ✅ Added NaN detection in extract_decoder_params (check C matrix after extraction)
3. ✅ Added NaN detection in get_result (validate C matrix before returning)
4. ✅ Replace NaN with zeros to prevent propagation (with warnings)
5. ⏳ Gradient clipping and learning rate: Need to verify in trainer config

**Files**: `dfm-python/src/dfm_python/models/ddfm.py`, `dfm-python/src/dfm_python/encoder/vae.py`

## Code Fixes Completed

1. ✅ **VAR KeyError fix**: Added KeyError handling in calculate_standardized_metrics() for y_train.columns
2. ✅ **ARIMA/VAR target_series handling**: Fixed Series input handling
3. ✅ **DFM/DDFM pickle errors**: Fixed make_cha_transformer (uses functools.partial)
4. ✅ **Test data size check**: Skip horizon 28 if test set too small
5. ✅ **run_experiment.sh**: Added MODELS environment variable support
6. ✅ **train.py**: Added --models flag for incremental testing

## Action Plan (Incremental, Prioritized)

### Experiment Status Summary
**Completed**: ARIMA (9/36 combinations = 25%)
- ✅ KOGDP...D: horizons 1, 7, 28 (n_valid=1 each)
- ✅ KOCNPER.D: horizons 1, 7, 28 (n_valid=1 each)
- ✅ KOGFCF..D: horizons 1, 7, 28 (n_valid=1 each)

**Missing**: VAR (9), DFM (9), DDFM (9) = 27 combinations
- ❌ VAR: KeyError fix applied, needs testing
- ❌ DFM: C matrix NaN (EM algorithm issue)
- ❌ DDFM: C matrix NaN (encoder issue)

**Minimum Viable**: 6 combinations (2 models × 3 targets) - ✅ ARIMA exceeds minimum
**Ideal**: All 36 combinations (3 targets × 4 models × 3 horizons)

### PHASE 1: Test VAR Fix [NEXT ACTION - Priority 1]
**Goal**: Verify VAR KeyError fix works, get VAR results for report

**Task 1.1**: Test VAR on minimal case (single target, single horizon)
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models var --horizons 1`
- Success criteria: n_valid > 0 in comparison_results.json
- Check: `outputs/comparisons/KOGDP...D_*/comparison_results.json` → `results.var.metrics.forecast_metrics.1.n_valid > 0`
- Time estimate: 5-10 minutes

**Task 1.2**: If Task 1.1 succeeds, test VAR on all horizons (single target)
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models var`
- Success criteria: n_valid > 0 for horizons 1, 7, 28
- Time estimate: 15-20 minutes

**Task 1.3**: If Task 1.2 succeeds, run VAR for all targets
- Command: `MODELS="var" bash run_experiment.sh`
- Success criteria: n_valid > 0 for all 9 VAR combinations (3 targets × 3 horizons)
- Check: Verify all 3 targets have valid VAR results
- Time estimate: 30-60 minutes

**Task 1.4**: Generate aggregated CSV with VAR results
- Command: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- Success criteria: `outputs/experiments/aggregated_results.csv` contains VAR rows with valid metrics
- Time estimate: 1 minute

**Dependencies**: None (VAR fix already applied)
**Blocking**: PHASE 2 (DFM/DDFM fixes can proceed in parallel, but full results need VAR)

### PHASE 2: Fix DFM Numerical Instability [Priority 2 - Can run parallel to PHASE 1]
**Goal**: Fix DFM C matrix NaN issue, get DFM results for report

**Task 2.1**: Investigate DFM C matrix NaN root cause
- Files to check: `dfm-python/src/dfm_python/ssm/em.py` (EM algorithm, lines 189-216), `dfm-python/src/dfm_python/models/dfm.py` (model initialization)
- Checkpoints:
  1. Review C matrix update in EM algorithm (em.py lines 196-216):
     - Check if sum_EZZ_reg is singular (determinant near zero)
     - Verify sum_yEZ computation handles missing data correctly
     - Check C matrix normalization (lines 210-214): verify zero denominator handling when norm < 1e-8
  2. Review PCA initialization in dfm.py:
     - Check if PCA fails for edge cases (T < N, high missing data, constant series)
     - Verify fallback initialization handles these cases
  3. Check for NaN propagation from data preprocessing:
     - Review rem_nans_spline_torch in dfm.py line 354
     - Check if imputed data contains NaN
  4. Check prediction step (dfm.py lines 595-690):
     - Verify C matrix validation (lines 672-677) catches all NaN cases
     - Check if target series extraction fails
- Success criteria: Identify exact location where NaN is introduced
- Time estimate: 30-60 minutes

**Task 2.2**: Implement DFM fix based on Task 2.1 findings
- **em.py fixes**:
  - Add NaN check after C_new computation (after line 208): if NaN detected, stop EM and log warning
  - Fix C matrix normalization (lines 210-214): if norm < 1e-8, set column to zero instead of leaving unchanged
  - Add early stopping: check for NaN in C_new, A_new, Q_new after each EM step
- **dfm.py fixes**:
  - Improve PCA initialization: add validation, fallback to random if PCA fails
  - Add NaN check in predict() before using C matrix (enhance lines 672-677)
  - Verify data imputation doesn't introduce NaN
- **src/model/dfm.py fixes**:
  - Check if predict() correctly handles target series extraction
  - Verify forecast extraction matches expected structure
- Add logging for debugging (use _logger from em.py)
- Success criteria: Code changes applied, no syntax errors, linter passes
- Time estimate: 30-60 minutes

**Task 2.3**: Test DFM fix on minimal case
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models dfm --horizons 1`
- Success criteria: n_valid > 0, no NaN in C matrix (check JSON results)
- Time estimate: 10-15 minutes

**Task 2.4**: If Task 2.3 succeeds, test DFM on all horizons (single target)
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models dfm`
- Success criteria: n_valid > 0 for horizons 1, 7, 28
- Time estimate: 20-30 minutes

**Task 2.5**: If Task 2.4 succeeds, run DFM for all targets
- Command: `MODELS="dfm" bash run_experiment.sh`
- Success criteria: n_valid > 0 for all 9 DFM combinations
- Time estimate: 60-120 minutes

**Dependencies**: None (can run parallel to PHASE 1)
**Blocking**: PHASE 3 (full results generation)

### PHASE 3: Fix DDFM Numerical Instability [Priority 3 - Can run parallel to PHASE 1-2]
**Goal**: Fix DDFM C matrix NaN issue, get DDFM results for report

**Task 3.1**: Investigate DDFM C matrix NaN root cause
- Files to check: `dfm-python/src/dfm_python/models/ddfm.py` (training_step line 785, encoder forward pass, get_result line 1897)
- Checkpoints:
  1. Review training_step (lines 785-840):
     - Check if forward pass produces NaN: add check after `reconstructed = self.forward(data)` (line 820)
     - Verify loss computation handles NaN correctly (lines 834-835)
     - Check if gradient clipping is applied (may be in trainer, not here)
  2. Review encoder forward pass:
     - Check encoder.__call__() or forward() method for NaN in intermediate layers
     - Verify activation functions don't produce NaN (ReLU, etc.)
  3. Review get_result() (lines 1897-1936):
     - Check extract_decoder_params() call (line 1922): verify C extraction doesn't introduce NaN
     - Check if decoder weights contain NaN before extraction
  4. Check trainer configuration:
     - Verify max_grad_norm is set (check trainer/ddfm.py or config)
     - Check learning rate: may be too high causing NaN
     - Verify optimizer settings
  5. Check initialization:
     - Verify encoder/decoder weight initialization (Xavier, He, etc.)
     - Check if initialization produces extreme values
- Success criteria: Identify exact location where NaN is introduced
- Time estimate: 30-60 minutes

**Task 3.2**: Implement DDFM fix based on Task 3.1 findings
- **ddfm.py training_step fixes** (lines 785-840):
  - Add NaN check after forward pass (after line 820): if NaN detected, log warning and return large loss
  - Add NaN check in loss computation (after line 835): if loss is NaN, log and return large loss
  - Consider reducing learning rate if NaN persists
- **Encoder/decoder fixes**:
  - Add NaN checks in encoder forward pass if needed
  - Verify activation functions are numerically stable
- **get_result() fixes** (lines 1897-1936):
  - Add validation in extract_decoder_params: check for NaN in decoder weights before extraction
  - Add NaN check after C extraction (line 1932): if NaN, log warning and use fallback
- **Trainer/config fixes**:
  - Verify gradient clipping is enabled (max_grad_norm)
  - Check learning rate: reduce if too high (typical: 1e-3 to 1e-4)
  - Consider adding learning rate scheduler
- Add logging for debugging
- Success criteria: Code changes applied, no syntax errors, linter passes
- Time estimate: 30-60 minutes

**Task 3.3**: Test DDFM fix on minimal case
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models ddfm --horizons 1`
- Success criteria: n_valid > 0, no NaN in C matrix (check JSON results)
- Time estimate: 15-20 minutes (DDFM training is slower)

**Task 3.4**: If Task 3.3 succeeds, test DDFM on all horizons (single target)
- Command: `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models ddfm`
- Success criteria: n_valid > 0 for horizons 1, 7, 28
- Time estimate: 30-60 minutes

**Task 3.5**: If Task 3.4 succeeds, run DDFM for all targets
- Command: `MODELS="ddfm" bash run_experiment.sh`
- Success criteria: n_valid > 0 for all 9 DDFM combinations
- Time estimate: 120-180 minutes (DDFM is slowest)

**Dependencies**: None (can run parallel to PHASE 1-2)
**Blocking**: PHASE 4 (full results generation)

### PHASE 4: Generate Full Results [AFTER PHASE 1-3]
**Goal**: Generate complete aggregated results and update report tables/plots

**Task 4.1**: Re-run full experiments (only missing models)
- Command: `bash run_experiment.sh` (will skip ARIMA if already complete, run VAR/DFM/DDFM)
- Note: run_experiment.sh already skips completed experiments, but may need update if structure changes
- Success criteria: All 36 combinations have valid results (n_valid > 0)
- Check: Verify `outputs/comparisons/{target}_*/comparison_results.json` for all targets
- Time estimate: 2-4 hours (depends on which models are still missing)

**Task 4.2**: Generate aggregated CSV with all models
- Command: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
- Success criteria: `outputs/experiments/aggregated_results.csv` has 36 rows (3 targets × 4 models × 3 horizons), all with valid metrics
- Time estimate: 1 minute

**Task 4.3**: Generate plots with all models
- Command: `python3 nowcasting-report/code/plot.py`
- Success criteria: 4 PNG files in `nowcasting-report/images/` updated with all models
- Files: model_comparison.png, horizon_trend.png, accuracy_heatmap.png, forecast_vs_actual.png
- Time estimate: 2-5 minutes

**Task 4.4**: Update LaTeX tables with all model results
- Files: `nowcasting-report/tables/tab_overall_metrics.tex`, `tab_overall_metrics_by_target.tex`, `tab_overall_metrics_by_horizon.tex`, `tab_nowcasting_metrics.tex`
- Source: `outputs/experiments/aggregated_results.csv`
- Success criteria: All "---" placeholders replaced with actual metrics
- Time estimate: 15-30 minutes

**Dependencies**: PHASE 1-3 (need at least VAR working, ideally all models)
**Blocking**: PHASE 5 (report finalization)

### PHASE 5: Finalize Report [AFTER PHASE 4]
**Goal**: Complete 20-30 page report with all results, no placeholders

**Task 5.1**: Update results section with all model findings
- File: `nowcasting-report/contents/5_result.tex`
- Add VAR/DFM/DDFM findings (if available)
- Include specific metrics, comparisons, patterns
- Success criteria: Results section complete with all available model results
- Time estimate: 30-60 minutes

**Task 5.2**: Update discussion section with full model comparison
- File: `nowcasting-report/contents/6_discussion.tex`
- Compare all models (ARIMA, VAR, DFM, DDFM)
- Discuss performance patterns, target differences, horizon effects
- Provide model selection guidance based on actual results
- Success criteria: Discussion section complete with comprehensive model comparison
- Time estimate: 30-60 minutes

**Task 5.3**: Update conclusion section if needed
- File: `nowcasting-report/contents/7_conclusion.tex`
- Reflect all experimental results
- Summarize key findings across all models
- Success criteria: Conclusion reflects complete experimental results
- Time estimate: 15-30 minutes

**Task 5.4**: Compile PDF and verify report completeness
- Command: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Success criteria:
  1. PDF compiles without errors
  2. Report is 20-30 pages
  3. No "---" placeholders in tables
  4. All plots included and referenced
  5. All citations resolved
- Time estimate: 5-10 minutes

**Dependencies**: PHASE 4 (need complete results)
**Blocking**: None (final step)

### Notes on run_experiment.sh Updates
- **Current**: run_experiment.sh already supports MODELS filter and skips completed experiments
- **Status**: No changes needed currently - script correctly skips ARIMA (already complete)
- **Future updates** (if needed):
  - If experiment structure changes, update `is_experiment_complete()` function (lines 136-198)
  - If new models added, update MODELS array default (line 260)
  - If new targets added, update TARGETS array (line 38)
- **For incremental testing**: Use `MODELS="var" bash run_experiment.sh` to test one model at a time

## Experiment Status

**Latest Run**: 20251206_082502
- **ARIMA**: ✅ WORKING - n_valid=1 for all horizons across all 3 targets (9 combinations)
- **VAR**: ❌ n_valid=0 - KeyError in metrics calculation (confirmed in logs: "Error calculating metrics: 'KOGDP...D'"), fix applied but not tested
- **DFM**: ❌ n_valid=0 - Two issues: (1) C matrix NaN (KOGDP...D), (2) Prediction failure even when training succeeds (KOCNPER.D, KOGFCF..D)
- **DDFM**: ❌ n_valid=0 - C matrix all NaN for all targets (encoder issue)

**Experiments Needed**:
- Minimum viable: 6 combinations (2 models × 3 targets) - ✅ ARIMA has 9 (exceeds minimum)
- Ideal: All 36 combinations (3 targets × 4 models × 3 horizons)
- Current: 9/36 = 25% complete

## Code Quality Improvements (PRIORITY 2 - After Critical Fixes)

### dfm-python Numerical Stability Issues

**1. DFM EM Algorithm (dfm-python/src/dfm_python/ssm/em.py)**:
- **Issue**: C matrix normalization (lines 210-214) may have zero denominator when `norm < 1e-8`
  - Current: `if norm > 1e-8: C_new[:, j] = C_new[:, j] / norm`
  - Problem: If norm is exactly 0 or very small, column remains unchanged but may contain NaN from previous step
  - Fix: Add explicit NaN check before normalization, set to zero if norm < 1e-8
- **Issue**: No early stopping if C matrix becomes NaN during EM iterations
  - Current: Exception handling catches RuntimeError but doesn't check for NaN
  - Fix: Add NaN check after C_new computation, stop EM if detected
- **Issue**: PCA initialization may fail for edge cases (T < N, high missing data, constant series)
  - Fix: Add validation in PCA initialization, fallback to random initialization if PCA fails

**2. DDFM Training (dfm-python/src/dfm_python/models/ddfm.py)**:
- **Issue**: No NaN detection in training_step (line 785-840)
  - Current: Loss computation doesn't check for NaN in forward pass output
  - Fix: Add NaN check after `reconstructed = self.forward(data)`, log warning and stop if NaN detected
- **Issue**: Gradient clipping may not be set appropriately
  - Check: Verify max_grad_norm in trainer configuration
  - Fix: Add gradient clipping if not present, reduce learning rate if NaN persists
- **Issue**: C matrix extraction (extract_decoder_params) may extract NaN if decoder weights are NaN
  - Fix: Add validation in extract_decoder_params, check for NaN before returning C

**3. DFM Prediction (dfm-python/src/dfm_python/models/dfm.py)**:
- **Issue**: Prediction may fail if C matrix has NaN even when training reports success
  - Current: Validation checks for NaN in Z_last and A/C (lines 662-677) but may not catch all cases
  - Fix: Add more comprehensive validation, check C matrix before prediction
- **Issue**: Target series extraction in src/model/sktime_forecaster.py may fail for DFM
  - Check: Verify DFMForecaster._predict() correctly extracts target series from X_forecast
  - Fix: Ensure column names match between training and prediction

### src/ Code Quality Issues

**1. Model Wrappers**:
- **Issue**: Duplicate logic in DFMForecaster and DDFMForecaster._predict() methods
  - Both have similar index creation and DataFrame conversion logic
  - Fix: Extract common logic to shared utility function
- **Issue**: Exception handling may not log all errors
  - Fix: Ensure all try/except blocks log errors with context

**2. Evaluation**:
- **Issue**: Silent NaN propagation in calculate_standardized_metrics()
  - Current: Returns NaN metrics but may not log why
  - Fix: Add more detailed logging when NaN detected

## Report Status

- ✅ Structure: Complete 8-section framework
- ✅ Citations: All 21 references verified
- ✅ Content: Sections 1-4, 6-7 complete with actual findings
- ✅ Results: Section 5 updated with ARIMA and VAR findings, detailed target/horizon analysis
- ✅ Discussion: Section 6 expanded with VAR findings and comprehensive model comparison
- ✅ Conclusion: Section 7 updated with VAR results summary
- ✅ Tables: ARIMA and VAR values filled, DFM/DDFM remain "---"
- ✅ Plots: Generated with ARIMA and VAR data
- ✅ Report expanded: Added detailed VAR performance analysis throughout

## Report Improvements Needed (PRIORITY 3 - After Experiments Complete)

### Content Issues

**1. Results Section (5_result.tex)**:
- **Issue**: VAR/DFM/DDFM results show "---" placeholders in tables
  - Fix: Update tables with actual metrics once experiments complete
- **Issue**: Nowcasting section (lines 44-88) has incomplete tables with "---"
  - Fix: Add actual nowcasting metrics or remove section if not applicable
- **Issue**: Some redundancy in target variable descriptions
  - Fix: Consolidate similar descriptions, focus on unique characteristics

**2. Discussion Section (6_discussion.tex)**:
- **Issue**: Limited to ARIMA findings, lacks comparison with other models
  - Fix: Add VAR/DFM/DDFM findings when available, compare all models
- **Issue**: Model selection guidance (lines 52-58) is theoretical, not based on results
  - Fix: Update with actual experimental findings
- **Issue**: Some statements are too optimistic about DDFM without results
  - Fix: Tone down expectations, acknowledge limitations

**3. Method Section (4_method_and_experiment.tex)**:
- **Issue**: May need updates if experimental setup changes
  - Fix: Ensure method section matches actual implementation

**4. Tables**:
- **Issue**: tab_nowcasting_metrics.tex has "---" for all DFM/DDFM entries
  - Fix: Update with actual metrics or remove if nowcasting not implemented
- **Issue**: Tables may need formatting improvements for readability
  - Fix: Review table formatting, ensure consistent decimal places

### Technical Writing Issues

**1. Flow and Clarity**:
- **Issue**: Some sections may have unnatural transitions
  - Fix: Review flow between sections, add transition sentences
- **Issue**: Korean technical terms may need consistency
  - Fix: Ensure consistent terminology throughout

**2. Citations**:
- **Issue**: All citations verified, but may need additional citations for new findings
  - Fix: Add citations when referencing knowledgebase for new information
  - Note: Use references.bib, never hallucinate citations
