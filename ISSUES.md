# Issues and Action Plan

## CRITICAL PRIORITY - REAL PROBLEMS TO FIX

### 1. DDFM MODELS PRODUCING CONSTANT PREDICTIONS - CRITICAL BUG (FIX APPLIED, NEEDS VERIFICATION)
**ACTUAL STATE**: DDFM backtest JSON files show `"status": "completed"` but produce constant predictions (same value for all months and both timepoints)
- **Problem**: DDFM models produce constant predictions instead of varying predictions, indicating Kalman filter re-run is failing and falling back to training state
- **Impact**: Table 3 and Plot4 show incorrect DDFM results (constant values), making nowcasting evaluation unreliable
- **Action**: Step 1 will detect DDFM constant predictions and re-run `bash agent_execute.sh backtest` to verify code fixes work
- **Verification**: After re-run, DDFM JSON files should contain varying forecast values (not constant) in `results_by_timepoint`
- **Status**: ⚠️ FIX APPLIED BUT NOT VERIFIED - Code fixes present but backtests NOT re-run to verify they work

### 2. TRAINING STATUS
**ACTUAL STATE**: checkpoint/ contains 12 model.pkl files (12/12 models trained)
- **Status**: ✅ DONE - All models are trained and available

### 3. DFM NOWCASTING WORKING, DDFM NOWCASTING NEEDS VERIFICATION
**ACTUAL STATE**: DFM models produce varying predictions (working), DDFM models producing constant predictions (code fixes applied but not verified)
- **DFM Status**: ✅ Working - KOEQUIPTE DFM shows varying predictions (7.5579, -0.2869, etc.)
- **DDFM Status**: ⚠️ Code fixes applied but NOT verified - All 3 DDFM models produce constant predictions:
  - KOEQUIPTE DDFM: All predictions = -0.16721546038528526 (constant)
  - KOIPALL.G DDFM: All predictions = 0.021669534372733057 (constant)
  - KOWRCCNSE DDFM: All predictions = 0.43302732169739405 (constant)
- **Root Causes Identified**:
  1. **Data filtering failing**: Dimension mismatch (117 vs 32 series) causing Kalman filter failures
  2. **Kalman filter re-run failing**: Inf values and dimension issues causing fallback to training state
  3. **No DDFM-specific fallback**: When Kalman filter fails, no encoder-based alternative path
- **Fixes Applied (This Iteration)**:
  1. ✅ **IMPROVED NaN/Inf HANDLING**: Added Inf detection and replacement at multiple stages (before/after standardization, in torch tensor) - `src/infer.py` lines 362-397
  2. ✅ **DDFM ENCODER FALLBACK**: Added DDFM-specific encoder path to extract factor state from last observation when Kalman filter fails - `src/infer.py` lines 444-477
  3. ✅ **ENHANCED ERROR LOGGING**: Improved error messages with detailed diagnostics (data shapes, parameter shapes, series_ids availability)
  4. ✅ **IMPROVED DATA FILTERING VALIDATION**: Enhanced validation with detailed logging for successful/failed filtering - `src/infer.py` lines 574-710
- **Impact**: Table 3 and Plot4 show incorrect DDFM results (constant values) - need regeneration after backtests are re-run
- **Action**: Re-run backtests to verify DDFM fixes work, then regenerate Table 3 and Plot4
- **Status**: ⚠️ FIX APPLIED BUT NOT VERIFIED - Code fixes applied for DDFM constant predictions. Needs re-run of backtests to verify fixes work.

## CODE FIXES APPLIED THIS ITERATION (NOT YET VERIFIED)

### Fix 1: UnboundLocalError in src/infer.py _get_current_factor_state()
- **Location**: `src/infer.py` lines 295-304
- **Fix**: Moved `Mx = result.Mx` and `Wx = result.Wx` extraction before validation checks
- **Enhanced**: Added explicit hasattr() checks for Mx/Wx before extraction to provide clearer RuntimeError messages if missing
- **Impact**: Should prevent "cannot access local variable 'Mx'" error that was causing all 12 backtests to fail
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest

### Fix 2: AttributeError in src/infer.py _get_current_factor_state()
- **Location**: `src/infer.py` lines 364-370
- **Issue**: Direct access to `Mx.shape` without checking if Mx has shape attribute (could fail for scalars)
- **Fix**: Added hasattr() check before accessing Mx.shape, with fallback for scalar Mx
- **Impact**: Should prevent AttributeError when Mx is a scalar or doesn't have shape attribute
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest

### Fix 3: ValueError in src/nowcasting.py _prepare_target()
- **Location**: `src/nowcasting.py` lines 1479-1504
- **Fix**: Handles empty Time_view gracefully with fallback values instead of raising ValueError
- **Impact**: Should prevent "Target period not found in Time index" errors
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest

### Fix 4: Numerical Instability (Inf values) in Kalman filter
- **Location**: `dfm-python/src/dfm_python/ssm/kalman.py` lines 222-228
- **Fix**: Added Inf detection and replacement with NaN in input Y matrix before forward pass
- **Impact**: Should prevent Inf propagation through matrix operations
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest

### Fix 5: Numerical Instability (Inf values) in data views
- **Location**: `src/nowcasting.py` lines 1451-1457
- **Fix**: Added Inf detection in X_view before processing, replaces Inf with NaN
- **Impact**: Should prevent numerical instability from Inf values in input data
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest

### Fix 6: Backtest Status Check in agent_execute.sh
- **Location**: `agent_execute.sh` line 68
- **Fix**: Changed to check `'status': 'completed'` for DFM/DDFM, or `'status': 'not_supported'` for ARIMA/VAR
- **Impact**: Should correctly identify completed backtests
- **Status**: ⚠️ Code fix present, will be verified when backtests run

### Fix 7: CRITICAL - Data Filtering Bug in _update_data_module_for_nowcasting()
- **Location**: `src/infer.py` lines 503-575
- **Problem**: Data filtering was failing silently - when config-based filtering failed, code continued with all 117 columns instead of filtering to model's expected 32 series. This caused dimension mismatch errors, Kalman filter failures, and fallback to training state (constant predictions).
- **Fix**: 
  - First tries to get series_ids from model result (most reliable)
  - Falls back to config if result not available
  - Uses Mx/Wx dimensions as last resort to determine expected series count
  - Raises errors if filtering cannot be done properly instead of silently continuing
  - Validates filtering worked before proceeding
- **Impact**: Should fix dimension mismatch (117 vs 32 series), allow Kalman filter to run properly, and produce varying predictions instead of constant values
- **Status**: ⚠️ Code fix present, NOT verified by successful backtest with non-constant predictions

### Fix 8: CRITICAL - DDFM Constant Predictions - Improved Kalman Filter Re-run and Encoder Fallback
- **Location**: `src/infer.py` lines 359-477
- **Problem**: Kalman filter re-run was failing for DDFM models, causing fallback to training state (constant predictions). Issues included:
  - Inf values in data causing Kalman filter failures
  - No DDFM-specific fallback path when Kalman filter fails
  - Encoder access path not handling both DDFMBase directly and DDFM wrapper (which stores model in _model)
  - Insufficient error logging to diagnose failures
- **Fix Applied**:
  1. ✅ **IMPROVED NaN/Inf HANDLING**: Added Inf detection and replacement with NaN before and after standardization, and in torch tensor (NaN is handled by Kalman filter, Inf is not)
  2. ✅ **DDFM ENCODER FALLBACK**: Added DDFM-specific encoder path that uses encoder to extract factor state from last observation if Kalman filter fails
  3. ✅ **ENCODER ACCESS PATH FIX**: Fixed encoder access to handle both DDFMBase directly (`dfm_model.encoder`) and DDFM wrapper (`dfm_model._model.encoder`)
  4. ✅ **ENHANCED ERROR LOGGING**: Improved error messages with detailed diagnostic information (data shapes, parameter shapes, series_ids availability)
  5. ✅ **IMPROVED DATA FILTERING VALIDATION**: Added detailed logging for successful/failed filtering with model type information
- **Code Changes**:
  - `src/infer.py` lines 362-397: Added Inf detection and replacement at multiple stages (before standardization, after standardization, in torch tensor)
  - `src/infer.py` lines 444-477: Added DDFM encoder fallback path that encodes last observation to get factor state, with proper encoder access (handles both DDFMBase and wrapper)
  - `src/infer.py` lines 687-710: Enhanced data filtering validation with detailed logging
- **Impact**: Should fix DDFM constant predictions by:
  - Handling Inf values that cause Kalman filter failures
  - Providing encoder-based fallback for DDFM when Kalman filter fails (with correct encoder access path)
  - Better diagnostics to identify remaining issues if any
- **Status**: ⚠️ Code fix applied, needs re-run of backtests to verify DDFM predictions vary (not constant)

## ACTION PLAN (PRIORITY ORDER)

### Phase 1: Fix DDFM Constant Predictions (CRITICAL - BLOCKING)
**GOAL**: Fix DDFM models producing constant predictions in nowcasting backtests

**Step 1.1: Inspect DDFM Result Structure**
- **Action**: Check if DDFM result has series_ids, Mx, Wx attributes
- **Location**: Inspect `src/infer.py` _update_data_module_for_nowcasting() lines 510-525
- **Code to add**: Log DDFM result structure (hasattr checks, type, shape of Mx/Wx)
- **Expected**: DDFM result should have same structure as DFM, or different structure needs handling

**Step 1.2: Inspect Data Filtering for DDFM**
- **Action**: Verify data filtering works correctly for DDFM models
- **Location**: `src/infer.py` _update_data_module_for_nowcasting() lines 540-575
- **Code to add**: Log before/after filtering: n_series_in_data, n_series_expected, model_series_ids length
- **Expected**: After filtering, data_monthly should have same number of columns as model expects

**Step 1.3: Inspect Kalman Filter Re-run for DDFM**
- **Action**: Check if Kalman filter re-run fails for DDFM and falls back to training state
- **Location**: `src/infer.py` _get_current_factor_state() lines 349-420
- **Code to add**: Log when fallback occurs (line 353-355), log Kalman filter errors for DDFM
- **Expected**: Kalman filter should succeed for DDFM, or alternative method needed

**Step 1.4: Fix DDFM-Specific Issues**
- **Action**: Based on inspection results, fix DDFM constant prediction issue
- **Possible fixes**:
  - If DDFM result structure differs: Add DDFM-specific handling in _update_data_module_for_nowcasting()
  - If data filtering fails for DDFM: Fix series_ids extraction for DDFM models
  - If Kalman filter fails: Add DDFM-specific factor state retrieval method
  - If encoder/decoder issue: Check if DDFM needs different prediction path than DFM
- **Verification**: Re-run DDFM backtests, verify predictions vary across months

### Phase 2: Training (DONE)
1. ✅ **Training complete**: checkpoint/ contains 12 model.pkl files (3 targets × 4 models)
2. **Status**: All models trained and available

### Phase 3: Re-run Backtesting (CRITICAL - BLOCKING)
1. **Step 1 automatically runs**: `bash agent_execute.sh backtest`
   - Detects DDFM constant predictions
   - Re-runs all 12 backtests with fixed DDFM code
   - Saves to outputs/backtest/{target}_{model}_backtest.json
2. **Verification**: 
   - Check JSON files have `"status": "completed"` (DFM/DDFM) or `"status": "no_results"` (ARIMA/VAR)
   - **CRITICAL**: Verify DDFM predictions vary (not constant) across months and timepoints
   - Verify DDFM shows different values for 4weeks vs 1week timepoints
3. **If backtesting fails**: Inspect logs in outputs/nowcast/, verify DDFM fixes work, fix remaining issues

### Phase 4: Table/Plot Generation (HIGH PRIORITY)
1. **After backtesting succeeds**: Regenerate tables and plots
   - Run: `python3 nowcasting-report/code/table.py` (generates all LaTeX tables)
   - Run: `python3 nowcasting-report/code/plot.py` (generates all plots)
2. **Verification**: Check outputs/experiments/tables/ and nowcasting-report/images/ for updated files
3. **If generation fails**: Check data format in outputs/backtest/, fix table/plot generation code

### Phase 5: Report Update (HIGH PRIORITY)
1. **After tables/plots generated**: Update report sections
   - Update nowcasting section in `nowcasting-report/contents/3_results_nowcasting.tex`
   - Reference Table 3 and Plot4 in report
   - Update discussion section with nowcasting results
2. **Verification**: Check report sections reference actual tables/plots, not placeholders
3. **If report update fails**: Verify table/plot file paths, fix LaTeX references

### Phase 6: Model Performance Inspection (MEDIUM PRIORITY)
1. **After all experiments complete**: Inspect results for anomalies
   - Check for near-perfect results (potential data leakage)
   - Check for extreme values (numerical instability)
   - Check for too-poor results (implementation errors)
2. **If anomalies found**: Fix code issues (data leakage, numerical stability, implementation bugs)
3. **Verification**: Results should be reasonable (not suspiciously good/poor)

### Phase 7: dfm-python Package Inspection (MEDIUM PRIORITY)
1. **Code quality review**: Check for consistency, naming patterns, documentation
2. **Numerical stability review**: Verify regularization, error handling, edge cases
3. **Theoretical correctness**: Verify implementation matches theory
4. **If issues found**: Fix code in dfm-python/ package
5. **Verification**: Package should have clean code, good numerical stability, correct implementation

## CURRENT STATE (ACTUAL - VERIFIED BY INSPECTION)

### Training
- **checkpoint/**: 12 model.pkl files exist (12/12 models trained)
- **Status**: ✅ DONE - All models trained and available

### Forecasting
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows)
- **outputs/comparisons/**: Contains comparison_results.json files
- **Status**: ✅ DONE - Results available (extreme VAR values filtered on load)

### Nowcasting
- **outputs/backtest/**: 12 JSON files exist
  - **DFM models (3)**: `"status": "completed"` ✅ - Working correctly, varying predictions (e.g., KOEQUIPTE DFM: 7.5579, -0.2869, etc.)
  - **DDFM models (3)**: `"status": "completed"` ⚠️ - Producing constant predictions (KOEQUIPTE: -0.167..., KOIPALL.G: 0.0216..., KOWRCCNSE: 0.433...)
  - **ARIMA/VAR models (6)**: `"status": "no_results"` ✅ - Expected (not supported for nowcasting)
- **Status**: ⚠️ PARTIAL - DFM working correctly, DDFM broken (constant predictions, code fixes applied but not verified)

### Tables/Plots
- **Table 1**: ✅ Generated (tab_dataset_params.tex)
- **Table 2**: ✅ Generated (tab_forecasting_results.tex)
- **Table 3**: ⚠️ Generated with N/A placeholders (needs nowcasting results)
- **Plot1-3**: ✅ Generated (forecast vs actual, heatmap, horizon trend)
- **Plot4**: ⚠️ Generated with placeholders (needs nowcasting results)

## MODEL PERFORMANCE ANALYSIS

**Forecasting Results Inspection**:
- All forecasting results have `n_valid=1` (single test point per horizon) - this is expected for current evaluation setup
- Some suspiciously good results detected (sMSE < 1e-4):
  - KOWRCCNSE,ARIMA,5: sMSE=0.00057 (single point, could be legitimate)
  - KOWRCCNSE,ARIMA,7: sMSE=0.00061 (single point, could be legitimate)
  - KOWRCCNSE,ARIMA,8: sMSE=0.00043 (single point, could be legitimate)
  - KOWRCCNSE,VAR,4: sMSE=0.00082 (single point, could be legitimate)
  - KOWRCCNSE,DDFM,7: sMSE=0.00016 (single point, could be legitimate)
- **Code handles these correctly**: `aggregate_overall_performance()` marks suspiciously good results as NaN (lines 1137-1156), and `generate_all_latex_tables()` also filters them (lines 1848-1883)
- **Note**: Current CSV file (aggregated_results.csv) contains these values because it was generated before the fix was added. When regenerated, these will be marked as NaN.
- With n_valid=1, these could be legitimate if prediction happened to be very close to actual value, but code marks them as NaN for reliability
- No evidence of data leakage: training period (1985-2019) is separate from test period
- Extreme values (> 1e10) are filtered out by validation code (already handled)

**Comparison Results**:
- All 3 targets show all 4 models completed successfully (no failed_models in comparison_results.json)
- VAR horizon 1 shows N/A in tables (expected - persistence predictions marked as NaN)
- DFM/DDFM horizon 22 shows NaN for some targets (n_valid=0, expected for long horizons)

## KNOWN LIMITATIONS

- ARIMA and VAR models return `"status": "not_supported"` for nowcasting (expected - only DFM/DDFM support nowcasting)
- Numerical stability: Some edge cases with sparse data may still need improvement (monitoring needed)
- Data filtering: Dimension mismatch checks added, but may need further validation (monitoring needed)
- Model performance: Single-point evaluation (n_valid=1) means results are sensitive to individual prediction accuracy
- Suspiciously good results: Some very small sMSE values exist but are flagged by validation - with n_valid=1, these could be legitimate or lucky single-point matches

## INSPECTION FINDINGS (This Iteration)

**Backtest Results Analysis**:
- **DFM models**: ✅ Working correctly - produce varying predictions (KOEQUIPTE DFM: 7.5579, -0.2869, etc.)
- **DDFM models**: ⚠️ Producing constant predictions (code fixes applied but NOT verified):
  - KOEQUIPTE DDFM: All 42 predictions (21 months × 2 timepoints) = -0.16721546038528526 (constant)
  - KOIPALL.G DDFM: All predictions = 0.021669534372733057 (constant)
  - KOWRCCNSE DDFM: All predictions = 0.43302732169739405 (constant)
- **Root cause hypothesis**: Data filtering or Kalman filter re-run failing for DDFM, causing fallback to training state (constant factor state)

**Code Inspection**:
- Data filtering code (lines 574-710) improved with multiple fallbacks (Mx → Wx → C matrix) and enhanced validation
- Kalman filter re-run (lines 349-420) has fallback to training state (line 353-355) - this may be triggering for DDFM
- DDFM encoder fallback added (lines 444-477) to handle both DDFMBase directly and DDFM wrapper
- Inf handling improved (lines 362-397) at multiple stages (before/after standardization, in torch tensor)

## NOTES

- **DO NOT claim "complete", "verified", "resolved" unless actually fixed**: DDFM constant predictions are a REAL bug - code fixes applied but NOT verified by re-running backtests
- **Priority**: Re-run backtests to verify DDFM fixes work BEFORE regenerating tables/plots
- **Step 1 automatically handles experiment execution**: Agent should NOT directly execute scripts, only modify code
- **Incremental improvement**: Code fixes applied this iteration - next iteration must re-run backtests to verify fixes work
- **This iteration**: Applied code fixes for DDFM constant predictions (encoder fallback, Inf handling, improved data filtering) - needs re-run to verify
