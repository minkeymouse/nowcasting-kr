# Issues and Action Plan

## CRITICAL PRIORITY - REAL PROBLEMS TO FIX

### 1. TRAINING NOT DONE - BLOCKING ALL EXPERIMENTS
**ACTUAL STATE**: checkpoint/ is EMPTY (0/12 models trained)
- **Problem**: All models need training before forecasting/nowcasting can run
- **Impact**: Blocks forecasting (needs models), blocks nowcasting (needs models)
- **Action**: Step 1 will automatically detect empty checkpoint/ and run `bash agent_execute.sh train`
- **Verification**: After training, checkpoint/ should contain 12 model.pkl files (3 targets × 4 models)
- **Status**: ❌ NOT FIXED - Training has not been run yet

### 2. NOWCASTING EXPERIMENTS FAILED - NEEDS RE-RUN
**ACTUAL STATE**: All 12 backtest JSON files have `"status": "no_results"` (0/12 completed)
- **Problem**: Nowcasting experiments failed, preventing Table 3 and Plot4 generation
- **Root Causes Identified**:
  - UnboundLocalError in src/infer.py _get_current_factor_state() - Code fix applied (Mx/Wx extraction moved before use)
  - ValueError in src/nowcasting.py _prepare_target() - Code fix applied (handles empty Time_view gracefully)
  - Numerical instability (Inf values) - Code fix applied (Inf detection added in Kalman filter and data views)
- **Code Fixes Applied**: Fixes are present in codebase but NOT verified by successful backtests
- **Action**: After training completes, Step 1 will detect "no_results" status and run `bash agent_execute.sh backtest`
- **Verification**: After backtesting, JSON files should have `"status": "completed"` for DFM/DDFM, or `"status": "not_supported"` for ARIMA/VAR
- **Status**: ⚠️ NOT VERIFIED - Code fixes present but backtests not re-run to verify they work

### 3. TABLE/PLOT GENERATION BLOCKED - WAITING FOR NOWCASTING RESULTS
**ACTUAL STATE**: Table 3 and Plot4 have N/A placeholders (nowcasting results missing)
- **Problem**: Cannot generate complete tables/plots without nowcasting results
- **Action**: After backtesting succeeds, regenerate:
  - Table 3: `python3 nowcasting-report/code/table.py` (generates tab_nowcasting_backtest.tex)
  - Plot4: `python3 nowcasting-report/code/plot.py` (generates nowcasting_comparison_*.png)
- **Status**: ⚠️ BLOCKED - Waiting for nowcasting results

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

## ACTION PLAN (PRIORITY ORDER)

### Phase 1: Training (CRITICAL - BLOCKING)
1. **Step 1 automatically runs**: `bash agent_execute.sh train`
   - Detects empty checkpoint/
   - Trains all 12 models (3 targets × 4 models)
   - Saves to checkpoint/{target}_{model}/model.pkl
2. **Verification**: Check checkpoint/ contains 12 model.pkl files
3. **If training fails**: Inspect logs in log/ directory, fix training code issues

### Phase 2: Backtesting (CRITICAL - BLOCKING)
1. **Step 1 automatically runs**: `bash agent_execute.sh backtest` (after training completes)
   - Detects "no_results" status in backtest JSON files
   - Re-runs all 12 backtests to verify code fixes work
   - Saves to outputs/backtest/{target}_{model}_backtest.json
2. **Verification**: Check JSON files have `"status": "completed"` (DFM/DDFM) or `"status": "not_supported"` (ARIMA/VAR)
3. **If backtesting fails**: Inspect logs in outputs/nowcast/, verify code fixes are correct, fix remaining issues

### Phase 3: Table/Plot Generation (HIGH PRIORITY)
1. **After backtesting succeeds**: Regenerate tables and plots
   - Run: `python3 nowcasting-report/code/table.py` (generates all LaTeX tables)
   - Run: `python3 nowcasting-report/code/plot.py` (generates all plots)
2. **Verification**: Check outputs/experiments/tables/ and nowcasting-report/images/ for updated files
3. **If generation fails**: Check data format in outputs/backtest/, fix table/plot generation code

### Phase 4: Report Update (HIGH PRIORITY)
1. **After tables/plots generated**: Update report sections
   - Update nowcasting section in `nowcasting-report/contents/3_results_nowcasting.tex`
   - Reference Table 3 and Plot4 in report
   - Update discussion section with nowcasting results
2. **Verification**: Check report sections reference actual tables/plots, not placeholders
3. **If report update fails**: Verify table/plot file paths, fix LaTeX references

### Phase 5: Model Performance Inspection (MEDIUM PRIORITY)
1. **After all experiments complete**: Inspect results for anomalies
   - Check for near-perfect results (potential data leakage)
   - Check for extreme values (numerical instability)
   - Check for too-poor results (implementation errors)
2. **If anomalies found**: Fix code issues (data leakage, numerical stability, implementation bugs)
3. **Verification**: Results should be reasonable (not suspiciously good/poor)

### Phase 6: dfm-python Package Inspection (MEDIUM PRIORITY)
1. **Code quality review**: Check for consistency, naming patterns, documentation
2. **Numerical stability review**: Verify regularization, error handling, edge cases
3. **Theoretical correctness**: Verify implementation matches theory
4. **If issues found**: Fix code in dfm-python/ package
5. **Verification**: Package should have clean code, good numerical stability, correct implementation

## CURRENT STATE (ACTUAL - VERIFIED BY INSPECTION)

### Training
- **checkpoint/**: EMPTY (0/12 models trained)
- **Status**: ❌ NOT DONE - All models need training

### Forecasting
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows)
- **outputs/comparisons/**: Contains comparison_results.json files
- **Status**: ✅ DONE - Results available (extreme VAR values filtered on load)

### Nowcasting
- **outputs/backtest/**: 12 JSON files with `"status": "no_results"` (0/12 completed)
- **Status**: ❌ NOT DONE - Code fixes present but not verified by successful backtests

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

## NOTES

- **DO NOT claim "complete", "verified", "resolved" unless actually fixed**: Training and backtesting have NOT been run yet
- **Code fixes are present but NOT verified**: All 6 fixes (UnboundLocalError, AttributeError, ValueError, Inf handling x2, backtest check) need successful experiments to verify
- **Step 1 automatically handles experiment execution**: Agent should NOT directly execute scripts, only modify code
- **Incremental improvement**: Focus on fixing one issue at a time, verify fixes work before moving to next issue
- **This iteration**: Code fixes applied but NO experiments run to verify - next iteration must run experiments
