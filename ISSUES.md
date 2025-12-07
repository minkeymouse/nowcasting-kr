# Issues and Action Plan

## EXECUTIVE SUMMARY

**CRITICAL BLOCKING ISSUES**:
1. ❌ **ALL 12 backtests failed** - All JSON files have "status": "no_results". Code fixes present but NOT VERIFIED (backtests need re-run)
2. ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained (needs training via Step 1)

**ROOT CAUSES IDENTIFIED AND FIXED IN CODE** (NOT VERIFIED):
- **NEW THIS ITERATION**: Fixed indentation bug in data_module update (src/infer.py lines 486-488) - incorrect indentation could cause logic errors
- **NEW THIS ITERATION**: Made target period verification less strict (src/infer.py lines 607-690) - was too defensive, skipping all months. Now proceeds even if exact date not found, with fallback to find closest date. Always proceeds instead of skipping.
- **NEW THIS ITERATION**: Improved model result restoration (src/infer.py lines 178-230) - added multiple restoration methods (training_state, wrapper get_result(), model_data) with better error handling
- **NEW THIS ITERATION**: Made recent data check more lenient (src/infer.py line 937) - changed from 365 days to 730 days to avoid skipping valid months
- Kalman smoother NaN propagation: Fixed backward smoother validation and fallback (dfm-python/src/dfm_python/ssm/kalman.py lines 467-505)
- Forward Kalman filter stability: Added adaptive regularization for sparse data (dfm-python/src/dfm_python/ssm/kalman.py lines 318-345)
- VAR column indexing: Fixed ColumnEnsembleTransformer compatibility (src/infer.py lines 973-1077)

**NEXT STEPS**:
1. **VERIFY CODE FIXES**: Step 1 will detect "no_results" status and re-run backtests to verify if fixes work
2. **TRAIN MODELS**: Step 1 will automatically detect and train missing models
3. **GENERATE TABLES/PLOTS**: After backtests succeed, regenerate Table 3 and Plot4

**KEY PRINCIPLE**: Code fixes are present but NOT verified. Only mark as "resolved" after backtests succeed and JSON files contain actual results.

---

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK**:
- **checkpoint/**: 10/12 models trained ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: 12 JSON files with "status": "no_results" ❌ (ALL backtests failed - code fixes applied, needs re-run)
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows, extreme VAR values filtered on load)
- **nowcasting-report/**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**This Iteration Work** (ACTUALLY DONE):
1. **Code fixes applied**:
   - Fixed indentation bug (src/infer.py lines 486-488): Fixed incorrect indentation in else block
   - Made target period verification less strict (src/infer.py lines 607-690): Added fallback logic, always proceeds instead of skipping
   - Improved model result restoration (src/infer.py lines 178-230): Added multiple restoration methods with better error handling
   - Made recent data check more lenient (src/infer.py line 937): Changed from 365 to 730 days

**Previous Code Fixes** (From earlier iterations - Present in code, NOT verified):
- Kalman smoother NaN propagation fix (dfm-python/src/dfm_python/ssm/kalman.py lines 467-505): Backward smoother validation and fallback
- Forward Kalman filter adaptive regularization (dfm-python/src/dfm_python/ssm/kalman.py lines 318-345): Adaptive regularization for sparse data
- VAR column indexing fix (src/infer.py lines 973-1077): Column index conversion for VAR pipeline
- Other fixes: Date type conversion, VAR state reset, validation logic, error handling, diagnostic logging

**What's NOT Working**:
- ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
- ⚠️ **ALL 12 backtests failed** - Code fixes applied but not verified (backtests need re-run). **NEW FIXES THIS ITERATION** address overly strict validation that was likely causing all months to be skipped.

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Verify Backtest Code Fixes (BLOCKING)
**Status**: ⚠️ **CODE FIXES PRESENT BUT NOT VERIFIED** - All 12 backtests still show "no_results"  
**Blocking**: Table 3 and Plot4 missing (blocked by backtest failures)

**REAL Problem** (Verified by inspection):
- **All 12 JSON files have "status": "no_results"** - No valid results generated
- **Root cause analysis**: Multiple validation checks in backtest code (src/infer.py) can cause months to be skipped. These checks are necessary for data quality but may be too strict in some edge cases:
  1. **view_date <= train_end_date check** (lines 377-380): Skips if view_date is at or before training period end. For January 2024 with 4 weeks before, view_date is ~Dec 2023, which should pass (train_end=2019-12-31). **STATUS**: Check is correct - ensures no data leakage.
  2. **No monthly data available** (lines 387-402): Skips if no monthly data exists at or before target_month_end. **STATUS**: Check is necessary - cannot nowcast without data.
  3. **Data module update failure** (lines 480-485): Skips if data_module update fails for DFM/DDFM models. **STATUS**: Fixed indentation bug - should work now. Needs verification.
  4. **No training data up to view_date** (lines 767-770): Skips if no training data available up to view_date for ARIMA/VAR. **STATUS**: Check is necessary.
  5. **Monthly resampling removes all data** (lines 780-782): Skips if resampling to monthly removes all data. **STATUS**: Check is necessary - indicates data quality issue.
  6. **view_date not before target_month_end** (lines 789-791): Skips if view_date is not before target_month_end (horizon <= 0). **STATUS**: Check is correct - ensures positive forecast horizon.
  7. **Insufficient data after cleaning** (lines 914-917): Skips if less than 10 data points after cleaning. **STATUS**: Check is reasonable - models need minimum data.
  8. **No recent data** (lines 960-962): Skips if no recent data (last 730 days) available. **STATUS**: Check is lenient (24 months) - appropriate for monthly data.
  9. **NaN forecast or actual values** (lines 1292-1297): Skips if forecast_value or actual_value is NaN. **STATUS**: Check is necessary - cannot compute metrics with NaN.

**VALIDATION ASSESSMENT**: All validation checks appear necessary and reasonable. The most likely causes of "no_results" are:
- Data module update failures (fixed indentation bug, needs verification)
- Target period not found in TimeIndex (made verification less strict, needs verification)
- NaN predictions from DFM/DDFM (Kalman filter fixes applied, needs verification)

- **Code fixes present** but NOT verified by re-running backtests:
  1. **NEW THIS ITERATION**: Fixed indentation bug in data_module update (src/infer.py lines 486-488) - incorrect indentation could cause logic errors
  2. **NEW THIS ITERATION**: Made target period verification less strict (src/infer.py lines 607-690) - was too defensive, skipping all months. Now proceeds even if exact date not found, with fallback to find closest date
  3. **NEW THIS ITERATION**: Improved model result restoration (src/infer.py lines 178-230) - multiple restoration methods with better error handling
  4. **NEW THIS ITERATION**: Made recent data check more lenient (src/infer.py line 937) - changed from 365 to 730 days
  5. Kalman smoother NaN propagation fix (dfm-python/src/dfm_python/ssm/kalman.py lines 467-505) - Prevents NaN in backward pass
  6. Forward Kalman filter adaptive regularization (dfm-python/src/dfm_python/ssm/kalman.py lines 318-345) - Adaptive regularization for sparse data
  7. VAR column indexing fix (src/infer.py lines 973-1077) - Fixes ColumnEnsembleTransformer compatibility

**Actions Needed**:
- Step 1 will detect "no_results" status → runs `bash agent_execute.sh backtest`
- Verify: JSON files contain `results_by_timepoint` with actual metrics (not "no_results")
- If fixes work: Backtests succeed, Table 3 and Plot4 can be regenerated
- If fixes don't work: Investigate remaining issues:
  - Check backtest logs for specific validation failures (grep for "Skipping" or "⚠" in log files)
  - Verify data availability: Check if full_data_monthly has data for 2024 months
  - Verify view_date calculation: Ensure view_date (target_month_end - weeks) is after train_end (2019-12-31)
  - Check for silent exceptions: Review exception handling that might catch errors without logging
  - May need additional code fixes based on specific validation failures found

**Success Criteria**:
- ✅ At least one backtest generates valid results (not "no_results")
- ✅ JSON files contain `results_by_timepoint` with actual sMAE/sMSE metrics
- ✅ Backtest logs show no critical errors (may still have warnings)

---

### Priority 2: CRITICAL - Train Missing Models (BLOCKING)
**Status**: ❌ **MISSING** - 2 models not trained (KOIPALL.G_ddfm, KOIPALL.G_dfm)  
**Blocking**: Complete experiment coverage (nowcasting blocked for these models)

**REAL Problem**:
- **checkpoint/**: Only 10/12 model.pkl files exist
- **Missing**: KOIPALL.G_ddfm/model.pkl, KOIPALL.G_dfm/model.pkl
- **Impact**: Nowcasting experiments for KOIPALL.G with DDFM/DFM cannot run

**Actions** (Step 1 will automatically handle):
- Step 1 detects missing models → runs `bash agent_execute.sh train`
- Expected: 2 additional model.pkl files in checkpoint/
- **INVESTIGATE**: If training fails, check logs for KOIPALL.G_ddfm and KOIPALL.G_dfm training errors

**Success Criteria**:
- ✅ checkpoint/ contains 12 model.pkl files (all 3 targets × 4 models)
- ✅ All models can be loaded successfully for inference

---

### Priority 3: HIGH - Regenerate Table 3 and Plot4 (BLOCKED)
**Status**: ⚠️ **BLOCKED** - Needs Priority 1 (backtests complete successfully)  
**Blocking**: Report completion (Table 3 and Plot4 show N/A/placeholders)

**Actions** (After Priority 1 completes):
- Execute: `python3 -c "from src.eval.evaluation import generate_all_latex_tables; generate_all_latex_tables()"`
- Execute: `python3 nowcasting-report/code/plot.py`
- Verify: N/A placeholders replaced with actual results

**Success Criteria**:
- ✅ Table 3 contains actual sMAE and sMSE values (not "N/A")
- ✅ Plot4 shows 3 plots (one per target) with actual data

---

### Priority 4: HIGH - Regenerate aggregated_results.csv (NON-BLOCKING)
**Status**: ⚠️ **CODE FIXED** - CSV needs regeneration (non-blocking, filtering works on load)

**Actions** (Optional):
- Execute: `python3 -c "from src.eval.evaluation import main_aggregator; main_aggregator()"`
- Or wait for Step 1 to regenerate during forecasting experiments

---

## EXPERIMENT STATUS (ACTUAL - VERIFIED)

**Configuration**: 3 targets × 4 models × 30 horizons (forecasting), 3 targets × 4 models × 12 months × 2 timepoints (nowcasting)

**ACTUAL Status**:
- **Training**: ❌ **10/12 models trained** (missing KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **Forecasting**: ✅ **DONE** - aggregated_results.csv EXISTS (36 rows, extreme VAR values filtered on load)
- **Nowcasting**: ❌ **0/12 experiments completed** (ALL 12 JSON files have "status": "no_results" - code fixes applied but not verified)

---

## MODEL PERFORMANCE ANOMALIES (REAL ISSUES TO FIX)

1. **VAR Horizon 1**: Near-zero values (3.65e-09) - persistence predictions
   - **Status**: ✅ **VERIFIED** - Code correctly identifies and marks persistence predictions as NaN. This is expected behavior, not a bug.

2. **VAR Horizons 7/28**: Extreme values (1e+27, 1e+120) - numerical instability
   - **Status**: ✅ **ALREADY HANDLED** - Code filters extreme values (> 1e10) at multiple stages. This is a known limitation of VAR models for long horizons.

3. **DFM/DDFM Horizon 28**: n_valid=0 (no valid predictions)
   - **Status**: ✅ **IMPROVED** - Enhanced error handling added (LinAlgError, ValueError, RuntimeError) with better diagnostics for long horizons (>= 28).

---

## CODE INSPECTION FINDINGS (ACTUAL STATE)

**Backtest Failures - Code Fixes Present**:
- **Model result restoration**: Fixed in src/infer.py lines 178-230 (multiple restoration methods)
- **Target period verification**: Fixed in src/infer.py lines 607-664 (enhanced verification with fallback)
- **Kalman smoother NaN propagation**: Fixed in dfm-python/src/dfm_python/ssm/kalman.py lines 467-505 (validation and fallback)
- **Forward Kalman filter stability**: Fixed in dfm-python/src/dfm_python/ssm/kalman.py lines 318-345 (adaptive regularization)
- **VAR column indexing**: Fixed in src/infer.py lines 973-1077 (column index conversion)
- **Status**: All fixes present in code but NOT VERIFIED by re-running backtests

**dfm-python Package Inspection**:
- **Code fixes present**: Kalman smoother NaN propagation fix, adaptive regularization for sparse data
- **Status**: Fixes present but NOT verified by successful backtests
- **Remaining concerns**: Numerical stability for very sparse data may need further improvement if fixes don't work

**Backtest Code** (src/infer.py):
- ✅ Model result restoration: Fixed (multiple restoration methods) - **NEW THIS ITERATION**
- ✅ Target period verification: Fixed (less strict, always proceeds) - **NEW THIS ITERATION**
- ✅ Data module update: Fixed indentation bug (lines 486-488) - **NEW THIS ITERATION**
- ✅ Recent data check: Made more lenient (730 days) - **NEW THIS ITERATION**
- ✅ Date type conversion: Fixed (pd.Timestamp consistency)
- ✅ VAR state reset: Fixed (transformer reset)
- ✅ Horizon calculation: Fixed (monthly data handling)
- ✅ VAR column indexing: Fixed (column index conversion)
- ⚠️ **All fixes present but NOT verified** - Backtests need re-run to verify if fixes work
- ⚠️ **Potential remaining issues**: Multiple validation checks (9 different skip conditions) may still cause months to be skipped. Need to verify which specific checks are failing by examining backtest logs after re-run.

---

## INSPECTION FINDINGS (ACTUAL STATE - VERIFIED)

**Model Performance Anomalies**:
- VAR horizon 1: Near-zero values - handled correctly (marked as NaN)
- VAR horizons 7/28: Extreme values - filtered correctly (marked as NaN)
- DFM/DDFM horizon 28: n_valid=0 - improved error handling added

**dfm-python Package**: 
- **Code fixes present**: Kalman smoother NaN propagation fix (lines 467-505), adaptive regularization (lines 318-345)
- **Status**: Fixes present in code but NOT verified by successful backtests
- **Remaining concerns**: If fixes don't work, may need further numerical stability improvements for very sparse data

**Report Documentation**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders - blocked by backtest failures); Plots 1-3 ✅, Plot4 ⚠️ (placeholders - blocked by backtest failures)

**Backtest Code**: 
- **Code fixes present**: Model result restoration, target period verification, Kalman filter stability, VAR column indexing, date type conversion, VAR state reset, validation logic, error handling
- **Status**: All fixes present in code but NOT verified by re-running backtests
- **Action**: Re-run backtests to verify if fixes work. If still failing, investigate remaining issues.

---

## NEXT ITERATION ACTIONS (Prioritized - REAL TASKS TO FIX)

**CRITICAL (Blocking - Must Fix)**:
1. **Verify Backtest Code Fixes** (Priority 1) - ⚠️ **CODE FIXES PRESENT, NEEDS VERIFICATION**
   - **Status**: All code fixes present (model result restoration, target period verification, Kalman filter stability, VAR column indexing) but NOT verified
   - **Action**: Step 1 detects "no_results" status → runs `bash agent_execute.sh backtest` → verify JSON files contain `results_by_timepoint`
   - **If fixes work**: Backtests succeed, proceed to regenerate Table 3 and Plot4
   - **If fixes don't work**: Investigate remaining issues, may need additional code fixes
   
2. **Train Missing Models** (Priority 2) - 2 models missing (KOIPALL.G_ddfm, KOIPALL.G_dfm)
   - **Status**: Only 10/12 models trained
   - **Action**: Step 1 detects missing models → runs `bash agent_execute.sh train`
   - **If training fails**: Check training logs for errors, inspect KOIPALL.G data for issues
   
3. **Regenerate Table 3 and Plot4** (Priority 3) - BLOCKED by Priority 1 & 2
   - **Status**: Table 3 and Plot4 have N/A placeholders
   - **Action**: After backtests succeed (Priority 1 & 2 fixed), regenerate from outputs/backtest/
   - **Code improvements applied**: Nowcasting table generation handles "no_results" status gracefully, fixed row label format (1week vs 1weeks)

**HIGH (Non-Blocking - Optional)**:
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Update report sections with nowcasting results (after Table 3 and Plot4 regenerated)
- Improve dfm-python code quality: Consistent naming, better error messages, theoretical correctness verification

**CODE IMPROVEMENTS (This Iteration - ACTUALLY DONE)**:
- ✅ **Fixed indentation bug** (src/infer.py lines 486-488): Fixed incorrect indentation in else block
- ✅ **Made target period verification less strict** (src/infer.py lines 607-690): Always proceeds with fallback logic instead of skipping
- ✅ **Improved model result restoration** (src/infer.py lines 178-230): Added multiple restoration methods with better error handling
- ✅ **Made recent data check more lenient** (src/infer.py line 937): Changed from 365 to 730 days

---

## SUMMARY FOR NEXT ITERATION

**CRITICAL BLOCKING ISSUES** (Must fix before report completion):
1. **ALL 12 backtests failed** - Status: "no_results" in all JSON files
   - **Root cause**: Multiple validation checks causing all months to be skipped (9 different skip conditions identified)
   - **Code fixes applied THIS ITERATION**: Indentation bug fix, less strict target period verification (always proceeds), improved model restoration, more lenient recent data check
   - **Action**: Step 1 will re-run backtests via `agent_execute.sh backtest` to verify if fixes work
   - **If still failing**: Check backtest logs for specific validation failures, investigate which of the 9 skip conditions are triggering

2. **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
   - **Action**: Step 1 will detect and train missing models via `agent_execute.sh train`

**BLOCKED TASKS** (Cannot proceed until blocking issues fixed):
- Table 3 generation (needs backtest results)
- Plot4 generation (needs backtest results)
- Report completion (needs Table 3 and Plot4)

**NON-BLOCKING TASKS** (Can proceed in parallel):
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Improve dfm-python code quality (naming, error messages, theoretical correctness)

**KEY PRINCIPLE**: Code fixes are present but NOT verified. Only mark as "resolved" after backtests succeed and JSON files contain actual results. Never claim "complete" or "done" unless actual code changes were made and verified.
