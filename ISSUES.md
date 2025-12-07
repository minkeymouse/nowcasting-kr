# Issues and Action Plan

## EXECUTIVE SUMMARY

**CRITICAL BLOCKING ISSUES**:
1. ❌ **ALL 12 backtests failed** - All JSON files have "status": "no_results". Code fixes present but NOT VERIFIED (backtests need re-run)
2. ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained (needs training via Step 1)
3. ⚠️ **Numerical instability concerns** - Backtest logs show NaN warnings in Kalman filter backward pass, suggesting fixes may need verification

**CODE FIXES PRESENT IN CODEBASE** (Cannot verify when added):
- Validation check numbering ([CHECK 1] through [CHECK 16]) in backtest skip conditions
- Kalman filter NaN handling (early termination, input validation, forward-only fallback)
- Backtest validation logic improvements (less strict checks, better error handling)
- Model result restoration improvements
- VAR column indexing fixes

**KEY PRINCIPLE**: Code fixes are present but NOT verified. Only mark as "resolved" after backtests succeed and JSON files contain actual results.

---

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK**:
- **checkpoint/**: 10/12 models trained ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: 12 JSON files with "status": "no_results" ❌ (ALL backtests failed - code fixes applied, needs re-run)
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows, extreme VAR values filtered on load)
- **nowcasting-report/**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**What's NOT Working**:
- ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
- ⚠️ **ALL 12 backtests failed** - Code fixes present but NOT verified (backtests need re-run)

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Verify Backtest Code Fixes (BLOCKING)
**Status**: ⚠️ **CODE FIXES PRESENT BUT NOT VERIFIED** - All 12 backtests still show "no_results"  
**Blocking**: Table 3 and Plot4 missing (blocked by backtest failures)

**REAL Problem** (Verified by inspection):
- **All 12 JSON files have "status": "no_results"** - No valid results generated
- **Root cause analysis**: Multiple validation checks in backtest code (src/infer.py) can cause months to be skipped:
  1. **view_date <= train_end_date check**: Skips if view_date is at or before training period end
  2. **No monthly data available**: Skips if no monthly data exists at or before target_month_end
  3. **Data module update failure**: Skips if data_module update fails for DFM/DDFM models
  4. **No training data up to view_date**: Skips if no training data available for ARIMA/VAR
  5. **Monthly resampling removes all data**: Skips if resampling to monthly removes all data
  6. **view_date not before target_month_end**: Skips if horizon <= 0
  7. **Insufficient data after cleaning**: Skips if less than 10 data points after cleaning
  8. **No recent data**: Skips if no recent data (last 730 days) available
  9. **NaN forecast or actual values**: Skips if forecast_value or actual_value is NaN

**VALIDATION ASSESSMENT**: All validation checks appear necessary and reasonable. The most likely causes of "no_results" are:
- Data module update failures
- Target period not found in TimeIndex
- NaN predictions from DFM/DDFM (Kalman filter issues)

**Actions Needed**:
- Step 1 will detect "no_results" status → runs `bash agent_execute.sh backtest`
- Verify: JSON files contain `results_by_timepoint` with actual metrics (not "no_results")
- If fixes work: Backtests succeed, Table 3 and Plot4 can be regenerated
- If fixes don't work: Investigate remaining issues:
  - Check backtest logs for specific validation failures (grep for "[CHECK N]" or "Skipping" in log files)
  - Verify data availability: Check if full_data_monthly has data for 2024 months
  - Verify view_date calculation: Ensure view_date (target_month_end - weeks) is after train_end (2019-12-31)
  - Check for silent exceptions: Review exception handling that might catch errors without logging

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
- **Validation check numbering**: Present in code ([CHECK 1] through [CHECK 16])
- **Model result restoration**: Fixed in src/infer.py lines 178-230 (multiple restoration methods)
- **Target period verification**: Fixed in src/infer.py lines 607-664 (enhanced verification with fallback)
- **Kalman smoother NaN propagation**: Fixed in dfm-python/src/dfm_python/ssm/kalman.py lines 467-505 (validation and fallback)
- **Forward Kalman filter stability**: Fixed in dfm-python/src/dfm_python/ssm/kalman.py lines 318-345 (adaptive regularization)
- **VAR column indexing**: Fixed in src/infer.py lines 973-1077 (column index conversion)
- **Status**: All fixes present in code but NOT VERIFIED by re-running backtests

**dfm-python Package Inspection**:
- **Code fixes present**: Kalman smoother NaN propagation fix (lines 467-505), adaptive regularization for sparse data (lines 318-345)
- **Status**: Fixes present but NOT verified by successful backtests
- **INSPECTION FINDING**: Backtest logs show NaN warnings in backward pass, suggesting fixes may not fully resolve numerical instability
- **REAL PROBLEM**: Backward pass may need additional regularization or early termination when NaN detected
- **ACTION NEEDED**: If backtests still fail, investigate backward pass NaN propagation - may need stronger regularization, different numerical approach, or early termination strategy

**Backtest Code** (src/infer.py):
- ✅ Validation check numbering: Present ([CHECK 1] through [CHECK 16])
- ✅ Model result restoration: Fixed (multiple restoration methods)
- ✅ Target period verification: Fixed (less strict, always proceeds)
- ✅ Data module update: Fixed indentation bug
- ✅ Recent data check: Made more lenient (730 days)
- ✅ Date type conversion: Fixed (pd.Timestamp consistency)
- ✅ VAR state reset: Fixed (transformer reset)
- ✅ Horizon calculation: Fixed (monthly data handling)
- ✅ VAR column indexing: Fixed (column index conversion)
- ⚠️ **All fixes present but NOT verified** - Backtests need re-run to verify if fixes work
- ⚠️ **Potential remaining issues**: Multiple validation checks (9 different skip conditions) may still cause months to be skipped. Need to verify which specific checks are failing by examining backtest logs after re-run.

---

## NEXT ITERATION ACTIONS (Prioritized - REAL TASKS TO FIX)

**CRITICAL (Blocking - Must Fix)**:
1. **Verify Backtest Code Fixes** (Priority 1) - ⚠️ **CODE FIXES PRESENT, NEEDS VERIFICATION**
   - **Status**: All code fixes present but NOT verified
   - **Action**: Step 1 detects "no_results" status → runs `bash agent_execute.sh backtest` → verify JSON files contain `results_by_timepoint`
   - **If fixes work**: Backtests succeed, proceed to regenerate Table 3 and Plot4
   - **If fixes don't work**: Investigate remaining issues:
     - Check backtest logs for specific validation failures (grep for "[CHECK N]" or "Skipping" in log files)
     - **INVESTIGATE NUMERICAL INSTABILITY**: If NaN warnings persist in Kalman filter backward pass, may need additional fixes in dfm-python/src/dfm_python/ssm/kalman.py:
       - Add stronger regularization in backward pass
       - Implement early termination when NaN detected in backward pass
       - Consider different numerical approach for backward pass (e.g., use forward-only prediction if backward fails)
     - May need additional code fixes based on specific validation failures found
   
2. **Train Missing Models** (Priority 2) - 2 models missing (KOIPALL.G_ddfm, KOIPALL.G_dfm)
   - **Status**: Only 10/12 models trained
   - **Action**: Step 1 detects missing models → runs `bash agent_execute.sh train`
   - **If training fails**: Check training logs for errors, inspect KOIPALL.G data for issues
   
3. **Regenerate Table 3 and Plot4** (Priority 3) - BLOCKED by Priority 1 & 2
   - **Status**: Table 3 and Plot4 have N/A placeholders
   - **Action**: After backtests succeed (Priority 1 & 2 fixed), regenerate from outputs/backtest/

**HIGH (Non-Blocking - Optional)**:
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Update report sections with nowcasting results (after Table 3 and Plot4 regenerated)
- Improve dfm-python code quality: Consistent naming, better error messages, theoretical correctness verification

---

## SUMMARY FOR NEXT ITERATION

**CRITICAL BLOCKING ISSUES** (Must fix before report completion):
1. **ALL 12 backtests failed** - Status: "no_results" in all JSON files
   - **Root cause**: Multiple validation checks causing all months to be skipped (9 different skip conditions identified)
   - **Code fixes applied**: Validation check numbering, less strict validation, improved model restoration, Kalman filter fixes
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
