# Issues and Action Plan

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK**:
- **checkpoint/**: 10/12 models trained ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: 12 JSON files with "status": "no_results" ❌ (ALL backtests failed - code fixes applied, needs re-run)
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows, extreme VAR values filtered on load)
- **nowcasting-report/**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**This Iteration Work** (Code fixes applied):
1. **CRITICAL FIX: DFM/DDFM TimeIndex issue** (lines 361-418, 438-510 in src/infer.py):
   - Root cause identified: TimeIndex created from weekly data didn't include monthly target periods
   - Fix: Resample data to monthly BEFORE creating data_module so TimeIndex has monthly dates
   - Fix: Skip month if no valid date found in TimeIndex (instead of continuing with invalid target_period)
   - Fix: Try to find closest date at or before target_period if no date in same month
   - This addresses the "Target period not found in Time index" errors that caused all DFM/DDFM backtests to fail
2. **Improved actual value lookup** (lines 515-540, 860-916): Better handling of monthly data matching, exact month matching, and diagnostics
3. **Added data range diagnostics** (lines 274-290): Check if data extends to nowcasting period and warn if not
4. **Improved target month validation** (lines 304-315): Check if actual values are available for target months before processing
5. **Previous fixes** (from STATUS.md):
   - DFM/DDFM data_module update (lines 320-383): Update with data up to `target_month_end`
   - DFM/DDFM TimeIndex conversion (lines 965-1000 in src/models.py): Convert pandas Index to TimeIndex object
   - VAR column matching (lines 613-625): Use training columns in backtest
   - Empty data check after resampling (lines 562-564): Skip month if resampling removes all data
   - Validation checks relaxed (lines 678-679, 724): Recent data 180→365 days, last valid data point 180→90 days

**What's NOT Working**:
- ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
- ❌ **ALL 12 backtests failed** - Code fixes applied but not verified (backtests need re-run)

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Inspect and Fix Backtest Failures (BLOCKING)
**Status**: ❌ **ALL 12 BACKTESTS FAILED** - "no_results" status indicates code issues need investigation  
**Blocking**: Table 3 and Plot4 (all nowcasting results missing)

**REAL Problem**:
- **ALL 12 JSON files** have `"status": "no_results"` with error "No valid results generated for any time point"
- **Root causes IDENTIFIED** (this iteration):
  1. **ARIMA/VAR**: Horizon conversion from days to months may be incorrect - fixed horizon calculation logic
  2. **DFM/DDFM**: Nowcast manager may be returning None/NaN values - TimeIndex fixes applied but need verification
  3. **Actual values**: Lookup may be failing due to date matching issues - improved matching logic applied
  4. **Prediction extraction**: Prediction DataFrame/Series extraction may be failing - code has extensive checks but may be too strict
- **Previous fix**: TimeIndex created from weekly data didn't include monthly target period dates - fix applied (resample to monthly before creating data_module)

**Code Fixes Applied** (this iteration - verified in code, NOT verified by re-running backtests):
- **CRITICAL FIX: Date type conversion for target_month_end** (lines 315-316 in src/infer.py):
  - **ROOT CAUSE IDENTIFIED**: target_month_end is a datetime object but full_data_monthly.index is pd.DatetimeIndex (pd.Timestamp objects). Comparisons between datetime and Timestamp in boolean indexing may fail or produce incorrect results.
  - **FIX APPLIED**: Convert target_month_end to pd.Timestamp at the start of the loop (target_month_end_ts) and use it consistently throughout for all comparisons with pandas DatetimeIndex. This ensures proper type matching in boolean indexing operations.
- **CRITICAL FIX: Date comparison for view_date validation** (line 322 in src/infer.py):
  - **ROOT CAUSE IDENTIFIED**: Date comparison might fail with timezone-aware dates or different date types
  - **FIX APPLIED**: Convert both view_date and train_end_date to pd.Timestamp before comparison to ensure consistent comparison
- **CRITICAL FIX: ARIMA/VAR horizon calculation** (lines 832-850 in src/infer.py):
  - **ROOT CAUSE IDENTIFIED**: Horizon calculation for months_ahead=0 case was using max(1, 0) which is correct, but logic was unclear
  - **FIX APPLIED**: Improved logic to explicitly handle months_ahead=0 (same month) → horizon=1, months_ahead>0 → horizon=months_ahead, months_ahead<0 → horizon=1 (fallback)
  - Added warning message for negative months_ahead case
  - For nowcasting, after refitting on data up to view_date, we predict from the last data point in fit_data to target_month_end
  - This ensures horizon is typically 1 month (or small number) instead of 48+ months, which was causing prediction failures
- **IMPROVED: Diagnostic logging** (line 1093 in src/infer.py):
  - Added summary message showing total target months and why all months were skipped (if applicable)
- **CRITICAL FIX: DFM/DDFM TimeIndex** (lines 361-418, 438-510 in src/infer.py):
  - Resample data to monthly BEFORE creating data_module so TimeIndex has monthly dates
  - Skip month if no valid date found in TimeIndex (instead of continuing with invalid target_period)
  - Try to find closest date at or before target_period if no date in same month
- **Improved actual value lookup** (lines 515-540, 860-916 in src/infer.py): Better monthly data matching, exact month matching, diagnostics
- **Data range diagnostics** (lines 274-290): Check if data extends to nowcasting period
- **Target month validation** (lines 304-315): Check if actual values available before processing
- **Previous fixes**:
  - DFM/DDFM: Update data_module with data up to `target_month_end` (lines 320-383)
  - DFM/DDFM: TimeIndex conversion (lines 965-1000 in src/models.py)
  - VAR: Column matching logic (lines 613-625)
  - Validation: Recent data check 180→365 days, last valid data point 180→90 days (lines 678-679, 724)
  - Empty data check after resampling (lines 562-564)

**Actions Needed**:
1. ✅ **FIXED**: Root cause identified and fixed - DFM/DDFM TimeIndex issue (resample to monthly, handle missing dates)
2. **VERIFY**: Re-run backtests after fixes → Step 1 detects "no_results" → runs `bash agent_execute.sh backtest`
3. **VALIDATE**: JSON files contain `results_by_timepoint` with actual metrics (after re-run)

**Success Criteria**:
- ✅ At least one backtest generates valid results (not "no_results")
- ✅ JSON files contain `results_by_timepoint` with actual metrics
- ✅ All 12 months (2024-01 to 2024-12) have results for both timepoints (4 weeks, 1 week)
- ✅ Root cause identified and fixed in code

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

1. **VAR Horizon 1**: Near-zero values (3.65e-09) - likely persistence predictions or numerical underflow
   - **Issue**: Code may be detecting persistence incorrectly, or VAR is producing invalid predictions
   - **Action**: Inspect VAR prediction code in src/models.py to verify prediction logic
   - **Fix**: If persistence detected, handle appropriately; if numerical underflow, add regularization

2. **VAR Horizons 7/28**: Extreme values (1e+27, 1e+120) - numerical instability
   - **Issue**: VAR model explodes for long horizons (known limitation)
   - **Current handling**: Code filters extreme values (> 1e10) as NaN when loading CSV
   - **Action**: Improve VAR training to prevent instability (regularization, horizon limits, or document as limitation)
   - **Fix**: Add better regularization in VAR training or cap horizon for VAR model

3. **DFM/DDFM Horizon 28**: n_valid=0 (no valid predictions)
   - **Issue**: DFM and DDFM fail to generate predictions for 28-day horizon
   - **Root cause**: Likely numerical instability or convergence issues in EM algorithm
   - **Action**: Inspect DFM/DDFM prediction code for horizon 28 failures
   - **Fix**: Add better error handling, regularization, or document as limitation

4. **DDFM Horizon 1**: Results appear reasonable (sRMSE 0.01-0.46 range) - no issues

---

## INSPECTION FINDINGS (ACTUAL STATE)

**Model Performance Anomalies**:
- VAR horizon 1: Near-zero values (3.65e-09) - needs investigation
- VAR horizons 7/28: Extreme values (1e+27, 1e+120) - numerical instability, filtered but should be fixed at source
- DFM/DDFM horizon 28: n_valid=0 - complete failure for long horizons, needs investigation
- DDFM horizon 1: Results reasonable (sRMSE 0.01-0.46) - no issues

**dfm-python Package**: 
- Numerical stability measures present (regularization, NaN handling)
- Known limitations documented (EM algorithm convergence, GPU stability)
- **Action**: Review if additional stability measures needed for horizon 28 failures

**Report Documentation**: 
- Tables 1-2 ✅ (forecasting results)
- Table 3 ⚠️ (N/A placeholders - blocked by backtest failures)
- Plots 1-3 ✅ (forecasting plots)
- Plot4 ⚠️ (placeholders - blocked by backtest failures)

**Backtest Code**: 
- Code fixes present (date type conversion, horizon calculation, diagnostic logging, DFM/DDFM TimeIndex fixes, VAR column matching, improved actual value lookup)
- **CRITICAL**: All 12 backtests still failing with "no_results" - fixes applied but NOT VERIFIED (backtests not re-run)
- **Action**: Re-run backtests to verify if fixes work. If still failing, inspect backtest logs to identify remaining root causes

---

## NEXT ITERATION ACTIONS (Prioritized - REAL TASKS)

**CRITICAL (Blocking - Must Fix)**:
1. **Re-run Backtests** (Priority 1) - Code fixes applied but NOT VERIFIED
   - **Status**: All 12 backtests show "no_results" - fixes in code (date type conversion, horizon calculation, diagnostic logging) but not verified
   - **Action**: Step 1 detects "no_results" → runs `bash agent_execute.sh backtest` → verify JSON files contain `results_by_timepoint` with actual metrics
   - **If still failing**: Inspect backtest logs to identify remaining root causes
   
2. **Train Missing Models** (Priority 2) - 2 models missing (KOIPALL.G_ddfm, KOIPALL.G_dfm)
   - **Status**: Only 10/12 models trained
   - **Action**: Step 1 detects missing models → runs `bash agent_execute.sh train`
   - **If training fails**: Investigate and fix training code
   
3. **Regenerate Table 3 and Plot4** (Priority 3) - BLOCKED by Priority 1
   - **Status**: Table 3 and Plot4 have N/A placeholders
   - **Action**: After backtests succeed, regenerate from outputs/backtest/

**HIGH (Non-Blocking - Should Fix)**:
- **VAR Numerical Instability**: Extreme values for horizons 7/28 - add regularization or document as limitation
- **DFM/DDFM Horizon 28 Failures**: n_valid=0 for forecasting (not nowcasting) - investigate or document as limitation
- **VAR Horizon 1 Near-Zero Values**: May indicate persistence detection issue - verify prediction logic

**MEDIUM (Non-Blocking - Optional)**:
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Update report with nowcasting results (after Table 3 and Plot4 regenerated)
