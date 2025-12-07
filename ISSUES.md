# Issues and Action Plan

## EXECUTIVE SUMMARY

**CRITICAL BLOCKING ISSUES**:
1. ❌ **ALL 12 backtests failed** - Code fixes applied but NOT VERIFIED (needs re-run via Step 1)
2. ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained (needs training via Step 1)

**NEXT STEPS**:
- Step 1 will automatically detect and run missing experiments (training + backtesting)
- After experiments complete, verify results and fix any remaining code issues
- Generate missing tables/plots (Table 3, Plot4) after backtests succeed

**KEY PRINCIPLE**: Find and FIX real problems in code, don't just document them. Only mark as "resolved" after actual code changes are made.

---

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK**:
- **checkpoint/**: 10/12 models trained ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: 12 JSON files with "status": "no_results" ❌ (ALL backtests failed - code fixes applied, needs re-run)
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows, extreme VAR values filtered on load)
- **nowcasting-report/**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**This Iteration Work** (Code fixes applied, NOT verified by re-running backtests):
1. **Type consistency fix** (src/infer.py): Convert `target_month_end` to `pd.Timestamp` for consistent date comparisons
2. **VAR internal state reset** (src/infer.py): Enhanced VAR forecaster reset to reset internal transformers_ when refitting
3. **DFM/DDFM error handling** (src/models.py): Added LinAlgError handling and improved error messages for horizon 28 failures
4. **Validation logic improvements** (src/infer.py): Fixed last_valid_idx check, simplified available data check, improved horizon calculation
5. **Diagnostic logging** (src/infer.py): Added summary messages to help identify why backtests fail

**What's NOT Working**:
- ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
- ❌ **ALL 12 backtests failed** - Code fixes applied but not verified (backtests need re-run)

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Inspect and Fix Backtest Failures (BLOCKING)
**Status**: ❌ **ALL 12 BACKTESTS FAILED** - "no_results" status  
**Blocking**: Table 3 and Plot4 (all nowcasting results missing)

**REAL Problem**:
- **ALL 12 JSON files** have `"status": "no_results"` with error "No valid results generated for any time point"
- **Root causes IDENTIFIED and fixed in code** (this iteration):
  1. Date type mismatch: target_month_end vs pd.Timestamp comparisons - FIXED
  2. VAR column index mismatch: Pipeline fitted with integer indices but data has string names - FIXED
  3. Validation logic issues: last_valid_idx check, available data check - FIXED
  4. Horizon calculation edge cases: months_ahead=0, negative values - FIXED
  5. DFM/DDFM TimeIndex: Resample to monthly before creating data_module - FIXED (previous iteration)

**Code Fixes Applied** (verified in code, NOT verified by re-running backtests):
- Date type conversion: Convert target_month_end to pd.Timestamp for consistent comparisons
- VAR internal state reset: Reset forecaster transformers_ when refitting
- Validation logic: Fixed last_valid_idx check, simplified available data check
- Horizon calculation: Improved logic for edge cases
- Diagnostic logging: Added summary messages

**Actions Needed**:
1. **VERIFY**: Re-run backtests → Step 1 detects "no_results" → runs `bash agent_execute.sh backtest`
2. **VALIDATE**: JSON files contain `results_by_timepoint` with actual metrics (after re-run)

**Success Criteria**:
- ✅ At least one backtest generates valid results (not "no_results")
- ✅ JSON files contain `results_by_timepoint` with actual metrics
- ✅ All 12 months (2024-01 to 2024-12) have results for both timepoints (4 weeks, 1 week)

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

## CODE INSPECTION TASKS (If backtests still fail after re-run)

**Backtest Code Inspection** (src/infer.py):
- Verify all validation checks are not too strict (may be skipping valid months unnecessarily)
- Check prediction extraction logic handles all DataFrame/Series formats
- Verify actual value lookup finds values correctly
- Check if horizon calculation for ARIMA/VAR is correct for monthly data
- Verify DFM/DDFM TimeIndex handling works correctly
- Check if all date type conversions are consistent (pd.Timestamp vs datetime)

---

## INSPECTION FINDINGS (ACTUAL STATE)

**Model Performance Anomalies**:
- VAR horizon 1: Near-zero values - handled correctly (marked as NaN)
- VAR horizons 7/28: Extreme values - filtered correctly (marked as NaN)
- DFM/DDFM horizon 28: n_valid=0 - improved error handling added

**dfm-python Package**: No critical issues found (from previous iteration)

**Report Documentation**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**Backtest Code**: 
- Code fixes present (date type conversion, VAR state reset, validation logic, error handling)
- **CRITICAL**: All 12 backtests still failing with "no_results" - fixes applied but NOT VERIFIED (backtests not re-run)
- **Action**: Re-run backtests to verify if fixes work. If still failing, inspect backtest logs to identify remaining root causes

---

## NEXT ITERATION ACTIONS (Prioritized - REAL TASKS TO FIX)

**CRITICAL (Blocking - Must Fix)**:
1. **Verify Backtest Fixes** (Priority 1) - Code fixes applied but NOT VERIFIED
   - **Status**: All 12 backtests show "no_results" - code fixes applied but not verified
   - **Action**: Step 1 detects "no_results" → runs `bash agent_execute.sh backtest` → verify JSON files contain `results_by_timepoint` with actual metrics
   - **If still failing after re-run**: Inspect latest backtest logs to identify remaining root causes
   
2. **Train Missing Models** (Priority 2) - 2 models missing (KOIPALL.G_ddfm, KOIPALL.G_dfm)
   - **Status**: Only 10/12 models trained
   - **Action**: Step 1 detects missing models → runs `bash agent_execute.sh train`
   - **If training fails**: Check training logs for errors, inspect KOIPALL.G data for issues
   
3. **Regenerate Table 3 and Plot4** (Priority 3) - BLOCKED by Priority 1
   - **Status**: Table 3 and Plot4 have N/A placeholders
   - **Action**: After backtests succeed, regenerate from outputs/backtest/

**HIGH (Non-Blocking - Optional)**:
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Update report sections with nowcasting results (after Table 3 and Plot4 regenerated)
