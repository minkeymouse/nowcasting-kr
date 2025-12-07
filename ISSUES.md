# Issues and Action Plan

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK** (Verified by inspection):
- **checkpoint/**: **10 model.pkl files** - **10/12 models trained** ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm) - **REAL PROBLEM: 2 models not trained**
- **outputs/backtest/**: **12 JSON files with "status": "no_results"** - **0/12 nowcasting experiments completed successfully** ❌ (ALL models failed: ARIMA/VAR/DFM/DDFM) - **REAL PROBLEM: All backtests failing**
- **outputs/experiments/aggregated_results.csv**: **EXISTS** (36 rows) - Contains extreme VAR values - **CODE FIXED, CSV NEEDS REGENERATION** (non-blocking, filtering works on load)
- **nowcasting-report/tables/**: 3 tables generated - Table 2 shows VAR-1 as N/A, Table 3 has N/A placeholders
- **nowcasting-report/images/**: 7 plots generated - Plot4 has placeholders
- **data/data.csv**: **VERIFIED** - Contains data up to 2025-11-28, so 2024 data exists

**CRITICAL FINDINGS**:
1. **2 models missing** (KOIPALL.G_ddfm, KOIPALL.G_dfm) - Training incomplete for these models
2. **All 12 backtests failed** with "status": "no_results" - Systematic failure affecting all models
3. **Data availability confirmed** - Data file contains 2024 dates, so data availability is not the issue
4. **Code fixes applied** but backtests still failing - Indicates validation logic or data filtering issue

**This Iteration Work**:
- **Status assessment** - Verified actual state: 10/12 models trained, all 12 backtests failed
- **Code verification** - Confirmed code fixes from previous iteration are present in codebase
- **Documentation** - Updated ISSUES.md to reflect honest current state

**Code Fixes from Previous Iteration** (Verified present in code):
- **src/infer.py**: Code fixes are present but backtests still failing
  - ✅ **PRESENT**: Horizon conversion fix at line 577 (`horizon_periods = max(1, round(horizon_days / 7.0))`)
  - ✅ **PRESENT**: Relaxed validation checks (180 days) at lines 515, 553
  - ✅ **PRESENT**: Debug logging throughout validation checks
  - ⚠️ **BUT INEFFECTIVE**: All 12 backtests still failing with "no_results" despite fixes

**What's NOT Working (REAL BLOCKING ISSUES)**:
- ❌ **ALL 12 nowcasting experiments FAILED** - All JSON files have "status": "no_results"
- ❌ **Error**: "No valid results generated for any time point" - All months are being skipped or failing
- ⚠️ **Code fixes applied but didn't resolve issue** - Indicates deeper problem needs investigation

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Fix Backtest Validation Logic (BLOCKING) - ⚠️ CODE FIXES PRESENT BUT INEFFECTIVE
**Status**: ⚠️ **CODE FIXES PRESENT BUT INEFFECTIVE** - Fixes are in code but backtests still failing  
**Blocking**: Table 3 and Plot4 (all nowcasting results missing)

**REAL Problem** (Verified by inspection):
- **ALL 12 JSON files** in `outputs/backtest/` have `"status": "no_results"` with error "No valid results generated for any time point"
- **Data confirmed available**: data/data.csv contains dates up to 2025-11-28, so 2024 data exists
- **Code fixes present**: Horizon conversion, relaxed validation, debug logging are all in code
- **But still failing**: Fixes didn't resolve the issue - need to identify why

**CODE FIXES FROM PREVIOUS ITERATION** (Verified present but ineffective):
1. ✅ **PRESENT**: Convert horizon_days to horizon_periods for weekly data (line 577)
   - Fix: `horizon_periods = max(1, round(horizon_days / 7.0))` - converts 28 days to 4 weeks
   - But backtests still failing
2. ✅ **PRESENT**: Relaxed recent data check from 90 days to 180 days (line 515)
3. ✅ **PRESENT**: Relaxed days_since_last check from 120 days to 180 days (line 553)
4. ✅ **PRESENT**: Debug logging for view_date validation, data availability, horizon calculation
5. ✅ **PRESENT**: Enhanced skip messages showing specific validation check that failed

**NEXT STEPS** (Investigation needed):
1. **Check log files** - Review `outputs/backtest/*.log` to see which validation check is actually failing
2. **Identify root cause** - Code fixes are present but ineffective, suggesting different issue
3. **Fix root cause** - Address the actual problem (likely different from what previous fixes addressed)
4. **Re-run backtests** - Step 1 will automatically run `bash agent_execute.sh backtest` after fix
5. **Verify fixes** - Check if backtests now generate valid results (not "no_results")

**Success Criteria**:
- ✅ At least one backtest generates valid results (not "no_results")
- ✅ JSON files contain `results_by_timepoint` with actual metrics
- ✅ All 12 months (2024-01 to 2024-12) have results for both timepoints (4 weeks, 1 week)

**Current Status**: ⚠️ **CODE FIXES PRESENT BUT INEFFECTIVE** - Need to identify why fixes didn't work and address root cause

---

### Priority 2: CRITICAL - Train Missing Models (BLOCKING)
**Status**: ❌ **MISSING** - 2 models not trained  
**Blocking**: Complete experiment coverage

**REAL Problem** (Verified by inspection):
- **checkpoint/** contains only **10/12 model.pkl files**
- **Missing models**: KOIPALL.G_ddfm, KOIPALL.G_dfm
- Training is incomplete - these models need to be trained before backtesting

**Actions** (Step 1 will automatically handle):
1. Step 1 checks `checkpoint/` → detects missing models (KOIPALL.G_ddfm, KOIPALL.G_dfm)
2. Step 1 automatically runs: `bash agent_execute.sh train` (which calls `run_train.sh`)
3. Expected output: 2 additional model.pkl files in `checkpoint/KOIPALL.G_ddfm/` and `checkpoint/KOIPALL.G_dfm/`

**Success Criteria**:
- ✅ `checkpoint/` contains 12 model.pkl files (all 3 targets × 4 models)
- ✅ Both KOIPALL.G_ddfm and KOIPALL.G_dfm models exist and are valid
- ✅ Training completes without errors (check logs)

**Current Status**: ❌ **MISSING** - 2 models need training (will be handled by Step 1)

---

### Priority 3: CRITICAL - Re-run Nowcasting Experiments (BLOCKED)
**Status**: ⚠️ **BLOCKED** - Needs Priority 1 and Priority 2 fixes first  
**Blocking**: Table 3 and Plot4 (all nowcasting results missing)

**REAL Problem**:
- All 12 backtest JSON files have "status": "no_results"
- Code fixes from previous iteration are present but didn't resolve the issue
- 2 models missing (KOIPALL.G_ddfm, KOIPALL.G_dfm) - training incomplete
- Backtests need re-run after Priority 1 and Priority 2 fixes

**Actions** (Step 1 will automatically handle after Priority 1):
1. Step 1 checks `outputs/backtest/` → detects JSON files with "no_results" status
2. After Priority 1 fix, Step 1 automatically runs: `bash agent_execute.sh backtest` (which calls `run_backtest.sh`)
3. Expected output: 12 JSON files in `outputs/backtest/{target}_{model}_backtest.json` (all models for 3 targets)

**Success Criteria**:
- ✅ Backtest completes without errors (check logs in outputs/backtest/)
- ✅ `outputs/backtest/` contains 12 JSON files with valid results (not "no_results")
- ✅ All JSON files contain `results_by_timepoint` with actual metrics
- ✅ All JSON files valid (test: `python3 -c "import json; json.load(open('outputs/backtest/KOEQUIPTE_arima_backtest.json'))"`)

**Current Status**: ⚠️ **BLOCKED** - Waiting for Priority 1 (identify why fixes didn't work) and Priority 2 (train missing models)

---

### Priority 4: HIGH - Regenerate Table 3 and Plot4 (BLOCKED)
**Status**: ⚠️ **CODE READY** - Needs Data  
**Blocking**: Report completion

**REAL Status**:
- Table 3 and Plot4 were generated but show **N/A/placeholders** (verified in nowcasting-report/tables/ and nowcasting-report/images/)
- Code is ready - just needs nowcasting results from `outputs/backtest/`
- **BLOCKED by Priority 1 and Priority 2** (all backtests must complete first)

**Actions** (After Priority 1 and Priority 2 complete):
1. Verify all 12 JSON files exist with valid results (not "no_results")
2. Regenerate Table 3: `python3 -c "from src.eval.evaluation import generate_all_latex_tables; generate_all_latex_tables()"`
3. Regenerate Plot4: `python3 nowcasting-report/code/plot.py`
4. Verify N/A placeholders replaced with actual results

**Success Criteria**:
- ✅ `outputs/backtest/` contains 12 JSON files (not just logs)
- ✅ All JSON files contain valid `results_by_timepoint` structure
- ✅ All timepoints (4weeks, 1weeks) have results for all 12 months (2024-01 to 2024-12)
- ✅ JSON files are valid (test: `python3 -c "import json; json.load(open('outputs/backtest/KOEQUIPTE_arima_backtest.json'))"`)

**Current Status**: ⚠️ **CODE READY** - **BLOCKED by Priority 1 and Priority 2** (backtests must complete first)

---

### Priority 5: HIGH - Regenerate aggregated_results.csv (NON-BLOCKING)
**Status**: ⚠️ **CODE FIXED** - CSV needs regeneration  
**Impact**: When regenerated, extreme VAR values will be marked as NaN (currently filtered on load, so non-blocking)

**REAL Problem**:
- `outputs/experiments/aggregated_results.csv` contains extreme VAR values (verified):
  - VAR horizon 7: sMSE = 5.746e+27, MSE = 1.100e+29
  - VAR horizon 28: sMSE = 1.414e+120, MSE = 2.707e+121
  - VAR horizon 1: sMSE = 3.654e-09 (persistence prediction - should be NaN)
- **Root cause**: VAR numerical instability for long horizons (known limitation)
- **FIX APPLIED**: Aggregation validates ALL metrics (MSE, MAE, RMSE). CSV loading also filters extreme values.

**Actions** (Optional - can wait for Step 1 or run manually):
1. **Option A**: Wait for Step 1 to regenerate during forecasting experiments (if forecasting is re-run)
2. **Option B**: Run manually: `python3 -c "from src.eval.evaluation import main_aggregator; main_aggregator()"`

**Success Criteria**:
- ✅ CSV regenerated with extreme values marked as NaN in ALL metrics
- ✅ All VAR horizon 7/28 values show NaN (not extreme numbers like 5.746e+27)
- ✅ All VAR horizon 1 values show NaN (not persistence values like 3.654e-09)
- ✅ Tables generated from CSV show "N/A" or "Unstable" instead of extreme values

**Current Status**: ⚠️ **CODE FIXED** - CSV needs regeneration (non-blocking, filtering works on load)

---

## EXPERIMENT STATUS (ACTUAL - VERIFIED)

**Configuration**:
- **Targets**: 3 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G)
- **Models**: 4 (ARIMA, VAR, DFM, DDFM)
- **Forecasting Horizons**: 1-30 days (table shows 1, 7, 30)
- **Nowcasting**: 12 months (2024-01 ~ 2024-12), 2 time points (4 weeks, 1 week before)

**ACTUAL Status** (Verified by inspection):
- **Training**: ⚠️ **10/12 models trained** (checkpoint/ has 10 model.pkl files, missing KOIPALL.G_ddfm and KOIPALL.G_dfm)
- **Forecasting**: ✅ **DONE** - aggregated_results.csv EXISTS with 36 rows (contains extreme VAR values, but filtering handles them when loading)
- **Nowcasting**: ❌ **0/12 experiments completed successfully** (ALL 12 JSON files exist but ALL have "status": "no_results" - systematic failure affecting all models)

**What Needs to Happen** (Step 1 will automatically handle):
1. ❌ **TRAIN MISSING MODELS** - checkpoint/ has only 10/12 model.pkl files (missing KOIPALL.G_ddfm, KOIPALL.G_dfm)
2. ⚠️ **INVESTIGATE BACKTEST FAILURES** - All 12 backtests failed with "no_results" despite code fixes being present. Code fixes are in codebase but ineffective - need to identify why and fix root cause
3. After fixing root cause, Step 1 will detect outputs/backtest/ has "no_results" → runs `bash agent_execute.sh backtest` → should populate outputs/backtest/ with 12 valid JSON files
4. After successful backtests, regenerate Table 3 from outputs/backtest/ (replace N/A with actual results)
5. After successful backtests, regenerate Plot4 from outputs/backtest/ (replace placeholders with actual plots)
6. Update report with actual nowcasting results

---

## MODEL PERFORMANCE ANOMALIES (CODE FIXED - CSV NEEDS REGENERATION)

1. **VAR Horizon 1 Suspicious Results**: 
   - **STATUS**: ✅ **FIXED** - Code marks persistence predictions as NaN
   - **REAL PROBLEM**: VAR horizon 1 shows extremely small values (e.g., 3.654e-09) in aggregated_results.csv
   - **Root cause**: VAR predicting persistence (last training value) - not data leakage
   - **FIX APPLIED**: `evaluate_forecaster()` marks metrics as NaN when VAR persistence is detected. Table generation also marks VAR-1 persistence values as NaN when loading CSV.
   - **Current Status**: Code fixed - CSV needs regeneration to apply fix (filtering works on load)

2. **VAR Numerical Instability (Horizons 7/28)**:
   - **STATUS**: ✅ **FIXED** - Aggregation validates all metrics
   - **REAL PROBLEM**: CSV contains extreme values (e.g., 5.746e+27, 1.414e+120)
   - **Root cause**: Known VAR limitation - becomes unstable for long horizons
   - **FIX APPLIED**: `aggregate_overall_performance()` validates ALL metrics (MSE, MAE, RMSE). CSV loading also filters extreme values.
   - **Current Status**: Code fixed - CSV needs regeneration (filtering works on load)

3. **DDFM Horizon 1 Results**:
   - **STATUS**: ✅ **NO ISSUES** - Results appear reasonable (sRMSE 0.01-0.46 range)
   - No anomalies detected - results are valid

---

## INSPECTION FINDINGS

**Model Performance Anomalies Inspection**:
- **STATUS**: ⚠️ **CODE FIXES PRESENT BUT INEFFECTIVE** - Code fixes from previous iteration are in codebase but backtests still failing
- **VAR horizon 1**: Code marks persistence predictions as NaN. Table 2 shows VAR-1 as N/A. CSV will show NaN when regenerated.
- **VAR horizons 7/28**: Code validates and marks extreme values (> 1e10) as NaN. CSV will show NaN when regenerated.
- **DDFM horizon 1**: Results appear reasonable (sRMSE 0.01-0.46 range) - no issues detected
- **Backtest failures**: All 12 JSON files have "no_results" status - systematic failure affecting all models despite code fixes

**dfm-python Package Inspection**:
- **STATUS**: ✅ **NO CRITICAL ISSUES FOUND** (from previous iteration)
- **Code Quality**: Clean structure, proper error handling, comprehensive validation
- **Numerical Stability**: Multiple stability measures (regularization, variance floors, NaN/Inf detection)
- **Theoretical Correctness**: EM algorithm, Kalman filtering, VAR estimation appear correct
- **Action**: Package appears functional - incremental improvements possible but non-blocking

**Report Documentation Status**:
- **STATUS**: ⚠️ **Tables and plots generated, but Table 3 and Plot4 have placeholders**
- Tables: Table 1 ✅, Table 2 ✅ (VAR-1 shows N/A), Table 3 ⚠️ (N/A placeholders - needs nowcasting results)
- Plots: Plot1 ✅, Plot2 ✅, Plot3 ✅, Plot4 ⚠️ (placeholders - needs nowcasting results)
- **Action**: Regenerate Table 3 and Plot4 after nowcasting experiments complete successfully

---

## NEXT ITERATION ACTIONS (Prioritized - Step 1 Will Handle Automatically)

**CRITICAL (Blocking - REAL PROBLEMS TO FIX)**:

1. **TRAIN MISSING MODELS** (Priority 2 - 2 MODELS MISSING):
   - **STATUS**: ❌ **MISSING** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
   - **ACTION**: Step 1 will detect missing models → runs `bash agent_execute.sh train`
   - **VERIFY**: After training, check `checkpoint/` contains 12 model.pkl files

2. **INVESTIGATE Systematic Backtest Failure** (Priority 1 - ALL 12 BACKTESTS FAILED):
   - **STATUS**: ⚠️ **INVESTIGATION NEEDED** - All 12 JSON files have "status": "no_results" (systematic failure)
   - **CURRENT STATE**: ALL models failed (ARIMA/VAR/DFM/DDFM) - code fixes from previous iteration are present but ineffective
   - **INVESTIGATION ACTIONS**:
     1. **Check log files** - Review `outputs/backtest/*.log` to see which validation check is actually failing (debug logging is present)
     2. **Verify code fixes** - Confirm fixes are actually being executed (horizon conversion, relaxed validation)
     3. **Check data filtering** - Verify `train_data_filtered` includes data up to view_date (2024 dates)
     4. **Check actual_value lookup** - Verify target_month_end exists in full_data.index
     5. **Check forecast_value extraction** - Verify ARIMA/VAR/DFM/DDFM predictions return valid values
     6. **Identify root cause** - Code fixes didn't work, so issue is likely different from what was fixed
   - **FIX ACTIONS** (after investigation):
     - Fix the actual root cause (likely different from what previous fixes addressed)
     - Test fix with single backtest before full re-run
   - **VERIFY**: After fix, check that at least one backtest generates valid results (not "no_results")

3. **Re-run Nowcasting Experiments** (After Priority 1 fix):
   - Step 1 checks `outputs/backtest/` → detects JSON files with "no_results" status
   - Step 1 runs: `bash agent_execute.sh backtest` (which calls `run_backtest.sh`)
   - Expected: 12 JSON files with valid `results_by_timepoint` in `outputs/backtest/{target}_{model}_backtest.json`
   - **VERIFY**: After Step 1, check `ls outputs/backtest/*.json | wc -l` should show 12 files
   - **VERIFY**: Check JSON validity and status: `python3 -c "import json; [print(f, json.load(open(f)).get('status', 'ok')) for f in __import__('glob').glob('outputs/backtest/*.json')]"`

4. **After Priority 3 completes - Regenerate Table 3**:
   - Execute: `python3 -c "from src.eval.evaluation import generate_all_latex_tables; generate_all_latex_tables()"`
   - **VERIFY**: Check `nowcasting-report/tables/tab_nowcasting_backtest.tex` - should show actual values (not N/A)

5. **After Priority 3 completes - Regenerate Plot4**:
   - Execute: `python3 nowcasting-report/code/plot.py`
   - **VERIFY**: Check `nowcasting-report/images/nowcasting_comparison_*.png` - should show 3 plots (one per target)

**HIGH (Important but Non-Blocking)**:
- **Optional - Regenerate aggregated_results.csv**: Execute `python3 -c "from src.eval.evaluation import main_aggregator; main_aggregator()"` (non-blocking, filtering works on load)
- **Update report**: After Table 3 and Plot4 are regenerated, update report with nowcasting results

**CRITICAL REMINDER**: DO NOT claim "complete", "verified", "resolved", "no issues", "production ready", "done", or "everything works" unless you actually FIXED or IMPROVED something in code or files. Always acknowledge there's room for improvement. Focus on REAL problems and FIX them, don't just document that "everything is fine".
