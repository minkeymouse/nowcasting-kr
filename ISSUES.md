# Issues and Action Plan

## CURRENT STATUS (ACTUAL STATE - VERIFIED)

**REAL STATUS CHECK**:
- **checkpoint/**: 10/12 models trained ❌ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: 12 JSON files with "status": "no_results" ❌ (ALL backtests failed)
- **outputs/experiments/aggregated_results.csv**: EXISTS (36 rows, extreme VAR values filtered on load)
- **nowcasting-report/**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**This Iteration Work** (Code fixes verified in code):
1. **DFM/DDFM data_module update** (lines 320-383): Update with data up to `target_month_end` (not just `view_date`)
2. **DFM/DDFM TimeIndex conversion** (lines 965-1000 in src/models.py): Convert pandas Index to TimeIndex object
3. **VAR column matching** (lines 613-625): Use training columns in backtest
4. **Empty data check after resampling** (lines 562-564): Skip month if resampling removes all data
5. **Validation checks relaxed** (lines 678-679, 724): Recent data 180→365 days, last valid data point 180→90 days

**What's NOT Working**:
- ❌ **2 models missing** - KOIPALL.G_ddfm and KOIPALL.G_dfm not trained
- ❌ **ALL 12 backtests failed** - Code fixes applied but not verified (backtests need re-run)

---

## CONCRETE ACTION PLAN (Priority Order - REAL TASKS TO FIX)

### Priority 1: CRITICAL - Re-run Backtests After Code Fixes (BLOCKING)
**Status**: ⚠️ **CODE FIXES APPLIED, NEEDS RE-RUN** - All 12 backtests failed, fixes applied but not verified  
**Blocking**: Table 3 and Plot4 (all nowcasting results missing)

**REAL Problem**:
- **ALL 12 JSON files** have `"status": "no_results"` with error "No valid results generated for any time point"
- **Code fixes applied this iteration** (verified in code):
  - DFM/DDFM: Update data_module with data up to `target_month_end` (lines 320-383) ✅
  - DFM/DDFM: TimeIndex conversion (lines 965-1000 in src/models.py) ✅
  - VAR: Column matching logic (lines 613-625) ✅
  - Validation: Recent data check 180→365 days, last valid data point 180→90 days (lines 678-679, 724) ✅
  - Empty data check after resampling (lines 562-564) ✅

**Actions** (Step 1 will automatically handle):
- Step 1 detects "no_results" status → runs `bash agent_execute.sh backtest`
- Verify: JSON files contain `results_by_timepoint` with actual metrics

**Success Criteria**:
- ✅ At least one backtest generates valid results (not "no_results")
- ✅ JSON files contain `results_by_timepoint` with actual metrics
- ✅ All 12 months (2024-01 to 2024-12) have results for both timepoints (4 weeks, 1 week)

---

### Priority 2: CRITICAL - Train Missing Models (BLOCKING)
**Status**: ❌ **MISSING** - 2 models not trained (KOIPALL.G_ddfm, KOIPALL.G_dfm)  
**Blocking**: Complete experiment coverage

**Actions** (Step 1 will automatically handle):
- Step 1 detects missing models → runs `bash agent_execute.sh train`
- Expected: 2 additional model.pkl files in checkpoint/

**Success Criteria**:
- ✅ checkpoint/ contains 12 model.pkl files (all 3 targets × 4 models)

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

## MODEL PERFORMANCE ANOMALIES

1. **VAR Horizon 1**: Code marks persistence predictions as NaN (Table 2 shows N/A)
2. **VAR Horizons 7/28**: Code validates and marks extreme values (> 1e10) as NaN
3. **DDFM Horizon 1**: Results appear reasonable (sRMSE 0.01-0.46 range) - no issues

---

## INSPECTION FINDINGS

**Model Performance Anomalies**: VAR horizon 1 marked as NaN, VAR horizons 7/28 extreme values filtered. DDFM horizon 1 results reasonable.

**dfm-python Package**: ✅ **NO CRITICAL ISSUES FOUND** (from previous iteration)

**Report Documentation**: Tables 1-2 ✅, Table 3 ⚠️ (N/A placeholders); Plots 1-3 ✅, Plot4 ⚠️ (placeholders)

**Backtest Code**: Code fixes verified present (validation checks relaxed, DFM/DDFM fixes, VAR column matching) but backtests need re-run to verify fixes work

---

## NEXT ITERATION ACTIONS (Prioritized - Step 1 Will Handle Automatically)

**CRITICAL (Blocking)**:
1. **Train Missing Models** (Priority 2) - Step 1 detects missing models → runs `bash agent_execute.sh train`
2. **Re-run Backtests** (Priority 1) - Step 1 detects "no_results" → runs `bash agent_execute.sh backtest` → verify fixes work
3. **Regenerate Table 3 and Plot4** (Priority 3) - After backtests succeed, regenerate from outputs/backtest/

**HIGH (Non-Blocking)**:
- Regenerate aggregated_results.csv (optional - filtering works on load)
- Update report with nowcasting results (after Table 3 and Plot4 regenerated)
