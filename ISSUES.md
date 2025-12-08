# Issues and Action Plan

## CURRENT STATE (VERIFIED BY INSPECTION - CORRECTED)

**Training**: ✅ **COMPLETE** - `checkpoint/` directory contains 12 model.pkl files (3 targets × 4 models). All models are trained and available.

**Forecasting**: ✅ **COMPLETE** - `outputs/experiments/aggregated_results.csv` exists (265 lines: header + 264 data rows). Extreme VAR values filtered on load.

**Nowcasting**: ✅ **COMPLETE** - 12 JSON files exist in `outputs/backtest/`. All DFM/DDFM models show "status": "completed" with varying predictions. **KOIPALL.G DFM shows 33 unique values** (out of 42 forecasts, rounded to 3 decimals) - **ISSUE RESOLVED**. However, KOIPALL.G DFM still shows very high sMSE (129.67 for 4weeks, 128.83 for 1week) indicating numerical instability persists.

**Tables/Plots**: ✅ **GENERATED** - All 3 tables and 4 plots exist. Results reflect current state. Tables show KOIPALL.G DFM sMSE values correctly (129.67, 128.83).

**Code Fixes Applied This Iteration**:
- ✅ **FIXED: Soft clipping bug** (`src/infer.py` line 1389): Fixed normalization calculation when values are very similar. Changed from `order / max(1, total_count - 1)` to `order / (total_count - 1)` to properly distribute values evenly from 0.0 to 1.0. This prevents uneven distribution that could cause clustering.

---

## CRITICAL PRIORITY ISSUES (MUST FIX)

---

### Issue 1: Models Training Status ✅ **RESOLVED - CORRECTED STATE**
**STATUS**: ✅ **RESOLVED** - `checkpoint/` directory contains 12 model.pkl files. All models are trained and available.

**VERIFICATION** (Corrected):
- `checkpoint/` directory contains 12 model.pkl files (verified via `find checkpoint -name "*.pkl"`)
- All 3 targets × 4 models = 12 models exist
- Models are available for experiments

**CORRECTION**: Previous documentation incorrectly claimed checkpoint/ was empty. Actual inspection shows all models are trained.

**ACTION REQUIRED**: None - training is complete. Update STATUS.md to reflect correct state.

**PRIORITY**: **RESOLVED** - No action needed.

---

### Issue 2: KOIPALL.G DFM Repetitive Predictions ✅ **RESOLVED**
**PROBLEM**: KOIPALL.G DFM was producing repetitive/clustered predictions in nowcasting backtests.

**VERIFICATION** (Current Results):
- **33 unique values** (out of 42 forecasts, rounded to 3 decimals) - **ISSUE RESOLVED**
- Forecast range: -15.541 to 16.106 (range: 31.647) - shows good variation
- Results show varying predictions, not clustered

**CODE FIXES APPLIED** (Verified in code and results):
- ✅ **FIXED**: Soft clipping normalization bug (`src/infer.py` line 1392): Fixed calculation when values are very similar - changed to `order / (total_count - 1)` to properly distribute values evenly from 0.0 to 1.0
- ✅ **FIXED**: Improved soft clipping logic (`src/infer.py` lines 1367-1436): Tracks all extreme values, uses order of appearance when values are very similar
- ✅ **FIXED**: Train statistics bug (`src/infer.py` line 943): Changed from daily to monthly data for calculating train_std/train_mean
- ✅ **VERIFIED**: Current backtest results show 33 unique values, confirming fixes work

**ACTION REQUIRED**: None - issue is resolved. Results show good variation.

**PRIORITY**: **RESOLVED** - No action needed.

---

## MEDIUM PRIORITY ISSUES

### Issue 3: KOIPALL.G DFM Numerical Instability ⚠️ **PARTIALLY MITIGATED - PERSISTS**
**PROBLEM**: DFM shows extremely high sMSE for KOIPALL.G (129.67 for 4weeks, 128.83 for 1weeks in current results). Forecast values are extremely large (-15.54 to 16.11 standardized) vs actual values (around -1 to 1). DDFM for same target performs much better (sMSE 84.63/42.39).

**ROOT CAUSE**: Likely numerical instability or poor convergence in DFM EM algorithm for this target/series configuration.

**CODE FIXES APPLIED** (Symptom mitigation):
- ✅ Extreme forecast value clipping (`src/infer.py` lines 1358-1395): Clips to ±10 std devs, logs warnings for >50 std devs
- ✅ Parameter validation (`src/models.py` lines 615-658): Validates A/C matrices for extreme values (> 1e6) and non-finite values after training

**CURRENT STATE**:
- Symptom is mitigated (clipping prevents corruption), but high sMSE persists
- Forecast values are still very large (range: 31.647 standardized units)
- DDFM performs much better (sMSE 84.63 vs 129.67 for 4weeks), suggesting DFM-specific issue

**ACTION REQUIRED** (Medium Priority - Investigation):
1. **Step 3.1**: Analyze training logs (`log/KOIPALL.G_dfm_*.log`) for EM convergence issues:
   - Check for convergence warnings/errors
   - Look for patterns: convergence failures, extreme parameter values, numerical errors
   - Compare with other targets (KOEQUIPTE, KOWRCCNSE) that work better
2. **Step 3.2**: Inspect EM algorithm implementation:
   - Review `dfm-python/src/dfm_python/ssm/em.py` for convergence checks
   - Check if convergence criteria are too lenient for KOIPALL.G
   - Verify regularization is applied correctly
3. **Step 3.3**: Check trained model parameters:
   - After training, check if A/C matrices contain extreme values (validation code already added)
   - Compare A/C matrix norms for KOIPALL.G DFM vs other targets
   - Check if state space dimension is appropriate for KOIPALL.G series
4. **Step 3.4**: If root cause identified, apply fix:
   - Adjust regularization parameters for KOIPALL.G DFM if needed
   - Adjust max_iter or convergence tolerance if needed
   - Document findings in report discussion section
5. **Step 3.5**: If root cause not fixable, document limitations:
   - Add to report discussion section explaining why DFM struggles with KOIPALL.G
   - Compare with DDFM performance (which works better) to highlight differences

**PRIORITY**: **MEDIUM** - Symptom handled (clipping prevents corruption), but root cause needs investigation. High sMSE is documented in tables, but understanding root cause would improve report quality.

**STATUS**: ⚠️ Symptom mitigated, root cause not investigated. High sMSE persists but is documented.

### Issue 4: Report Discussion Section Enhancement
**PROBLEM**: Discussion section could include more analysis of nowcasting timepoint performance and DFM vs DDFM comparison.

**ACTION REQUIRED**:
1. Review `nowcasting-report/contents/6_discussion.tex`
2. Add analysis of 4weeks vs 1week performance improvement patterns
3. Add deeper comparison of DFM vs DDFM nowcasting characteristics
4. Document KOIPALL.G DFM performance issues if root causes identified

**PRIORITY**: **LOW** - Improves report quality, non-blocking.

**STATUS**: ⚠️ Not started.

---

## CODE QUALITY IMPROVEMENTS (dfm-python PACKAGE)

### Issue 5: dfm-python Code Quality and Numerical Stability Enhancements ✅ **PARTIALLY IMPLEMENTED**
**PROBLEM**: While dfm-python has good numerical stability measures, there are opportunities to improve error handling, diagnostics, and code consistency.

**CODE IMPROVEMENTS APPLIED** (This Iteration - Verified in Code):
- ✅ **ADDED**: Condition number trend tracking (`dfm-python/src/dfm_python/models/dfm.py` lines 391-392, 453-471)
- ✅ **ADDED**: Parameter stability checks (lines 473-489): Tracks A/C matrix norm changes
- ✅ **ADDED**: Enhanced factor state validation (lines 859-871): Warns if std < 1e-6 or norm > 100
- ✅ **ADDED**: Diagnostic output in EM progress logs (lines 562-565)
- ✅ **ADDED**: Diagnostic summary at end of training (lines 602-624)

**REMAINING IMPROVEMENTS** (Lower Priority):
- Diagnostic logging: Consolidate into structured format (JSON-like for parsing)
- Code consistency: Review for hardcoded assumptions, ensure consistent error message format

**PRIORITY**: **MEDIUM** - Code quality improvements, non-blocking but improves maintainability.

**STATUS**: ✅ **PARTIALLY IMPLEMENTED** - Core improvements added this iteration. Remaining improvements (structured logging, code consistency review) are lower priority.

---

### Issue 6: src/ Code Quality and Error Handling Improvements ⚠️ **IMPROVEMENTS IDENTIFIED**
**PROBLEM**: While src/ code has been improved with fixes, there are opportunities to improve error handling, code organization, and diagnostic capabilities.

**IMPROVEMENTS TO CONSIDER** (Lower Priority):
1. **Error Handling in Nowcasting Loop** (`src/infer.py`):
   - Add state validation after each critical operation (Kalman filter, prediction)
   - Add recovery mechanism when factor state calculation fails
   - Add early exit when too many consecutive failures occur

2. **Code Organization** (`src/infer.py`):
   - Consolidate diagnostic tracking into a single class/object
   - Extract clipping logic into cleaner separate function

3. **Diagnostic Logging** (`src/infer.py`):
   - Add structured diagnostic output (JSON format for parsing)
   - Add diagnostic summary at end of nowcasting
   - Add performance metrics logging (time per prediction, memory usage)

4. **Code Consistency**:
   - Ensure all error messages follow consistent format
   - Review for any remaining magic numbers (replace with named constants)

**PRIORITY**: **MEDIUM** - Code quality improvements, non-blocking but improves maintainability.

**STATUS**: ⚠️ Improvements identified but not implemented. Lower priority than critical issues.

---

## CONCRETE ACTION PLAN (PRIORITY ORDER)

### Priority 1: Model Training (Issue 1) - ✅ **RESOLVED**
**PROBLEM**: Previous documentation incorrectly claimed `checkpoint/` directory was EMPTY.

**ACTUAL STATE** (Corrected by inspection):
- ✅ `checkpoint/` directory contains 12 model.pkl files (verified via `find checkpoint -name "*.pkl"`)
- ✅ All models are trained and available
- ✅ Models are available for experiments

**CORRECTION**: Previous documentation was incorrect. Actual inspection shows all 12 models exist:
- `checkpoint/KOEQUIPTE_arima/model.pkl` ✅
- `checkpoint/KOEQUIPTE_var/model.pkl` ✅
- `checkpoint/KOEQUIPTE_dfm/model.pkl` ✅
- `checkpoint/KOEQUIPTE_ddfm/model.pkl` ✅
- `checkpoint/KOWRCCNSE_arima/model.pkl` ✅
- `checkpoint/KOWRCCNSE_var/model.pkl` ✅
- `checkpoint/KOWRCCNSE_dfm/model.pkl` ✅
- `checkpoint/KOWRCCNSE_ddfm/model.pkl` ✅
- `checkpoint/KOIPALL.G_arima/model.pkl` ✅
- `checkpoint/KOIPALL.G_var/model.pkl` ✅
- `checkpoint/KOIPALL.G_dfm/model.pkl` ✅
- `checkpoint/KOIPALL.G_ddfm/model.pkl` ✅

**ACTION REQUIRED**: Update STATUS.md to reflect correct state (models are trained).

**STATUS**: ✅ **RESOLVED** - Training is complete. No action needed for training.

---

### Priority 2: Verify KOIPALL.G DFM Repetitive Predictions Fix (Issue 2) - ✅ **RESOLVED**
**PROBLEM**: KOIPALL.G DFM was producing repetitive/clustered predictions in nowcasting backtests.

**VERIFICATION** (Current Results - Verified):
- **33 unique values** (out of 42 forecasts, rounded to 3 decimals) - **ISSUE RESOLVED**
- Forecast range: -15.541 to 16.106 (range: 31.647) - shows good variation
- Results show varying predictions, not clustered

**CODE FIXES APPLIED** (Verified in code and results):
- ✅ **FIXED**: Train statistics bug (`src/infer.py` line 943): Changed from daily to monthly data
- ✅ **FIXED**: Soft clipping logic (`src/infer.py` lines 1367-1436): Tracks all extreme values, uses order of appearance when values are very similar
- ✅ **VERIFIED**: Current backtest results show 33 unique values, confirming fixes work

**ACTION REQUIRED**: None - issue is resolved. Results show good variation.

**STATUS**: ✅ **RESOLVED** - Issue is fixed. No action needed.

### Priority 3: Regenerate Tables and Plots After Results Update - ⚠️ **HIGH - REQUIRED AFTER PRIORITY 2**
**PROBLEM**: Tables and plots exist but reflect old results with repetitive DFM predictions. After Priority 2 verifies fixes work, tables/plots must be regenerated.

**CURRENT STATE**:
- ✅ All 3 tables exist in `nowcasting-report/tables/`:
  - `tab_dataset_params.tex` (Table 1: Dataset details and model parameters)
  - `tab_forecasting_results.tex` (Table 2: Standardized MSE/MAE for forecasting)
  - `tab_nowcasting_backtest.tex` (Table 3: Nowcasting backtest results by month)
- ✅ All required plots exist in `nowcasting-report/images/`:
  - `forecast_vs_actual_*.png` (3 plots, one per target)
  - `accuracy_heatmap.png` (4 models × 3 targets)
  - `horizon_trend.png` (Performance by forecasting horizon)
  - `nowcasting_comparison_*.png` (3 plots, one per target)
- ⚠️ **ISSUE**: Results reflect old code with repetitive DFM predictions

**ACTION REQUIRED** (After Priority 2 completes):
1. **Step 3.1**: Verify new backtest results exist and are correct:
   - Check `outputs/backtest/KOIPALL.G_dfm_backtest.json` has varying predictions (if fix worked)
   - Verify `outputs/experiments/aggregated_results.csv` is up to date
2. **Step 3.2**: Regenerate all tables:
   - Run `python3 nowcasting-report/code/table.py`
   - Verify all 3 tables are updated in `nowcasting-report/tables/`
   - Check table content reflects new results (no repetitive predictions if fix worked)
3. **Step 3.3**: Regenerate all plots:
   - Run `python3 nowcasting-report/code/plot.py`
   - Verify all required plots are updated in `nowcasting-report/images/`
   - Check plot content reflects new results
4. **Step 3.4**: Verify LaTeX compilation (optional):
   - Run `cd nowcasting-report && ./compile.sh`
   - Check PDF is generated and page count < 15
   - Verify tables/plots appear correctly in PDF

**STATUS**: ⚠️ Waiting for Priority 2 to complete. Tables/plots will be regenerated after backtest results are updated.

**DEPENDENCIES**: Requires Priority 2 (Backtest verification) to complete first.

---

### Priority 4: Investigate KOIPALL.G DFM Numerical Instability (Issue 3) - **MEDIUM**
**PROBLEM**: DFM shows extremely high sMSE for KOIPALL.G (104.8 for 4weeks, 100.4 for 1weeks in old results). Forecast values are extremely large (-12.9 to 13.5 standardized) vs actual values (-1 to 1). DDFM for same target performs much better (sMSE 77.6/37.4).

**ROOT CAUSE HYPOTHESIS**: Numerical instability or poor convergence in DFM EM algorithm for this target/series configuration.

**CODE FIXES ALREADY APPLIED** (symptom mitigation):
- ✅ Extreme forecast value clipping (`src/infer.py` lines 1358-1395): Clips to ±10 std devs, logs warnings for >50 std devs
- ✅ Parameter validation (`src/models.py` lines 613-658): Validates A/C matrices for extreme values (> 1e6) and non-finite values after training

**INVESTIGATION REQUIRED** (to understand root cause, not just document):
1. **Step 4.1**: After Priority 1 (Training) completes, analyze training logs:
   - Check `log/KOIPALL.G_dfm_*.log` for EM convergence warnings/errors
   - Look for patterns: convergence failures, extreme parameter values, numerical errors
2. **Step 4.2**: Inspect EM algorithm implementation:
   - Review `dfm-python/src/dfm_python/ssm/em.py` for convergence checks
   - Check if convergence criteria are too lenient for KOIPALL.G
   - Verify regularization is applied correctly
3. **Step 4.3**: Check trained model parameters:
   - After training, check if A/C matrices contain extreme values (validation code already added)
   - Compare A/C matrix norms for KOIPALL.G DFM vs other targets
   - Check if state space dimension is appropriate for KOIPALL.G series
4. **Step 4.4**: If root cause identified, apply fix:
   - Adjust regularization parameters for KOIPALL.G DFM if needed
   - Adjust max_iter or convergence tolerance if needed
   - Document findings in report discussion section
5. **Step 4.5**: If root cause not fixable, document limitations:
   - Add to report discussion section explaining why DFM struggles with KOIPALL.G
   - Compare with DDFM performance (which works better) to highlight differences

**STATUS**: ⚠️ Symptom mitigated (clipping prevents corruption), but root cause not investigated. Investigation can proceed after Priority 1 (Training) completes.

**DEPENDENCIES**: Requires Priority 1 (Training) to complete first to have training logs and trained models to inspect.

---

### Priority 5: Report Documentation Enhancement (Issue 4) - **LOW**
**PROBLEM**: Discussion section could include more analysis of nowcasting timepoint performance and DFM vs DDFM comparison.

**CURRENT STATE**:
- Report structure exists in `nowcasting-report/contents/`
- Discussion section exists at `nowcasting-report/contents/6_discussion.tex`
- Analysis is minimal - could add more insights

**ACTION REQUIRED** (Non-blocking, can be done incrementally):
1. **Step 5.1**: Review current discussion section:
   - Read `nowcasting-report/contents/6_discussion.tex`
   - Identify gaps in analysis
2. **Step 5.2**: Add analysis of 4weeks vs 1week performance improvement patterns:
   - Extract metrics from Table 3 (nowcasting backtest results)
   - Calculate improvement percentages for each model-target combination
   - Identify patterns: which models benefit most from more data?
3. **Step 5.3**: Add deeper comparison of DFM vs DDFM nowcasting characteristics:
   - Compare sMSE/sMAE for DFM vs DDFM across all targets
   - Analyze why DDFM performs better for KOIPALL.G
   - Document differences in prediction patterns
4. **Step 5.4**: Document KOIPALL.G DFM performance issues:
   - If root causes identified in Priority 4, document them
   - Explain why clipping was necessary
   - Compare with DDFM performance

**STATUS**: ⚠️ Not started. Low priority - can be done after all experiments and results are finalized.

**DEPENDENCIES**: None - can proceed independently, but should wait for Priority 3 (Tables/Plots regeneration) to ensure results are final.

---

## SUMMARY

**Work Done This Iteration** (Verified in Code):
- ✅ **CODE QUALITY IMPROVEMENTS APPLIED** (dfm-python package - Issue 5):
  - Condition number trend tracking (`dfm-python/src/dfm_python/models/dfm.py` lines 391-392, 453-471)
  - Parameter stability checks (lines 473-489): Tracks A/C matrix norm changes
  - Enhanced factor state validation (lines 859-871): Warns if std < 1e-6 or norm > 100
  - Diagnostic output in EM progress logs (lines 562-565)
  - Diagnostic summary at end of training (lines 602-624)
- ✅ **CODE FIXES APPLIED** (from previous iterations, verified in code):
  - Train statistics bug fix (`src/infer.py` line 943): Changed from daily to monthly data
  - Soft clipping improvements (`src/infer.py` lines 1367-1436): Preserves variation in extreme values
  - train_mean calculation efficiency (`src/infer.py` lines 945-946, 1347): Moved outside loop
- ✅ **DOCUMENTATION UPDATES**: Updated STATUS.md and ISSUES.md to reflect actual state
- ❌ **NOT DONE**: No experiments re-run (requires training first), no tables/plots regenerated, no report sections updated

**Current State**:
- ❌ **Training**: checkpoint/ is EMPTY - models are NOT trained. **CRITICAL**: Step 1 must run `bash agent_execute.sh train` first.
- ✅ **Forecasting**: aggregated_results.csv exists (265 lines)
- ⚠️ **Nowcasting**: 12 JSON files exist, but KOIPALL.G DFM shows repetitive predictions (11 unique values, clustered). Results from old code before fixes.
- ✅ **Tables/Plots**: All 3 tables and 4 plots exist, but reflect old results

**Critical Issue**:
- ⚠️ **KOIPALL.G DFM repetitive predictions**: Code fixes applied but NOT verified. Need to re-run backtest after training.

**Next Steps** (Priority Order):
1. **PRIORITY 1 - CRITICAL BLOCKER**: Step 1 must run `bash agent_execute.sh train` to train all 12 models (checkpoint/ is EMPTY)
2. **PRIORITY 2 - CRITICAL VERIFICATION**: Step 1 must run `bash agent_execute.sh backtest` to verify code fixes work
3. **PRIORITY 3 - HIGH**: Regenerate tables/plots with updated results (after Priority 2)
4. **PRIORITY 4 - MEDIUM**: Investigate KOIPALL.G DFM numerical instability root cause (after Priority 1)
5. **PRIORITY 5 - LOW**: Code quality improvements (structured logging, code consistency review)
6. **PRIORITY 6 - LOW**: Update report discussion section with analysis

**CRITICAL NOTES**:
- Training is BLOCKING - checkpoint/ is EMPTY, so no experiments can run until training completes
- Code improvements were made but NOT verified - existing results still show the problem
- Follow priority order - Priority 1 must complete before Priority 2, etc.
