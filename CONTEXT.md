# Project Context

## Project Overview
This project compares 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 3 Korean macroeconomic targets (KOIPALL.G, KOEQUIPTE, KOWRCCNSE) for forecasting (22 months) and nowcasting (22 months, 2 timepoints) tasks.

## Current Experiment State

### Training Status
- **checkpoint/**: EMPTY - No models trained. **CRITICAL**: Training must be run first via `bash agent_execute.sh train`
- **Impact**: All experiments require trained models. Cannot proceed with forecasting/nowcasting until training completes.

### Forecasting Status
- **outputs/experiments/aggregated_results.csv**: EXISTS (265 lines)
  - VAR: Valid results for all 3 targets × 22 horizons
  - DFM: Valid results for all 3 targets (21 horizons for KOIPALL.G/KOEQUIPTE, 22 for KOWRCCNSE)
  - DDFM: Valid results for all 3 targets (21 horizons for KOIPALL.G/KOEQUIPTE, 22 for KOWRCCNSE)
  - ARIMA: n_valid=0 for all targets/horizons (no valid results)
- **Tables**: All forecasting tables regenerated (tab_forecasting_results.tex, tab_appendix_forecasting_*.tex)
- **Plots**: All forecasting plots regenerated (forecast_vs_actual_*.png, accuracy_heatmap.png, horizon_trend.png)

### Nowcasting Status
- **outputs/backtest/**: 6 JSON files exist (DFM/DDFM for 3 targets)
  - **Status**: ALL FAILED - All 6 files show "status": "failed" with CUDA tensor conversion errors
  - **Code Fix**: CUDA tensor conversion errors fixed in code (`.cpu().numpy()` pattern added)
  - **Action Required**: Re-run backtest experiments after training to verify fix works
- **Tables**: tab_nowcasting_backtest.tex regenerated (correctly shows N/A for all failed backtests)
- **Plots**: Nowcasting plots regenerated (placeholders since all backtests failed)

## Code Improvements Applied (Not Yet Tested)

### CUDA Tensor Conversion Fix
- **Files Modified**: `src/models/models_utils.py`, `src/evaluation/evaluation_forecaster.py`, `src/evaluation/evaluation_metrics.py`
- **Change**: All tensor conversions now use `.cpu().numpy()` pattern to move CUDA tensors to CPU before numpy conversion
- **Impact**: Should fix all DFM/DDFM backtest failures
- **Status**: Fixed in code, needs verification by re-running experiments after training

### DDFM Improvements for KOEQUIPTE
- **File Modified**: `src/train.py` (lines 363-397)
- **Changes**:
  - Automatically uses deeper encoder `[64, 32, 16]` for KOEQUIPTE (instead of default `[16, 4]`)
  - Automatically uses tanh activation for KOEQUIPTE (instead of default 'relu')
  - Increased epochs to 150 for KOEQUIPTE with deeper encoder (from default 100)
- **Rationale**: KOEQUIPTE shows identical performance to DFM (sMAE=1.14), suggesting encoder may be too small or activation function limiting
- **Status**: Implemented in code, needs experiments to test effectiveness

### Huber Loss Support
- **Files Modified**: `dfm-python/src/dfm_python/models/ddfm.py`, `src/models/models_forecasters.py`
- **Change**: Added `loss_function` parameter: 'mse' (default) or 'huber'
- **Change**: Added `huber_delta` parameter (default 1.0) for Huber loss transition point
- **Rationale**: Huber loss is more robust to outliers than MSE
- **Status**: Implemented in code, needs experiments to test robustness

## Report Status

### Tables Generated
- tab_dataset_params.tex: Dataset details and model parameters
- tab_forecasting_results.tex: Forecasting results by model-target (average across horizons)
- tab_appendix_forecasting_koipall_g.tex: Detailed results for KOIPALL.G (all horizons)
- tab_appendix_forecasting_koequipte.tex: Detailed results for KOEQUIPTE (all horizons)
- tab_appendix_forecasting_kowrccnse.tex: Detailed results for KOWRCCNSE (all horizons)
- tab_appendix_forecasting_all.tex: Average results across all targets (all horizons)
- tab_nowcasting_backtest.tex: Nowcasting backtest results (shows N/A for all failed backtests)

### Plots Generated
- forecast_vs_actual_koipall_g.png: Forecast vs actual for KOIPALL.G
- forecast_vs_actual_koequipte.png: Forecast vs actual for KOEQUIPTE
- forecast_vs_actual_kowrccnse.png: Forecast vs actual for KOWRCCNSE
- accuracy_heatmap.png: Standardized RMSE heatmap (4 models × 3 targets)
- horizon_trend.png: Performance trend by forecast horizon (1-22 months)
- nowcasting_comparison_*.png: Nowcasting comparison plots (3 targets, placeholders)
- nowcasting_trend_error_*.png: Nowcasting trend and error plots (3 targets, placeholders)

### Report Sections
- All report sections reference tables/plots correctly
- Report accurately reflects current experimental state (ARIMA excluded, backtest results noted as failed)
- Report structure finalized (methodology title fixed, results hierarchy corrected)
- Tanh activation documented across relevant sections

## Next Steps

### Priority 1 (Critical - BLOCKING)
1. **Train models** - checkpoint/ is EMPTY, blocking all experiments
   - Action: Step 1 must run `bash agent_execute.sh train` to train all 12 models (3 targets × 4 models)
   - Verification: Check `checkpoint/` contains 12 model.pkl files after training

### Priority 2 (Critical - Verification)
2. **Verify CUDA tensor conversion fixes** - Re-run backtest to verify if fixes work
   - Action: Step 1 must run `bash agent_execute.sh backtest` after training completes
   - If fix works: All 6 DFM/DDFM backtest results should show "status": "completed" instead of "failed"
   - If fix works: Regenerate tables/plots with fixed results

### Priority 3 (High)
3. **Test DDFM improvements** - Verify deeper encoder and tanh activation effectiveness
   - Action: After training, check if KOEQUIPTE DDFM performance improves with deeper encoder (target: sMAE < 1.03 from baseline 1.14)
   - Action: Optionally test Huber loss for robustness to outliers

### Priority 4 (Medium)
4. **Regenerate tables/plots** - After Priority 2 verifies CUDA fixes work
   - Action: Run table/plot generation scripts to reflect successful backtest results

### Priority 5 (Low)
5. **Investigate ARIMA failures** - All ARIMA results have n_valid=0
   - Action: After training, investigate ARIMA training/prediction pipeline
   - Check ARIMA training logs in `log/` directory
   - Verify ARIMA model instantiation and fitting in `src/models/`

## Known Issues

1. **Models NOT trained** - checkpoint/ is empty, blocking all experiments
2. **All DFM/DDFM backtest results failed** - CUDA tensor conversion errors (code fixed, needs re-run)
3. **ARIMA produces no valid results** - n_valid=0 for all targets/horizons (requires investigation)

See ISSUES.md for detailed issue tracking and research plans.

## Key Files

- **Experiment Results**: `outputs/experiments/aggregated_results.csv` (forecasting), `outputs/backtest/*.json` (nowcasting)
- **Tables**: `nowcasting-report/tables/*.tex`
- **Plots**: `nowcasting-report/images/*.png`
- **Report**: `nowcasting-report/contents/*.tex`
- **Status Tracking**: `STATUS.md`, `ISSUES.md`, `CONTEXT.md`
- **Code**: `src/` (max 15 files), `dfm-python/` (core DFM/DDFM package)
