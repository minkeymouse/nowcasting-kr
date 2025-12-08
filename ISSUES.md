# Issues

## CRITICAL: Models Not Trained (checkpoint/ is empty)
- **Problem**: The `checkpoint/` directory is empty, meaning no models have been trained.
- **Impact**: 
  - All forecasting and nowcasting experiments are using non-existent or outdated models
  - Results in `outputs/experiments/aggregated_results.csv` may be from previous runs or invalid
  - Cannot generate new predictions or retrain models
  - DDFM improvements (deeper encoder, Huber loss) cannot be tested
- **Required Action**: Step 1 must run `bash agent_execute.sh train` to generate model checkpoints
- **Status**: **BLOCKING** - Must be resolved before any new experiments can be run
- **Verification**: After training, check that `checkpoint/` contains 12 model.pkl files (3 targets × 4 models)

## CRITICAL: DFM/DDFM Backtest CUDA Tensor Conversion Error (Fixed in Code, Needs Re-run)
- **Problem**: All DDFM and DFM backtest results failed with error: "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
- **Root Cause**: The `_convert_predictions_to_dataframe` function and other prediction conversion functions used `np.asarray()` directly on CUDA tensors without first moving them to CPU.
- **Impact**: All 22 months of backtest results for DFM and DDFM models failed (6 JSON files × 22 months = 132 failed predictions). All backtest JSON files in `outputs/backtest/` show "status": "failed" with CUDA errors.
- **Resolution** (This Iteration): Fixed tensor conversion in multiple locations:
  - `src/models/models_utils.py`: `_convert_predictions_to_dataframe()` and `_validate_predictions()` - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_forecaster.py`: Prediction value extraction - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_metrics.py`: Metric calculation - Added `.cpu().numpy()` pattern
  - All now convert CUDA tensors to CPU before numpy conversion using `.cpu().numpy()`
- **Status**: ✅ **FIXED IN CODE** (verified in code). **Backtest experiments MUST be re-run** via `bash agent_execute.sh backtest` after training to generate valid results. Current backtest JSON files all show "failed" status with CUDA errors.

## ARIMA Forecasting Results Missing
- **Problem**: ARIMA model produces no valid results (n_valid=0) for all three targets (KOIPALL.G, KOEQUIPTE, KOWRCCNSE) across all 22 forecast horizons.
- **Impact**: ARIMA cannot be included in forecasting comparison. All ARIMA entries in aggregated_results.csv have empty sMSE, sMAE, sRMSE values. Report sections have been updated to reflect that ARIMA is excluded from comparison.
- **Possible Causes**:
  1. ARIMA model training/prediction pipeline failure
  2. Data preprocessing issues (missing values, transformation errors)
  3. Model fitting convergence issues
  4. Prediction generation errors (shape mismatches, index alignment)
- **Investigation Needed** (After training):
  - Check ARIMA training logs in `log/` directory
  - Verify ARIMA model instantiation and fitting in `src/models/`
  - Check prediction generation code for ARIMA
  - Verify data compatibility (index alignment, missing value handling)
- **Report Updates** (This Iteration):
  - Fixed inconsistencies in report sections (7_issues.tex, 6_discussion.tex) - removed incorrect ARIMA performance analysis
  - Updated plot captions to reflect that ARIMA is excluded (3_results_forecasting.tex)
  - Updated result completeness statistics to accurately reflect ARIMA's status
- **Status**: Unresolved. Requires investigation and fix before ARIMA can be included in results. Report has been updated to accurately reflect current state. Investigation should be done after training completes.

## DDFM Metrics Improvement (Implemented, Needs Testing)
- **Current Performance** (from aggregated_results.csv):
  - KOIPALL.G: sMAE=0.69, sMSE=0.61, sRMSE=0.69 (21 horizons) - **Excellent**
  - KOWRCCNSE: sMAE=0.50, sMSE=0.49, sRMSE=0.50 (22 horizons) - **Excellent**
  - KOEQUIPTE: sMAE=1.14, sMSE=2.12, sRMSE=1.14 (21 horizons) - **Moderate** (identical to DFM)
- **Implemented Improvements** (This Iteration - Verified in Code):
  1. **Target-Specific Encoder Architectures** (Implemented in `src/train.py` lines 363-397):
     - KOEQUIPTE: Automatically uses deeper encoder `[64, 32, 16]` instead of default `[16, 4]` when not explicitly specified
     - Increased epochs to 150 for KOEQUIPTE with deeper encoder (from default 100)
     - Rationale: KOEQUIPTE shows identical performance to DFM, suggesting current encoder is too small to capture useful nonlinear features
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments after training to test effectiveness
  2. **Huber Loss Support** (Implemented in `dfm-python/src/dfm_python/models/ddfm.py`, `src/models/models_forecasters.py`):
     - Added `loss_function` parameter to DDFM: 'mse' (default) or 'huber'
     - Added `huber_delta` parameter (default 1.0) for Huber loss transition point
     - Huber loss is more robust to outliers than MSE, which may help with volatile series
     - Can be configured via model_params in training config: `loss_function: 'huber'`
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments after training to test robustness
  3. **Enhanced Training Configuration**:
     - Loss function and Huber delta can be specified in model_params
     - Target-specific hyperparameter adjustments for KOEQUIPTE
     - Better logging of loss function and architecture choices
- **Areas for Further Improvement** (After Testing Current Improvements):
  1. **KOEQUIPTE Performance**: If deeper encoder doesn't improve performance, consider:
     - Try different activation functions (tanh, sigmoid) for KOEQUIPTE
     - Investigate if KOEQUIPTE has strong linear relationships that don't benefit from nonlinear encoding
     - Consider ensemble methods (combining DFM and DDFM) for KOEQUIPTE
  2. **Horizon 22 Missing**: KOIPALL.G and KOEQUIPTE are missing horizon 22 results. Investigate after training:
     - Check if predictions are generated but fail validation
     - Verify if model fails to generate predictions for the longest horizon
     - Check for numerical instability at long horizons
  3. **Performance Consistency**: While DDFM performs excellently on KOIPALL.G and KOWRCCNSE, investigate:
     - Why performance is much better than DFM on these targets but identical on KOEQUIPTE
     - Whether encoder architecture can be optimized for all targets simultaneously
- **Status**: ✅ **IMPROVEMENTS IMPLEMENTED IN CODE** (verified). Need to re-run experiments after training to test:
  - KOEQUIPTE with deeper encoder `[64, 32, 16]` and 150 epochs
  - Optional: Test Huber loss for all targets to assess robustness
  - Current performance is good for 2/3 targets, improvement needed for KOEQUIPTE.

## Report Finalization (Completed This Iteration)
- **Status**: Report sections have been finalized and corrected for accuracy
- **Improvements Made**:
  - Fixed ARIMA inconsistencies: Removed incorrect performance analysis from 7_issues.tex since ARIMA has no valid results (n_valid=0)
  - Updated plot captions: Modified 3_results_forecasting.tex to reflect that ARIMA is excluded from plots due to no valid results
  - Corrected result completeness statistics: Updated 7_issues.tex and 6_discussion.tex to accurately reflect ARIMA's status
  - Verified report structure: All sections properly referenced, tables and figures correctly linked
- **Files Modified**:
  - `nowcasting-report/contents/3_results_forecasting.tex`: Updated plot captions
  - `nowcasting-report/contents/6_discussion.tex`: Fixed ARIMA references
  - `nowcasting-report/contents/7_issues.tex`: Replaced incorrect ARIMA performance analysis with failure analysis
- **Note**: Report accurately reflects current experimental state. Missing experiments (training, nowcasting backtest) are noted and will be run by Step 1 automatically.
