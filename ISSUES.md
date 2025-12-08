# Issues

## CRITICAL: Models Not Trained (checkpoint/ is empty)
- **Problem**: The `checkpoint/` directory is empty, meaning no models have been trained.
- **Impact**: 
  - All forecasting and nowcasting experiments are using non-existent or outdated models
  - Results in `outputs/experiments/aggregated_results.csv` may be from previous runs or invalid
  - Cannot generate new predictions or retrain models
  - DDFM improvements (deeper encoder, tanh activation, Huber loss) cannot be tested
- **Required Action**: Step 1 must run `bash agent_execute.sh train` to generate model checkpoints
- **Status**: **BLOCKING** - Must be resolved before any new experiments can be run
- **Verification**: After training, check that `checkpoint/` contains 12 model.pkl files (3 targets × 4 models)

## CRITICAL: DFM/DDFM Backtest CUDA Tensor Conversion Error (Fixed in Code, Needs Re-run)
- **Problem**: All DDFM and DFM backtest results failed with error: "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
- **Root Cause**: The `_convert_predictions_to_dataframe` function and other prediction conversion functions used `np.asarray()` directly on CUDA tensors without first moving them to CPU.
- **Impact**: All 22 months of backtest results for DFM and DDFM models failed (6 JSON files × 22 months = 132 failed predictions). All backtest JSON files in `outputs/backtest/` show "status": "failed" with CUDA errors.
- **Resolution** (Previous Iteration): Fixed tensor conversion in multiple locations:
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
- **Report Updates** (Previous Iteration):
  - Fixed inconsistencies in report sections (7_issues.tex, 6_discussion.tex) - removed incorrect ARIMA performance analysis
  - Updated plot captions to reflect that ARIMA is excluded (3_results_forecasting.tex)
  - Updated result completeness statistics to accurately reflect ARIMA's status
- **Status**: Unresolved. Requires investigation and fix before ARIMA can be included in results. Report has been updated to accurately reflect current state. Investigation should be done after training completes.

## DDFM Metrics Improvement (Research Plan)
- **Current Performance Analysis** (from aggregated_results.csv):
  - **KOIPALL.G**: sMAE=0.69, sMSE=0.61, sRMSE=0.69 (21 horizons) - **Excellent**
    - Horizon-specific: Very low error in short-term (1-6 months, sMAE < 0.15), moderate in mid-term (7-12 months, sMAE 0.2-0.9), stable in long-term (13-21 months, sMAE 0.5-1.3)
    - Missing horizon 22 (n_valid=0) - needs investigation
  - **KOWRCCNSE**: sMAE=0.50, sMSE=0.49, sRMSE=0.50 (22 horizons) - **Excellent**
    - Horizon-specific: Very low error in short-term (1, 4-8 months, sMAE < 0.15), moderate in mid-term (9-16 months, sMAE 0.2-0.5), some spikes at horizons 14, 19-20
    - Horizon 17 has missing sMSE/sMAE but very small MSE/MAE values (possible numerical precision issue)
  - **KOEQUIPTE**: sMAE=1.14, sMSE=2.12, sRMSE=1.14 (21 horizons) - **Moderate** (identical to DFM)
    - Horizon-specific: Moderate error across all horizons (sMAE 1.0-1.5 for short-term, similar for mid-term), identical to DFM at every horizon
    - Missing horizon 22 (n_valid=0) - needs investigation
    - **Critical observation**: DDFM and DFM show nearly identical performance at every single horizon, suggesting encoder is learning linear relationships only

- **Root Cause Analysis**:
  - **ReLU Activation Limitation**: ReLU activation (`max(0, x)`) zeros negative values, which may prevent the encoder from learning negative correlations between factors and observations. KOEQUIPTE may have negative factor loadings that ReLU cannot capture.
  - **Encoder Capacity**: Default encoder `[16, 4]` may be too small to capture complex nonlinear relationships for KOEQUIPTE.
  - **Linear Collapse**: The encoder may be collapsing to linear behavior (equivalent to PCA/DFM) due to insufficient capacity or activation function limitations.

- **Implemented Improvements** (Previous Iteration - Verified in Code):
  1. **Target-Specific Encoder Architectures** (`src/train.py` lines 363-397):
     - KOEQUIPTE: Automatically uses deeper encoder `[64, 32, 16]` instead of default `[16, 4]`
     - Increased epochs to 150 for KOEQUIPTE with deeper encoder (from default 100)
     - Rationale: Current encoder may be too small or learning only linear features
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments after training to test effectiveness
  2. **Activation Function Support** (`src/models/models_forecasters.py`, `src/train.py`):
     - Added `activation` parameter support to DDFMForecaster (default: 'relu')
     - KOEQUIPTE: Automatically uses 'tanh' activation instead of 'relu' (unless explicitly overridden)
     - Rationale: Tanh activation can capture negative correlations that ReLU cannot (ReLU zeros negative values)
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments after training to test effectiveness
  3. **Huber Loss Support** (`dfm-python/src/dfm_python/models/ddfm.py`, `src/models/models_forecasters.py`):
     - Added `loss_function` parameter: 'mse' (default) or 'huber'
     - Added `huber_delta` parameter (default 1.0) for transition point
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments to test robustness to outliers

- **Concrete Research Plan for DDFM Metrics Improvement**:
  
  **Phase 1: Test Current Improvements (After Training)**
  1. **KOEQUIPTE Deeper Encoder + Tanh Activation Test**:
     - Re-run training with deeper encoder `[64, 32, 16]`, tanh activation, and 150 epochs
     - Compare sMAE/sMSE/sRMSE with current results (baseline: sMAE=1.14)
     - **Success criterion**: sMAE improvement of at least 10% (target: sMAE < 1.03)
     - **If no improvement**: Investigate other causes (data characteristics, factor structure, etc.)
     - **Expected improvement**: Tanh activation should allow encoder to learn negative correlations that ReLU cannot capture
  
  2. **Activation Function Comparison** (If Phase 1 shows improvement):
     - Compare 'relu' vs 'tanh' vs 'sigmoid' for KOEQUIPTE
     - Document which activation works best for each target
     - **Hypothesis**: Tanh should outperform ReLU for KOEQUIPTE if negative correlations are important
  
  3. **Huber Loss Robustness Test** (Optional):
     - Test Huber loss (`loss_function: 'huber'`) for all three targets
     - Compare with MSE baseline, especially for volatile horizons (KOIPALL.G horizons 11-13, KOWRCCNSE horizons 14, 19-20)
     - **Success criterion**: Reduced error at volatile horizons without degrading overall performance
  
  4. **Horizon 22 Investigation**:
     - Check if predictions are generated but fail validation (check logs)
     - Verify numerical stability at longest horizon (22 months)
     - **Action**: Fix validation or numerical issues to complete all 22 horizons

  **Phase 2: Advanced Improvements (If Phase 1 Doesn't Improve KOEQUIPTE)**
  1. **Activation Function Experimentation**:
     - Test different activation functions for KOEQUIPTE encoder: tanh, sigmoid, LeakyReLU
     - Current default is likely ReLU - may not capture negative correlations well
     - **Hypothesis**: KOEQUIPTE may have negative correlations that ReLU cannot capture
     - **Implementation**: Add `activation` parameter to DDFM encoder config
  
  2. **Linear Relationship Investigation**:
     - Analyze correlation structure of KOEQUIPTE vs KOIPALL.G/KOWRCCNSE
     - Check if KOEQUIPTE has stronger linear relationships in factor space
     - **If confirmed**: Consider hybrid approach (linear DFM for KOEQUIPTE, DDFM for others)
  
  3. **Encoder Architecture Optimization**:
     - Try different architectures: `[32, 16, 8]`, `[128, 64, 32]`, `[64, 32, 16, 8]`
     - Use hyperparameter search (grid search or Bayesian optimization)
     - **Focus**: Find architecture that improves KOEQUIPTE without degrading KOIPALL.G/KOWRCCNSE

  **Phase 3: Ensemble and Advanced Techniques** (If Phase 2 Still Doesn't Improve)
  1. **Ensemble Methods**:
     - Weighted average of DFM and DDFM predictions for KOEQUIPTE
     - Learn ensemble weights on validation set
     - **Rationale**: If both models learn similar features, ensemble may reduce variance
  
  2. **Regularization Techniques**:
     - Add dropout to encoder layers for KOEQUIPTE
     - L1/L2 regularization on encoder weights
     - **Hypothesis**: Prevent overfitting to linear features, encourage nonlinear feature learning
  
  3. **Feature Engineering**:
     - Pre-process KOEQUIPTE data differently (different transformations)
     - Add interaction features before encoding
     - **Rationale**: If raw data has weak nonlinear signal, engineered features may help

- **Success Metrics**:
  - **Primary**: KOEQUIPTE sMAE improvement from 1.14 to < 1.0 (12% improvement)
  - **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
  - **Tertiary**: Complete all 22 horizons for all targets (currently missing horizon 22 for KOIPALL.G and KOEQUIPTE)

- **Status**: ✅ **IMPROVEMENTS IMPLEMENTED IN CODE** (verified). **RESEARCH PLAN DEFINED**. Need to:
  1. Re-run training with deeper encoder and tanh activation for KOEQUIPTE
  2. Evaluate if improvements increase KOEQUIPTE performance (target: sMAE < 1.03)
  3. If not, proceed with Phase 2 research plan
  4. Document findings and update report with improvement results
