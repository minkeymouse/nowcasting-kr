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
- **Note**: Report sections have been updated to document current state and implemented improvements. Once experiments are run, tables and plots should be regenerated to reflect new results.

## CRITICAL: DFM/DDFM Backtest CUDA Tensor Conversion Error (Fixed in Code, Needs Re-run)
- **Problem**: All DDFM and DFM backtest results failed with error: "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
- **Root Cause**: The `_convert_predictions_to_dataframe` function and other prediction conversion functions used `np.asarray()` directly on CUDA tensors without first moving them to CPU.
- **Impact**: All 22 months of backtest results for DFM and DDFM models failed (6 JSON files × 22 months = 132 failed predictions). All backtest JSON files in `outputs/backtest/` show "status": "failed" with CUDA errors.
- **Resolution** (Previous Iteration): Fixed tensor conversion in multiple locations:
  - `src/models/models_utils.py`: `_convert_predictions_to_dataframe()` and `_validate_predictions()` - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_forecaster.py`: Prediction value extraction - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_metrics.py`: Metric calculation - Added `.cpu().numpy()` pattern
  - All now convert CUDA tensors to CPU before numpy conversion using `.cpu().numpy()`
- **Status**: ✅ **FIXED IN CODE** (code changes verified by inspection). **NOT VERIFIED BY EXPERIMENTS** - Backtest experiments MUST be re-run via `bash agent_execute.sh backtest` after training to verify fix works. Current backtest JSON files all show "failed" status with CUDA errors. Fix may work, but needs experimental verification.

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
  4. **Weight Decay (L2 Regularization)** (`dfm-python/src/dfm_python/models/ddfm.py`, `src/train.py`, `src/models/models_forecasters.py`):
     - Added `weight_decay` parameter to DDFM model and DDFMForecaster (default: 0.0)
     - KOEQUIPTE: Automatically uses weight_decay=1e-4 to prevent encoder from collapsing to linear behavior
     - Rationale: L2 regularization encourages encoder to learn diverse features, preventing overfitting to linear PCA-like solutions
     - Applied to all optimizer instances (configure_optimizers, _create_optimizer, pre_train)
     - Status: ✅ **IMPLEMENTED IN CODE** - Needs experiments after training to test effectiveness
  5. **Gradient Clipping Improvements** (`dfm-python/src/dfm_python/models/ddfm.py`, `dfm-python/src/dfm_python/models/mcmc.py`):
     - Added `grad_clip_val` parameter to DDFM model (default: 1.0)
     - Gradient clipping now uses configurable value instead of hardcoded 1.0
     - Applied to pre_train, MCMC training, and Lightning training_step (via gradient norm logging)
     - Rationale: Prevents training instability and gradient explosion that can cause NaN values or linear collapse
     - Status: ✅ **IMPLEMENTED IN CODE** - Improves training stability

- **Concrete Research Plan for DDFM Metrics Improvement**:
  
  **Phase 0: Pre-Experiment Data Analysis (Can be done now, before training)**
  1. **Correlation Structure Analysis**:
     - **Action**: Create script to analyze correlation matrices of KOEQUIPTE vs KOIPALL.G/KOWRCCNSE
     - **Location**: Add to `src/evaluation/` or create `src/analysis/correlation_analysis.py`
     - **Metrics**: 
       - Cross-correlation between target series and all input series
       - Factor loading patterns (if available from DFM training)
       - Negative correlation count and magnitude
     - **Hypothesis**: If KOEQUIPTE has more negative correlations, tanh activation should help
     - **Output**: Report/document correlation patterns to inform activation function choice
  
  2. **Factor Space Linearity Check**:
     - **Action**: Compare DFM factor loadings with DDFM encoder outputs (if available from previous training)
     - **Location**: Analyze existing model checkpoints or factor outputs
     - **Method**: 
       - Extract factor loadings from DFM (linear)
       - Extract encoder weights from DDFM (if saved)
       - Compare similarity using cosine similarity or correlation
     - **Hypothesis**: If encoder weights are highly similar to DFM loadings, encoder is learning linear features
     - **Output**: Quantify linearity of learned features

  **Phase 1: Test Current Improvements (After Training)**
  1. **KOEQUIPTE Deeper Encoder + Tanh Activation Test**:
     - **Action**: Re-run training with deeper encoder `[64, 32, 16]`, tanh activation, and 150 epochs
     - **Code location**: Already implemented in `src/train.py` lines 363-380 (automatic for KOEQUIPTE)
     - **Execution**: `bash agent_execute.sh train` (will automatically use new settings for KOEQUIPTE)
     - **Evaluation**: Compare sMAE/sMSE/sRMSE with current results (baseline: sMAE=1.14)
     - **Success criterion**: sMAE improvement of at least 10% (target: sMAE < 1.03)
     - **Comparison method**: 
       - Load old results from `outputs/experiments/aggregated_results.csv`
       - Compare with new results after re-training
       - Generate comparison table/plot
     - **If no improvement**: Proceed to Phase 2 root cause analysis
     - **Expected improvement**: Tanh activation should allow encoder to learn negative correlations that ReLU cannot capture
  
  2. **Activation Function Ablation Study** (If Phase 1 shows improvement):
     - **Action**: Systematically compare 'relu' vs 'tanh' vs 'sigmoid' vs 'leaky_relu' for KOEQUIPTE
     - **Implementation**: Modify `src/train.py` to accept activation parameter override, or create config variants
     - **Method**: Train 4 models with different activations, compare metrics
     - **Documentation**: Create table showing activation vs performance for each target
     - **Hypothesis**: Tanh should outperform ReLU for KOEQUIPTE if negative correlations are important
     - **Output**: Best activation function per target, documented in report
  
  3. **Huber Loss Robustness Test**:
     - **Action**: Test Huber loss (`loss_function: 'huber'`) for all three targets
     - **Implementation**: Already supported in code (`src/train.py` line 383-384), add config option
     - **Method**: Train models with `loss_function: 'huber'` and `huber_delta: [0.5, 1.0, 2.0]`
     - **Comparison**: Compare with MSE baseline, especially for volatile horizons
     - **Target horizons**: KOIPALL.G horizons 11-13, KOWRCCNSE horizons 14, 19-20, KOEQUIPTE horizons 7-8, 13-14
     - **Success criterion**: Reduced error at volatile horizons without degrading overall performance (>5% improvement at volatile horizons)
     - **Output**: Best huber_delta per target, performance comparison table
  
  4. **Horizon 22 Investigation**:
     - **Action**: Investigate why horizon 22 fails for KOIPALL.G and KOEQUIPTE
     - **Method**: 
       - Check training logs in `log/` directory for horizon 22 errors
       - Check if predictions are generated but fail validation
       - Verify numerical stability at longest horizon (22 months)
     - **Code locations**: 
       - `src/evaluation/evaluation_forecaster.py` (validation logic)
       - `src/models/models_forecasters.py` (prediction generation)
     - **Fix**: Address validation or numerical issues to complete all 22 horizons
     - **Output**: Fixed validation logic or numerical stability improvements

  **Phase 2: Advanced Improvements (If Phase 1 Doesn't Improve KOEQUIPTE)**
  1. **Encoder Architecture Grid Search**:
     - **Action**: Systematically test different encoder architectures for KOEQUIPTE
     - **Architectures to test**: `[32, 16, 8]`, `[64, 32, 16]`, `[128, 64, 32]`, `[64, 32, 16, 8]`, `[128, 64, 32, 16]`
     - **Implementation**: Create script `src/experiments/grid_search_encoder.py` or modify `src/train.py` to accept architecture list
     - **Method**: Train models with each architecture, compare performance
     - **Constraint**: Must not degrade KOIPALL.G and KOWRCCNSE performance (maintain sMAE < 0.7)
     - **Output**: Best architecture per target, performance vs architecture table
  
  2. **Factor Loading Analysis**:
     - **Action**: Analyze learned factor loadings from DFM vs DDFM for KOEQUIPTE
     - **Method**: 
       - Extract factor loadings from trained DFM model
       - Extract encoder weights/activations from trained DDFM model
       - Compare using PCA, correlation analysis, or visualization
     - **Hypothesis**: If DDFM encoder learns similar loadings to DFM, it's learning linear features
     - **Implementation**: Create `src/analysis/factor_analysis.py` to extract and compare factors
     - **Output**: Factor loading comparison plots, linearity score
  
  3. **Regularization Experiments**:
     - **Action**: Test dropout and L1/L2 regularization for KOEQUIPTE encoder
     - **Implementation**: 
       - Add dropout parameter to `dfm-python/src/dfm_python/models/ddfm.py` Encoder class
       - Add weight_decay parameter for L2 regularization in optimizer
     - **Method**: Train models with different regularization strengths
     - **Hypothesis**: Regularization may prevent overfitting to linear features, encourage nonlinear learning
     - **Output**: Best regularization settings, performance comparison

  **Phase 3: Ensemble and Advanced Techniques** (If Phase 2 Still Doesn't Improve)
  1. **Ensemble Methods**:
     - **Action**: Implement weighted ensemble of DFM and DDFM predictions for KOEQUIPTE
     - **Implementation**: Create `src/models/ensemble_forecaster.py`
     - **Method**: 
       - Train both DFM and DDFM
       - Learn ensemble weights on validation set (simple linear regression or grid search)
       - Generate ensemble predictions: `w * DDFM + (1-w) * DFM`
     - **Rationale**: If both models learn similar features, ensemble may reduce variance
     - **Output**: Best ensemble weights, ensemble performance metrics
  
  2. **Feature Engineering**:
     - **Action**: Pre-process KOEQUIPTE data with different transformations or interaction features
     - **Implementation**: Modify `src/data/preprocessing.py` to add feature engineering step
     - **Methods to test**:
       - Different transformations (log, sqrt, box-cox)
       - Interaction features (product, ratio of key series)
       - Lagged features (if not already included)
     - **Rationale**: If raw data has weak nonlinear signal, engineered features may help
     - **Output**: Best feature engineering approach, performance comparison
  
  3. **Hybrid Approach**:
     - **Action**: Use DFM for KOEQUIPTE (if it's truly linear) and DDFM for others
     - **Implementation**: Modify `src/train.py` to conditionally use DFM for KOEQUIPTE
     - **Rationale**: If KOEQUIPTE is fundamentally linear, simpler model may be better
     - **Output**: Performance comparison of hybrid vs pure DDFM approach

- **Success Metrics**:
  - **Primary**: KOEQUIPTE sMAE improvement from 1.14 to < 1.0 (12% improvement)
  - **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
  - **Tertiary**: Complete all 22 horizons for all targets (currently missing horizon 22 for KOIPALL.G and KOEQUIPTE)

- **Status**: ✅ **IMPROVEMENTS IMPLEMENTED IN CODE** (code changes verified by inspection). **NOT TESTED** - Improvements cannot be tested until models are trained. Research plan defined but Phase 1 testing blocked by lack of trained models.
  
  **Immediate Next Steps** (in priority order):
  1. **BLOCKING**: Train models first - `checkpoint/` is empty, blocking all experiments
     - Execute: `bash agent_execute.sh train` (will automatically use new settings for KOEQUIPTE)
     - Verification: Check `checkpoint/` contains 12 model.pkl files
  2. **Phase 1**: Test implemented improvements after training completes
     - Evaluate: Compare new KOEQUIPTE DDFM results with baseline (target: sMAE < 1.03 from 1.14)
     - Compare: Load baseline from `outputs/experiments/aggregated_results.csv`, compare with new results
     - If improvement: Document and proceed to activation ablation study
     - If no improvement: Proceed to Phase 2 root cause analysis
  3. **Phase 1**: Investigate horizon 22 failures (can be done in parallel after training)
     - Check logs, fix validation/numerical issues
  4. **Phase 0** (Optional): Perform correlation structure analysis (can be done now, before training)
     - Create `src/analysis/correlation_analysis.py` to analyze KOEQUIPTE correlation patterns
     - Document findings to inform activation function choice
  5. **Documentation**: Update report sections with research findings and improvement results (after Phase 1)
     - Update `nowcasting-report/contents/6_discussion.tex` with Phase 1 results
     - Update `nowcasting-report/contents/7_issues.tex` with resolved issues
