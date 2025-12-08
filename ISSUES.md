# Issues

## Quick Summary

**Current State:**
- ✅ **Code improvements implemented**: Deeper encoder, tanh activation, weight decay, gradient clipping, Huber loss, improved initialization, increased pre-training, batch size optimization
- ⚠️ **Testing pending**: Models exist (12 .pkl files) but were trained before latest improvements - re-training recommended to test improvements
- 📊 **Baseline metrics**: KOEQUIPTE DDFM sMAE=1.14 (identical to DFM), KOIPALL.G sMAE=0.69 (excellent), KOWRCCNSE sMAE=0.50 (excellent)

**Immediate Next Steps (Priority Order - Concrete Actions):**

**IMMEDIATE (Can be done now, no training required - < 20 minutes total):**
1. **Phase 0: Correlation Structure Analysis** - Execute correlation analysis for all 3 targets
   - **Status**: ✅ Function implemented in `src/evaluation/evaluation_aggregation.py` (lines 1156-1300)
   - **Action**: Run correlation analysis to compare structural differences between targets
   - **Execution**: 
     ```bash
     mkdir -p outputs/analysis
     for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
       python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
     done
     ```
   - **Expected Output**: 3 JSON files in `outputs/analysis/` with correlation statistics
   - **Decision Criteria**:
     - If KOEQUIPTE `negative_fraction > 0.3` and others < 0.2: tanh activation justified
     - If KOEQUIPTE `mean_correlation < 0.1` and others > 0.2: deeper encoder needed
     - If structures are similar: investigate why DDFM works for others but not KOEQUIPTE
   - **Next Action**: Update report discussion section with findings, adjust improvement strategy if needed

**AFTER PHASE 0 (Requires training - ~30-60 minutes per model):**
2. **Phase 1: Re-train Models** - Re-train with latest improvements to test effectiveness
   - **Status**: ⚠️ Models exist (12 .pkl files) but were trained before latest improvements
   - **Action**: Step 1 runs `bash agent_execute.sh train` (automatically applies KOEQUIPTE-specific settings)
   - **KOEQUIPTE-Specific Settings** (automatically applied):
     - Encoder: `[64, 32, 16]` (default: `[16, 4]`)
     - Activation: `tanh` (default: `relu`)
     - Epochs: `150` (default: `100`)
     - Weight decay: `1e-4` (default: `0.0`)
     - Pre-training multiplier: `2` (default: `1`)
     - Batch size: `64` (default: `100`)
   - **Verification**: Check logs for target-specific settings, verify checkpoint files exist
   - **Expected Time**: ~30-60 minutes per model (12 models total, may run in parallel)

3. **Phase 1: Run Forecasting** - Generate new results with improved models
   - **Action**: Step 1 runs `bash agent_execute.sh forecast` after training
   - **Baseline Preservation**: Before forecasting, backup current results:
     ```bash
     cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv
     ```
   - **Results Location**: `outputs/experiments/aggregated_results.csv`
   - **Automatic Analysis**: `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` run automatically
   - **Analysis Output**: `outputs/experiments/ddfm_linearity_analysis.json`

4. **Phase 1: Compare Results** - Compare new results with baseline
   - **Action**: Run comparison script (see Phase 1 section below for full script)
   - **Success Criteria**: 
     - Primary: sMAE improvement ≥ 10% (target: sMAE < 1.03 from baseline 1.14) AND DDFM ≥ 5% better than DFM
     - Secondary: No degradation > 5% at any horizon
     - Tertiary: Horizon 22 completed (n_valid=1)
   - **Decision Tree**: See Phase 1 section for detailed decision criteria

**CONDITIONAL (If Phase 1 shows improvement):**
5. **Phase 1.2: Activation Ablation Study** - Systematic comparison of activation functions
   - **Prerequisite**: Phase 1 shows ≥ 5% improvement
   - **Action**: Train 4 DDFM models for KOEQUIPTE with different activations (relu, tanh, sigmoid, leaky_relu)
   - **Expected Time**: ~2-4 hours (4 models × 30-60 minutes each)

**PARALLEL (Can be done alongside Phase 1):**
6. **Phase 1.3: Horizon 22 Investigation** - Fix missing horizon 22 results
   - **Status**: Currently n_valid=0 for KOIPALL.G and KOEQUIPTE at horizon 22
   - **Action**: Investigate validation logic, numerical stability, index alignment issues
   - **Code Locations**: `src/evaluation/evaluation_forecaster.py`, `src/models/models_forecasters.py`

**Success Criteria:**
- **Primary**: KOEQUIPTE sMAE improvement from 1.14 to < 1.03 (≥10% improvement)
- **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
- **Tertiary**: Complete all 22 horizons for all targets

See detailed plan below for specific actions and execution commands.

---

## CRITICAL: Models May Not Reflect Latest Code Improvements
- **Problem**: Models exist (12 .pkl files from Dec 9 02:35-02:47) but were trained before latest code improvements (mult_epoch_pretrain, batch_size optimization)
- **Impact**: 
  - Current models may not have latest DDFM improvements (mult_epoch_pretrain=2, batch_size=64 for KOEQUIPTE)
  - Cannot verify if latest improvements improve performance without re-training
  - Forecasting results may not reflect latest code improvements
- **Required Action**: Step 1 should run `bash agent_execute.sh train` to re-train models with latest improvements
- **Status**: ⚠️ **MODELS EXIST BUT MAY BE OUTDATED** - Re-training recommended to test latest improvements
- **Verification**: After re-training, check that `checkpoint/` contains 12 model.pkl files with recent timestamps
- **Note**: Models are trained but may not have latest improvements. Re-training will ensure all improvements are applied.

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
- **Status**: Unresolved. Requires investigation and fix before ARIMA can be included in results. Investigation should be done after training completes.

## DDFM Metrics Improvement (Research Plan)
- **Current Performance Analysis** (from aggregated_results.csv - quantitative analysis):
  - **KOIPALL.G**: sMAE=0.69, sMSE=0.61, sRMSE=0.69 (21 horizons) - **Excellent**
    - Horizon-specific: Very low error in short-term (1-6 months, sMAE < 0.15), moderate in mid-term (7-12 months, sMAE 0.2-0.9), stable in long-term (13-21 months, sMAE 0.5-1.3)
    - Missing horizon 22 (n_valid=0) - needs investigation
    - DDFM outperforms DFM by 21.7x (DFM sMAE=14.97 vs DDFM sMAE=0.69)
  - **KOWRCCNSE**: sMAE=0.50, sMSE=0.49, sRMSE=0.50 (22 horizons) - **Excellent**
    - Horizon-specific: Very low error in short-term (1, 4-8 months, sMAE < 0.15), moderate in mid-term (9-16 months, sMAE 0.2-0.5), some spikes at horizons 14, 19-20
    - Horizon 17 has missing sMSE/sMAE but very small MSE/MAE values (possible numerical precision issue)
    - DDFM outperforms DFM by 5.6x (DFM sMAE=2.78 vs DDFM sMAE=0.50)
  - **KOEQUIPTE**: sMAE=1.14, sMSE=2.12, sRMSE=1.14 (21 horizons) - **Moderate** (identical to DFM)
    - Horizon-specific: Moderate error across all horizons (sMAE 1.03-1.07 for short-term 1-3 months, sMAE 0.33-0.76 for mid-term 4-12 months, spikes at horizons 7-8: sMAE 2.33, horizons 13-14: sMAE 3.21-3.28)
    - Missing horizon 22 (n_valid=0) - needs investigation
    - **Critical observation**: DDFM and DFM show nearly identical performance at every single horizon:
      - Maximum sMAE difference: 0.00212 (horizon 2: DFM=1.45051, DDFM=1.44847)
      - Minimum sMAE difference: 0.00001 (horizon 15: DFM=0.08423, DDFM=0.08548)
      - Average sMAE difference: 0.00085 across all 21 horizons
      - Relative difference: < 0.15% at every horizon (|DDFM - DFM| / DFM < 0.0015)
      - This suggests encoder is learning linear relationships only (equivalent to PCA/DFM)

- **Root Cause Analysis**:
  - **ReLU Activation Limitation**: ReLU activation (`max(0, x)`) zeros negative values, which may prevent the encoder from learning negative correlations between factors and observations. KOEQUIPTE may have negative factor loadings that ReLU cannot capture.
  - **Encoder Capacity**: Default encoder `[16, 4]` may be too small to capture complex nonlinear relationships for KOEQUIPTE.
  - **Linear Collapse**: The encoder may be collapsing to linear behavior (equivalent to PCA/DFM) due to insufficient capacity or activation function limitations.

- **Implemented Improvements** (Verified in Code):
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
  6. **Improved Encoder Weight Initialization** (`dfm-python/src/dfm_python/encoder/vae.py`):
     - Added Xavier/Kaiming initialization for encoder layers based on activation function
     - Kaiming initialization for ReLU activations (better for ReLU networks)
     - Xavier initialization for tanh/sigmoid activations (better for symmetric activations)
     - Smaller initialization (gain=0.1) for output layer to prevent large initial factors
     - Rationale: Better weight initialization improves training stability and convergence, especially for deeper networks
     - Status: ✅ **IMPLEMENTED IN CODE** - Improves training stability and convergence
  7. **Factor Order Configuration** (`src/models/models_forecasters.py`, `src/train.py`):
     - Added `factor_order` parameter to DDFMForecaster (default: 1, supports 1 or 2)
     - Allows VAR(2) factor dynamics for targets that may benefit from longer memory
     - VAR(2) can capture multi-period dependencies but requires more data
     - Rationale: Some targets may have complex multi-period dynamics that VAR(1) cannot capture
     - Status: ✅ **IMPLEMENTED IN CODE** - Configurable via model_params['factor_order'], not yet tested (blocked by lack of trained models)
  8. **Increased Pre-Training for KOEQUIPTE** (`src/train.py` lines 407-418):
     - Added `mult_epoch_pretrain` parameter support to DDFMForecaster (default: 1)
     - KOEQUIPTE: Automatically uses `mult_epoch_pretrain=2` (double pre-training epochs)
     - Rationale: More pre-training epochs give encoder more time to learn nonlinear features before MCMC iterations
     - Status: ✅ **IMPLEMENTED IN CODE** - Not tested (models need re-training)
  9. **Batch Size Optimization for KOEQUIPTE** (`src/train.py` lines 420-426):
     - KOEQUIPTE: Automatically uses `batch_size=64` instead of default 100
     - Rationale: Smaller batch sizes provide more diverse gradients per epoch, helping encoder learn nonlinear features
     - Status: ✅ **IMPLEMENTED IN CODE** - Not tested (models need re-training)
  10. **Enhanced Training Stability** (`dfm-python/src/dfm_python/models/ddfm.py`):
     - Improved input clipping for deeper networks (tighter clipping range for networks with >2 layers)
     - Better numerical stability handling in training step
     - Rationale: Deeper networks are more sensitive to extreme values, tighter clipping improves stability
     - Status: ✅ **IMPLEMENTED IN CODE** - Improves training stability for deeper architectures

- **Concrete Research Plan for DDFM Metrics Improvement**:
  
  **Phase 0: Pre-Experiment Data Analysis (Can be done now, before training)**
  1. **Correlation Structure Analysis** (IMMEDIATE ACTION - No training required):
     - **Status**: ✅ Function already implemented in `src/evaluation/evaluation_aggregation.py` - `analyze_correlation_structure()`
     - **Action**: Execute correlation analysis for all 3 targets to compare structural differences
     - **Execution**: 
       ```bash
       mkdir -p outputs/analysis
       for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
         python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
       done
       ```
     - **Key Metrics to Extract and Compare**:
       - `negative_fraction`: Fraction of negative correlations (if KOEQUIPTE > others, tanh activation justified)
       - `strong_negative_count`: Count of correlations < -0.3 (indicates strong negative relationships)
       - `mean_correlation`: Average correlation magnitude (if KOEQUIPTE < others, may indicate weaker signal)
       - `std_correlation`: Correlation distribution spread (indicates structural complexity)
     - **Hypothesis Testing**:
       - If KOEQUIPTE has `negative_fraction > 0.3` and others < 0.2: tanh activation should help
       - If KOEQUIPTE has `mean_correlation < 0.1` and others > 0.2: deeper encoder needed
       - If KOEQUIPTE has similar structure to others: investigate why DDFM works for others but not KOEQUIPTE
     - **Output Actions**:
       - Save results to `outputs/analysis/correlation_analysis_{target}.json` for each target
       - Create comparison table/documentation in report discussion section
       - Update ISSUES.md with findings and decision on improvement strategy
     - **Expected Time**: < 5 minutes per target (15 minutes total)
     - **Decision Point**: Based on correlation analysis, confirm or adjust improvement strategy before training
  
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
  1. **KOEQUIPTE Deeper Encoder + Tanh Activation Test** (REQUIRES TRAINING):
     - **Status**: ✅ Code implemented in `src/train.py` lines 363-426 (automatic for KOEQUIPTE)
     - **Action**: Re-train models with latest improvements (deeper encoder, tanh, weight decay, etc.)
     - **Execution**: Step 1 runs `bash agent_execute.sh train` (automatically applies KOEQUIPTE-specific settings)
     - **KOEQUIPTE-Specific Settings Applied Automatically**:
       - Encoder: `[64, 32, 16]` (default: `[16, 4]`)
       - Activation: `tanh` (default: `relu`)
       - Epochs: `150` (default: `100`)
       - Weight decay: `1e-4` (default: `0.0`)
       - Pre-training multiplier: `2` (default: `1`)
       - Batch size: `64` (default: `100`)
     - **Baseline Preservation** (before training):
       ```bash
       cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv
       ```
     - **Evaluation Method** (after forecasting completes):
       - Load baseline: `outputs/experiments/aggregated_results_baseline.csv`
       - Load new results: `outputs/experiments/aggregated_results.csv`
       - Run comparison script (see lines 424-449 in ISSUES.md) to calculate:
         - Overall sMAE improvement percentage
         - DDFM vs DFM difference (must be ≥ 5% for success)
         - Horizon-specific improvements (especially volatile horizons 7-8, 13-14)
         - Horizon 22 completion status
     - **Success Criteria** (quantitative):
       - **Primary**: sMAE improvement ≥ 10% (target: sMAE < 1.03 from baseline 1.14) AND DDFM ≥ 5% better than DFM
       - **Secondary**: No degradation > 5% at any horizon
       - **Tertiary**: Horizon 22 completed (n_valid=1, currently n_valid=0)
     - **Automatic Analysis**: After forecasting, `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` automatically run
       - Results saved to `outputs/experiments/ddfm_linearity_analysis.json`
       - Provides linearity scores, improvement ratios, and recommendations
     - **Decision Tree**:
       - ✅ **SUCCESS** (improvement ≥ 10% AND DDFM > 5% better): Document improvement, proceed to Phase 1.2 (activation ablation)
       - ⚠️ **PARTIAL** (improvement < 10% but > 0% AND DDFM > 5% better): Document findings, proceed to Phase 1.2
       - ❌ **NEEDS INVESTIGATION** (improvement < 10% OR DDFM still identical): Check logs, verify settings applied, proceed to Phase 2
       - ❌ **FAILURE** (no improvement or degradation): Check logs for errors, investigate root cause, proceed to Phase 2
  
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

- **Success Metrics** (based on quantitative analysis of current results):
  - **Primary**: KOEQUIPTE sMAE improvement from 1.14 to < 1.03 (≥10% improvement, target: 0.90-1.00)
    - Current baseline: sMAE=1.14 (identical to DFM at all 21 horizons, average difference 0.00085)
    - Target: sMAE < 1.03 (10% improvement) or ideally < 1.0 (12% improvement)
    - Success criterion: DDFM sMAE must be at least 5% lower than DFM sMAE (currently 0% difference)
    - Horizon-specific targets: Improve at volatile horizons (7-8, 13-14) where sMAE spikes to 2.33-3.28
  - **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
    - KOIPALL.G: Current sMAE=0.69 (excellent, maintain or improve)
    - KOWRCCNSE: Current sMAE=0.50 (excellent, maintain or improve)
  - **Tertiary**: Complete all 22 horizons for all targets (currently missing horizon 22 for KOIPALL.G and KOEQUIPTE)
    - Fix validation or numerical issues preventing horizon 22 predictions

- **Status**: ✅ **IMPROVEMENTS IMPLEMENTED IN CODE** (code changes verified by inspection). **NOT TESTED** - Improvements cannot be tested until models are re-trained with latest improvements. Research plan defined but Phase 1 testing requires re-training to test latest improvements.

- **Metrics Research Improvements** (Already Implemented):
  1. **Enhanced DDFM Linearity Detection** (`src/evaluation/evaluation_aggregation.py`):
     - Enhanced `detect_ddfm_linearity()` function with performance improvement metrics
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Will automatically detect linearity and improvement when results are aggregated
  2. **DDFM Prediction Quality Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_ddfm_prediction_quality()` function for detailed DDFM performance analysis
     - Includes CV, consistency metrics, linear collapse risk assessment, horizon degradation detection
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Additional diagnostic metrics available for DDFM performance analysis
  3. **Missing Horizons Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_missing_horizons()` function to identify validation failures (n_valid=0)
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Will automatically analyze missing horizons when results are aggregated
  4. **Enhanced Error Distribution Metrics** (`src/evaluation/evaluation_metrics.py`):
     - Added error distribution analysis: skewness, kurtosis, bias-variance decomposition, error concentration
     - Status: ✅ **IMPLEMENTED IN CODE** - Enhanced metrics available in per-horizon evaluation results
  5. **Horizon Error Correlation Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_horizon_error_correlation()` function to analyze error patterns across horizons
     - Status: ✅ **IMPLEMENTED IN CODE** - Available for analyzing DDFM error patterns across horizons

- **What Can Be Done Now (Before Training)**:
  - ✅ **Phase 0: Correlation structure analysis** - `analyze_correlation_structure()` function exists in `src/evaluation/evaluation_aggregation.py`. Can be run on existing data.csv to analyze correlation patterns before training.
  - Code review: Verify all improvements are correctly implemented
  - Planning: Refine research plan based on current results analysis

- **What Requires Training First**:
  - Phase 1: Test implemented improvements (deeper encoder, tanh activation, weight decay)
  - Phase 1.2: Activation function ablation study
  - Phase 1.3: Horizon 22 investigation (needs trained models to generate predictions)
  - Phase 2: Advanced improvements (architecture grid search, factor loading analysis)
  - Phase 3: Ensemble and advanced techniques
  
  **Immediate Next Steps** (in priority order):
  
  **Phase 0: Pre-Training Analysis (Can be done NOW, before training)**
  1. **Correlation Structure Analysis** - Run before training to inform improvement strategy
     - **Status**: ✅ Function implemented, ready to execute (no training required)
     - **Priority**: HIGH - Can be done immediately to inform training strategy
     - **Expected time**: < 5 minutes per target
     - **Action**: Execute `analyze_correlation_structure()` function from `src/evaluation/evaluation_aggregation.py`
     - **Code location**: `src/evaluation/evaluation_aggregation.py` - function `analyze_correlation_structure()`
     - **Execution command** (single target): 
       ```bash
       python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', 'KOEQUIPTE', output_path='outputs/analysis/correlation_analysis_KOEQUIPTE.json'); print(json.dumps(result['summary'], indent=2))"
       ```
     - **Execution command** (all 3 targets for comparison): 
       ```bash
       mkdir -p outputs/analysis
       for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
         python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
       done
       ```
     - **Key metrics to analyze**: 
       - Negative correlation fraction (count of negative correlations / total correlations)
       - Strong negative count (correlations < -0.3)
       - Mean correlation magnitude
       - Correlation distribution (histogram of correlation values)
       - Compare KOEQUIPTE vs KOIPALL.G/KOWRCCNSE to identify structural differences
     - **Hypothesis validation**: 
       - If KOEQUIPTE has higher negative correlation fraction than others, tanh activation should help
       - If KOEQUIPTE has lower correlation magnitude overall, may indicate weaker signal requiring deeper encoder
       - If correlation structure is similar across targets, investigate why DDFM works for KOIPALL.G/KOWRCCNSE but not KOEQUIPTE
     - **Output**: 
       - Save results to JSON file (`outputs/analysis/correlation_analysis_{target}.json`)
       - Document findings in report discussion section (`nowcasting-report/contents/6_discussion.tex`)
       - Update ISSUES.md with correlation analysis findings
     - **Decision**: Based on correlation analysis results, confirm or adjust improvement strategy before training
  
  **Phase 1: Training and Initial Testing (RECOMMENDED - Models exist but may be outdated)**
  2. **Re-train models** - Models exist but may not reflect latest code improvements
     - **Status**: ⚠️ **RECOMMENDED** - Models exist (12 .pkl files) but were trained before latest improvements
     - **Priority**: HIGH - Re-training ensures latest improvements are applied
     - **Expected training time**: ~30-60 minutes per model (12 models total, may run in parallel)
     - **Execution**: Step 1 automatically runs `bash agent_execute.sh train` (automatically uses new settings for KOEQUIPTE)
     - **Code location**: `src/train.py` lines 363-426 (target-specific settings for KOEQUIPTE)
     - **KOEQUIPTE-specific improvements that will be applied automatically**:
       - Deeper encoder: `[64, 32, 16]` (default: `[16, 4]`)
       - Activation: `tanh` (default: `relu`)
       - Epochs: `150` (default: `100`)
       - Weight decay: `1e-4` (default: `0.0`)
       - Pre-training multiplier: `2` (default: `1`)
       - Batch size: `64` (default: `100`)
     - **Baseline preservation** (before training): Backup current results
       ```bash
       cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv
       ```
     - **Verification steps** (after training):
       ```bash
       # Check checkpoint directory
       ls -la checkpoint/*/model.pkl | wc -l  # Should be 12 files
       
       # Verify KOEQUIPTE DDFM checkpoint exists
       test -f checkpoint/KOEQUIPTE_DDFM/model.pkl && echo "KOEQUIPTE DDFM trained" || echo "Missing"
       
       # Check training logs for KOEQUIPTE DDFM
       grep -i "target-specific\|tanh\|weight_decay\|epochs\|mult_epoch\|batch_size" log/KOEQUIPTE_ddfm_*.log | tail -20
       ```
     - **Expected log messages** (in `log/KOEQUIPTE_ddfm_*.log`):
       - "Using target-specific encoder architecture [64, 32, 16]"
       - "Using tanh activation for KOEQUIPTE"
       - "Using weight_decay=1e-4 for KOEQUIPTE"
       - "Increased epochs to 150 for KOEQUIPTE"
       - "Using mult_epoch_pretrain=2 for KOEQUIPTE"
       - "Using batch_size=64 for KOEQUIPTE"
   
  3. **Test implemented improvements** - After training completes
     - **Status**: ⚠️ **BLOCKED** - Requires training and forecasting experiments to complete
     - **Priority**: HIGH - Critical for validating improvement strategy
     - **Prerequisites**: 
       - Models re-trained with latest improvements (Step 2)
       - Forecasting experiments run via `bash agent_execute.sh forecast` (Step 1 automatically runs this)
     - **Baseline**: Current results in `outputs/experiments/aggregated_results_baseline.csv` (KOEQUIPTE DDFM: sMAE=1.14, sMSE=2.12, 21 horizons, identical to DFM)
     - **New results**: After training and forecasting, results in `outputs/experiments/aggregated_results.csv`
     - **Comparison metrics** (compare baseline vs new results):
       - Overall sMAE: Baseline 1.14 → Target < 1.03 (≥10% improvement)
       - DDFM vs DFM difference: Baseline 0% (identical) → Target ≥ 5% (DDFM better than DFM)
       - Horizon-specific: Check improvement at volatile horizons (7-8, 13-14) where sMAE spikes to 2.33-3.28
       - Horizon 22 completion: Baseline n_valid=0 → Target n_valid=1
     - **Comparison script** (run after forecasting completes):
       ```python
       import pandas as pd
       import numpy as np
       
       baseline = pd.read_csv('outputs/experiments/aggregated_results_baseline.csv')
       new = pd.read_csv('outputs/experiments/aggregated_results.csv')
       
       # KOEQUIPTE DDFM comparison
       koequipte_ddfm_baseline = baseline[(baseline['target']=='KOEQUIPTE') & (baseline['model']=='DDFM')]
       koequipte_ddfm_new = new[(new['target']=='KOEQUIPTE') & (new['model']=='DDFM')]
       koequipte_dfm_new = new[(new['target']=='KOEQUIPTE') & (new['model']=='DFM')]
       
       baseline_smae = koequipte_ddfm_baseline['sMAE'].mean()
       new_smae = koequipte_ddfm_new['sMAE'].mean()
       dfm_smae = koequipte_dfm_new['sMAE'].mean()
       improvement_pct = (baseline_smae - new_smae) / baseline_smae * 100
       ddfm_vs_dfm_diff = (dfm_smae - new_smae) / dfm_smae * 100
       
       print(f"Baseline sMAE: {baseline_smae:.4f}")
       print(f"New DDFM sMAE: {new_smae:.4f}")
       print(f"New DFM sMAE: {dfm_smae:.4f}")
       print(f"Improvement vs baseline: {improvement_pct:.2f}%")
       print(f"DDFM vs DFM difference: {ddfm_vs_dfm_diff:.2f}%")
       print(f"Horizon 22 status: baseline={koequipte_ddfm_baseline[koequipte_ddfm_baseline['horizon']==22]['n_valid'].values[0]}, new={koequipte_ddfm_new[koequipte_ddfm_new['horizon']==22]['n_valid'].values[0]}")
       ```
     - **Success criteria** (quantitative): 
       - **Primary**: sMAE improvement ≥ 10% (target: sMAE < 1.03 from baseline 1.14) AND DDFM sMAE must be ≥ 5% lower than DFM sMAE
       - **Secondary**: Maintain performance at all horizons (no degradation > 5% at any horizon)
       - **Tertiary**: Complete horizon 22 (currently n_valid=0, should be n_valid=1)
       - **Horizon-specific**: Improve at volatile horizons (7-8, 13-14) where current sMAE spikes to 2.33-3.28
     - **Decision tree**:
       - **If improvement ≥ 10% AND DDFM > 5% better than DFM**: ✅ SUCCESS - Document percentage improvement, proceed to Phase 1.2 (activation ablation study)
       - **If improvement < 10% but > 0% AND DDFM > 5% better than DFM**: ⚠️ PARTIAL SUCCESS - Document findings, proceed to Phase 1.2
       - **If improvement < 10% OR DDFM still identical to DFM**: ❌ NEEDS INVESTIGATION - Check training logs, verify encoder architecture used, verify all improvements were applied, proceed to Phase 2 root cause analysis
       - **If no improvement or degradation**: ❌ FAILURE - Check training logs for errors, verify all improvements were applied, investigate why improvements didn't work, proceed to Phase 2
     - **Documentation**: 
       - Update `nowcasting-report/contents/6_discussion.tex` with improvement percentage, DDFM vs DFM comparison, and findings
       - Update `nowcasting-report/contents/3_results_forecasting.tex` with new metrics if improved
       - Update `ISSUES.md` with Phase 1 results and next steps
       - Regenerate tables/plots if results change significantly
   
  **Phase 1.2: Activation Ablation Study (If Phase 1 shows improvement)**
  4. **Systematic Activation Function Comparison** (REQUIRES TRAINING):
     - **Status**: ⚠️ **CONDITIONAL** - Only if Phase 1 shows improvement (≥ 5% improvement)
     - **Method**: Train 4 DDFM models for KOEQUIPTE with different activations
     - **Activations to Test**: 'relu', 'tanh', 'sigmoid', 'leaky_relu'
     - **Implementation**: 
       - Modify `src/train.py` to accept activation parameter override via config
       - Or create separate config files for each activation
       - Keep all other settings constant (encoder [64, 32, 16], epochs 150, weight_decay 1e-4, etc.)
     - **Execution**: Train 4 models sequentially or in parallel (if resources allow)
     - **Evaluation**: Compare sMAE/sMSE/sRMSE across all 4 activations
     - **Output**: 
       - Performance comparison table (activation vs metrics)
       - Best activation function identification
       - Update report with findings
       - Document in `nowcasting-report/contents/6_discussion.tex`
     - **Expected Time**: ~2-4 hours (4 models × 30-60 minutes each)
     - **Success Criterion**: Identify activation that gives best performance for KOEQUIPTE
   
  **Phase 1.3: Horizon 22 Investigation (Parallel with Phase 1)**
  5. **Fix Missing Horizon 22 Results** (REQUIRES TRAINING):
     - **Status**: ⚠️ **INVESTIGATION NEEDED** - Currently n_valid=0 for KOIPALL.G and KOEQUIPTE at horizon 22
     - **Current State**: KOWRCCNSE has horizon 22 (n_valid=1), but KOIPALL.G and KOEQUIPTE missing
     - **Investigation Steps**:
       1. Check training logs: `grep -i "horizon.*22\|error\|failed" log/*_ddfm_*.log | grep -i "KOEQUIPTE\|KOIPALL"`
       2. Check evaluation logs: `grep -i "horizon.*22\|validation\|n_valid" log/*evaluation*.log`
       3. Verify prediction generation: Check if predictions are generated but fail validation
       4. Check numerical stability: Verify if predictions are NaN or inf at horizon 22
     - **Code Locations to Check**:
       - `src/evaluation/evaluation_forecaster.py`: Validation logic (lines checking n_valid)
       - `src/models/models_forecasters.py`: Prediction generation for horizon 22
       - `src/evaluation/evaluation_metrics.py`: Metric calculation that might filter horizon 22
     - **Possible Causes**:
       - Validation threshold too strict (prediction variance too high)
       - Numerical instability at longest horizon (22 months)
       - Index alignment issue (prediction index doesn't match test index)
       - Shape mismatch (prediction shape incorrect for horizon 22)
     - **Fix Actions**:
       - If validation threshold issue: Adjust validation criteria for horizon 22
       - If numerical instability: Add numerical stability checks or clipping
       - If index alignment: Fix index alignment in prediction generation
       - If shape mismatch: Fix prediction shape calculation for horizon 22
     - **Verification**: After fix, re-run forecasting and verify n_valid=1 for horizon 22
     - **Expected Time**: 1-2 hours (investigation + fix + verification)
   
  **Phase 2: Advanced Improvements (If Phase 1 fails)**
  6. **Architecture grid search** - Test different encoder architectures
  7. **Factor loading analysis** - Compare DFM vs DDFM learned factors
  8. **Regularization experiments** - Test dropout, L1/L2 combinations
   
  **Documentation Updates (After each phase)**
  9. **Update report sections** with findings:
     - `nowcasting-report/contents/6_discussion.tex`: Add Phase 0/1 results, improvement percentages, correlation analysis findings
     - `nowcasting-report/contents/7_issues.tex`: Mark resolved issues, update research status, document Phase 1 results
     - `nowcasting-report/contents/3_results_forecasting.tex`: Update metrics if improved, regenerate tables/plots if results change
     - `STATUS.md`: Update experiment status, document Phase 0/1 findings, update next iteration priorities

---

## DDFM Metrics Improvement Summary

**Current State:**
- ✅ **Code improvements implemented**: Deeper encoder, tanh activation, weight decay, gradient clipping, Huber loss, improved initialization, increased pre-training, batch size optimization
- ⚠️ **Testing pending**: Models exist (12 .pkl files) but were trained before latest improvements - re-training recommended to test improvements
- 📊 **Baseline metrics**: KOEQUIPTE DDFM sMAE=1.14 (identical to DFM), KOIPALL.G sMAE=0.69 (excellent), KOWRCCNSE sMAE=0.50 (excellent)

**Improvement Plan Structure:**
1. **Phase 0 (Pre-training)**: Correlation structure analysis - can be done now, no training required
2. **Phase 1 (Initial testing)**: Test implemented improvements after training - requires trained models
3. **Phase 1.2 (Ablation)**: Activation function comparison - if Phase 1 shows improvement
4. **Phase 1.3 (Investigation)**: Horizon 22 fix - parallel with Phase 1
5. **Phase 2 (Advanced)**: Architecture grid search, factor analysis, regularization - if Phase 1 fails
6. **Phase 3 (Advanced techniques)**: Ensemble, feature engineering, hybrid approach - if Phase 2 fails

**Success Criteria:**
- **Primary**: KOEQUIPTE sMAE improvement from 1.14 to < 1.03 (≥10% improvement)
- **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
- **Tertiary**: Complete all 22 horizons for all targets (currently missing horizon 22 for KOIPALL.G and KOEQUIPTE)

**Next Immediate Actions (Priority Order - Concrete Steps):**

**IMMEDIATE (Can be done now, no training required):**
1. **Phase 0: Correlation Structure Analysis** (15 minutes total)
   - **Execute**: Run correlation analysis for all 3 targets:
     ```bash
     mkdir -p outputs/analysis
     for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
       python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
     done
     ```
   - **Analyze Results**: Compare `negative_fraction`, `strong_negative_count`, `mean_correlation` across targets
   - **Decision**: If KOEQUIPTE has `negative_fraction > 0.3` and others < 0.2, tanh activation is justified
   - **Document**: Update `nowcasting-report/contents/6_discussion.tex` with correlation analysis findings
   - **Update**: Add findings to ISSUES.md Phase 0 section

**AFTER TRAINING (Requires models to be re-trained):**
2. **Phase 1: Re-train Models** (Step 1 automatically runs `bash agent_execute.sh train`)
   - **Before Training**: Backup baseline results
     ```bash
     cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv
     ```
   - **After Training**: Verify KOEQUIPTE DDFM settings applied
     ```bash
     # Check checkpoint exists
     test -f checkpoint/KOEQUIPTE_DDFM/model.pkl && echo "KOEQUIPTE DDFM trained" || echo "Missing"
     
     # Check logs for target-specific settings
     grep -i "target-specific\|tanh\|weight_decay\|epochs\|mult_epoch\|batch_size" log/KOEQUIPTE_ddfm_*.log | tail -20
     ```
   - **Expected Log Messages**: "Using target-specific encoder architecture [64, 32, 16]", "Using tanh activation for KOEQUIPTE", etc.

3. **Phase 1: Run Forecasting** (Step 1 automatically runs `bash agent_execute.sh forecast`)
   - **Results Location**: `outputs/experiments/aggregated_results.csv`
   - **Automatic Analysis**: `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` run automatically
   - **Analysis Output**: `outputs/experiments/ddfm_linearity_analysis.json` (linearity scores, improvement ratios)

4. **Phase 1: Compare Results with Baseline** (Run after forecasting completes)
   - **Execute Comparison Script**:
     ```python
     import pandas as pd
     import numpy as np
     
     baseline = pd.read_csv('outputs/experiments/aggregated_results_baseline.csv')
     new = pd.read_csv('outputs/experiments/aggregated_results.csv')
     
     # KOEQUIPTE DDFM comparison
     koequipte_ddfm_baseline = baseline[(baseline['target']=='KOEQUIPTE') & (baseline['model']=='DDFM')]
     koequipte_ddfm_new = new[(new['target']=='KOEQUIPTE') & (new['model']=='DDFM')]
     koequipte_dfm_new = new[(new['target']=='KOEQUIPTE') & (new['model']=='DFM')]
     
     baseline_smae = koequipte_ddfm_baseline['sMAE'].mean()
     new_smae = koequipte_ddfm_new['sMAE'].mean()
     dfm_smae = koequipte_dfm_new['sMAE'].mean()
     improvement_pct = (baseline_smae - new_smae) / baseline_smae * 100
     ddfm_vs_dfm_diff = (dfm_smae - new_smae) / dfm_smae * 100
     
     print(f"Baseline sMAE: {baseline_smae:.4f}")
     print(f"New DDFM sMAE: {new_smae:.4f}")
     print(f"New DFM sMAE: {dfm_smae:.4f}")
     print(f"Improvement vs baseline: {improvement_pct:.2f}%")
     print(f"DDFM vs DFM difference: {ddfm_vs_dfm_diff:.2f}%")
     print(f"Horizon 22 status: baseline={koequipte_ddfm_baseline[koequipte_ddfm_baseline['horizon']==22]['n_valid'].values[0]}, new={koequipte_ddfm_new[koequipte_ddfm_new['horizon']==22]['n_valid'].values[0]}")
     ```
   - **Success Criteria**: sMAE improvement ≥ 10% (target < 1.03) AND DDFM ≥ 5% better than DFM
   - **Documentation**: 
     - Update `nowcasting-report/contents/6_discussion.tex` with improvement percentage and findings
     - Update `nowcasting-report/contents/3_results_forecasting.tex` if metrics improved
     - Update ISSUES.md with Phase 1 results and next steps
     - Regenerate tables/plots if results change significantly

**Key Files:**
- **Code**: `src/train.py` (lines 363-426) - target-specific settings for KOEQUIPTE
- **Analysis**: `src/evaluation/evaluation_aggregation.py` - `analyze_correlation_structure()` function
- **Results**: `outputs/experiments/aggregated_results.csv` - forecasting results
- **Baseline**: `outputs/experiments/aggregated_results_baseline.csv` - backup of current results
- **Report**: `nowcasting-report/contents/6_discussion.tex` - discussion section with improvement analysis
