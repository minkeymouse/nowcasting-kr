# Issues

## Quick Summary

**Current State:**
- ✅ **Code improvements implemented** (previous iterations): Deeper encoder, tanh activation, weight decay, gradient clipping, Huber loss, improved initialization, increased pre-training, batch size optimization
- ❌ **Models NOT trained**: `checkpoint/` is EMPTY - no model.pkl files exist. Training is REQUIRED before any experiments can proceed.
- 📊 **Baseline metrics** (from aggregated_results.csv):
  - **KOEQUIPTE**: DDFM sMAE=1.1441, DFM sMAE=1.1439 (identical, linear collapse confirmed)
  - **KOIPALL.G**: DDFM sMAE=0.69 (21.7x better than DFM sMAE=14.97) - excellent
  - **KOWRCCNSE**: DDFM sMAE=0.50 (5.6x better than DFM sMAE=2.78) - excellent
- ⚠️ **Phase 0 not executed**: Correlation structure analysis function exists but has not been run yet - can be done immediately without training (~15 minutes)
- ⚠️ **This iteration**: Tables/plots regenerated (Dec 9 09:03), documentation updated (STATUS.md, ISSUES.md, CONTEXT.md). No code changes, no experiments run.

**Quick Action Reference:**
1. **REQUIRED (Before experiments)**: Train models via `bash agent_execute.sh train` - `checkpoint/` is EMPTY, training is REQUIRED
2. **IMMEDIATE (No training needed)**: Run Phase 0 correlation analysis - see "PRIORITY 0" below (~15 min)
3. **IMMEDIATE (No training needed)**: Run baseline metrics analysis - see "PRIORITY 0" below (~5 min)
4. **After training**: Run forecasting via `bash agent_execute.sh forecast` to test improvements
5. **After training**: Run backtesting via `bash agent_execute.sh backtest` to verify CUDA fixes

**Immediate Next Steps (Priority Order - Concrete Actions):**

**PRIORITY 0: Phase 0 Correlation Analysis (IMMEDIATE - Can be done now, no training required - ~15 minutes total)**

⚠️ **CRITICAL: This should be executed BEFORE training to inform improvement strategy and save experimental time.**

1. **Execute Phase 0: Correlation Structure Analysis** - Run correlation analysis for all 3 targets
   - **Status**: ✅ Function implemented in `src/evaluation/evaluation_aggregation.py` (lines 1156-1300), ⚠️ **NOT YET EXECUTED**
   - **Why Now**: This analysis can inform improvement strategy before training, saving experimental time. Results will help validate whether tanh activation and deeper encoder strategies are appropriate for KOEQUIPTE.
   - **Action Required**: Execute correlation analysis to compare structural differences between targets
   - **Execution Command** (Step 1 should run this automatically, or can be run manually):
     ```bash
     mkdir -p outputs/analysis
     for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
       python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
     done
     ```
   - **Expected Output**: 3 JSON files in `outputs/analysis/correlation_analysis_{target}.json` with correlation statistics
   - **Key Metrics to Extract**:
     - `negative_fraction`: Fraction of negative correlations (if KOEQUIPTE > 0.3 and others < 0.2: tanh activation justified)
     - `strong_negative_count`: Count of correlations < -0.3 (indicates strong negative relationships)
     - `mean_correlation`: Average correlation magnitude (if KOEQUIPTE < 0.1 and others > 0.2: deeper encoder needed)
     - `std_correlation`: Correlation distribution spread (indicates structural complexity)
   - **Decision Criteria** (after analysis):
     - If KOEQUIPTE `negative_fraction > 0.3` and others < 0.2: tanh activation is justified, proceed with current strategy
     - If KOEQUIPTE `mean_correlation < 0.1` and others > 0.2: deeper encoder is needed, current strategy is appropriate
     - If structures are similar: investigate why DDFM works for others but not KOEQUIPTE, may need Phase 2 approaches
   - **Next Action After Analysis**: 
     - Update `nowcasting-report/contents/6_discussion.tex` with correlation analysis findings
     - Update this ISSUES.md Phase 0 section with actual results
     - Adjust improvement strategy if needed based on findings
   - **Time Estimate**: < 5 minutes per target (15 minutes total)

**AFTER PHASE 0 (Training required - ~30-60 minutes per model):**
2. **Phase 1: Train Models** - Models do not exist, training is REQUIRED
   - **Status**: ❌ Models do not exist - `checkpoint/` is EMPTY - training is REQUIRED
   - **Action**: Step 1 must run `bash agent_execute.sh train` (automatically applies KOEQUIPTE-specific settings)
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

## Models NOT Trained (CRITICAL - Training Required)
- **Current State**: `checkpoint/` directory is EMPTY - no model.pkl files exist
- **Issue**: Training is REQUIRED before any experiments can proceed
- **Impact**: 
  - Cannot run forecasting experiments (no models to generate predictions)
  - Cannot run backtesting experiments (no models to verify CUDA fixes)
  - Cannot test DDFM improvements (no models to compare performance)
- **Action Required**: Step 1 must run `bash agent_execute.sh train` to train all 12 models (3 targets × 4 models: ARIMA, VAR, DFM, DDFM) with latest improvements
- **Status**: ❌ **NOT TRAINED** - `checkpoint/` is EMPTY - training is REQUIRED
- **Verification**: `checkpoint/` is EMPTY (no model.pkl files exist)
- **Note**: All DDFM improvements are implemented in code and will be automatically applied during training (deeper encoder, tanh, weight_decay, mult_epoch_pretrain=2, batch_size=64 for KOEQUIPTE)

## CRITICAL: DFM/DDFM Backtest CUDA Tensor Conversion Error (Fixed in Code, Needs Training + Re-run)
- **Problem**: All DDFM and DFM backtest results failed with error: "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
- **Root Cause**: The `_convert_predictions_to_dataframe` function and other prediction conversion functions used `np.asarray()` directly on CUDA tensors without first moving them to CPU.
- **Impact**: All 22 months of backtest results for DFM and DDFM models failed (6 JSON files × 22 months = 132 failed predictions). All backtest JSON files in `outputs/backtest/` show "status": "failed" with CUDA errors.
- **Resolution** (Previous Iteration): Fixed tensor conversion in multiple locations:
  - `src/models/models_utils.py`: `_convert_predictions_to_dataframe()` and `_validate_predictions()` - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_forecaster.py`: Prediction value extraction - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_metrics.py`: Metric calculation - Added `.cpu().numpy()` pattern
  - All now convert CUDA tensors to CPU before numpy conversion using `.cpu().numpy()`
- **Status**: ✅ **FIXED IN CODE** (code changes verified by inspection). **NOT VERIFIED BY EXPERIMENTS** - Models exist, backtest experiments MUST be re-run via `bash agent_execute.sh backtest` to verify fix works. Current backtest JSON files all show "failed" status with CUDA errors. Fix may work, but needs experimental verification.


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

**EXECUTIVE SUMMARY - Concrete Actionable Plan:**
- **Problem**: KOEQUIPTE DDFM shows linear collapse (sMAE=1.14, identical to DFM at all 21 horizons, max difference 0.00212)
- **Goal**: Improve KOEQUIPTE DDFM sMAE from 1.14 to < 1.03 (≥10% improvement) AND make DDFM ≥ 5% better than DFM
- **Current Status**: 
  - ✅ Code improvements implemented (deeper encoder, tanh, weight_decay, etc.)
  - ✅ Models exist in `checkpoint/` (12 files, trained Dec 9) but trained BEFORE latest improvements
  - ⚠️ Phase 0 correlation analysis NOT executed (can be done now, ~15 min, no training required)
  - ❌ Latest improvements NOT tested (models need re-training to apply improvements)
- **Next Immediate Action**: Phase 0 correlation analysis (15 minutes, no training required) → Then Phase 1 re-training and testing
- **Success Metrics**: sMAE < 1.03, DDFM > 5% better than DFM, linearity score < 0.95, horizon 22 completed
- **Metrics Research Strategy**: Use existing comprehensive DDFM metrics analysis functions (already implemented) to guide improvements:
  - **Phase 0 (Pre-training)**: `analyze_correlation_structure()` - Analyze data structure to inform improvement strategy (~15 min, no training required)
  - **Phase 1 (Post-training)**: `detect_ddfm_linearity()` + `analyze_ddfm_prediction_quality()` - Automatically run after aggregation to monitor improvements
  - **Key Metrics to Track**:
    - Linearity score: Target < 0.95 (current: ~0.99 for KOEQUIPTE, estimated from identical performance)
    - Improvement ratio: Target > 10% (current: ~0% for KOEQUIPTE, computed from aggregated_results.csv)
    - Error distribution differences: Target skewness diff > 0.2, kurtosis diff > 1.0 (indicates DDFM learning different patterns)
    - Horizon-weighted improvement: Target > 5% (prioritizes short-term 1-6 months)
    - Consistency metric: Target > 0.7 (measures consistent improvement across horizons)

**What Can Be Done NOW (No Training Required):**
1. **Phase 0: Correlation Structure Analysis** - Execute `analyze_correlation_structure()` for all 3 targets (~15 min)
   - Analyze correlation patterns to validate tanh activation and deeper encoder strategy
   - Compare KOEQUIPTE with KOIPALL.G and KOWRCCNSE to identify structural differences
   - Update report sections with findings before training
2. **Analyze Current Results** - Use existing `aggregated_results.csv` to compute baseline metrics
   - Run `detect_ddfm_linearity()` on current results to get baseline linearity score
   - Run `analyze_ddfm_prediction_quality()` on current results to get baseline improvement ratio
   - Document baseline metrics in ISSUES.md for comparison after re-training

**What Requires Training:**
1. **Phase 1: Re-train and Test** - Train models with latest improvements, then run forecasting
   - Apply KOEQUIPTE-specific settings (deeper encoder, tanh, weight_decay, etc.)
   - Compare new results with baseline to measure improvement
   - Use automatic analysis functions to track metrics

**CONCRETE EXECUTION CHECKLIST (Priority Order):**

**✅ IMMEDIATE (No training required - Can execute now):**
- [ ] **Phase 0: Correlation Structure Analysis** (~15 minutes)
  - **Status**: ⚠️ **NOT EXECUTED** - Function exists, ready to run
  - **Execute**: `bash -c "mkdir -p outputs/analysis && for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do python3 -c \"from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '\$target', output_path='outputs/analysis/correlation_analysis_\${target}.json'); print(f'\\n=== \$target ==='); print(json.dumps(result['summary'], indent=2))\"; done"`
  - **Analyze**: Compare `negative_fraction`, `strong_negative_count`, `mean_correlation`, `std_correlation` across targets
  - **Quantitative Decision Criteria**:
    - If KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2 → tanh activation justified (ReLU limitation confirmed)
    - If KOEQUIPTE `strong_negative_count > 10` AND others < 5 → tanh activation critical (strong negative relationships)
    - If KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2 → deeper encoder needed (weak signal requires more capacity)
    - If KOEQUIPTE `std_correlation < 0.15` AND others > 0.25 → structural complexity differs (may need different approach)
  - **Document**: 
    - Update `nowcasting-report/contents/6_discussion.tex` with correlation findings and decision rationale
    - Update ISSUES.md Phase 0 section with actual metrics and decision
    - If findings contradict current strategy, adjust improvement plan before training

- [ ] **Baseline Metrics Analysis** (~5 minutes)
  - **Status**: ⚠️ **NOT EXECUTED** - Can run now on existing aggregated_results.csv
  - **Execute**: Load `outputs/experiments/aggregated_results.csv`, run `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()`, save to `outputs/analysis/baseline_*.json`
  - **Document**: Save baseline metrics to ISSUES.md for comparison after training
  - **Purpose**: Establish quantitative baseline to measure improvement after Phase 1

**⏳ AFTER TRAINING (Requires trained models):**
- [ ] **Phase 1: Baseline Preservation** (Before training)
  - Execute: `cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv`
  - Verify: Baseline file exists with 265 lines
  - **Also preserve baseline analysis**: Copy `outputs/analysis/baseline_linearity.json` and `outputs/analysis/baseline_quality.json` if they exist
  
- [ ] **Phase 1: Train Models** (Step 1 runs automatically via `bash agent_execute.sh train`)
  - Verify: `checkpoint/` contains 12 model.pkl files after training
  - Check logs: `grep -i "target-specific\|tanh\|weight_decay\|epochs\|mult_epoch\|batch_size" log/KOEQUIPTE_ddfm_*.log | tail -20`
  - Expected: "Using target-specific encoder architecture [64, 32, 16]", "Using tanh activation for KOEQUIPTE", etc.

- [ ] **Phase 1: Run Forecasting** (Step 1 runs automatically via `bash agent_execute.sh forecast`)
  - Results: `outputs/experiments/aggregated_results.csv`
  - Auto-analysis: `outputs/experiments/ddfm_linearity_analysis.json` (generated automatically)

- [ ] **Phase 1: Compare Results** (After forecasting completes)
  - **Automatic Analysis**: `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` run automatically after aggregation
    - Results saved to `outputs/experiments/ddfm_linearity_analysis.json`
    - Check this file for automatic analysis results
  - **Manual Comparison Script** (if needed): Load baseline/new results, run `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()`, compare linearity scores, improvement ratios, collapse risk, and sMAE. See full script in previous iterations if needed.
  - **Quantitative Success Criteria** (using automatic analysis results):
    - **Primary**: sMAE improvement ≥ 10% (target < 1.03 from baseline 1.14) AND DDFM ≥ 5% better than DFM
    - **Secondary**: No degradation > 5% at any horizon (all horizons must have `(baseline - new) / baseline >= -0.05`)
    - **Tertiary**: Horizon 22 completed (n_valid=1, currently n_valid=0)
    - **Metrics from automatic analysis** (check `outputs/experiments/ddfm_linearity_analysis.json`):
      - Linearity score: Target < 0.95 (current baseline: ~0.99 for KOEQUIPTE)
      - Improvement ratio: Target > 10% (current baseline: ~0% for KOEQUIPTE)
      - Consistency metric: Target > 0.7 (measures consistent improvement across horizons)
      - Linear collapse risk: Target < 0.5 (current baseline: ~0.95+ for KOEQUIPTE)
      - Error pattern similarity: Target < 0.6 (indicates DDFM learning different patterns from DFM)
      - Horizon error correlation: Target < 0.5 (low correlation indicates nonlinear behavior)
      - Error distribution differences: Target skewness diff > 0.2, kurtosis diff > 1.0 (indicates DDFM learning different patterns)
      - Horizon-weighted improvement: Target > 5% (prioritizes short-term 1-6 months)
      - Relative improvement consistency: Target > 0.7 (DDFM improves over DFM at >70% of horizons)
      - Improvement persistence: Target > 0.7 (improvements are consistent, not transient)
  - **Decision Tree** (based on automatic analysis metrics):
    - ✅ **SUCCESS** (improvement ≥ 10% AND DDFM > 5% better AND linearity < 0.95 AND collapse_risk < 0.5): Document percentage, proceed to Phase 1.2
    - ⚠️ **PARTIAL** (improvement 5-10% AND DDFM > 5% better AND linearity 0.95-0.98): Document findings, investigate, proceed to Phase 1.2 with caution
    - ❌ **NEEDS INVESTIGATION** (improvement < 10% OR DDFM still identical OR linearity ≥ 0.95 OR collapse_risk ≥ 0.5): Check logs, verify settings, proceed to Phase 2
    - ❌ **FAILURE** (no improvement or degradation): Check logs for errors, investigate root cause, proceed to Phase 2
  - **Document**: 
    - Update report sections (`6_discussion.tex`, `3_results_forecasting.tex`) with improvement percentage and all metrics from automatic analysis
    - Update ISSUES.md with Phase 1 results and next steps
    - Regenerate tables/plots if results change significantly

**DDFM Metrics Research Plan:**

**Current Baseline Metrics** (from `outputs/experiments/aggregated_results.csv`):
- **KOEQUIPTE**: DDFM sMAE=1.1441, DFM sMAE=1.1439 (identical, linear collapse confirmed)
- **KOIPALL.G**: DDFM sMAE=0.69 (21.7x better than DFM sMAE=14.97) - excellent
- **KOWRCCNSE**: DDFM sMAE=0.50 (5.6x better than DFM sMAE=2.78) - excellent

**Key Observation**: KOEQUIPTE shows linear collapse (encoder learning only linear features), while others show strong nonlinear benefits.

**Root Cause**: ReLU activation limitation (zeros negative values), insufficient encoder capacity, or training dynamics causing linear collapse.

**Implemented Improvements** (Verified in Code):
  1. **KOEQUIPTE-Specific Settings** (auto-applied in `src/train.py`):
     - Encoder: `[64, 32, 16]` (default: `[16, 4]`), Activation: `tanh` (default: `relu`)
     - Epochs: `150` (default: `100`), Weight decay: `1e-4` (default: `0.0`)
     - Pre-training multiplier: `2` (default: `1`), Batch size: `64` (default: `100`)
  2. **General Improvements**: Huber loss, gradient clipping, improved weight initialization, factor order configuration, enhanced training stability
  3. **Status**: ✅ **ALL IMPLEMENTED IN CODE** - Needs experiments after training to test effectiveness

- **Concrete Research Plan for DDFM Metrics Improvement** (Based on Current Results Analysis):
  
  **Current Quantitative Baseline (from aggregated_results.csv, computed Dec 2024):**
  - **KOEQUIPTE DDFM**: sMAE=1.1441 (21 horizons), DFM sMAE=1.1439 (21 horizons)
    - Average absolute difference: 0.00085 (0.074% of average sMAE)
    - Maximum difference: 0.00212 (horizon 2)
    - All horizons show differences < 0.003 (within numerical precision)
    - **Conclusion**: Encoder is learning linear relationships only (equivalent to PCA/DFM)
  - **KOIPALL.G DDFM**: sMAE=0.69 (21 horizons), DFM sMAE=14.97 (21 horizons)
    - DDFM outperforms DFM by 21.7x (excellent nonlinear benefit)
    - Missing horizon 22 (n_valid=0)
  - **KOWRCCNSE DDFM**: sMAE=0.50 (22 horizons), DFM sMAE=2.78 (22 horizons)
    - DDFM outperforms DFM by 5.6x (excellent nonlinear benefit)
    - All 22 horizons completed
  - **Key Observation**: KOEQUIPTE shows linear collapse (encoder learning only linear features), while others show strong nonlinear benefits. This suggests KOEQUIPTE data structure or training dynamics differ from other targets.
  
  **DDFM Metrics Research Strategy - Using Existing Analysis Functions:**
  
  The codebase already includes comprehensive DDFM metrics analysis functions (all implemented in `src/evaluation/evaluation_aggregation.py` and `src/evaluation/evaluation_metrics.py`). These functions automatically run after result aggregation and provide actionable insights:
  
  **1. Phase 0: Correlation Structure Analysis** (`analyze_correlation_structure()`) - **IMMEDIATE ACTION**
     - **Status**: ✅ Implemented, ⚠️ NOT YET EXECUTED (can run now, ~15 min, no training required)
     - **Purpose**: Understand why KOEQUIPTE shows linear collapse while others show strong nonlinear benefits
     - **Key Metrics**: `negative_fraction`, `strong_negative_count`, `mean_correlation`, `std_correlation`
     - **Decision Criteria**: 
       - If KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2 → tanh activation strategy confirmed
       - If KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2 → deeper encoder strategy confirmed
     - **Action**: Execute before training to inform improvement strategy
  
  **2. Phase 1: Automatic Analysis After Training** (Runs automatically after `main_aggregator()`)
     - **Linearity Detection** (`detect_ddfm_linearity()`):
       - **Current**: KOEQUIPTE linearity score ~0.99 (near-perfect linear collapse)
       - **Target**: < 0.95 after improvements
       - **Output**: `outputs/experiments/ddfm_linearity_analysis.json`
     - **Prediction Quality Analysis** (`analyze_ddfm_prediction_quality()`):
       - **Current**: KOEQUIPTE improvement ratio ~0% (no improvement over DFM)
       - **Target**: Improvement ratio > 10%, consistency > 0.7, linear collapse risk < 0.5
       - **Tracks**: Improvement ratio, consistency, horizon-weighted improvement, error pattern similarity, horizon error correlation
     - **Error Distribution Analysis** (via `calculate_standardized_metrics()`):
       - **Current**: KOEQUIPTE DDFM and DFM error distributions nearly identical
       - **Target**: After improvements, DDFM should differ (skewness diff > 0.2, kurtosis diff > 1.0)
       - **Tracks**: Error skewness, kurtosis, bias-variance decomposition, error concentration
  
  **3. Metrics-Driven Decision Tree** (Quantitative Criteria):
     - **If linearity score < 0.95 AND improvement ratio > 10% AND consistency > 0.7**: ✅ SUCCESS → Proceed to Phase 1.2 (activation ablation)
     - **If improvement 5-10% OR linearity 0.95-0.98 OR consistency 0.5-0.7**: ⚠️ PARTIAL → Investigate, proceed to Phase 1.2 with caution
     - **If improvement < 5% OR linearity > 0.98 OR consistency < 0.5**: ❌ FAILURE → Proceed to Phase 2 (advanced improvements)
     - **If error distribution differences (skewness diff < 0.2, kurtosis diff < 1.0)**: ⚠️ WARNING → DDFM still learning similar patterns to DFM, may need Phase 2
     - **If horizon-weighted improvement < 5%**: ⚠️ WARNING → Short-term performance not improving, may need different strategy
  
  **Metrics-Driven Improvement Workflow:**
  
  1. **Phase 0 (Pre-Training)**: Run `analyze_correlation_structure()` for all 3 targets
     - Extract correlation metrics (negative_fraction, mean_correlation, etc.)
     - Compare KOEQUIPTE with KOIPALL.G and KOWRCCNSE
     - Confirm or adjust improvement strategy based on correlation patterns
     - **Output**: JSON files with correlation analysis, decision on activation function and encoder architecture
  
  2. **Phase 1 (After Training)**: Run forecasting experiments with improved models
     - Run `detect_ddfm_linearity()` automatically after aggregation
     - Run `analyze_ddfm_prediction_quality()` automatically after aggregation
     - Extract key metrics:
       - Linearity score (target: < 0.95, current: ~0.99)
       - Improvement ratio (target: > 10%, current: ~0%)
       - Consistency metric (target: > 0.7)
       - Error distribution differences (skewness diff > 0.2, kurtosis diff > 1.0)
     - **Decision Criteria**: 
       - ✅ SUCCESS: Linearity < 0.95 AND improvement > 10% AND consistency > 0.7 → Proceed to Phase 1.2
       - ⚠️ PARTIAL: Improvement 5-10% OR linearity 0.95-0.98 → Investigate, proceed to Phase 1.2 with caution
       - ❌ FAILURE: Improvement < 5% OR linearity > 0.98 → Proceed to Phase 2
  
  3. **Phase 1.2 (If Phase 1 Shows Improvement)**: Activation function ablation study
     - Train 4 DDFM models for KOEQUIPTE with different activations (relu, tanh, sigmoid, leaky_relu)
     - Run `analyze_ddfm_prediction_quality()` for each activation
     - Compare improvement ratios, linearity scores, and error distributions
     - **Decision Criteria**: Select activation with best improvement ratio and lowest linearity score
  
  4. **Phase 2 (If Phase 1 Fails)**: Advanced improvements
     - Use error distribution metrics to guide improvements:
       - High bias component → Adjust model architecture or loss function
       - High variance component → Apply regularization or ensemble methods
       - High error concentration → Investigate data quality or model assumptions
     - Run `analyze_horizon_error_correlation()` to identify systematic vs horizon-specific issues
     - **Decision Criteria**: If error patterns improve (bias decreases, variance stable, concentration decreases), proceed to Phase 3
  
  5. **Continuous Monitoring**: After each experiment iteration
     - Run all analysis functions automatically
     - Track metrics over time to monitor improvement trends
     - Use metrics to guide next iteration's improvements
     - **Output**: JSON files with analysis results, recommendations for next steps
  
  **Phase 0: Pre-Experiment Data Analysis (IMMEDIATE - Can be done now, before training)**
  1. **Correlation Structure Analysis** (IMMEDIATE ACTION - No training required, ~15 minutes total):
     - **Status**: ✅ Function implemented in `src/evaluation/evaluation_aggregation.py` (lines 1156-1300), ⚠️ **NOT YET EXECUTED**
     - **Purpose**: Understand why KOEQUIPTE shows linear collapse while others show strong nonlinear benefits
     - **Execution Command** (Step 1 should run automatically, or manual):
       ```bash
       mkdir -p outputs/analysis
       for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
         python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
       done
       ```
     - **Key Metrics to Extract and Compare** (quantitative thresholds):
       - `negative_fraction`: Fraction of negative correlations
         - **Decision threshold**: If KOEQUIPTE > 0.3 AND others < 0.2 → tanh activation justified
         - **Rationale**: ReLU zeros negative values, tanh can capture negative correlations
       - `strong_negative_count`: Count of correlations < -0.3
         - **Decision threshold**: If KOEQUIPTE > 10 AND others < 5 → tanh activation critical
         - **Rationale**: Strong negative relationships require symmetric activation
       - `mean_correlation`: Average correlation magnitude
         - **Decision threshold**: If KOEQUIPTE < 0.1 AND others > 0.2 → deeper encoder needed
         - **Rationale**: Weak signal requires more capacity to extract nonlinear patterns
       - `std_correlation`: Correlation distribution spread
         - **Decision threshold**: If KOEQUIPTE < 0.15 AND others > 0.25 → structural complexity differs
         - **Rationale**: Low spread suggests simpler structure, may need different approach
     - **Hypothesis Testing** (quantitative decision tree):
       - **Scenario A**: KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2
         - **Action**: Confirm tanh activation strategy, proceed with Phase 1
         - **Expected**: tanh should help capture negative correlations
       - **Scenario B**: KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2
         - **Action**: Confirm deeper encoder strategy, may need even deeper ([128, 64, 32])
         - **Expected**: More capacity needed to extract weak nonlinear signal
       - **Scenario C**: KOEQUIPTE structure similar to others (all metrics within 20% of each other)
         - **Action**: Investigate training dynamics (learning rate, initialization, convergence)
         - **Expected**: Issue may be in training procedure, not data structure
     - **Output Actions**:
       - Save results to `outputs/analysis/correlation_analysis_{target}.json` (3 files)
       - Update `nowcasting-report/contents/6_discussion.tex` with correlation findings and decision rationale
       - Update ISSUES.md Phase 0 section with actual metrics and decision
       - Adjust improvement strategy if findings contradict current approach
     - **Expected Time**: < 5 minutes per target (15 minutes total)
     - **Decision Point**: Based on correlation analysis, confirm or adjust improvement strategy before training
     - **Success Criteria**: All 3 JSON files created, metrics extracted, decision documented in ISSUES.md
  
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
     - **Baseline Metrics** (from aggregated_results.csv, computed Dec 2024):
       - KOEQUIPTE DDFM: sMAE=1.1441 (21 horizons, identical to DFM)
       - KOEQUIPTE DFM: sMAE=1.1439 (21 horizons)
       - Average absolute difference: 0.00085 (0.074% of average sMAE)
       - Maximum horizon difference: 0.00212 (horizon 2: DFM=1.45051, DDFM=1.44847)
       - Minimum horizon difference: 0.00001 (horizon 15: DFM=0.08423, DDFM=0.08548)
       - Relative difference: < 0.15% at every horizon
       - Volatile horizons: 7-8 (sMAE ~2.33), 13-14 (sMAE ~3.21-3.28)
       - Horizon 22: Both models have NaN (n_valid=0) - validation failure
     - **Action**: Re-train models with latest improvements (deeper encoder, tanh, weight decay, etc.)
     - **Execution**: Step 1 runs `bash agent_execute.sh train` (automatically applies KOEQUIPTE-specific settings)
     - **KOEQUIPTE-Specific Settings Applied Automatically** (verified in code):
       - Encoder: `[64, 32, 16]` (default: `[16, 4]`) - 4x capacity increase
       - Activation: `tanh` (default: `relu`) - symmetric activation for negative correlations
       - Epochs: `150` (default: `100`) - 50% more training
       - Weight decay: `1e-4` (default: `0.0`) - L2 regularization to prevent linear collapse
       - Pre-training multiplier: `2` (default: `1`) - double pre-training epochs
       - Batch size: `64` (default: `100`) - smaller batches for gradient diversity
     - **Baseline Preservation** (before training): `cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv`
     - **Evaluation Method** (after forecasting): Load baseline/new results, calculate sMAE improvement percentage, DDFM vs DFM difference, check horizon-specific improvements (especially volatile 7-8, 13-14), verify horizon 22 completion
     - **Success Criteria** (quantitative thresholds):
       - **Primary**: sMAE improvement ≥ 10% (target: sMAE < 1.03 from baseline 1.14) AND DDFM ≥ 5% better than DFM
         - **Calculation**: `improvement_pct >= 10.0 AND ddfm_vs_dfm_diff >= 5.0`
       - **Secondary**: No degradation > 5% at any horizon (all horizons must have `(baseline - new) / baseline >= -0.05`)
       - **Tertiary**: Horizon 22 completed (n_valid=1, currently n_valid=0)
       - **Volatile Horizon Target**: Improve horizons 7-8, 13-14 by ≥ 15% (from sMAE 2.33-3.28 to < 2.0-2.8)
     - **Automatic Analysis**: After forecasting, `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` automatically run, results saved to `outputs/experiments/ddfm_linearity_analysis.json`. Check linearity score (< 0.95), improvement ratio (> 10%), consistency (> 0.5)
     - **Decision Tree**: ✅ SUCCESS (≥10% improvement AND DDFM >5% better AND linearity <0.95) → proceed to Phase 1.2. ⚠️ PARTIAL (5-10% improvement) → investigate, proceed with caution. ❌ NEEDS INVESTIGATION (<10% OR identical) → check logs, verify settings, proceed to Phase 2
  
  2. **Activation Function Ablation Study** (If Phase 1 shows improvement): Compare 'relu' vs 'tanh' vs 'sigmoid' vs 'leaky_relu' for KOEQUIPTE, train 4 models, compare metrics
  3. **Huber Loss Robustness Test**: Test `loss_function: 'huber'` with `huber_delta: [0.5, 1.0, 2.0]` for all targets, compare with MSE baseline at volatile horizons
  4. **Horizon 22 Investigation**: Investigate why horizon 22 fails for KOIPALL.G and KOEQUIPTE, check logs, validation logic, numerical stability

  **Phase 2: Advanced Improvements (If Phase 1 Doesn't Improve KOEQUIPTE)**
  1. **Encoder Architecture Grid Search**: Test architectures `[32, 16, 8]`, `[64, 32, 16]`, `[128, 64, 32]`, `[64, 32, 16, 8]`, `[128, 64, 32, 16]` for KOEQUIPTE, maintain others < 0.7
  2. **Factor Loading Analysis**: Extract DFM vs DDFM factor loadings, compare using PCA/correlation, identify linear collapse
  3. **Regularization Experiments**: Test dropout and L1/L2 regularization for KOEQUIPTE encoder

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

- **Success Metrics** (based on quantitative analysis of current results, computed Dec 2024):
  - **Primary**: KOEQUIPTE sMAE improvement from 1.1441 to < 1.03 (≥10% improvement, target: 0.90-1.00)
    - Current baseline: DDFM sMAE=1.1441, DFM sMAE=1.1439 (identical at all 21 horizons, average difference 0.00085 = 0.074% of average)
    - Target: DDFM sMAE < 1.03 (10% improvement from 1.1441) or ideally < 1.0 (12.6% improvement)
    - Success criterion: DDFM sMAE must be at least 5% lower than DFM sMAE (currently 0.017% difference, essentially 0%)
    - Horizon-specific targets: Improve at volatile horizons (7-8: sMAE ~2.33, 13-14: sMAE ~3.21-3.28) by ≥ 15%
    - Linearity score target: < 0.95 (currently likely > 0.99 based on identical performance)
  - **Secondary**: Maintain KOIPALL.G and KOWRCCNSE performance (sMAE < 0.7)
    - KOIPALL.G: Current sMAE=0.69 (excellent, maintain or improve)
    - KOWRCCNSE: Current sMAE=0.50 (excellent, maintain or improve)
  - **Tertiary**: Complete all 22 horizons for all targets (currently missing horizon 22 for KOIPALL.G and KOEQUIPTE)
    - Fix validation or numerical issues preventing horizon 22 predictions

- **Status**: ✅ **IMPROVEMENTS IMPLEMENTED IN CODE** (code changes verified by inspection). **NOT TESTED** - Improvements cannot be tested until models are re-trained with latest improvements. Research plan defined but Phase 1 testing requires re-training to test latest improvements.

- **Metrics Research Improvements** (Already Implemented + Current Iteration):
  1. **Enhanced DDFM Linearity Detection** (`src/evaluation/evaluation_aggregation.py`):
     - Enhanced `detect_ddfm_linearity()` function with performance improvement metrics
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Will automatically detect linearity and improvement when results are aggregated
  1a. **Enhanced DDFM Metrics Calculation** (`src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - **Improved improvement ratio calculation**: Enhanced edge case handling for zero DFM errors, very small differences, and NaN values. Added clipping to reasonable range to avoid extreme values from numerical issues.
     - **Factor dynamics stability integration**: Integrated `calculate_factor_dynamics_stability()` into `analyze_ddfm_prediction_quality()` to detect VAR factor dynamics issues (oscillations, exponential growth/decay, numerical instability) from prediction patterns.
     - **Enhanced linear collapse risk assessment**: Added 7th risk factor (error distribution similarity) to better detect linear collapse when DDFM and DFM have similar error distributions (skewness/kurtosis similarity). Updated risk factor weights to include error distribution similarity.
     - **Better recommendations**: Added factor dynamics stability recommendations and enhanced linear collapse warnings with error distribution similarity.
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Enhanced DDFM metrics provide more reliable and comprehensive performance analysis
  2. **DDFM Prediction Quality Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_ddfm_prediction_quality()` function for detailed DDFM performance analysis
     - Includes CV, consistency metrics, linear collapse risk assessment, horizon degradation detection
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Additional diagnostic metrics available for DDFM performance analysis
  3. **Horizon-Weighted Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_horizon_weighted_metrics()` function for horizon-weighted evaluation
     - Short-term horizons (1-6): 2x weight, mid-term (7-12): 1x weight, long-term (13-22): 0.5x weight
     - Calculates weighted averages prioritizing short-term forecasting (more important for practical use)
     - Integrated into `analyze_ddfm_prediction_quality()` for weighted improvement analysis
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Will automatically be used when results are aggregated
  4. **Training-Aligned Metrics** (`src/evaluation/evaluation_metrics.py` - Current Iteration):
     - Added `calculate_training_aligned_metrics()` function for training-aware metrics
     - Calculates metrics aligned with training loss function (MSE or Huber)
     - Provides standardized training loss for comparison with evaluation metrics
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Available for use in evaluation pipeline
  3. **Missing Horizons Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_missing_horizons()` function to identify validation failures (n_valid=0)
     - Automatically runs after aggregating results via `main_aggregator()`
     - Status: ✅ **IMPLEMENTED IN CODE** - Will automatically analyze missing horizons when results are aggregated
  4. **Enhanced Error Distribution Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added error distribution analysis: skewness, kurtosis, bias-variance decomposition, error concentration, prediction_bias, directional_accuracy, theil_u, mape
     - **NEW (Current Iteration)**: Enhanced `aggregate_overall_performance()` to store diagnostic metrics (error_skewness, error_kurtosis, error_bias_squared, error_variance, error_concentration, prediction_bias, directional_accuracy, theil_u, mape) in aggregated_results.csv
     - **NEW (Current Iteration)**: Enhanced `analyze_ddfm_prediction_quality()` to analyze error distribution patterns and provide recommendations based on:
       - Error skewness similarity between DDFM and DFM (indicates linear behavior)
       - Error kurtosis similarity (indicates similar error tail behavior)
       - Bias-variance decomposition (identifies systematic vs random errors)
       - Error concentration (identifies concentrated error patterns)
       - Prediction bias (identifies systematic over/underprediction)
     - These metrics help diagnose DDFM linear collapse and provide actionable recommendations
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Enhanced metrics now stored in aggregated results and used in analysis
  5. **Horizon Error Correlation Analysis** (`src/evaluation/evaluation_aggregation.py`):
     - Added `analyze_horizon_error_correlation()` function to analyze error patterns across horizons
     - Status: ✅ **IMPLEMENTED IN CODE** - Available for analyzing DDFM error patterns across horizons
  6. **Enhanced DDFM Linear Collapse Risk Assessment** (`src/evaluation/evaluation_aggregation.py` - Previous Iteration):
     - Enhanced `analyze_ddfm_prediction_quality()` with improved linear collapse risk assessment
     - Added 5 risk factors (instead of 3): improvement ratio, similarity, consistency, error pattern similarity (NEW), horizon error correlation (NEW)
     - Added error pattern similarity metric (sMSE/sMAE ratio similarity between DDFM and DFM)
     - Added horizon error correlation metric (correlation of errors across horizons between DDFM and DFM)
     - Added sMSE/sMAE ratio stability metrics (CV and variance) to detect unstable prediction error structure
     - Enhanced recommendations with pattern-specific and correlation-specific guidance
     - Status: ✅ **IMPLEMENTED IN CODE** (previous iteration) - Provides more accurate linear collapse detection
  7. **Horizon-Weighted Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_horizon_weighted_metrics()` function for horizon-weighted evaluation
     - Short-term horizons (1-6): 2x weight, mid-term (7-12): 1x weight, long-term (13-22): 0.5x weight
     - Calculates weighted averages of sMAE, sMSE, sRMSE prioritizing short-term forecasting (more important for practical use)
     - Integrated into `analyze_ddfm_prediction_quality()` for weighted improvement analysis
     - Provides weighted improvement percentages (DDFM vs DFM) in analysis results
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Will automatically be used when results are aggregated
  8. **Training-Aligned Metrics** (`src/evaluation/evaluation_metrics.py` - Previous Iteration):
     - Added `calculate_training_aligned_metrics()` function for training-aware metrics
     - Calculates metrics aligned with training loss function (MSE or Huber)
     - Provides standardized training loss for comparison with evaluation metrics
     - Helps ensure evaluation metrics match what the model was optimized for during training
     - Status: ✅ **IMPLEMENTED IN CODE** (previous iteration) - Available for use in evaluation pipeline
  9. **Relative Error Stability Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_relative_error_stability()` function to analyze how DDFM vs DFM relative performance changes across horizons
     - Calculates stability score (0-1), coefficient of variation, and trend analysis (improving/degrading/stable)
     - Detects systematic patterns in relative performance that may indicate encoder issues or factor dynamics problems
     - Integrated into `analyze_ddfm_prediction_quality()` for automatic analysis
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Will automatically analyze relative error stability when results are aggregated
  10. **Improvement Persistence Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_improvement_persistence()` function to detect if DDFM improvements are persistent (consistent) or transient (noise)
     - Calculates persistence score (0-1), improvement fraction, consecutive improvement streaks, and improvement clusters
     - Helps distinguish between systematic improvements and random noise at specific horizons
     - Integrated into `analyze_ddfm_prediction_quality()` for automatic analysis
     - Status: ✅ **IMPLEMENTED IN CODE** (current iteration) - Will automatically analyze improvement persistence when results are aggregated
  11. **Temporal Consistency Metrics** (`src/evaluation/evaluation_metrics.py` - Previous Iteration):
     - Added `calculate_temporal_consistency_metrics()` function to detect sudden jumps in predictions across consecutive horizons
     - Calculates temporal consistency score (0-1), jump count, jump fraction, and jump magnitudes
     - Helps detect model instability or factor dynamics issues that cause inconsistent predictions
     - Available for use in evaluation pipeline (can be integrated into analysis functions as needed)
     - Status: ✅ **IMPLEMENTED IN CODE** (previous iteration) - Available for use in evaluation pipeline
  12. **Robust Statistics and Bootstrap Confidence Intervals** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_robust_metrics()` function for median-based metrics (more resistant to outliers)
       - Calculates robust_sMAE, robust_sMSE, robust_sRMSE using median instead of mean
       - Provides IQR-based metrics and outlier detection using IQR method
       - Uses MAD (Median Absolute Deviation) for robust normalization
     - Added `calculate_bootstrap_confidence_intervals()` function for uncertainty quantification
       - Provides 95% confidence intervals for sMAE, sMSE, sRMSE using bootstrap resampling
       - Uses 1000 bootstrap samples by default for statistical reliability
       - Helps assess uncertainty in DDFM performance evaluation
     - **Enhanced Integration** (Current Iteration):
       - `analyze_ddfm_prediction_quality()` now automatically calculates robust metrics alongside mean-based metrics
       - Automatically uses robust (median-based) metrics when coefficient of variation > 0.5 (indicating outliers)
       - Provides both mean-based and robust improvement percentages for comparison
       - Recommendations indicate when robust metrics are used and explain why
       - Summary logging includes robust metrics when outliers are detected
       - Improves reliability of DDFM performance evaluation, especially for targets with volatile horizons
     - Added `aggregate_robust_metrics_across_horizons()` function for robust horizon aggregation
       - Aggregates metrics across horizons using median instead of mean
       - Provides IQR statistics for metric spread across horizons
       - More resistant to outliers from specific problematic horizons
     - **Rationale**: Mean-based metrics are sensitive to outliers from numerical instability or model issues at specific horizons. Robust statistics provide more reliable performance evaluation, especially for DDFM where some horizons may have extreme errors. Bootstrap confidence intervals provide uncertainty quantification for more reliable performance comparisons.
     - **Status**: ✅ **IMPLEMENTED IN CODE** (current iteration) - Robust metrics improve DDFM performance evaluation reliability and provide uncertainty quantification
  14. **Forecast Skill Score and Information Gain Metrics** (`src/evaluation/evaluation_metrics.py`, `src/evaluation/evaluation_aggregation.py` - Current Iteration):
     - Added `calculate_forecast_skill_score()` function for comparing DDFM to naive baseline
       - Skill score ranges from -inf to 1.0 (1.0 = perfect, 0.0 = same as baseline, < 0.0 = worse than baseline)
       - Calculates skill scores for MSE, MAE, and RMSE
       - Provides percentage improvement over baseline (persistence or mean forecast)
       - Helps quantify forecast improvement relative to simple baselines
     - Added `calculate_information_gain()` function for measuring information gain of DDFM over DFM
       - Two methods: KL divergence between error distributions, or mutual information between predictions and true values
       - Quantifies value of nonlinear features learned by DDFM encoder
       - Helps identify when DDFM is learning different patterns from DFM
     - **Enhanced horizon improvement tracking**: Added horizon categorization in `analyze_ddfm_prediction_quality()`
       - Categorizes horizons by improvement level: significant (>10%), moderate (5-10%), marginal (0-5%), no improvement, degradation
       - Calculates improvement distribution statistics (fractions, counts per category)
       - Provides actionable insights on which horizons benefit most from DDFM
       - Recommendations include horizon-specific guidance based on improvement distribution
     - **Rationale**: Skill score provides standardized measure of forecast improvement relative to naive baselines, making DDFM performance more interpretable. Information gain quantifies the value of nonlinear features. Enhanced horizon tracking helps identify which forecast horizons benefit most from DDFM improvements.
     - **Status**: ✅ **IMPLEMENTED IN CODE** (current iteration) - New metrics provide additional insights for DDFM performance evaluation and improvement tracking
  13. **Factor Dynamics Stability Inference** (`src/evaluation/evaluation_metrics.py` - Current Iteration):
     - Added `calculate_factor_dynamics_stability()` function to infer VAR factor dynamics stability from prediction patterns
     - **Metrics included**:
       - Oscillation detection: Detects oscillatory patterns (indicates complex eigenvalues in VAR transition matrix)
       - Exponential growth/decay detection: Detects exponential trends (indicates eigenvalues outside unit circle)
       - Prediction smoothness score: Measures smoothness using second derivative variance (0-1, higher = smoother)
       - Divergence/convergence detection: Detects if predictions are diverging or converging across horizons
       - Overall stability score: Combined stability metric (0-1, higher = more stable)
       - Stability interpretation: Text interpretation ('stable', 'oscillatory', 'unstable', 'diverging', 'converging')
     - **Rationale**: VAR factor dynamics stability cannot be directly accessed in evaluation pipeline (transition matrix not available). This function infers stability from prediction patterns across horizons. Unstable factor dynamics can cause oscillations, exponential growth/decay, or numerical instability. Early detection helps identify if VAR transition matrix has eigenvalues outside unit circle or complex eigenvalues with large imaginary parts.
     - **Status**: ✅ **IMPLEMENTED IN CODE** (current iteration) - Factor dynamics stability inference available for use in evaluation pipeline when predictions by horizon are available

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
     - **Status**: ✅ Function implemented in `src/evaluation/evaluation_aggregation.py` (lines 1156-1300), ⚠️ **NOT YET EXECUTED**
     - **Priority**: HIGH - Can be done immediately to inform training strategy, saves experimental time
     - **Expected time**: < 5 minutes per target (15 minutes total for all 3 targets)
     - **Action Required**: Execute correlation analysis for all 3 targets (Step 1 should run this automatically, or can be run manually)
     - **Code location**: `src/evaluation/evaluation_aggregation.py` - function `analyze_correlation_structure()`
     - **Execution command** (all 3 targets for comparison - recommended): 
       ```bash
       mkdir -p outputs/analysis
       for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
         python3 -c "from src.evaluation.evaluation_aggregation import analyze_correlation_structure; import json; result = analyze_correlation_structure('data/data.csv', '$target', output_path='outputs/analysis/correlation_analysis_${target}.json'); print(f'\n=== $target ==='); print(json.dumps(result['summary'], indent=2))"
       done
       ```
     - **Expected Output**: 3 JSON files in `outputs/analysis/correlation_analysis_{target}.json` with correlation statistics
     - **Key metrics to extract and compare**: 
       - `negative_fraction`: Fraction of negative correlations (if KOEQUIPTE > 0.3 and others < 0.2: tanh activation justified)
       - `strong_negative_count`: Count of correlations < -0.3 (indicates strong negative relationships)
       - `mean_correlation`: Average correlation magnitude (if KOEQUIPTE < 0.1 and others > 0.2: deeper encoder needed)
       - `std_correlation`: Correlation distribution spread (indicates structural complexity)
     - **Decision criteria** (after analysis): 
       - If KOEQUIPTE `negative_fraction > 0.3` and others < 0.2: tanh activation is justified, proceed with current strategy
       - If KOEQUIPTE `mean_correlation < 0.1` and others > 0.2: deeper encoder is needed, current strategy is appropriate
       - If structures are similar: investigate why DDFM works for others but not KOEQUIPTE, may need Phase 2 approaches
     - **Next actions after analysis**: 
       - Update `nowcasting-report/contents/6_discussion.tex` with correlation analysis findings
       - Update this ISSUES.md Phase 0 section with actual results
       - Adjust improvement strategy if needed based on findings
  
  **Phase 1: Training and Initial Testing (Training required)**
  2. **Train models** - Models do not exist, training is REQUIRED
     - **Status**: ❌ **MODELS NOT TRAINED** - `checkpoint/` is EMPTY - no model.pkl files exist
     - **Priority**: HIGH - Training is REQUIRED before any experiments can proceed
     - **Expected training time**: ~30-60 minutes per model (12 models total, may run in parallel)
     - **Execution**: Step 1 must run `bash agent_execute.sh train` (automatically uses new settings for KOEQUIPTE)
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
     - **Comparison**: Load baseline/new results, calculate sMAE improvement percentage, DDFM vs DFM difference, check horizon 22 completion status
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
- ❌ **Models NOT trained**: `checkpoint/` is EMPTY - no model.pkl files exist. Training is REQUIRED before any experiments can proceed
- 📊 **Baseline metrics** (from old aggregated_results.csv): KOEQUIPTE DDFM sMAE=1.14 (identical to DFM), KOIPALL.G sMAE=0.69, KOWRCCNSE sMAE=0.50

**Plan**: Phase 0 (correlation analysis, ~15 min) → Phase 1 (train & test, requires training) → Phase 1.2/1.3 (ablation/investigation) → Phase 2/3 (advanced if needed)

**Success Criteria**: KOEQUIPTE sMAE < 1.03 (≥10% improvement), maintain others < 0.7, complete all 22 horizons

**Key Files**: `src/train.py` (KOEQUIPTE settings), `src/evaluation/evaluation_aggregation.py` (analysis functions), `outputs/experiments/aggregated_results.csv` (results)

---

## DDFM Metrics Research: Analysis Functions

**Key Analysis Functions** (all in `src/evaluation/evaluation_aggregation.py`):
- `detect_ddfm_linearity()` - Detects linear collapse (output: `outputs/experiments/ddfm_linearity_analysis.json`)
- `analyze_ddfm_prediction_quality()` - Comprehensive performance analysis (runs automatically after aggregation)
- `analyze_correlation_structure()` - Pre-training data structure analysis (output: `outputs/analysis/correlation_analysis_{target}.json`)

**Key Metrics to Track**:
- Linearity score: < 0.95 (currently ~0.99 for KOEQUIPTE)
- Improvement ratio: > 10% (currently ~0% for KOEQUIPTE)
- Linear collapse risk: < 0.5 (currently ~0.95+ for KOEQUIPTE)
- Consistency: > 0.7 (measures consistent improvement across horizons)

**Workflow**:
1. **Before Training (Phase 0)**: Run `analyze_correlation_structure()` for all targets to inform improvement strategy
2. **After Training + Forecasting (Phase 1)**: Check `outputs/experiments/ddfm_linearity_analysis.json` (auto-generated) for metrics
3. **Iterative Improvement**: Use metrics to guide next iteration's improvements

**Execution Commands**: See Phase 0 and Phase 1 sections above for detailed execution commands.
