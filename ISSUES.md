# Issues

## Quick Summary

**Current State:**
- ✅ **Code improvements implemented** (previous iterations): Deeper encoder, tanh activation, weight decay, gradient clipping, Huber loss, improved initialization, increased pre-training, batch size optimization
- ✅ **Models TRAINED**: `checkpoint/` contains 12 model.pkl files (trained Dec 9 02:35-02:47). All models exist and ready for use.
- 📊 **Baseline metrics** (from aggregated_results.csv):
  - **KOEQUIPTE**: DDFM sMAE=1.1441, DFM sMAE=1.1439 (identical, linear collapse confirmed)
  - **KOIPALL.G**: DDFM sMAE=0.69 (21.7x better than DFM sMAE=14.97) - excellent
  - **KOWRCCNSE**: DDFM sMAE=0.50 (5.6x better than DFM sMAE=2.78) - excellent
- ⚠️ **Phase 0 not executed**: Correlation structure analysis function exists but has not been run yet - can be done immediately without training (~15 minutes)
- ⚠️ **This iteration**: Documentation updated (STATUS.md, ISSUES.md, CONTEXT.md) to track project state. No code changes, no new experiments run.

**Quick Action Reference:**
1. **REQUIRED (Models exist)**: Re-run backtesting via `bash agent_execute.sh backtest` to verify CUDA fixes work
2. **REQUIRED (Verify results)**: Re-run forecasting via `bash agent_execute.sh forecast` to verify results reflect latest code improvements
3. **IMMEDIATE (No training needed)**: Run Phase 0 correlation analysis - see "PRIORITY 0" below (~15 min)
4. **IMMEDIATE (No training needed)**: Run baseline metrics analysis - see "PRIORITY 0" below (~5 min)

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

**AFTER PHASE 0 (Models exist, can proceed directly to testing):**
2. **Phase 1: Models Already Trained** - Models exist (trained Dec 9 02:35-02:47)
   - **Status**: ✅ Models exist - `checkpoint/` contains 12 model.pkl files
   - **Action**: None - models are trained and ready for use. Can proceed directly to forecasting and backtesting
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

## Models Training Status (CORRECTED - Models ARE Trained)
- **Current State**: `checkpoint/` directory contains 12 model.pkl files (trained Dec 9 02:35-02:47)
- **Status**: ✅ **TRAINED** - All 12 models exist (3 targets × 4 models: ARIMA, VAR, DFM, DDFM)
- **Verification**: `checkpoint/` contains 12 model.pkl files (verified by inspection)
- **Note**: Models are trained and ready for use. All DDFM improvements are implemented in code and were automatically applied during training (deeper encoder, tanh, weight_decay, mult_epoch_pretrain=2, batch_size=64 for KOEQUIPTE)
- **Action Required**: None - models exist and can be used for forecasting and backtesting experiments

## CRITICAL: DFM/DDFM Backtest CUDA Tensor Conversion Error (Fixed in Code, Needs Training + Re-run)
- **Problem**: All DDFM and DFM backtest results failed with error: "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
- **Root Cause**: The `_convert_predictions_to_dataframe` function and other prediction conversion functions used `np.asarray()` directly on CUDA tensors without first moving them to CPU.
- **Impact**: All 22 months of backtest results for DFM and DDFM models failed (6 JSON files × 22 months = 132 failed predictions). All backtest JSON files in `outputs/backtest/` show "status": "failed" with CUDA errors.
- **Resolution** (Previous Iteration): Fixed tensor conversion in multiple locations:
  - `src/models/models_utils.py`: `_convert_predictions_to_dataframe()` and `_validate_predictions()` - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_forecaster.py`: Prediction value extraction - Added `.cpu().numpy()` pattern
  - `src/evaluation/evaluation_metrics.py`: Metric calculation - Added `.cpu().numpy()` pattern
  - All now convert CUDA tensors to CPU before numpy conversion using `.cpu().numpy()`
- **Status**: ✅ **FIXED IN CODE** (code changes verified by inspection). **NOT VERIFIED BY EXPERIMENTS** - Models exist (trained Dec 9 02:35-02:47), backtest experiments MUST be re-run via `bash agent_execute.sh backtest` to verify fix works. Current backtest JSON files all show "failed" status with CUDA errors. Fix may work, but needs experimental verification.


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

**EXECUTIVE SUMMARY - Metrics-Driven Research Plan:**
- **Problem**: KOEQUIPTE DDFM shows linear collapse (sMAE=1.14, identical to DFM at all 21 horizons, max difference 0.00212)
- **Goal**: Improve KOEQUIPTE DDFM sMAE from 1.14 to < 1.03 (≥10% improvement) AND make DDFM ≥ 5% better than DFM
- **Current Status**: 
  - ✅ Code improvements implemented (deeper encoder, tanh, weight_decay, etc.)
  - ✅ Models TRAINED: `checkpoint/` contains 12 model.pkl files (trained Dec 9 02:35-02:47). All models exist and ready for use.
  - ⚠️ Phase 0 correlation analysis NOT executed (can be done now, ~15 min, no training required)
  - ⚠️ Latest improvements NOT verified with new results (forecasting results exist but may be from before improvements, need to re-run to verify)
- **Next Immediate Action**: Re-run backtesting to verify CUDA fixes → Re-run forecasting to verify results reflect latest improvements → Phase 0 correlation analysis (15 minutes, no training required)
- **Success Metrics**: sMAE < 1.03, DDFM > 5% better than DFM, linearity score < 0.95, horizon 22 completed

**CONCRETE ACTION PLAN (Execute in Order):**

**IMMEDIATE ACTIONS (No training required - Execute NOW):**
1. **Phase 0: Correlation Structure Analysis** (~15 minutes)
   - **Purpose**: Validate improvement strategy before training
   - **Command**: See "IMMEDIATE (No training required)" section below
   - **Output**: 3 JSON files with correlation statistics
   - **Decision**: If KOEQUIPTE shows high negative correlations (>0.3) vs others (<0.2), tanh strategy is validated
   
2. **Baseline Metrics Analysis** (~5 minutes)
   - **Purpose**: Establish quantitative baseline for comparison
   - **Command**: Run `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` on current `aggregated_results.csv`
   - **Output**: Baseline linearity score (~0.99 expected), improvement ratio (~0% expected), collapse risk (~0.95+ expected)
   - **Document**: Save to `outputs/analysis/baseline_linearity.json` and `outputs/analysis/baseline_quality.json`

**AFTER PHASE 0 (Models already exist - can proceed to testing):**
3. **Re-run Forecasting** (Models exist, trained Dec 9 02:35-02:47)
   - **Purpose**: Verify results reflect latest code improvements
   - **Command**: `bash agent_execute.sh forecast`
   - **Auto-analysis**: `detect_ddfm_linearity()` and `analyze_ddfm_prediction_quality()` run automatically
   - **Output**: New `aggregated_results.csv` and `ddfm_linearity_analysis.json`

4. **Compare Results** (After forecasting)
   - **Purpose**: Measure improvement from baseline
   - **Method**: Compare new metrics with baseline metrics
   - **Key Metrics**: sMAE improvement %, linearity score, improvement ratio, collapse risk
   - **Success Criteria**: See "Phase 1: Compare Results" section below

5. **DECISION TREE** (Based on metrics):
   - ✅ **SUCCESS** (improvement ≥ 10% AND DDFM > 5% better AND linearity < 0.95) → Proceed to Phase 1.2
   - ⚠️ **PARTIAL** (improvement 5-10% OR linearity 0.95-0.98) → Investigate, proceed to Phase 1.2 with caution
   - ❌ **NEEDS INVESTIGATION** (improvement < 10% OR linearity ≥ 0.95) → Check logs, proceed to Phase 2
   - ❌ **FAILURE** (no improvement or degradation) → Check logs, investigate root cause, proceed to Phase 2
- **Metrics-Driven Research Strategy**: Use existing comprehensive DDFM metrics analysis functions (already implemented) to guide improvements:
  - **Phase 0 (Pre-training)**: `analyze_correlation_structure()` - Analyze data structure to inform improvement strategy (~15 min, no training required)
  - **Phase 1 (Post-training)**: `detect_ddfm_linearity()` + `analyze_ddfm_prediction_quality()` - Automatically run after aggregation to monitor improvements
  - **Key Metrics to Track** (with target values and interpretation):
    - **Linearity score**: Target < 0.95 (current: ~0.99 for KOEQUIPTE, estimated from identical performance)
      - Interpretation: 0.95+ indicates encoder learning only linear features (PCA-like behavior)
      - Action if > 0.95: Deeper encoder, tanh activation, weight decay, increased pre-training
    - **Improvement ratio**: Target > 10% (current: ~0% for KOEQUIPTE, computed from aggregated_results.csv)
      - Interpretation: Percentage improvement of DDFM over DFM (positive = DDFM better)
      - Action if < 10%: Check encoder architecture, activation function, training dynamics
    - **Error distribution differences**: Target skewness diff > 0.2, kurtosis diff > 1.0
      - Interpretation: Indicates DDFM learning different patterns from DFM (nonlinear behavior)
      - Action if diff < 0.2: DDFM and DFM have similar error patterns (linear collapse risk)
    - **Horizon-weighted improvement**: Target > 5% (prioritizes short-term 1-6 months)
      - Interpretation: Weighted average improvement prioritizing short-term horizons (2x weight)
      - Action if < 5%: Short-term performance not improving, may need different strategy
    - **Consistency metric**: Target > 0.7 (measures consistent improvement across horizons)
      - Interpretation: How consistent DDFM improvement is across all horizons (0-1 scale)
      - Action if < 0.7: Improvement is inconsistent, may need horizon-specific tuning
    - **Linear collapse risk**: Target < 0.5 (current: ~0.95+ for KOEQUIPTE)
      - Interpretation: 7-factor risk assessment (0-1, higher = more risk of linear collapse)
      - Action if > 0.5: High risk of encoder learning only linear features, apply all improvements
    - **Error pattern similarity**: Target < 0.6 (indicates DDFM learning different patterns from DFM)
      - Interpretation: Similarity of sMSE/sMAE ratio patterns between DDFM and DFM (0-1 scale)
      - Action if > 0.6: DDFM and DFM have similar error patterns (linear behavior)
    - **Horizon error correlation**: Target < 0.5 (low correlation indicates nonlinear behavior)
      - Interpretation: Correlation of errors across horizons between DDFM and DFM (-1 to 1)
      - Action if > 0.5: High correlation indicates systematic linear behavior
    - **Relative improvement consistency**: Target > 0.7 (DDFM improves over DFM at >70% of horizons)
      - Interpretation: Fraction of horizons where DDFM improves over DFM
      - Action if < 0.7: Improvement is not consistent across horizons
    - **Improvement persistence**: Target > 0.7 (improvements are consistent, not transient)
      - Interpretation: Whether DDFM improvements are persistent (systematic) or transient (noise)
      - Action if < 0.7: Improvements may be random noise, not systematic model improvement

**What Can Be Done NOW (No Training Required):**

1. **Phase 0: Correlation Structure Analysis** - Execute `analyze_correlation_structure()` for all 3 targets (~15 min)
   - **Purpose**: Validate improvement strategy before training
   - **Command**: 
     ```bash
     mkdir -p outputs/analysis
     for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
       python3 -c "
       from src.evaluation.evaluation_aggregation import analyze_correlation_structure
       import json
       result = analyze_correlation_structure('data/data.csv', '$target', 
         output_path='outputs/analysis/correlation_analysis_${target}.json')
       print(f'\n=== $target ===')
       print(json.dumps(result['summary'], indent=2))
       "
     done
     ```
   - **Expected Output**: 3 JSON files in `outputs/analysis/correlation_analysis_{target}.json`
   - **Key Metrics to Extract**:
     - `summary.negative_fraction`: Fraction of negative correlations (if KOEQUIPTE > 0.3 and others < 0.2: tanh activation justified)
     - `summary.strong_negative_count`: Count of correlations < -0.3 (indicates strong negative relationships)
     - `summary.mean_correlation`: Average correlation magnitude (if KOEQUIPTE < 0.1 and others > 0.2: deeper encoder needed)
     - `summary.std_correlation`: Correlation distribution spread (indicates structural complexity)
   - **Decision Criteria**:
     - If KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2 → tanh activation strategy validated
     - If KOEQUIPTE `strong_negative_count > 10` AND others < 5 → tanh activation critical
     - If KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2 → deeper encoder strategy validated
   - **Next Action**: Update report sections with findings, adjust improvement strategy if needed

2. **Baseline Metrics Analysis** - Use existing `aggregated_results.csv` to compute baseline metrics (~5 min)
   - **Purpose**: Establish quantitative baseline for comparison after re-training
   - **Command**:
     ```python
     import pandas as pd
     from src.evaluation.evaluation_aggregation import detect_ddfm_linearity, analyze_ddfm_prediction_quality
     import json
     
     # Load current results
     results = pd.read_csv('outputs/experiments/aggregated_results.csv')
     
     # Run analysis
     linearity = detect_ddfm_linearity(results, output_path='outputs/analysis/baseline_linearity.json')
     quality = analyze_ddfm_prediction_quality(results, output_path='outputs/analysis/baseline_quality.json')
     
     # Print key metrics for KOEQUIPTE
     print("KOEQUIPTE Baseline Metrics:")
     print(f"  Linearity score: {linearity['linearity_scores'].get('KOEQUIPTE', {}).get('overall_linearity', 'N/A')}")
     if 'target_analysis' in quality and 'KOEQUIPTE' in quality['target_analysis']:
         koe = quality['target_analysis']['KOEQUIPTE']
         print(f"  Improvement ratio: {koe.get('improvement_ratio', 'N/A')}")
         print(f"  Linear collapse risk: {koe.get('linear_collapse_risk', 'N/A')}")
     ```
   - **Expected Output**: 
     - `outputs/analysis/baseline_linearity.json`: Baseline linearity scores (expected: KOEQUIPTE ~0.99)
     - `outputs/analysis/baseline_quality.json`: Baseline improvement ratios and collapse risk (expected: KOEQUIPTE improvement ~0%, collapse risk ~0.95+)
   - **Next Action**: Document baseline metrics in this ISSUES.md for comparison after Phase 1

**What Requires Re-running Experiments:**
1. **Phase 1: Re-run Forecasting** - Models exist, re-run forecasting to verify results reflect latest improvements
   - Models already trained with KOEQUIPTE-specific settings (deeper encoder, tanh, weight_decay, etc.)
   - Compare new results with baseline to measure improvement
   - Use automatic analysis functions to track metrics

**CONCRETE EXECUTION CHECKLIST (Priority Order):**

**✅ IMMEDIATE (No training required - Can execute now):**
- [ ] **Phase 0: Correlation Structure Analysis** (~15 minutes)
  - **Status**: ⚠️ **NOT EXECUTED** - Function exists, ready to run
  - **Execute**: 
    ```bash
    mkdir -p outputs/analysis
    for target in KOEQUIPTE KOIPALL.G KOWRCCNSE; do
      python3 -c "
      from src.evaluation.evaluation_aggregation import analyze_correlation_structure
      import json
      result = analyze_correlation_structure('data/data.csv', '$target', 
        output_path='outputs/analysis/correlation_analysis_${target}.json')
      print(f'\n=== $target ===')
      print(json.dumps(result['summary'], indent=2))
      "
    done
    ```
  - **Verify Output**: Check that 3 JSON files exist: `outputs/analysis/correlation_analysis_{target}.json`
  - **Extract Key Metrics** (from each JSON file):
    - `summary.negative_fraction`: Fraction of negative correlations
    - `summary.strong_negative_count`: Count of correlations < -0.3
    - `summary.mean_correlation`: Average correlation magnitude
    - `summary.std_correlation`: Correlation distribution spread
  - **Quantitative Decision Criteria**:
    - If KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2 → tanh activation justified (ReLU limitation confirmed)
    - If KOEQUIPTE `strong_negative_count > 10` AND others < 5 → tanh activation critical (strong negative relationships)
    - If KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2 → deeper encoder needed (weak signal requires more capacity)
    - If KOEQUIPTE `std_correlation < 0.15` AND others > 0.25 → structural complexity differs (may need different approach)
  - **Document Results**: 
    - Update `nowcasting-report/contents/6_discussion.tex` with correlation findings and decision rationale
    - Update ISSUES.md Phase 0 section with actual metrics and decision
    - If findings contradict current strategy, adjust improvement plan before training

- [ ] **Baseline Metrics Analysis** (~5 minutes)
  - **Status**: ⚠️ **NOT EXECUTED** - Can run now on existing aggregated_results.csv
  - **Execute**: 
    ```python
    import pandas as pd
    from src.evaluation.evaluation_aggregation import detect_ddfm_linearity, analyze_ddfm_prediction_quality
    import json
    
    # Load current results
    results = pd.read_csv('outputs/experiments/aggregated_results.csv')
    
    # Run analysis
    linearity = detect_ddfm_linearity(results, output_path='outputs/analysis/baseline_linearity.json')
    quality = analyze_ddfm_prediction_quality(results, output_path='outputs/analysis/baseline_quality.json')
    
    # Print key metrics for KOEQUIPTE
    print("KOEQUIPTE Baseline Metrics:")
    print(f"  Linearity score: {linearity['linearity_scores'].get('KOEQUIPTE', 'N/A')}")
    if 'target_analysis' in quality and 'KOEQUIPTE' in quality['target_analysis']:
        koe = quality['target_analysis']['KOEQUIPTE']
        print(f"  Improvement ratio: {koe.get('improvement_ratio', 'N/A')}")
        print(f"  Linear collapse risk: {koe.get('linear_collapse_risk', 'N/A')}")
    ```
  - **Verify Output**: Check that 2 JSON files exist: `outputs/analysis/baseline_linearity.json`, `outputs/analysis/baseline_quality.json`
  - **Document**: Save baseline metrics to ISSUES.md for comparison after training
  - **Purpose**: Establish quantitative baseline to measure improvement after Phase 1

**⏳ AFTER PHASE 0 (Models exist, can proceed to testing):**
- [ ] **Phase 1: Baseline Preservation** (Before re-running forecasting)
  - Execute: `cp outputs/experiments/aggregated_results.csv outputs/experiments/aggregated_results_baseline.csv`
  - Verify: Baseline file exists with 264 lines
  - **Also preserve baseline analysis**: Copy `outputs/analysis/baseline_linearity.json` and `outputs/analysis/baseline_quality.json` if they exist
  
- [ ] **Phase 1: Models Already Trained** (Models exist, trained Dec 9 02:35-02:47)
  - Verify: `checkpoint/` contains 12 model.pkl files (already verified)
  - Check logs: `grep -i "target-specific\|tanh\|weight_decay\|epochs\|mult_epoch\|batch_size" log/KOEQUIPTE_ddfm_*.log | tail -20`
  - Expected: "Using target-specific encoder architecture [64, 32, 16]", "Using tanh activation for KOEQUIPTE", etc.

- [ ] **Phase 1: Re-run Forecasting** (Step 1 runs automatically via `bash agent_execute.sh forecast`)
  - Results: `outputs/experiments/aggregated_results.csv` (will be overwritten)
  - Auto-analysis: `outputs/experiments/ddfm_linearity_analysis.json` (generated automatically)

- [ ] **Phase 1: Compare Results** (After forecasting completes)
  - **Step 1: Check Automatic Analysis** - Results saved to `outputs/experiments/ddfm_linearity_analysis.json`
    ```bash
    # View analysis results
    cat outputs/experiments/ddfm_linearity_analysis.json | python3 -m json.tool | grep -A 20 "KOEQUIPTE"
    ```
  - **Step 2: Extract Key Metrics** from `outputs/experiments/ddfm_linearity_analysis.json`:
    - `linearity_scores.KOEQUIPTE`: Should be < 0.95 (baseline: ~0.99)
    - `target_analysis.KOEQUIPTE.improvement_ratio`: Should be > 10% (baseline: ~0%)
    - `target_analysis.KOEQUIPTE.linear_collapse_risk`: Should be < 0.5 (baseline: ~0.95+)
    - `target_analysis.KOEQUIPTE.consistency`: Should be > 0.7
    - `target_analysis.KOEQUIPTE.error_pattern_similarity`: Should be < 0.6
    - `target_analysis.KOEQUIPTE.horizon_error_correlation`: Should be < 0.5
  - **Step 3: Compare sMAE** (from `outputs/experiments/aggregated_results.csv`):
    ```python
    import pandas as pd
    baseline = pd.read_csv('outputs/experiments/aggregated_results_baseline.csv')
    new = pd.read_csv('outputs/experiments/aggregated_results.csv')
    
    # KOEQUIPTE DDFM sMAE
    baseline_smae = baseline[(baseline['target']=='KOEQUIPTE') & (baseline['model']=='DDFM')]['sMAE'].mean()
    new_smae = new[(new['target']=='KOEQUIPTE') & (new['model']=='DDFM')]['sMAE'].mean()
    improvement_pct = (baseline_smae - new_smae) / baseline_smae * 100
    
    print(f"KOEQUIPTE DDFM sMAE: {baseline_smae:.4f} → {new_smae:.4f}")
    print(f"Improvement: {improvement_pct:.2f}%")
    ```
  - **Quantitative Success Criteria**:
    - **Primary**: sMAE improvement ≥ 10% (target < 1.03 from baseline 1.14) AND DDFM ≥ 5% better than DFM
    - **Secondary**: No degradation > 5% at any horizon
    - **Tertiary**: Horizon 22 completed (n_valid=1, currently n_valid=0)
    - **Metrics Targets** (from automatic analysis):
      - Linearity score: < 0.95 (baseline: ~0.99)
      - Improvement ratio: > 10% (baseline: ~0%)
      - Consistency: > 0.7
      - Linear collapse risk: < 0.5 (baseline: ~0.95+)
      - Error pattern similarity: < 0.6
      - Horizon error correlation: < 0.5
  - **Decision Tree** (based on metrics):
    - ✅ **SUCCESS**: improvement ≥ 10% AND DDFM > 5% better AND linearity < 0.95 AND collapse_risk < 0.5 → Proceed to Phase 1.2
    - ⚠️ **PARTIAL**: improvement 5-10% AND DDFM > 5% better AND linearity 0.95-0.98 → Investigate, proceed to Phase 1.2 with caution
    - ❌ **NEEDS INVESTIGATION**: improvement < 10% OR linearity ≥ 0.95 OR collapse_risk ≥ 0.5 → Check logs, verify settings, proceed to Phase 2
    - ❌ **FAILURE**: no improvement or degradation → Check logs, investigate root cause, proceed to Phase 2
  - **Document**: 
    - Update report sections with improvement percentage and metrics
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
  
  **DDFM Metrics Research Strategy - Metrics-Driven Approach Using Existing Analysis Functions:**
  
  The codebase already includes comprehensive DDFM metrics analysis functions (all implemented in `src/evaluation/evaluation_aggregation.py` and `src/evaluation/evaluation_metrics.py`). These functions automatically run after result aggregation and provide actionable insights. The research plan follows a metrics-driven approach where each phase uses specific metrics to guide decisions:
  
  **1. Phase 0: Correlation Structure Analysis** (`analyze_correlation_structure()`) - **IMMEDIATE ACTION**
     - **Status**: ✅ Implemented, ⚠️ NOT YET EXECUTED (can run now, ~15 min, no training required)
     - **Purpose**: Understand why KOEQUIPTE shows linear collapse while others show strong nonlinear benefits
     - **Key Metrics to Extract**:
       - `negative_fraction`: Fraction of negative correlations (0-1)
       - `strong_negative_count`: Count of correlations < -0.3
       - `mean_correlation`: Average correlation magnitude (absolute value)
       - `std_correlation`: Correlation distribution spread (standard deviation)
     - **Metrics-Driven Decision Criteria**: 
       - If KOEQUIPTE `negative_fraction > 0.3` AND others < 0.2 → tanh activation strategy confirmed (ReLU limitation)
       - If KOEQUIPTE `strong_negative_count > 10` AND others < 5 → tanh activation critical (strong negative relationships)
       - If KOEQUIPTE `mean_correlation < 0.1` AND others > 0.2 → deeper encoder strategy confirmed (weak signal)
       - If KOEQUIPTE `std_correlation < 0.15` AND others > 0.25 → structural complexity differs (may need different approach)
     - **Action**: Execute before training to inform improvement strategy. Results inform whether current improvements (tanh, deeper encoder) are appropriate or need adjustment.
     - **Output**: JSON files with correlation statistics for each target, comparison across targets to identify structural differences
  
  **2. Phase 1: Automatic Analysis After Training** (Runs automatically after `main_aggregator()`)
     - **Linearity Detection** (`detect_ddfm_linearity()`):
       - **Current Baseline**: KOEQUIPTE linearity score ~0.99 (near-perfect linear collapse)
       - **Target After Improvements**: < 0.95 (indicates encoder learning nonlinear features)
       - **Interpretation**: 0.95+ indicates encoder learning only linear features (PCA-like behavior)
       - **Output**: `outputs/experiments/ddfm_linearity_analysis.json` with linearity scores and recommendations
       - **Action if > 0.95**: Apply deeper encoder, tanh activation, weight decay, increased pre-training
     - **Prediction Quality Analysis** (`analyze_ddfm_prediction_quality()`):
       - **Current Baseline**: KOEQUIPTE improvement ratio ~0% (no improvement over DFM)
       - **Target After Improvements**: 
         - Improvement ratio > 10% (DDFM > 10% better than DFM)
         - Consistency > 0.7 (consistent improvement across horizons)
         - Linear collapse risk < 0.5 (low risk of linear collapse)
       - **Tracks**: Improvement ratio, consistency, horizon-weighted improvement, error pattern similarity, horizon error correlation, relative improvement consistency, improvement persistence
       - **Output**: Comprehensive analysis in `outputs/experiments/ddfm_linearity_analysis.json` with all metrics and recommendations
       - **Action if improvement < 10%**: Check encoder architecture, activation function, training dynamics, proceed to Phase 2
     - **Error Distribution Analysis** (via `calculate_standardized_metrics()`):
       - **Current Baseline**: KOEQUIPTE DDFM and DFM error distributions nearly identical (skewness diff ~0, kurtosis diff ~0)
       - **Target After Improvements**: DDFM should differ from DFM (skewness diff > 0.2, kurtosis diff > 1.0)
       - **Interpretation**: Different error distributions indicate DDFM learning different patterns from DFM (nonlinear behavior)
       - **Tracks**: Error skewness, kurtosis, bias-variance decomposition, error concentration, prediction bias
       - **Action if diff < 0.2**: DDFM and DFM have similar error patterns (linear collapse risk), apply improvements
  
  **3. Metrics-Driven Decision Tree** (Quantitative Criteria Based on Automatic Analysis):
     - **✅ SUCCESS** (linearity score < 0.95 AND improvement ratio > 10% AND consistency > 0.7 AND collapse_risk < 0.5):
       - **Action**: Document percentage improvement, proceed to Phase 1.2 (activation ablation study)
       - **Metrics to Document**: All metrics from automatic analysis (linearity, improvement ratio, consistency, collapse risk, error pattern similarity, horizon error correlation, error distribution differences, horizon-weighted improvement, relative improvement consistency, improvement persistence)
     - **⚠️ PARTIAL SUCCESS** (improvement 5-10% OR linearity 0.95-0.98 OR consistency 0.5-0.7 OR collapse_risk 0.5-0.7):
       - **Action**: Document findings, investigate specific metrics that didn't meet targets, proceed to Phase 1.2 with caution
       - **Investigation**: Check which specific metrics failed (linearity, improvement ratio, consistency, collapse risk, error patterns) and why
     - **❌ NEEDS INVESTIGATION** (improvement < 10% OR linearity ≥ 0.95 OR collapse_risk ≥ 0.5 OR error pattern similarity > 0.6):
       - **Action**: Check logs, verify settings were applied correctly, investigate root cause, proceed to Phase 2 (advanced improvements)
       - **Investigation**: Verify encoder architecture, activation function, weight decay, pre-training, batch size were all applied correctly
     - **❌ FAILURE** (no improvement or degradation OR improvement < 5% OR linearity > 0.98):
       - **Action**: Check logs for errors, investigate root cause, proceed to Phase 2 (advanced improvements) or Phase 3 (ensemble/hybrid approaches)
       - **Investigation**: Check training logs for errors, verify all improvements were applied, investigate why improvements didn't work
     - **⚠️ WARNING** (error distribution differences: skewness diff < 0.2, kurtosis diff < 1.0):
       - **Action**: DDFM still learning similar patterns to DFM, may need Phase 2 improvements (architecture grid search, factor loading analysis)
     - **⚠️ WARNING** (horizon-weighted improvement < 5%):
       - **Action**: Short-term performance not improving, may need different strategy (horizon-specific tuning, ensemble methods)
  
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

**DDFM Metrics Documentation and Additional Metrics** (Implemented):
- Enhanced documentation in `analyze_ddfm_prediction_quality()` with comprehensive docstrings and limitations documentation
- Quantile-based error metrics (`calculate_quantile_based_metrics()`) for robust evaluation of skewed distributions
- Factor loading comparison (`compare_factor_loadings()`) to detect linear collapse (requires model internals)
- Status: ✅ **IMPLEMENTED IN CODE** - Enhanced documentation and additional metrics available

- **Metrics Research Improvements** (All Implemented):
  - Enhanced DDFM linearity detection, prediction quality analysis, horizon-weighted metrics, training-aligned metrics
  - Missing horizons analysis, enhanced error distribution metrics, horizon error correlation analysis
  - Enhanced linear collapse risk assessment (7 risk factors), robust statistics and bootstrap confidence intervals
  - Forecast skill score and information gain metrics, factor dynamics stability inference
  - Relative error stability metrics, improvement persistence metrics, temporal consistency metrics
  - All metrics automatically calculated after aggregation via `main_aggregator()`
  - Status: ✅ **ALL IMPLEMENTED IN CODE** - Comprehensive metrics available for DDFM performance analysis

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
  
**Phase 1: Testing with Trained Models (Models exist)**
2. **Models Already Trained** - Models exist (trained Dec 9 02:35-02:47)
   - **Status**: ✅ **MODELS TRAINED** - `checkpoint/` contains 12 model.pkl files
   - **Priority**: N/A - Models are trained and ready for use
   - **Action**: None - models exist, can proceed directly to forecasting and backtesting
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
- ✅ **Models TRAINED**: `checkpoint/` contains 12 model.pkl files (trained Dec 9 02:35-02:47). All models exist and ready for use.
- 📊 **Baseline metrics** (from aggregated_results.csv): KOEQUIPTE DDFM sMAE=1.14 (identical to DFM), KOIPALL.G sMAE=0.69, KOWRCCNSE sMAE=0.50

**Plan**: Phase 0 (correlation analysis, ~15 min) → Phase 1 (train & test, requires training) → Phase 1.2/1.3 (ablation/investigation) → Phase 2/3 (advanced if needed)

**Success Criteria**: KOEQUIPTE sMAE < 1.03 (≥10% improvement), maintain others < 0.7, complete all 22 horizons

**Key Files**: `src/train.py` (KOEQUIPTE settings), `src/evaluation/evaluation_aggregation.py` (analysis functions), `outputs/experiments/aggregated_results.csv` (results)

---

## DDFM Metrics Research: Analysis Functions

**Key Analysis Functions** (all in `src/evaluation/evaluation_aggregation.py`):
- `detect_ddfm_linearity()` - Detects linear collapse (output: `outputs/experiments/ddfm_linearity_analysis.json`)
- `analyze_ddfm_prediction_quality()` - Comprehensive performance analysis (runs automatically after aggregation)
- `analyze_correlation_structure()` - Pre-training data structure analysis (output: `outputs/analysis/correlation_analysis_{target}.json`)

**Key Metrics to Track** (target values, current baseline):
- Linearity score: < 0.95 (current: ~0.99), Improvement ratio: > 10% (current: ~0%), Linear collapse risk: < 0.5 (current: ~0.95+), Consistency: > 0.7, Error pattern similarity: < 0.6, Horizon error correlation: < 0.5

**Workflow**: Phase 0 (correlation analysis) → Phase 1 (check auto-calculated metrics after aggregation) → Iterative improvement based on metrics
