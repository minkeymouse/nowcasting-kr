# Issues and Action Plan

## Executive Summary (2025-01-XX)

**Current State**: ARIMA working with complete results (9 combinations). Report improved with actual findings. VAR KeyError fix applied, DFM/DDFM numerical instability remains.  
**Goal**: Complete 20-30 page report with actual results, finalize dfm-python package  
**Strategy**: Incremental approach - test VAR fix first, then investigate DFM/DDFM numerical issues  
**Next Action**: Test VAR KeyError fix, then investigate DFM/DDFM C matrix NaN issues

## Critical Issues (PRIORITY 1)

### 1. VAR n_valid=0 [FIXED - NEEDS TESTING]
**Status**: KeyError fix applied in calculate_standardized_metrics() for y_train.columns lookup  
**Root Cause**: When target_series is string and not in y_train.columns, KeyError raised at line 139  
**Fix Applied**: Added KeyError handling with fallback to use column index from y_true or column 0  
**Action**: Test VAR on single target/horizon to verify fix works
- Command: `MODELS="var" bash run_experiment.sh` or `.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models var --horizons 1`
- Success: n_valid > 0 after fix

### 2. DFM Numerical Instability [INVESTIGATE]
**Status**: C matrix contains NaN (confirmed in JSON results)  
**Root Cause**: EM algorithm C matrix update produces NaN
- KOGDP...D: first row all NaN
- KOCNPER.D: multiple NaN values
- KOGFCF..D: Training succeeds (loglik=135.76) but n_valid=0

**Fix Needed**:
1. Add NaN detection/early stopping in dfm-python/src/dfm_python/ssm/em.py
2. Check C matrix normalization (||C[:,j]|| = 1) handles zero denominator gracefully
3. Verify PCA initialization handles edge cases (T < N, high missing data, constant series)

**Files**: `dfm-python/src/dfm_python/ssm/em.py`, `dfm-python/src/dfm_python/models/dfm.py`

### 3. DDFM Numerical Instability [INVESTIGATE]
**Status**: C matrix has NaN for all targets (confirmed in JSON)  
**Root Cause**: PyTorch encoder produces NaN during training

**Fix Needed**:
1. Check encoder forward pass in dfm-python/src/dfm_python/models/ddfm.py
2. Add NaN detection in training_step
3. Verify gradient clipping and initialization

**Files**: `dfm-python/src/dfm_python/models/ddfm.py`

## Code Fixes Completed

1. ✅ **VAR KeyError fix**: Added KeyError handling in calculate_standardized_metrics() for y_train.columns
2. ✅ **ARIMA/VAR target_series handling**: Fixed Series input handling
3. ✅ **DFM/DDFM pickle errors**: Fixed make_cha_transformer (uses functools.partial)
4. ✅ **Test data size check**: Skip horizon 28 if test set too small
5. ✅ **run_experiment.sh**: Added MODELS environment variable support
6. ✅ **train.py**: Added --models flag for incremental testing

## Action Plan (Incremental, Prioritized)

### PHASE 1: Test VAR Fix [NEXT ACTION]
1. ⏳ Test VAR on single target/horizon: `MODELS="var" bash run_experiment.sh`
2. ⏳ Verify n_valid > 0 after fix
3. ⏳ Re-run VAR for all targets if fix works

### PHASE 2: Fix DFM/DDFM Numerical Instability [AFTER VAR]
4. ⏳ Investigate DFM C matrix NaN: Check EM algorithm in em.py
5. ⏳ Investigate DDFM C matrix NaN: Check encoder in ddfm.py
6. ⏳ Add NaN detection and early stopping
7. ⏳ Test fixes on single target/horizon

### PHASE 3: Generate Full Results [AFTER PHASE 1-2]
8. ⏳ Re-run full experiments: `bash run_experiment.sh` (will skip ARIMA, run VAR/DFM/DDFM)
9. ⏳ Generate aggregated CSV: `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
10. ⏳ Update tables/plots with all models

### PHASE 4: Finalize Report [AFTER PHASE 3]
11. ⏳ Update results section with VAR/DFM/DDFM findings
12. ⏳ Update discussion with all model comparisons
13. ⏳ Finalize report: Compile PDF, verify 20-30 pages

## Experiment Status

**Latest Run**: 20251206_082502
- **ARIMA**: ✅ WORKING - n_valid=1 for all horizons across all 3 targets (9 combinations)
- **VAR**: ❌ n_valid=0 - KeyError fix applied, needs testing
- **DFM**: ❌ n_valid=0 - C matrix NaN (numerical instability)
- **DDFM**: ❌ n_valid=0 - C matrix NaN (encoder issue)

**Experiments Needed**:
- Minimum viable: 6 combinations (2 models × 3 targets) - ✅ ARIMA has 9 (exceeds minimum)
- Ideal: All 36 combinations (3 targets × 4 models × 3 horizons)
- Current: 9/36 = 25% complete

## Code Quality Improvements (PRIORITY 2 - After Critical Fixes)

1. **dfm-python Numerical Stability**:
   - Add early stopping if C matrix becomes NaN/Inf during EM iterations
   - Verify regularization constants handle edge cases
   - Check C matrix normalization handles zero denominator

2. **src/ Code Quality**:
   - Consolidate duplicate logic in model wrappers
   - Verify all exceptions logged
   - Check for silent NaN propagation

## Report Status

- ✅ Structure: Complete 8-section framework
- ✅ Citations: All 21 references verified
- ✅ Content: Sections 1-4, 6-7 complete with actual findings
- ✅ Results: Section 5 updated with ARIMA findings and specific metrics
- ✅ Discussion: Section 6 improved with actual ARIMA insights and model selection guidance
- ✅ Conclusion: Section 7 updated to reflect actual experimental results
- ⚠️ Tables: ARIMA values filled, VAR/DFM/DDFM remain "---"
- ⚠️ Plots: Generated with ARIMA data, can be updated when other models work
