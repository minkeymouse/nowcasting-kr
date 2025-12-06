# Project Status

## Current State (2025-12-06 - Start of New Experiment Phase)

**Current Summary**: Project restructured with new targets and condensed report. Experiments need to be run for 4 new targets (KOEQUIPTE, KOWRCCNSE, KOIPALL.G, KOMPRI30G). Report structure updated to 6 sections focusing on production, investment, and consumption models. All configuration files updated. Test script verified and ready.

**What's Done**:
- ✅ Configuration updated: 4 new target configs created (koequipte_report, kowrccnse_report, koipallg_report, kompri30g_report)
- ✅ Series configs updated: All 101 series configs have `block: null` (only global block used)
- ✅ Data path fixed: All configs use `data/data.csv`
- ✅ Report structure updated: 6 sections (Introduction, Methodology, Production Model, Investment Model, Consumption Model, Conclusion) - condensed to under 15 pages
- ✅ Scripts finalized: `run_experiment.sh` and `run_test_experiment.sh` ready for new targets
- ✅ Test script verified: ARIMA and VAR tests passing

**What's Not Done**:
- ⏳ Experiments: Need to run experiments for 4 new targets (48 combinations: 4 targets × 4 models × 3 horizons)
- ⏳ Report content: Need to populate with actual experiment results
- ⏳ Tables: Need to update with actual metrics from experiments
- ⏳ Plots: Need to generate with new target data

**Status for Next Iteration**: 
- ✅ Configuration ready: All 4 target configs exist, series configs updated, scripts ready
- ✅ Test verification: Test script working correctly (ARIMA/VAR verified)
- ⏳ **NEXT**: Run experiments using `run_test_experiment.sh` for verification, then `run_experiment.sh` for full run

**Next Steps**: 
1. **Critical**: Run test experiments to verify setup (`./run_test_experiment.sh`)
2. **Critical**: Run full experiments (`./run_experiment.sh`) for all 4 targets
3. **After experiments**: Update report with actual results, generate plots, compile PDF

### Project Overview
Systematic comparison framework for nowcasting Korean macroeconomic variables (Production: KOIPALL.G, KOMPRI30G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE) using 4 forecasting models (ARIMA, VAR, DFM, DDFM) across 3 forecast horizons (1, 7, 28 days). Goal: Complete under 15 page LaTeX report with experimental results and finalized dfm-python package.

### Experiment Status

**New Configuration**:
- **Targets**: 4 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G, KOMPRI30G)
- **Models**: 4 (ARIMA, VAR, DFM, DDFM)
- **Horizons**: 3 (1, 7, 28 days)
- **Total**: 48 combinations (4 × 4 × 3)

**Status**: Experiments not yet run for new targets. Use `run_test_experiment.sh` to verify setup before full run.

**Configuration Details**:
- All series configs use `block: null` (only global block for DFM/DDFM)
- Data file: `data/data.csv`
- Config files: `config/experiment/{koequipte,kowrccnse,koipallg,kompri30g}_report.yaml`

### Code Status

**Package Status**:
- ✅ **dfm-python**: Finalized with consistent naming (PascalCase classes, snake_case functions), clean code patterns
- ✅ **src/**: 15 files (max 15 required), all modules working correctly
- ✅ **Config**: All 4 target configs created, series configs updated (block: null)
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized and verified

### Report Status

**Structure** (6 sections, under 15 pages):
- ✅ **Introduction**: Overview of nowcasting and research objectives
- ✅ **Methodology**: DFM, monthly index estimation, high-frequency models, deep learning models
- ✅ **Production Model**: KOIPALL.G, KOMPRI30G - Data composition, DFM estimation, nowcasting performance, model comparisons
- ✅ **Investment Model**: KOEQUIPTE - Data composition, DFM estimation, nowcasting performance, model comparisons
- ✅ **Consumption Model**: KOWRCCNSE - Data composition, DFM estimation, nowcasting performance, model comparisons
- ✅ **Conclusion**: Summary and future research directions

**Content Status**:
- ⏳ **Tables**: Structure ready, need actual results from experiments
- ⏳ **Plots**: Need to generate with new target data
- ⏳ **Results**: Need to populate with actual experiment metrics

## Project Structure

**Source Code (`src/`)**: 15 files - Entry points (train.py, infer.py), model wrappers (ARIMA/VAR/DFM/DDFM), evaluation, preprocessing
**DFM Package (`dfm-python/`)**: Finalized - DFM (EM algorithm), DDFM (PyTorch Lightning), clean code patterns, consistent naming
**Report (`nowcasting-report/`)**: Structure ready - 6 LaTeX sections, tables structure ready, need results
**Experiment Pipeline**: Hydra configs, run_experiment.sh, run_test_experiment.sh, outputs/comparisons/, outputs/experiments/

## Work Completed Summary

**Summary**: Project restructured with new targets and condensed report. All configuration files updated. Test script verified. Ready to run experiments.

## Next Steps

### ⏳ Immediate Tasks (High Priority)
**Run Experiments** [Status: Ready to Start]
- **Goal**: Run experiments for 4 new targets, compare 4 models, generate results
- **Tasks** (execute in order):
  1. **Task 1.1**: Verify setup with test script (`./run_test_experiment.sh`)
  2. **Task 1.2**: Run full experiments (`./run_experiment.sh`) for all 4 targets
  3. **Task 1.3**: Generate aggregated results and plots
  4. **Task 1.4**: Update report with actual results
  5. **Task 1.5**: Compile PDF (under 15 pages target)

## Experiment Configuration

- **Targets**: 4 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G, KOMPRI30G)
- **Models**: 4 (ARIMA, VAR, DFM, DDFM)
- **Horizons**: 3 (1, 7, 28 days)
- **Total**: 48 combinations (4 × 4 × 3)
- **Series Config**: All series use `block: null` (only global block)
- **Data File**: `data/data.csv`
