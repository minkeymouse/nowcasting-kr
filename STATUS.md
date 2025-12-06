# Project Status

## Current State (2025-12-06)

### Project Overview
This project implements a systematic comparison framework for nowcasting Korean macroeconomic variables (GDP, Consumption, Investment) using 4 forecasting models (ARIMA, VAR, DFM, DDFM) across 3 forecast horizons (1, 7, 28 days). The goal is to produce a complete 20-30 page LaTeX report with actual experimental results and finalize the dfm-python package.

### Experiment Status

**Latest Update**: 2025-12-06

**Completed** (18/36 = 50%):
- ✅ **ARIMA**: All 9 combinations (3 targets × 3 horizons, n_valid=1)
  - Overall: sMSE 0.171, sMAE 0.366, sRMSE 0.366
  - By target: GDP (sRMSE 0.314), Consumption (sRMSE 0.229), Investment (sRMSE 0.555)
  - By horizon: 1-day (sRMSE 0.445), 7-day (0.376), 28-day (0.277)
- ✅ **VAR**: All 9 combinations (3 targets × 3 horizons, n_valid=1)
  - Overall: sMSE 0.004, sMAE 0.046, sRMSE 0.046
  - By target: GDP (sRMSE 0.056), Consumption (sRMSE 0.055), Investment (sRMSE 0.028)
  - By horizon: 1-day (sRMSE 0.006), 7-day (0.036), 28-day (0.098)

**Ready for Full Run** (18/36 = 50%):
- ⏳ **DFM**: Fixes applied and tested successfully
  - Fix: Added target_series to training data, improved evaluation fallback logic
  - Test result: KOGDP...D horizon 1 - n_valid=1, sRMSE=0.713, converged=True
  - Status: Ready for full run (all 9 combinations)
- ⏳ **DDFM**: Fixes applied and tested successfully
  - Fixes: Gradient clipping (1.0), learning rate (0.005 with exponential decay), pre-training, ReLU activation, batch size (100), input clipping, NaN batch skipping
  - Test result: C matrix no NaN (mean=-0.11, std=0.38, nonzero=76/76), training successful
  - Status: Ready for full run (all 9 combinations)

**Total Progress**: 18/36 = 50% complete

### Code Status

**Fixes Applied**:
- ✅ VAR: target_series handling, KeyError handling
- ✅ DFM: target_series handling, evaluation fallback, DataFrame column preservation
- ✅ DDFM: Gradient clipping, learning rate scheduler (exponential decay, gamma=0.96), pre-training, ReLU activation, batch size 100, input clipping, NaN batch skipping
- ✅ Config: DDFM configs updated (learning_rate=0.005, activation=relu, batch_size=100, decay_learning_rate=true)
- ✅ Logger: DFMTrainer logger enabled (creates lightning_logs/dfm/ folder)
- ✅ Legacy code: Cleaned up deprecated comments and backward compatibility notes

**Package Status**:
- ✅ dfm-python: Finalized with consistent naming, clean code patterns
- ✅ src/: 15 files (max 15 required), all fixes verified
- ✅ Tests: All pytest tests passing (133 passed, 8 skipped)

### Report Status

**Completed**:
- ✅ Structure: All 8 sections complete
- ✅ Citations: 21 references verified
- ✅ Results: ARIMA and VAR findings integrated with detailed analysis
- ✅ Discussion: ARIMA and VAR findings included with comprehensive comparison
- ✅ Tables: ARIMA and VAR values filled (DFM/DDFM remain "---")
- ✅ Plots: Generated with ARIMA and VAR data

**Pending**:
- ⏳ DFM/DDFM results: Tables show "---" placeholders (waiting for full experiments)
- ⏳ Full model comparison: Discussion needs DFM/DDFM results

## Project Structure

**1. Source Code (`src/`) - 15 files**
- Entry Points: `train.py` (compare command), `infer.py` (nowcast command)
- Core Module: Unified training via sktime forecasters
- Model Wrappers: ARIMA/VAR (sktime), DFM (EM algorithm), DDFM (PyTorch Lightning)
- Evaluation: Standardized metrics (sMSE, sMAE, sRMSE), aggregation

**2. DFM Package (`dfm-python/`) - Finalized**
- Core Models: DFM (EM algorithm), DDFM (PyTorch encoder + VAE)
- Lightning Integration: DataModule, KalmanFilter, EMAlgorithm
- Features: Clock-based mixed-frequency, block-structured factors, Hydra YAML config
- Status: Code finalized, legacy code cleaned up

**3. Report (`nowcasting-report/`) - Structure Complete**
- Sections: 8 LaTeX sections
- Tables: 4 tables (ARIMA/VAR filled, DFM/DDFM pending)
- Plots: 4 PNG images (ARIMA/VAR data)
- Citations: 21 references verified

**4. Experiment Pipeline**
- Config: Hydra YAML configs in `config/experiment/`, `config/model/`, `config/series/`
- Execution: `run_experiment.sh` with parallel processing, MODELS filter support
- Results: Per-target JSON, aggregated CSV, model pickles

## Next Steps

1. **Run full DFM experiments**: `MODELS="dfm" bash run_experiment.sh` (8 remaining combinations)
2. **Run full DDFM experiments**: `MODELS="ddfm" bash run_experiment.sh` (8 remaining combinations)
3. **Generate aggregated results**: Update CSV, plots, and LaTeX tables
4. **Finalize report**: Update results/discussion sections, compile PDF

## Experiment Configuration

- **3 Targets**: KOGDP...D (GDP), KOCNPER.D (Consumption), KOGFCF..D (Investment)
- **4 Models**: arima, var, dfm, ddfm
- **3 Horizons**: 1, 7, 28 days
- **Total**: 36 combinations (18 complete, 18 pending)
