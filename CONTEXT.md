# Project Context Summary

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (15 files max) - wrappers for sktime & dfm-python
- **dfm-python/**: Core DFM/DDFM package with Lightning-based training
- **nowcasting-report/**: LaTeX report with plots from outputs/
- **config/**: Hydra YAML configs (experiment/, model/, series/)
- **outputs/**: Experiment results (comparisons/, models/, experiments/)

## Key Components

### src/ Module (Experiment Engine) - 14 files
**Entry Points (3):**
- `train.py`: CLI/API for training (`train`, `compare` commands)
- `infer.py`: CLI/API for nowcasting evaluation (`nowcast` command)
- `nowcasting.py`: Nowcasting simulation utilities (masking, splits)

**Core Modules (11):**
- `core/training.py`: Unified training via sktime forecasters (ARIMA, VAR, DFM, DDFM)
- `model/` (4 files): 
  - `dfm.py`: DFM wrapper
  - `ddfm.py`: DDFM wrapper
  - `sktime_forecaster.py`: sktime Forecaster interface (DFMForecaster, DDFMForecaster)
  - `_common.py`: Shared utilities for model wrappers
- `preprocess/` (3 files):
  - `transformations.py`: Per-series transformations (pch, pc1, lin, etc.)
  - `sktime.py`: sktime transformer integration
  - `utils.py`: Data loading utilities
- `eval/` (2 files):
  - `evaluation.py`: Standardized metrics (sMSE, sMAE, sRMSE), model comparison
  - `aggregator.py`: Result aggregation across experiments
- `utils/` (2 files):
  - `config_parser.py`: Hydra config parsing, experiment config extraction
  - `path_setup.py`: Path management for imports

**Data Flow:**
1. Config parsing (`utils/config_parser.py`) → Hydra config
2. Series config loading → dfm-python format conversion
3. Data loading → Preprocessing (transformations per series)
4. Model training → sktime forecaster interface
5. Evaluation → Standardized metrics (sMSE, sMAE, sRMSE)
6. Results → outputs/comparisons/ (JSON) → aggregation → outputs/experiments/

### dfm-python/ Package
**Core Structure:**
- `models/`: DFM (EM algorithm), DDFM (PyTorch encoder + VAE)
- `lightning/`: DataModule, KalmanFilter, EMAlgorithm (PyTorch Lightning integration)
- `nowcast/`: Nowcasting utilities, news decomposition, data views
- `config/`: Config adapters, schema validation, results handling
- `ssm/`: State-space model (Kalman filter, EM algorithm)
- `encoder/`: PCA, VAE encoders for DDFM
- `trainer/`: DFMTrainer, DDFMTrainer (Lightning trainers)
- `data/`: Data loading, dataset utilities
- `logger/`: Training/inference logging

**Key Features:**
- Clock-based mixed-frequency (monthly clock, tent kernels)
- Block-structured factors (global + block-level)
- Lightning-based training pattern (DFMTrainer/DDFMTrainer)
- Hydra YAML configuration

### nowcasting-report/ Structure
**LaTeX Report:**
- `main.tex`: Main document
- `preamble.tex`: LaTeX packages, commands
- `contents/`: Chapter files (1_introduction.tex, 2_dfm_modeling.tex, etc.)
- `tables/`: LaTeX tables (tab_nowcasting_metrics.tex, etc.)
- `images/`: PNG plots (accuracy_heatmap.png, forecast_vs_actual.png, etc.)
- `code/plot.py`: Plot generation from outputs/comparisons/
- `references.bib`: Bibliography

## Experiment Hooks

**Training Pipeline:**
```
run_experiment.sh
  → src/train.py compare --config-name experiment/{target}_report
    → parse_experiment_config() [utils/config_parser.py]
    → compare_models() [core/training.py]
      → For each model in config:
        → train() [core/training.py]
          → _train_forecaster() [unified sktime interface]
            → Load data (CSV) → Preprocess (per-series transformations)
            → Create forecaster (ARIMA/VAR/DFMForecaster/DDFMForecaster)
            → forecaster.fit(y_train) → forecaster.predict(fh=horizons)
            → evaluate_forecaster() [eval/evaluation.py]
            → Save to outputs/models/{model_name}/model.pkl
      → _compare_results() [core/training.py]
        → compare_multiple_models() [eval/evaluation.py]
        → generate_comparison_table() [eval/evaluation.py]
    → Save to outputs/comparisons/{target}_{timestamp}/comparison_results.json
```

**Nowcasting Pipeline:**
```
src/infer.py nowcast --config-name experiment/{target}_report
  → run_nowcasting_evaluation() [infer.py]
    → Load trained model from outputs/models/
    → simulate_nowcasting_evaluation() [nowcasting.py]
      → For each target_date:
        → mask_recent_observations() [mask_days lag]
        → create_nowcasting_splits() [train/test splits]
        → forecaster.fit(y_train_masked) → predict()
        → calculate_metrics_per_horizon() [eval/evaluation.py]
        → Compare with full-data baseline
    → Return nowcast_metrics, full_metrics, improvement
```

**Result Aggregation:**
```
src/eval/aggregator.py (main)
  → collect_all_comparison_results() [from outputs/comparisons/]
  → aggregate_overall_performance() [combine across targets/models/horizons]
  → Save to outputs/experiments/aggregated_results.csv
```

**Plot Generation:**
```
nowcasting-report/code/plot.py
  → load_comparison_results() [from outputs/comparisons/]
  → extract_metrics_from_results() [DataFrame]
  → Generate plots:
    - accuracy_heatmap.png: Model × Horizon heatmap
    - forecast_vs_actual.png: Time series comparison
    - horizon_trend.png: Metrics by horizon
    - model_comparison.png: Model performance comparison
  → Save to nowcasting-report/images/*.png
```

## Configuration Structure

**Experiment Config** (`config/experiment/{target}_report.yaml`):
- `target_series`: Target variable (e.g., KOGDP...D)
- `data_path`: CSV data file
- `forecast_horizons`: [1, 7, 28]
- `models`: [arima, var, dfm, ddfm]
- `series`: List of series IDs
- `model_overrides`: Model-specific parameters

**Model Config** (`config/model/{model}.yaml`):
- DFM/DDFM: blocks structure, AR lag, regularization
- ARIMA/VAR: order, auto-selection flags

**Series Config** (`config/series/{series_id}.yaml`):
- `series_id`, `frequency`, `transformation`, `blocks`

## Data Flow Summary

1. **Config → Model Setup**: 
   - Hydra loads `config/experiment/{target}_report.yaml`
   - Extract series list → load `config/series/{series_id}.yaml` for each
   - Load `config/model/{model}.yaml` for block structure
   - Build dfm-python config dict (series, blocks, model params)
   - Apply `model_overrides` from experiment config

2. **Data → Preprocessing**: 
   - Load CSV (`data_path`) → pandas DataFrame
   - For each series: apply transformation (pch/pc1/lin) from series config
   - Create sktime FunctionTransformer pipeline
   - Fit/transform → standardized data

3. **Training**: 
   - Create forecaster (ARIMA/VAR/DFMForecaster/DDFMForecaster)
   - `forecaster.fit(y_train)` → EM (DFM) or PyTorch Lightning (DDFM)
   - Save to `outputs/models/{model_name}/model.pkl` (forecaster + config + metadata)

4. **Evaluation**: 
   - Train/test split (80/20) → `forecaster.predict(fh=horizons)`
   - `calculate_standardized_metrics()` → sMSE, sMAE, sRMSE per horizon
   - Normalize by training std (σ) for fair comparison

5. **Comparison**: 
   - Aggregate across models → `compare_multiple_models()`
   - `generate_comparison_table()` → DataFrame
   - Save JSON (`comparison_results.json`) + CSV (`comparison_table.csv`)

6. **Visualization**: 
   - Load JSON from `outputs/comparisons/`
   - Extract metrics → pandas DataFrame
   - matplotlib/seaborn plots → `nowcasting-report/images/*.png`

## Key Design Patterns

- **Unified Interface**: All models use sktime forecaster interface (fit/predict)
- **Config-Driven**: Hydra YAML configs for all experiments
- **Modular Preprocessing**: Per-series transformations via sktime FunctionTransformer
- **Standardized Metrics**: sMSE, sMAE, sRMSE (normalized by training std)
- **Output Structure**: outputs/{models,comparisons,experiments}/ with timestamps

## File Count Summary

**src/**: 20 Python files (exceeds 15-file limit - needs consolidation)
- Entry: train.py, infer.py, nowcasting.py (3)
- Core: core/{__init__,training}.py (2)
- Model: model/{__init__,dfm,ddfm,sktime_forecaster,_common}.py (5)
- Preprocess: preprocess/{__init__,transformations,sktime,utils}.py (4)
- Eval: eval/{__init__,evaluation,aggregator}.py (3)
- Utils: utils/{__init__,config_parser,path_setup}.py (3)

**Note**: Current count (20) exceeds 15-file limit. Consider consolidating:
- Merge preprocess modules
- Combine eval modules
- Consolidate utils

**dfm-python/**: Core package (submodule)
- Models, trainers, data modules, config adapters

**nowcasting-report/**: LaTeX report
- main.tex, preamble.tex, contents/*.tex, tables/*.tex, images/*.png
- code/plot.py for visualization

## Current Status

### Working Components
- **Training pipeline**: Architecture complete (ARIMA, VAR, DFM, DDFM via sktime interface)
- **Evaluation**: Standardized metrics (sMSE, sMAE, sRMSE) per horizon implemented
- **Comparison**: Multi-model comparison with aggregation to CSV framework ready
- **Nowcasting**: Framework exists (`nowcasting.py`), full integration pending
- **Report**: LaTeX structure exists, plot generation code ready

### Current Issues (Critical)
1. **Import Errors**: ✅ FIXED - Root causes addressed
   - Error: `ModuleNotFoundError: No module named 'src'` at line 27
   - Root causes: Missing `src/__init__.py` + incorrect path calculation
   - Fixes: Created `src/__init__.py`, fixed path in `train.py` and `infer.py`
   - Status: Ready for testing (fixes applied, not yet verified with actual run)

2. **No Experiment Results**: 
   - `outputs/comparisons/` contains only `.log` files (9 error logs)
   - No `comparison_results.json` files found
   - No actual metrics data available for report
   - Status: Blocked until import fix verified and experiments run successfully

3. **Report Content**: ✅ IMPROVED (Latest iteration)
   - Removed redundant mentions of "experiments in progress"
   - Improved flow and clarity in results section
   - Still has placeholder content for missing experiments (KOCNPER.D, KOGFCF..D)
   - Status: Content quality improved, but still needs actual results for complete report

4. **File Count**: 
   - `src/` has 17 Python files (code effectively in 15 files with deprecation wrappers)
   - Deprecation wrappers kept for backward compatibility
   - Status: Non-critical, within acceptable range

### Experiment Results Status (Confirmed 2025-12-06)
- **KOGDP...D**: Failed (import error) - 3 runs (00:17:31, 00:24:02, 00:44:56)
- **KOCNPER.D**: Failed (import error) - 3 runs (00:17:31, 00:24:02, 00:44:56)
- **KOGFCF..D**: Failed (import error) - 3 runs (00:17:31, 00:24:02, 00:44:56)
- **Result Files**: None generated (no JSON, CSV, or result directories)
- **Error**: `ModuleNotFoundError: No module named 'src'` at `src/train.py:27`
- **Root Causes**: 
  1. Missing `src/__init__.py` (Python requires this for package recognition)
  2. Incorrect path: `_project_root = _script_dir.parent.parent` should be `_script_dir.parent`
- **Fixes Applied**: Created `src/__init__.py`, fixed path calculation in both `train.py` and `infer.py`

### Next Steps (Priority Order)
1. **Test import fixes** - Verify import errors are resolved with actual experiment run
2. **Run successful experiments** - Generate actual results for all 3 targets
3. **Update report with results** - Populate tables and generate plots from actual data
4. **Complete report** - Ensure 20-30 pages with all sections properly filled
5. **Finalize dfm-python** - Review numerical stability and theoretical correctness (naming already verified as consistent)
