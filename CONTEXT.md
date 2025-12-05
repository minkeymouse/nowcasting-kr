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

**src/**: 17 Python files (exceeds 15-file limit by 2 files)
- Entry: train.py, infer.py, nowcasting.py (3) [nowcasting.py is deprecated wrapper]
- Core: core/{__init__,training}.py (2)
- Model: model/{__init__,dfm,ddfm,sktime_forecaster}.py (4)
- Preprocess: preprocess/{__init__,transformations,utils}.py (3) [transformations.py is deprecated wrapper]
- Eval: eval/{__init__,evaluation}.py (2) [aggregator functionality in evaluation.py as main_aggregator()]
- Utils: utils/{__init__,config_parser}.py (2)

**Note**: Current count (17) exceeds 15-file limit by 2 files. Two deprecated wrapper files (nowcasting.py, preprocess/transformations.py) cannot be deleted per project rules. Effective code is in 15 files.

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

### Current Issues
1. **Missing Dependencies**: ⚠️ CURRENT BLOCKER
   - Missing `hydra-core` dependency - all 36 runs failed
   - Code fixes complete, ready once dependencies installed

2. **No Experiment Results**: 
   - No result files, no trained models, no aggregated results
   - Blocked until dependencies installed and experiments run

3. **Report Placeholders**: 
   - Content enhanced, but placeholders remain for KOCNPER.D and KOGFCF..D
   - Requires actual experiment results

### Current Status Summary
- **Experiments**: 0/3 targets complete, 36 failed runs - all due to missing `hydra-core` dependency
- **Code**: All import/path issues fixed, dfm-python finalized (consistent naming, clean patterns)
- **Report**: Structure complete, content enhanced, placeholders for KOCNPER.D and KOGFCF..D
- **Blocker**: Missing Python dependencies (hydra-core, omegaconf) - install before running experiments
- **File System**: No result directories, no JSON/CSV files, `outputs/models/` doesn't exist

## Project Overview
Comprehensive nowcasting framework for Korean macroeconomic variables:
- **4 Models**: ARIMA, VAR, DFM, DDFM
- **3 Targets**: KOGDP...D (GDP), KOCNPER.D (Consumption), KOGFCF..D (Investment)
- **3 Horizons**: 1, 7, 28 days
- **Goal**: Complete 20-30 page report with actual results

### Experiment Configuration
- **3 targets** × **4 models** × **3 horizons** = 36 combinations
- Targets: KOGDP...D (55 series), KOCNPER.D (50 series), KOGFCF..D (19 series)
- Models: arima, var, dfm, ddfm
- Horizons: [1, 7, 28] days
- **Status**: 0/3 targets complete (all failed due to missing dependencies)

### Experiment Execution Flow

**run_experiment.sh Logic:**
1. Validates environment (venv, data file, config files, dependencies)
2. Checks for completed experiments via `is_experiment_complete()`:
   - Looks for `outputs/comparisons/{target}_{timestamp}/comparison_results.json`
   - Skips if latest result directory contains JSON file
3. Runs incomplete targets in parallel (max 5 concurrent)
4. Each experiment: `python3 src/train.py compare --config-name experiment/{target}_report`
5. After completion: Aggregates results via `main_aggregator()` from `src.eval.evaluation`

**Expected Output Structure:**
```
outputs/
├── comparisons/
│   ├── KOGDP...D_YYYYMMDD_HHMMSS/
│   │   ├── comparison_results.json      # Full results with metrics per model/horizon
│   │   └── comparison_table.csv         # Summary table
│   ├── KOCNPER.D_YYYYMMDD_HHMMSS/
│   └── KOGFCF..D_YYYYMMDD_HHMMSS/
├── models/
│   ├── {target}_arima/model.pkl
│   ├── {target}_var/model.pkl
│   ├── {target}_dfm/model.pkl
│   └── {target}_ddfm/model.pkl
└── experiments/
    └── aggregated_results.csv           # Combined results across all targets
```

### Experiment Status
- **0/3 targets complete**: All 36 runs failed due to missing `hydra-core` dependency
- **No result files**: No JSON/CSV, no result directories, no trained models
- **Error progression**: Relative import → missing src → missing hydra (all code issues fixed)
- **Latest error**: `ImportError: Required dependencies not available: No module named 'hydra'`

### Result File Structure (When Experiments Succeed)
**Per Target (`outputs/comparisons/{target}_{timestamp}/`):**
- `comparison_results.json`: Full results with structure:
  ```json
  {
    "target_series": "KOGDP...D",
    "models": ["arima", "var", "dfm", "ddfm"],
    "horizons": [1, 7, 28],
    "results": {
      "dfm": {
        "status": "completed",
        "metrics": {
          "forecast_metrics": {
            "1": {"sMSE": 2.502, "sMAE": 1.5818, "sRMSE": 1.5818, "n_valid": 32},
            "7": {"sMSE": 0.0018, "sMAE": 0.0419, "sRMSE": 0.0419, "n_valid": 26},
            "28": {"sMSE": null, "sMAE": null, "sRMSE": null, "n_valid": 0}
          }
        }
      },
      ...
    },
    "comparison": {...},
    "timestamp": "2025-12-06T...",
    "failed_models": []
  }
  ```
- `comparison_table.csv`: Summary table with model × horizon metrics

**Aggregated (`outputs/experiments/aggregated_results.csv`):**
- CSV with columns: target, model, horizon, sMSE, sMAE, sRMSE, n_valid
- 36 rows (3 targets × 4 models × 3 horizons)

**Trained Models (`outputs/models/{target}_{model}/model.pkl`):**
- Pickled sktime forecasters (12 total: 3 targets × 4 models)

### Report Update Plan (Once Results Available)

**Step 1: Generate Plots**
- Run: `python3 nowcasting-report/code/plot.py`
- Expected outputs:
  - `images/accuracy_heatmap.png`: Model × Target heatmap (sRMSE)
  - `images/model_comparison.png`: Bar chart comparing models (sMSE, sMAE, sRMSE)
  - `images/horizon_trend.png`: Line plot showing performance by horizon
  - `images/forecast_vs_actual.png`: Time series comparison (currently placeholder)

**Step 2: Update LaTeX Tables**
- `tables/tab_overall_metrics.tex`: Overall averages across all targets/horizons
- `tables/tab_overall_metrics_by_target.tex`: Per-target averages (KOGDP...D, KOCNPER.D, KOGFCF..D)
- `tables/tab_overall_metrics_by_horizon.tex`: Per-horizon averages (1, 7, 28 days)
- `tables/tab_nowcasting_metrics.tex`: Nowcasting results (if nowcasting evaluation is run)

**Step 3: Update Report Content**
- `contents/5_result.tex`: Replace placeholder text with actual findings
  - Remove "향후 연구에서 다룰 예정" for KOCNPER.D and KOGFCF..D
  - Add actual metrics and analysis for all 3 targets
  - Reference specific numbers from tables (e.g., "DFM achieved sRMSE=0.0419 for 7-day horizon")
- `contents/6_discussion.tex`: Discuss actual findings with real numbers
- `main.tex`: Update abstract if needed to reflect all 3 targets

**Step 4: Verify Report Completeness**
- Compile PDF: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Check page count (target: 20-30 pages)
- Verify all figures/tables referenced in text exist
- Ensure no placeholder text remains

### run_experiment.sh Status
- ✅ Skip logic verified: Checks for `comparison_results.json`, automatically skips completed targets
- ✅ Script is correct, no updates needed unless partial failures occur
- After dependencies installed, will run all 3 targets (all currently incomplete)

## Next Steps (Priority Order)
1. **Install dependencies** (CRITICAL - unblocks everything)
2. **Run experiments** (3 targets, 4 models each)
3. **Generate plots** (from experiment results)
4. **Update tables** (from aggregated results)
5. **Update report content** (replace placeholders)
6. **Finalize report** (compile, verify, polish)
