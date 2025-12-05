# Project Context Summary

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (15 files, max 15 required) - wrappers for sktime & dfm-python
  - ✅ transformations.py removed (no active imports found)
  - Entry points: train.py (compare), infer.py (nowcast)
  - Core modules: core/training.py, eval/evaluation.py, model/, preprocess/, utils/
- **dfm-python/**: Core DFM/DDFM package (submodule) - ✅ Finalized
  - Lightning-based training, EM algorithm (DFM), PyTorch encoder (DDFM)
  - Consistent naming: snake_case functions, PascalCase classes
- **nowcasting-report/**: LaTeX report (20-30 pages target)
  - Contents: 8 sections (intro, lit review, theory, method, results, discussion, conclusion, acknowledgement)
  - Tables: 4 tables with placeholders (tab_overall_metrics, tab_by_target, tab_by_horizon, tab_nowcasting)
  - Images: 4 plots (model_comparison, horizon_trend, accuracy_heatmap, forecast_vs_actual)
  - Code: plot.py generates plots from outputs/
- **config/**: Hydra YAML configs
  - experiment/: 3 target configs (kogdp_report, kocnper_report, kogfcf_report)
  - model/: Model-specific parameters (arima, var, dfm, ddfm)
  - series/: 100+ series configs (frequency, transformation, blocks)
- **outputs/**: Experiment results
  - comparisons/: Per-target results (comparison_results.json, comparison_table.csv)
  - models/: Trained models (model.pkl per target/model)
  - experiments/: Aggregated results (aggregated_results.csv - MISSING)

### Key Components

**src/ Module (Experiment Engine):**
- Entry: `train.py` (compare command), `infer.py` (nowcast command)
- Core: `core/training.py` - Unified training via sktime forecasters
- Model: `model/{dfm,ddfm,sktime_forecaster,_common}.py` - Model wrappers
- Preprocess: `preprocess/{sktime,utils,transformations}.py` - Data preprocessing
- Eval: `eval/evaluation.py` - Standardized metrics (sMSE, sMAE, sRMSE), aggregation
- Utils: `utils/config_parser.py` - Hydra config parsing

**dfm-python/ Package:**
- Models: DFM (EM algorithm), DDFM (PyTorch encoder + VAE)
- Lightning: DataModule, KalmanFilter, EMAlgorithm (PyTorch Lightning integration)
- Features: Clock-based mixed-frequency, block-structured factors, Hydra YAML config

**nowcasting-report/ Structure:**
- LaTeX report: `main.tex`, `preamble.tex`, `contents/*.tex`
- Tables: `tables/tab_*.tex` (4 tables with placeholders)
- Images: `images/*.png` (4 plots, currently placeholders)
- Code: `code/plot.py` - Plot generation from outputs/

## Experiment Pipeline

**Training Flow:**
```
run_experiment.sh
  → src/train.py compare --config-name experiment/{target}_report
    → parse_experiment_config() → compare_models()
      → For each model: train() → _train_forecaster()
        → Load data → Preprocess → Create forecaster → fit() → predict()
        → evaluate_forecaster() → Save to outputs/models/
      → _compare_results() → Save to outputs/comparisons/{target}_{timestamp}/
```

**Result Structure:**
- Per-target: `outputs/comparisons/{target}_{timestamp}/comparison_results.json`
- Aggregated: `outputs/experiments/aggregated_results.csv` (36 rows: 3 targets × 4 models × 3 horizons)
- Models: `outputs/models/{target}_{model}/model.pkl` (12 total)

**Configuration:**
- Experiment: `config/experiment/{target}_report.yaml` - Target, models, horizons, series
- Model: `config/model/{model}.yaml` - Model-specific parameters
- Series: `config/series/{series_id}.yaml` - Frequency, transformation, blocks

## Current Status (2025-01-XX - Iteration Summary)

### Experiment Results Status
- **No valid results exist** - Only log files in outputs/comparisons/ (no comparison_results.json)
- **Ready for execution** - All code fixes verified, script ready to run
- **Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations
- **No Aggregated Results**: `outputs/experiments/aggregated_results.csv` does NOT exist (will be generated after experiments)

### Code Status (All Fixes Verified)
- **ARIMA**: ✅ Position-based matching implemented (evaluation.py:336-343)
- **VAR**: ✅ Forward-fill imputation implemented (training.py:253-259), frequency setting verified (training.py:264-274)
- **DFM/DDFM**: ✅ Frequency hierarchy check implemented (training.py:689-720)
- **All critical bugs fixed** - Ready for experiment execution

### Report Status
- **Structure**: ✅ Complete framework (1456 lines total across 8 sections)
- **Tables**: ⚠️ All 4 tables contain "---" placeholders (blocked until experiments complete)
- **Plots**: ⚠️ 4 placeholder images exist (will generate placeholders if no valid data)
- **Content**: ✅ Sections 1-4, 6-7 complete; ⚠️ Section 5 (results) has placeholders
- **Citations**: ✅ All verified in references.bib (20+ references)

### Code Quality Status
- **src/ Module**: ✅ 15 files (max 15 required) - transformations.py removed, all imports fixed
- **dfm-python/ Package**: ✅ Finalized - consistent naming (snake_case functions, PascalCase classes), clean patterns, no TODOs
- **run_experiment.sh**: ✅ Verified - ready to run all 3 targets

## Data Flow

1. **Config → Model Setup**: Hydra loads experiment config → Extract series → Build dfm-python config
2. **Data → Preprocessing**: Load CSV → Apply per-series transformations → Standardize
3. **Training**: Create forecaster → fit() → EM (DFM) or PyTorch Lightning (DDFM)
4. **Evaluation**: Train/test split (80/20) → predict() → calculate_standardized_metrics()
5. **Comparison**: Aggregate across models → generate_comparison_table() → Save JSON/CSV
6. **Visualization**: Load JSON → Extract metrics → Generate plots → Save PNG

## Key Design Patterns

- **Unified Interface**: All models use sktime forecaster interface (fit/predict)
- **Config-Driven**: Hydra YAML configs for all experiments
- **Modular Preprocessing**: Per-series transformations via sktime FunctionTransformer
- **Standardized Metrics**: sMSE, sMAE, sRMSE (normalized by training std)
- **Output Structure**: outputs/{models,comparisons,experiments}/ with timestamps

## Project Structure Details

### src/ Module (15 files - max 15 required)
```
src/
├── __init__.py
├── train.py                    # CLI entry: compare command
├── infer.py                    # CLI entry: nowcast command
├── core/
│   ├── __init__.py
│   └── training.py            # Unified training via sktime forecasters
├── eval/
│   ├── __init__.py
│   └── evaluation.py          # Standardized metrics, aggregation
├── model/
│   ├── __init__.py
│   ├── dfm.py                 # DFM model wrapper
│   ├── ddfm.py                # DDFM model wrapper
│   └── sktime_forecaster.py   # sktime forecaster adapters
├── preprocess/
│   ├── __init__.py
│   └── utils.py               # Preprocessing utilities
└── utils/
    ├── __init__.py
    └── config_parser.py        # Hydra config parsing
```

### Experiment Configuration
- **3 Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **4 Models**: arima, var, dfm, ddfm
- **3 Horizons**: 1, 7, 28 days
- **Total Combinations**: 3 × 4 × 3 = 36

### Data Flow
1. **Config Loading**: Hydra loads experiment config → extracts series list → builds dfm-python config
2. **Data Loading**: Load CSV (data/sample_data.csv) → 101 series, 2,538 observations
3. **Preprocessing**: Per-series transformations (log, diff, standardization) → sktime FunctionTransformer
4. **Training**: Create forecaster → fit() → EM (DFM) or PyTorch Lightning (DDFM)
5. **Evaluation**: Train/test split (80/20) → predict() → calculate_standardized_metrics() → n_valid check
6. **Comparison**: Aggregate across models → generate_comparison_table() → Save JSON/CSV
7. **Visualization**: Load JSON → Extract metrics → Generate plots → Save PNG
8. **Report**: Update LaTeX tables from aggregated CSV → Compile PDF

## Critical Issues Status

### All Critical Issues Resolved
1. **ARIMA n_valid=0**: ✅ FIXED - Position-based matching implemented
2. **VAR Missing Data**: ✅ FIXED - Forward-fill imputation implemented
3. **DFM/DDFM Shape Mismatch**: ✅ FIXED - Frequency hierarchy check implemented

### Missing Components (Blocked by Experiments)
- **Aggregated Results CSV**: Does not exist (will be generated after experiments)
- **Valid Metrics**: No results available yet (experiments not run)
- **Report Tables**: All contain placeholders (will be updated after experiments)
- **Report Plots**: Will be placeholders if generated now (will be updated after experiments)

## Next Steps (Priority Order - Incremental Approach)

### PHASE 1: Fix Model Issues [COMPLETED]
1. ✅ **Fix ARIMA n_valid=0** - Position-based matching implemented
2. ✅ **Fix VAR Missing Data** - Forward-fill imputation implemented
3. ✅ **Fix DFM/DDFM Shape Mismatch** - Frequency hierarchy check implemented

### PHASE 2: Execute Experiments [READY]
4. **Re-run Experiments** → `bash run_experiment.sh` (all fixes applied)
5. **Verify Results** → Check n_valid > 0 for at least 2 models per target

### PHASE 3: Generate Results [BLOCKED by Phase 2]
6. **Generate Aggregated CSV** → `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
7. **Generate Plots** → `python3 nowcasting-report/code/plot.py` (4 PNG files)
8. **Update Tables** → From aggregated_results.csv (replace "---" placeholders)

### PHASE 4: Update Report [BLOCKED by Phase 3]
9. **Update Results Section** → `contents/5_result.tex` with actual numbers from tables
10. **Update Discussion** → `contents/6_discussion.tex` with real findings
11. **Finalize Report** → Compile PDF, verify 20-30 pages, no placeholders

## Report Update Plan (After Experiments Complete)

**Step 1: Generate Plots**
- Run: `python3 nowcasting-report/code/plot.py`
- Outputs: 4 PNG files (accuracy_heatmap, model_comparison, horizon_trend, forecast_vs_actual)

**Step 2: Update LaTeX Tables**
- `tables/tab_overall_metrics.tex`: Overall averages
- `tables/tab_overall_metrics_by_target.tex`: Per-target averages
- `tables/tab_overall_metrics_by_horizon.tex`: Per-horizon averages
- `tables/tab_nowcasting_metrics.tex`: Nowcasting results (if evaluated)

**Step 3: Update Report Content**
- `contents/5_result.tex`: Replace placeholders with actual findings
- `contents/6_discussion.tex`: Discuss actual findings with real numbers
- Reference specific metrics from tables

**Step 4: Finalize Report**
- Compile PDF: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Verify: Page count (20-30), all figures/tables exist, no placeholders remain
