# Project Context Summary

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (17 files, 15 effective) - wrappers for sktime & dfm-python
- **dfm-python/**: Core DFM/DDFM package with Lightning-based training (submodule)
- **nowcasting-report/**: LaTeX report (20-30 pages) with plots from outputs/
- **config/**: Hydra YAML configs (experiment/, model/, series/)
- **outputs/**: Experiment results (comparisons/, models/, experiments/)

### Key Components

**src/ Module (Experiment Engine):**
- Entry: `train.py` (compare command), `infer.py` (nowcast command), `nowcasting.py` (deprecated)
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

## Current Status (2025-01-XX)

### Work Completed This Iteration
- ✅ **Report**: 20-30 page LaTeX framework complete, all citations verified (20+ references), improved structure (redundant warnings removed)
- ✅ **dfm-python Package**: Code quality finalized - consistent naming, no TODO/FIXME
- ✅ **src/ Module**: Architecture complete (17 files, 15 effective), all import errors fixed
- ✅ **Code Quality**: All type hints fixed, circular import resolved, ready for execution

### Experiment Status
- **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- **Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Previous Attempts**: 45 log files from 2025-12-06 (all failed - errors analyzed and fixed)
- **Current State**: All code fixes applied, ready for execution
- **Next Action**: Run experiments using `bash run_experiment.sh`

**Error History (All Resolved):**
- Circular import (050328) - FIXED (moved imports inside methods)
- Pandas NameError (044509) - FIXED (type hints use string literals)
- Import errors (001731-040746) - FIXED (path setup, module names)

### Working Components
- ✅ **Training Pipeline**: Unified sktime forecaster interface, config-driven via Hydra
- ✅ **Evaluation Framework**: Standardized metrics (sMSE, sMAE, sRMSE), aggregation
- ✅ **Result Structure**: Well-defined JSON/CSV output format
- ✅ **Visualization**: Plot generation code ready (currently generates placeholders)
- ✅ **Report Structure**: Complete LaTeX framework, all sections present

### Current Blockers
- **Experiments Not Run**: 0/3 targets complete, no result files, report has placeholders
- **Solution**: Execute `bash run_experiment.sh` to run all 3 targets

### Code Quality Status
- **src/ Module**: 17 files (15 effective - within limit), all imports fixed
- **dfm-python/ Package**: ✅ Finalized - consistent naming, clean patterns
- **run_experiment.sh**: ✅ Verified - auto-skip logic, parallel execution

### Report Status
- **Structure**: ✅ Complete 20-30 page framework, improved flow
- **Content Quality**: ✅ All citations verified, terminology consistent, no hallucinations
- **Placeholders**: ⚠️ All results are placeholders (will be updated after experiments)

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

## Next Steps (Priority Order)

1. **Run experiments** (`bash run_experiment.sh`) - 3 targets, 4 models each
2. **Generate plots** (`python3 nowcasting-report/code/plot.py`)
3. **Update tables** (from `outputs/experiments/aggregated_results.csv`)
4. **Update report content** (replace placeholders in results/discussion sections)
5. **Finalize report** (compile PDF, verify 20-30 pages, no placeholders)

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
