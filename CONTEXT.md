# Project Context Summary

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (16 files) - wrappers for sktime & dfm-python (transformations.py deprecated but kept for compatibility)
- **dfm-python/**: Core DFM/DDFM package with Lightning-based training (submodule) - ✅ Finalized, consistent naming (snake_case functions, PascalCase classes)
- **nowcasting-report/**: LaTeX report (20-30 pages) with plots from outputs/
- **config/**: Hydra YAML configs (experiment/, model/, series/)
- **outputs/**: Experiment results (comparisons/, models/, experiments/)

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

## Current Status (2025-01-XX)

### Work Completed This Iteration
- ✅ **Report Content Review**: Reviewed all report sections - introduction, literature review, theoretical background, method, results, discussion, conclusion are well-structured
- ✅ **dfm-python Verification**: Verified naming consistency (snake_case functions, PascalCase classes) - consistent throughout
- ✅ **Code Structure Review**: Reviewed src/ structure - transformations.py is deprecated (re-exports from utils.py) but kept for backward compatibility
- ✅ **Status Files Update**: Updated STATUS.md, CONTEXT.md, ISSUES.md for next iteration with current state

### Experiment Status
- **0/3 targets executed** - No valid results exist, ready to run with fixed code
- **Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Code Status**: ✅ All critical bugs fixed
  - ARIMA: ✅ Fixed prediction matching using position-based approach
  - VAR: ✅ Fixed frequency error by setting freq on DatetimeIndex
  - DFM/DDFM: ✅ Fixed weekly series filter - excludes weekly series from monthly blocks
- **Action Required**: Run experiments with `bash run_experiment.sh` to generate valid results

### Code Quality Status
- **src/ Module**: 16 files (transformations.py deprecated but kept for compatibility), all imports fixed
- **dfm-python/ Package**: ✅ Finalized - consistent naming (snake_case functions, PascalCase classes), clean patterns, no TODOs
- **run_experiment.sh**: ✅ Verified - auto-skip logic, parallel execution

### Report Status
- **Structure**: ✅ Complete 20-30 page framework, improved flow
- **Content Quality**: ✅ All citations verified, terminology consistent, redundant warnings removed
- **Placeholders**: ⚠️ Tables contain placeholders (will be updated after experiments)

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

1. **Re-run Experiments** → `bash run_experiment.sh` (ready to run, all fixes complete)
2. **Verify Results** → Check n_valid > 0 for at least some model/horizon combinations
3. **Generate Aggregated CSV** → Create `outputs/experiments/aggregated_results.csv` from comparison results
4. **Generate Plots** → `python3 nowcasting-report/code/plot.py` (4 PNG files)
5. **Update Tables** → From aggregated_results.csv
6. **Update Report Content** → Replace placeholders in `contents/5_result.tex`, `contents/6_discussion.tex`
7. **Finalize Report** → Compile PDF, verify 20-30 pages, no placeholders

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
