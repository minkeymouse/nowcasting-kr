# Project Context Summary

## Project Overview

**Goal**: Complete under 15 page LaTeX report comparing 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 4 Korean macroeconomic targets (Production: KOIPALL.G, KOMPRI30G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE) across 3 forecast horizons (1, 7, 28 days). Finalize dfm-python package.

**Experiment Configuration**:
- **4 Targets**: KOEQUIPTE (Equipment Investment Index), KOWRCCNSE (Wholesale and Retail Trade Sales), KOIPALL.G (Industrial Production Index, All Industries), KOMPRI30G (Manufacturing Production Index)
- **4 Models**: ARIMA (sktime), VAR (sktime), DFM (EM algorithm), DDFM (PyTorch Lightning)
- **3 Horizons**: 1, 7, 28 days
- **Total**: 48 combinations (4 × 4 × 3)

**Current Status (2025-12-06 - New Experiment Phase)**: 
- ✅ **Configuration**: All 4 target configs created, series configs updated (block: null), data path fixed
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized and verified
- ✅ **Report Structure**: 6 sections ready (condensed to under 15 pages)
- ⏳ **Experiments**: Need to run for 4 new targets (0/48 complete)
- ⏳ **Report Content**: Need to populate with actual experiment results

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (15 files, max 15 required) - wrappers for sktime & dfm-python
  - Entry points: train.py (compare), infer.py (nowcast)
  - Core modules: core/training.py, eval/evaluation.py, model/, preprocess/, utils/
- **dfm-python/**: Core DFM/DDFM package (submodule) - Finalized
  - Lightning-based training, EM algorithm (DFM), PyTorch encoder (DDFM)
  - Consistent naming: snake_case functions, PascalCase classes
- **nowcasting-report/**: LaTeX report (under 15 pages target)
  - Contents: 6 sections (Introduction, Methodology, Production Model, Investment Model, Consumption Model, Conclusion)
  - Tables: Structure ready (tab_overall_metrics, tab_by_target, tab_by_horizon)
  - Images: Need to generate with new target data
  - Code: plot.py generates plots from outputs/
- **config/**: Hydra YAML configs
  - experiment/: 4 target configs (koequipte_report, kowrccnse_report, koipallg_report, kompri30g_report)
  - model/: Model-specific parameters (arima, var, dfm, ddfm)
  - series/: 101 series configs (frequency, transformation, block: null - only global block)
- **outputs/**: Experiment results
  - comparisons/: Per-target results (comparison_results.json, comparison_table.csv)
  - models/: Trained models (model.pkl per target/model)
  - experiments/: Aggregated results (aggregated_results.csv)

### Key Components

**src/ Module (Experiment Engine)**:
- Entry: `train.py` (compare command), `infer.py` (nowcast command)
- Core: `core/training.py` - Unified training via sktime forecasters
- Model: `model/{dfm,ddfm,sktime_forecaster,_common}.py` - Model wrappers
- Preprocess: `preprocess/{sktime,utils,transformations}.py` - Data preprocessing
- Eval: `eval/evaluation.py` - Standardized metrics (sMSE, sMAE, sRMSE), aggregation
- Utils: `utils/config_parser.py` - Hydra config parsing

**dfm-python/ Package**:
- Models: DFM (EM algorithm), DDFM (PyTorch encoder + VAE)
- Lightning: DataModule, KalmanFilter, EMAlgorithm (PyTorch Lightning integration)
- Features: Clock-based mixed-frequency, block-structured factors (only global block now), Hydra YAML config

## Experiment Pipeline

**Training Flow**:
```
run_experiment.sh
  → src/train.py compare --config-name experiment/{target}_report [--models var dfm]
    → parse_experiment_config() → compare_models()
      → For each model: train() → _train_forecaster()
        → Load data → Preprocess → Create forecaster → fit() → predict()
        → evaluate_forecaster() → Save to outputs/models/
      → _compare_results() → Save to outputs/comparisons/{target}_{timestamp}/
```

**Result Structure**:
- Per-target: `outputs/comparisons/{target}_{timestamp}/comparison_results.json`
- Aggregated: `outputs/experiments/aggregated_results.csv` (48 rows: 4 targets × 4 models × 3 horizons)
- Models: `outputs/models/{target}_{model}/model.pkl` (16 total)

**Configuration**:
- Experiment: `config/experiment/{target}_report.yaml` - Target, models, horizons, series
- Model: `config/model/{model}.yaml` - Model-specific parameters
- Series: `config/series/{series_id}.yaml` - Frequency, transformation, block: null (only global block)

## Data Flow

1. **Config → Model Setup**: Hydra loads experiment config → Extract series → Build dfm-python config (only global block)
2. **Data → Preprocessing**: Load CSV (`data/data.csv`) → Apply per-series transformations → Standardize
3. **Training**: Create forecaster → fit() → EM (DFM) or PyTorch Lightning (DDFM)
4. **Evaluation**: Train/test split (80/20) → predict() → calculate_standardized_metrics()
5. **Comparison**: Aggregate across models → generate_comparison_table() → Save JSON/CSV
6. **Visualization**: Load JSON → Extract metrics → Generate plots → Save PNG
7. **Report**: Update tables → Compile PDF (under 15 pages)

## Key Design Patterns

- **Unified Interface**: All models use sktime forecaster interface (fit/predict)
- **Config-Driven**: Hydra YAML configs for all experiments
- **Modular Preprocessing**: Per-series transformations via sktime FunctionTransformer
- **Standardized Metrics**: sMSE, sMAE, sRMSE (normalized by training std)
- **Output Structure**: outputs/{models,comparisons,experiments}/ with timestamps
- **Block Structure**: Only global block used (all series have `block: null`)

## Usage

**Run all experiments**:
```bash
bash run_experiment.sh
```

**Run test verification**:
```bash
bash run_test_experiment.sh
```

**Run specific models** (for incremental testing):
```bash
MODELS="var dfm" bash run_experiment.sh
```

**Run single model/target**:
```bash
.venv/bin/python3 src/train.py compare --config-name experiment/koequipte_report --models var
```

**Generate aggregated results**:
```bash
python3 -c "from src.eval import main_aggregator; main_aggregator()"
```

**Generate plots**:
```bash
python3 nowcasting-report/code/plot.py
```

## Report Update Workflow

1. Run test verification → `./run_test_experiment.sh` (verify all targets/models)
2. Run experiments → `./run_experiment.sh` (all 4 targets × 4 models × 3 horizons)
3. Generate aggregated CSV → `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
4. Generate plots → `python3 nowcasting-report/code/plot.py`
5. Update LaTeX tables → From aggregated_results.csv (populate with actual metrics)
6. Update result sections → `contents/3_production_model.tex`, `contents/4_investment_model.tex`, `contents/5_consumption_model.tex` with specific numbers
7. Update conclusion → `contents/6_conclusion.tex` to reflect actual results
8. Finalize report → Compile PDF, verify under 15 pages, no placeholders

## Latest Updates (2025-12-06)

**Completed**:
- ✅ Configuration updated: 4 new target configs created, series configs updated (block: null)
- ✅ Data path fixed: All configs use `data/data.csv`
- ✅ Report structure: 6 sections ready (condensed to under 15 pages)
- ✅ Scripts finalized: `run_experiment.sh` and `run_test_experiment.sh` ready
- ✅ Test verification: ARIMA and VAR tests passing

**For Next Iteration**: 
- ⏳ **NEXT**: Run test verification (`./run_test_experiment.sh`) to verify all targets and models
- ⏳ **After verification**: Run full experiments (`./run_experiment.sh`) for all 48 combinations
- ⏳ **After experiments**: Update report with actual results, generate plots, compile PDF

**Status**: Configuration ready, experiments pending. Report structure ready, need results to populate content.
