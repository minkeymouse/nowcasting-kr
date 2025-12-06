# Project Context Summary

## Project Overview

**Goal**: Complete 20-30 page LaTeX report comparing 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 3 Korean macroeconomic targets (GDP, Consumption, Investment) across 3 forecast horizons (1, 7, 28 days). Finalize dfm-python package.

**Experiment Configuration**:
- **3 Targets**: KOGDP...D (GDP, 55 series), KOCNPER.D (Consumption, 50 series), KOGFCF..D (Investment, 19 series)
- **4 Models**: ARIMA (sktime), VAR (sktime), DFM (EM algorithm), DDFM (PyTorch Lightning)
- **3 Horizons**: 1, 7, 28 days
- **Total**: 36 combinations (3 × 4 × 3)

**Current Status (2025-12-06 - End of Iteration 39)**: 
- ✅ ARIMA: Complete (9/9 combinations, sRMSE=0.3662)
- ✅ VAR: Complete (9/9 combinations, sRMSE=0.0465) - best overall performance
- ⚠️ DFM: Partial (4/9 combinations) - KOGDP...D h1,h7; KOGFCF..D h1,h7 available; KOCNPER.D all horizons failed (numerical instability); all h28 unavailable (test set too small)
- ⚠️ DDFM: Partial (6/9 combinations) - all targets h1,h7 available; all h28 unavailable (test set too small)
- ✅ Report: Complete with all available results (28/36 combinations, 77.8%) integrated, all metric values verified and corrected, all LaTeX cross-references verified, all images confirmed
- ✅ Package: dfm-python finalized with consistent naming, clean code patterns
- ✅ Results Analysis: All comparison results analyzed and verified - all metric values match aggregated_results.csv exactly, all breakdowns confirmed, no errors or discrepancies found (0 discrepancies)
- ✅ Quality Improvements: All Phase 1-5 tasks complete, all Priority 1-5 incremental improvements complete, all verification tasks complete
- ✅ Experiment Completion Verification: Complete - all 3 targets verified with comparison_results.json files, run_experiment.sh verified and correctly configured
- ✅ Comparison Results Re-Analysis: All results in outputs/comparisons/ analyzed, all metric values verified to match aggregated_results.csv exactly, all expected limitations confirmed
- ✅ Report Finalization (Iteration 39): All LaTeX cross-references verified, all images confirmed, all tracking files under 1000 lines
- ⏳ Next: PDF compilation (external dependency - requires LaTeX installation) [BLOCKER]

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (15 files, max 15 required) - wrappers for sktime & dfm-python
  - Entry points: train.py (compare), infer.py (nowcast)
  - Core modules: core/training.py, eval/evaluation.py, model/, preprocess/, utils/
- **dfm-python/**: Core DFM/DDFM package (submodule) - Finalized
  - Lightning-based training, EM algorithm (DFM), PyTorch encoder (DDFM)
  - Consistent naming: snake_case functions, PascalCase classes
- **nowcasting-report/**: LaTeX report (20-30 pages target)
  - Contents: 8 sections (intro, lit review, theory, method, results, discussion, conclusion, acknowledgement)
  - Tables: 4 tables (tab_overall_metrics, tab_by_target, tab_by_horizon, tab_nowcasting)
  - Images: 4 plots (model_comparison, horizon_trend, accuracy_heatmap, forecast_vs_actual)
  - Code: plot.py generates plots from outputs/
- **config/**: Hydra YAML configs
  - experiment/: 3 target configs (kogdp_report, kocnper_report, kogfcf_report)
  - model/: Model-specific parameters (arima, var, dfm, ddfm)
  - series/: 100+ series configs (frequency, transformation, blocks)
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
- Features: Clock-based mixed-frequency, block-structured factors, Hydra YAML config

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
- Aggregated: `outputs/experiments/aggregated_results.csv` (36 rows: 3 targets × 4 models × 3 horizons)
- Models: `outputs/models/{target}_{model}/model.pkl` (12 total)

**Configuration**:
- Experiment: `config/experiment/{target}_report.yaml` - Target, models, horizons, series
- Model: `config/model/{model}.yaml` - Model-specific parameters
- Series: `config/series/{series_id}.yaml` - Frequency, transformation, blocks

## Data Flow

1. **Config → Model Setup**: Hydra loads experiment config → Extract series → Build dfm-python config
2. **Data → Preprocessing**: Load CSV → Apply per-series transformations → Standardize
3. **Training**: Create forecaster → fit() → EM (DFM) or PyTorch Lightning (DDFM)
4. **Evaluation**: Train/test split (80/20) → predict() → calculate_standardized_metrics()
5. **Comparison**: Aggregate across models → generate_comparison_table() → Save JSON/CSV
6. **Visualization**: Load JSON → Extract metrics → Generate plots → Save PNG
7. **Report**: Update tables → Compile PDF

## Key Design Patterns

- **Unified Interface**: All models use sktime forecaster interface (fit/predict)
- **Config-Driven**: Hydra YAML configs for all experiments
- **Modular Preprocessing**: Per-series transformations via sktime FunctionTransformer
- **Standardized Metrics**: sMSE, sMAE, sRMSE (normalized by training std)
- **Output Structure**: outputs/{models,comparisons,experiments}/ with timestamps

## Usage

**Run all experiments**:
```bash
bash run_experiment.sh
```

**Run specific models** (for incremental testing):
```bash
MODELS="var dfm" bash run_experiment.sh
```

**Run single model/target**:
```bash
.venv/bin/python3 src/train.py compare --config-name experiment/kogdp_report --models var --horizons 1
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

1. Run experiments → `bash run_experiment.sh` (or with MODELS filter)
2. Generate aggregated CSV → `python3 -c "from src.eval import main_aggregator; main_aggregator()"`
3. Generate plots → `python3 nowcasting-report/code/plot.py`
4. Update LaTeX tables → From aggregated_results.csv (replace "---" placeholders)
5. Update results section → `contents/5_result.tex` with specific numbers
6. Update discussion → `contents/6_discussion.tex` with real findings and insights
7. Update conclusion → `contents/7_conclusion.tex` to reflect actual results
8. Finalize report → Compile PDF, verify 20-30 pages, no placeholders

## Latest Updates (Iteration 39 - 2025-12-06)

**Completed**:
- ✅ Report Finalization: Verified all LaTeX cross-references (\ref{}, \cite{}) are consistent and resolve correctly
- ✅ Image Verification: Confirmed all 4 images exist in images/ directory (model_comparison.png, horizon_trend.png, accuracy_heatmap.png, forecast_vs_actual.png)
- ✅ File Size Verification: Confirmed all tracking files under 1000 lines (STATUS.md: 138, ISSUES.md: 296, CONTEXT.md: 166)
- ✅ Report Ready: All content complete, all sections verified, all metric values match aggregated_results.csv exactly, ready for PDF compilation

**For Next Iteration (Iteration 40)**: All development, verification, and results analysis work complete. All experiments verified (28/36 = 77.8% complete). All comparison results analyzed and verified (0 discrepancies). All report content verified, all LaTeX cross-references verified, all images confirmed. All code finalized. run_experiment.sh verified and correctly configured. All tracking files verified under 1000 lines. Report is ready for PDF compilation (Tasks 2.1-2.4) which requires LaTeX installation (external dependency). Once LaTeX is installed, proceed with: Task 2.1 (LaTeX Environment Setup) → Task 2.2 (Initial PDF Compilation) → Task 2.3 (PDF Quality Verification) → Task 2.4 (PDF Finalization).

## Previous Updates Summary (Iterations 11-35)

**Completed**:
- ✅ All experiments verified (28/36 = 77.8%) with comparison_results.json files for all 3 targets
- ✅ All report content verified (all metric values match aggregated_results.csv exactly, all sections complete, all citations verified)
- ✅ All code finalized (src/ 15 files, dfm-python finalized, all tests passing)
- ✅ All incremental improvements complete (Priority 1-5: report refinement, code quality, documentation, numerical stability, final verification)
- ✅ All comparison results analyzed and verified (all metric values match, breakdowns confirmed, no discrepancies)
- ✅ Metric value corrections completed (DDFM sRMSE discrepancy fixed)
- ✅ Experiment completion verification completed (all 3 targets verified)
- ✅ Comparison results analysis completed (all results verified, all metric values match)

**Status**: All report content is complete and verified. All metric values match aggregated_results.csv exactly. All citations verified. LaTeX syntax verified. Code finalized. All Phase 1-5 quality improvements complete. Report finalization complete (all cross-references verified, all images confirmed). Ready for PDF compilation (external dependency - requires LaTeX installation).
