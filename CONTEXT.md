# Project Context Summary

## Project Overview

**Goal**: Complete under 15 page LaTeX report comparing 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 3 Korean macroeconomic targets (Production: KOIPALL.G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE) across forecasting horizons (1-30 days) and nowcasting evaluation with multiple time points. Finalize dfm-python package.

**Experiment Configuration**:
- **3 Targets**: KOEQUIPTE (Equipment Investment Index), KOWRCCNSE (Wholesale and Retail Trade Sales), KOIPALL.G (Industrial Production Index, All Industries)
- **4 Models**: ARIMA (sktime), VAR (sktime), DFM (EM algorithm), DDFM (PyTorch Lightning)
- **Forecasting Horizons**: 1-30 days (table shows 1, 7, 30)
- **Nowcasting Configuration**: 
  - All models (ARIMA, VAR, DFM, DDFM) and all targets (3)
  - 12 target months (2024-01 ~ 2024-12)
  - 2 time points per month (4 weeks before, 1 week before month end)
  - Release date based masking
  - 1 horizon forecast at each time point
  - Total: 3 targets × 4 models × 12 months × 2 timepoints = 288 nowcasting predictions

**ACTUAL Current Status**:
- **checkpoint/**: **10 model.pkl files** - **10/12 models trained** ⚠️ (Missing: KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/backtest/**: **12 JSON files with "status": "no_results"** - **0/12 nowcasting experiments completed** ⚠️
  - Code fixes present in codebase (validation check numbering, Kalman filter NaN handling, backtest validation improvements)
  - Code fixes NOT verified by re-running backtests
- **outputs/experiments/aggregated_results.csv**: **EXISTS** - Forecasting results aggregated (36 rows, extreme VAR values filtered on load)
- **outputs/comparisons/**: Contains comparison_results.json files - forecasting completed successfully
- **nowcasting-report/tables/**: **3 tables generated** (tab_dataset_params.tex, tab_forecasting_results.tex, tab_nowcasting_backtest.tex)
- **nowcasting-report/images/**: **7 plots generated** (forecast_vs_actual_*.png × 3, accuracy_heatmap.png, horizon_trend.png, nowcasting_comparison_*.png × 3)

**What This Means**:
- ⚠️ **Training INCOMPLETE** - Only 10/12 models trained (missing KOIPALL.G_ddfm, KOIPALL.G_dfm)
- ⚠️ **Nowcasting experiments FAILED** - All 12 JSON files have "no_results" status. Code fixes present but NOT verified.
- ✅ **Forecasting results exist** - Table 2 can be generated (extreme values filtered when loading)
- ⚠️ **Tables and plots generated** - Table 3 and Plot4 have placeholders (nowcasting results missing)

**Next Steps** (Step 1 will automatically handle):
1. Check checkpoint/ - if empty, run `bash agent_execute.sh train`
2. Check outputs/experiments/aggregated_results.csv - if missing, run `bash agent_execute.sh forecast`
3. Check outputs/backtest/ - if missing, run `bash agent_execute.sh backtest`

---

## Architecture Overview

### Directory Structure
- **src/**: Experiment engine (15 files, max 15 required) - wrappers for sktime & dfm-python
  - Entry points: train.py (train, compare), infer.py (backtest)
  - Core modules: core/training.py, eval/evaluation.py, model/, preprocess/, utils/, nowcast/
  - Status: Code should work, but models NOT trained
- **dfm-python/**: Core DFM/DDFM package (submodule)
  - Lightning-based training, EM algorithm (DFM), PyTorch encoder (DDFM)
  - Status: Package structure ready
- **nowcasting-report/**: LaTeX report (under 15 pages target)
  - Contents: 4 sections (Introduction, Methodology, Results, Discussion)
  - Results section: Forecasting, Nowcasting, Performance subsections
  - Tables: Table 1 (dataset/params), Table 2 (forecasting), Table 3 (nowcasting - **NOT generated yet**)
  - Images: Plot1 (forecast vs actual), Plot2 (heatmap), Plot3 (horizon trend), Plot4 (nowcasting comparison - **NOT generated yet**)
  - Code: plot.py generates plots from outputs/
- **config/**: Hydra YAML configs
  - experiment/: 3 target configs (koequipte_report, kowrccnse_report, koipallg_report)
  - model/: Model-specific parameters (arima, var, dfm, ddfm)
  - series/: Series configs (frequency, transformation, block: null - only global block)
- **checkpoint/**: **10 model.pkl files** - Trained models saved here (12 models needed, missing KOIPALL.G_ddfm, KOIPALL.G_dfm)
- **outputs/**: Experiment results
  - comparisons/: Per-target forecasting results (exists - used to generate Plot1-3)
  - experiments/: Aggregated forecasting results (aggregated_results.csv - **EXISTS**, 36 rows)
  - backtest/: **12 JSON files with "status": "no_results"** - Nowcasting backtest results (all failed, code fixes present but NOT verified)

---

## Experiment Pipeline

**Training Flow** (NOT RUN YET):
```
run_train.sh (or agent_execute.sh train)
  → src/train.py train --config-name experiment/{target}_report --model {model}
    → _train_forecaster() → Load data (1985-2019) → Preprocess → Create forecaster → fit()
    → Save to checkpoint/{target}_{model}/model.pkl
```
**Status**: checkpoint/ has 10/12 model.pkl files - training INCOMPLETE (missing KOIPALL.G_ddfm, KOIPALL.G_dfm)

**Forecasting Flow** (DONE):
```
run_forecast.sh (or agent_execute.sh forecast)
  → src/train.py compare --config-name experiment/{target}_report
    → Load models from checkpoint/ → Generate forecasts for all horizons (1-30)
    → evaluate_forecaster() → Save to outputs/comparisons/
    → Aggregate results → Save to outputs/experiments/aggregated_results.csv
```
**Status**: aggregated_results.csv EXISTS with 36 rows (contains extreme VAR values, but filtering handles them when loading)

**Nowcasting Flow** (NOT RUN YET):
```
run_backtest.sh (or agent_execute.sh backtest)
  → src/infer.py backtest --config-name experiment/{target}_report --model {model} --weeks-before 4 1
    → Load model from checkpoint/ → For each target month (2024-01 ~ 2024-12):
      → For each time point (4 weeks, 1 week before):
        → Calculate view_date = target_month_end - weeks
        → Mask data based on release dates
        → Generate 1 horizon forecast
        → Calculate sMSE, sMAE
    → Save to outputs/backtest/{target}_{model}_backtest.json
```
**Status**: outputs/backtest/ has 12 JSON files with "status": "no_results" - nowcasting FAILED (code fixes present but NOT verified)

---

## Result Structure

**Forecasting**:
- Per-target: `outputs/comparisons/{target}_{timestamp}/comparison_results.json` (exists - used to generate Plot1-3)
- Aggregated: `outputs/experiments/aggregated_results.csv` (**EXISTS**, 36 rows)
- Models: `checkpoint/{target}_{model}/model.pkl` (**10 files - checkpoint/ has 10/12 model.pkl files**)

**Nowcasting**:
- Per-target-model: `outputs/backtest/{target}_{model}_backtest.json` (**12 files - all with "status": "no_results"**)
- Structure: Should contain `results_by_timepoint` with "4weeks" and "1weeks" keys, but currently all show "no_results"
- Each timepoint should contain: monthly_results, overall_sMAE, overall_sMSE, n_months

---

## Usage

**Training (save models to checkpoint/)** - **INCOMPLETE**:
```bash
bash agent_execute.sh train
# Or: bash run_train.sh
# Trains all models for all targets, saves to checkpoint/{target}_{model}/model.pkl
# Training data: 1985-01-01 to 2019-12-31 (no data leakage)
# Status: checkpoint/ has 10/12 model.pkl files - missing KOIPALL.G_ddfm, KOIPALL.G_dfm
```

**Forecasting (load from checkpoint/, generate forecasts)** - **DONE**:
```bash
bash agent_execute.sh forecast
# Or: bash run_forecast.sh
# Loads models from checkpoint/, generates forecasts for all horizons (1-30)
# Aggregates results to outputs/experiments/aggregated_results.csv
# Status: aggregated_results.csv EXISTS with 36 rows
```

**Nowcasting Backtest (multiple time points)** - **FAILED**:
```bash
bash agent_execute.sh backtest
# Or: bash run_backtest.sh
# Runs backtest for all models and targets
# For each target month (2024-01 ~ 2024-12), predicts at 4 weeks and 1 week before
# Saves to outputs/backtest/{target}_{model}_backtest.json
# Status: outputs/backtest/ has 12 JSON files with "status": "no_results" - code fixes present but NOT verified
```

**Generate tables**:
```bash
python3 -c "from src.eval.evaluation import generate_all_latex_tables; generate_all_latex_tables()"
# Status: Tables generated - Table 3 shows N/A (nowcasting results missing)
```

**Generate plots**:
```bash
python3 nowcasting-report/code/plot.py
# Status: Plots generated - Plot4 shows placeholders (nowcasting results missing)
```

---

## Report Update Workflow

**Forecasting Workflow**:
1. Train models → `bash agent_execute.sh train` (saves to checkpoint/) - **INCOMPLETE** (10/12 models)
2. Run forecasts → `bash agent_execute.sh forecast` (loads from checkpoint/, generates forecasts for 1-30 horizons) - **DONE**
3. Generate aggregated CSV → `outputs/experiments/aggregated_results.csv` - **EXISTS** (36 rows)
4. Generate plots → `python3 nowcasting-report/code/plot.py` (Plot1, Plot2, Plot3) - **DONE**
5. Update LaTeX tables → Table 1 (dataset/params), Table 2 (forecasting results) - **DONE**

**Nowcasting Workflow**:
1. Run backtests → `bash agent_execute.sh backtest` (all models, all targets, 4 weeks and 1 week before) - **FAILED** (all 12 JSON files have "no_results")
2. Generate table 3 → From outputs/backtest/*_backtest.json (8 rows × 7 columns) - **GENERATED WITH N/A** (needs regeneration after backtests succeed)
3. Generate plot4 → From outputs/backtest/*_backtest.json (3 pairs, 6 plots total) - **GENERATED WITH PLACEHOLDERS** (needs regeneration after backtests succeed)
4. Update report → Nowcasting section in 3_results.tex with table 3 and plot4 - **BLOCKED** (needs actual results)

**Finalization**:
1. Update result sections → `contents/3_results.tex` with forecasting and nowcasting results - **PARTIALLY DONE** (nowcasting results missing)
2. Update discussion → `contents/4_discussion.tex` with nowcasting timepoint analysis - **STRUCTURE READY, RESULTS MISSING**
3. Finalize report → Compile PDF, verify under 15 pages, check for placeholders - **CANNOT BE DONE YET** (nowcasting results missing)

---

## Latest Updates

**Code Structure**:
- Nowcasting experiment structure defined in WORKFLOW.md
- Code implementation ready (src/infer.py supports weeks_before)
- Scripts ready (run_train.sh, run_forecast.sh, run_backtest.sh, agent_execute.sh)
- Report structure ready (methodology, results, discussion sections)

**What's Actually Missing**:
- ⚠️ **Training INCOMPLETE** - Only 10/12 models trained (missing KOIPALL.G_ddfm, KOIPALL.G_dfm)
- ⚠️ **Nowcasting experiments FAILED** - All 12 JSON files have "no_results" status (code fixes present but NOT verified)
- ⚠️ **Table 3 generated but shows N/A** (needs regeneration after backtests succeed)
- ⚠️ **Plot4 generated but shows placeholders** (needs regeneration after backtests succeed)

**Next Steps**:
- Step 1 will automatically check and run needed experiments
- Training must complete before nowcasting
- Nowcasting must complete before Table 3 and Plot4
