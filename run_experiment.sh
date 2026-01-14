#!/bin/bash
# Run NeuralForecast-based models: TFT, PatchTST, iTransformer, TimeMixer
# Includes training and forecasting for both short-term and long-term experiments
# Usage: ./run_experiment.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# =============================================================================
# SHORT-TERM EXPERIMENTS - INVESTMENT
# =============================================================================
echo "========================================="
echo "SHORT-TERM EXPERIMENTS - INVESTMENT"
echo "========================================="
echo "Note: Training and experiment will run together (train=true experiment=short_term)"
echo ""

echo "Training and running short-term experiment: PatchTST..."
python -m src.main model=patchtst/investment data=investment train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: TFT..."
python -m src.main model=tft/investment data=investment train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: iTransformer..."
python -m src.main model=itf/investment data=investment train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: TimeMixer..."
python -m src.main model=timemixer/investment data=investment train=true experiment=short_term

echo ""

# =============================================================================
# SHORT-TERM EXPERIMENTS - PRODUCTION
# =============================================================================
echo "========================================="
echo "SHORT-TERM EXPERIMENTS - PRODUCTION"
echo "========================================="
echo "Note: Training and experiment will run together (train=true experiment=short_term)"
echo ""

echo "Training and running short-term experiment: PatchTST..."
python -m src.main model=patchtst/production data=production train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: TFT..."
python -m src.main model=tft/production data=production train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: iTransformer..."
python -m src.main model=itf/production data=production train=true experiment=short_term

echo ""
echo "Training and running short-term experiment: TimeMixer..."
python -m src.main model=timemixer/production data=production train=true experiment=short_term

echo ""

# =============================================================================
# LONG-TERM EXPERIMENTS - INVESTMENT
# =============================================================================
echo "========================================="
echo "LONG-TERM EXPERIMENTS - INVESTMENT"
echo "========================================="
echo "Note: Models are trained with max horizon (40 weeks) to support all horizons (4w-40w)"
echo "      Training and experiment will run together"
echo ""

echo "Training and running long-term experiment: PatchTST..."
python -m src.main model=patchtst/investment data=investment train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: TFT..."
python -m src.main model=tft/investment data=investment train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: iTransformer..."
python -m src.main model=itf/investment data=investment train=true experiment=long_term \
    model.n_forecasts=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: TimeMixer..."
python -m src.main model=timemixer/investment data=investment train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""

# =============================================================================
# LONG-TERM EXPERIMENTS - PRODUCTION
# =============================================================================
echo "========================================="
echo "LONG-TERM EXPERIMENTS - PRODUCTION"
echo "========================================="
echo "Note: Models are trained with max horizon (40 weeks) to support all horizons (4w-40w)"
echo "      Training and experiment will run together"
echo ""

echo "Training and running long-term experiment: PatchTST..."
python -m src.main model=patchtst/production data=production train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: TFT..."
python -m src.main model=tft/production data=production train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: iTransformer..."
python -m src.main model=itf/production data=production train=true experiment=long_term \
    model.n_forecasts=40 model.horizon=40

echo ""
echo "Training and running long-term experiment: TimeMixer..."
python -m src.main model=timemixer/production data=production train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40

echo ""
echo "========================================="
echo "All NeuralForecast models training and forecasting completed!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Short-term: outputs/short_term/{investment,production}/{patchtst,tft,itf,timemixer}/"
echo "  - Long-term: outputs/long_term/{investment,production}/{patchtst,tft,itf,timemixer}/horizon_<N>w/"
echo ""
echo "Metrics saved in metrics.json files in each output directory."