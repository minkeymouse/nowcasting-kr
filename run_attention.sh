#!/bin/bash
# Run NeuralForecast-based models: TFT, PatchTST, iTransformer
# Includes training and forecasting for both short-term and long-term experiments
# Usage: ./run_attention.sh [--test]
#   --test: Override config to use max_epochs=1 for fast testing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parse arguments
TEST_MODE=false
if [ "$1" == "--test" ]; then
    TEST_MODE=true
    echo "========================================="
    echo "TEST MODE: Using max_epochs=1"
    echo "========================================="
    echo ""
fi

# Set epoch override for test mode
if [ "$TEST_MODE" = true ]; then
    EPOCH_OVERRIDE="model.max_epochs=1"
else
    EPOCH_OVERRIDE=""
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
python -m src.main model=patchtst/investment data=investment train=true experiment=short_term $EPOCH_OVERRIDE

echo ""
echo "Training and running short-term experiment: TFT..."
python -m src.main model=tft/investment data=investment train=true experiment=short_term $EPOCH_OVERRIDE

echo ""
echo "Training and running short-term experiment: iTransformer..."
python -m src.main model=itf/investment data=investment train=true experiment=short_term $EPOCH_OVERRIDE

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
python -m src.main model=patchtst/production data=production train=true experiment=short_term $EPOCH_OVERRIDE

echo ""
echo "Training and running short-term experiment: TFT..."
python -m src.main model=tft/production data=production train=true experiment=short_term $EPOCH_OVERRIDE

echo ""
echo "Training and running short-term experiment: iTransformer..."
python -m src.main model=itf/production data=production train=true experiment=short_term $EPOCH_OVERRIDE

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
    model.prediction_length=40 model.horizon=40 $EPOCH_OVERRIDE

echo ""
echo "Training and running long-term experiment: TFT..."
python -m src.main model=tft/investment data=investment train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40 $EPOCH_OVERRIDE

echo ""
echo "Training and running long-term experiment: iTransformer..."
python -m src.main model=itf/investment data=investment train=true experiment=long_term \
    model.n_forecasts=40 model.horizon=40 $EPOCH_OVERRIDE

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
    model.prediction_length=40 model.horizon=40 $EPOCH_OVERRIDE

echo ""
echo "Training and running long-term experiment: TFT..."
python -m src.main model=tft/production data=production train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40 $EPOCH_OVERRIDE

echo ""
echo "Training and running long-term experiment: iTransformer..."
python -m src.main model=itf/production data=production train=true experiment=long_term \
    model.n_forecasts=40 model.horizon=40 $EPOCH_OVERRIDE

echo ""
echo "========================================="
echo "All NeuralForecast models training and forecasting completed!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Short-term: outputs/short_term/{investment,production}/{patchtst,tft,itf}/"
echo "  - Long-term: outputs/long_term/{investment,production}/{patchtst,tft,itf}/horizon_<N>w/"
echo ""
echo "Metrics saved in metrics.json files in each output directory."