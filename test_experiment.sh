#!/bin/bash
# Test script for Mamba model training and forecasting
# Usage: ./test_experiment.sh

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
echo "SHORT-TERM EXPERIMENTS - INVESTMENT (MAMBA)"
echo "========================================="
echo "Note: Training and experiment will run together (train=true experiment=short_term)"
echo "      Using max_epochs=1 for testing"
echo ""

echo "Training and running short-term experiment: Mamba..."
python -m src.main model=mamba/investment data=investment train=true experiment=short_term \
    model.max_epochs=1

echo ""

# =============================================================================
# SHORT-TERM EXPERIMENTS - PRODUCTION
# =============================================================================
echo "========================================="
echo "SHORT-TERM EXPERIMENTS - PRODUCTION (MAMBA)"
echo "========================================="
echo "Note: Training and experiment will run together (train=true experiment=short_term)"
echo "      Using max_epochs=1 for testing"
echo ""

echo "Training and running short-term experiment: Mamba..."
python -m src.main model=mamba/production data=production train=true experiment=short_term \
    model.max_epochs=1

echo ""

# =============================================================================
# LONG-TERM EXPERIMENTS - INVESTMENT
# =============================================================================
echo "========================================="
echo "LONG-TERM EXPERIMENTS - INVESTMENT (MAMBA)"
echo "========================================="
echo "Note: Models are trained with max horizon (40 weeks) to support all horizons (4w-40w)"
echo "      Training and experiment will run together"
echo "      Using max_epochs=1 for testing"
echo ""

echo "Training and running long-term experiment: Mamba..."
python -m src.main model=mamba/investment data=investment train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40 model.max_epochs=1

echo ""

# =============================================================================
# LONG-TERM EXPERIMENTS - PRODUCTION
# =============================================================================
echo "========================================="
echo "LONG-TERM EXPERIMENTS - PRODUCTION (MAMBA)"
echo "========================================="
echo "Note: Models are trained with max horizon (40 weeks) to support all horizons (4w-40w)"
echo "      Training and experiment will run together"
echo "      Using max_epochs=1 for testing"
echo ""

echo "Training and running long-term experiment: Mamba..."
python -m src.main model=mamba/production data=production train=true experiment=long_term \
    model.prediction_length=40 model.horizon=40 model.max_epochs=1

echo ""
echo "========================================="
echo "All Mamba model training and forecasting completed!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Short-term: outputs/short_term/{investment,production}/mamba/"
echo "  - Long-term: outputs/long_term/{investment,production}/mamba/horizon_<N>w/"
echo ""
echo "Metrics saved in metrics.json files in each output directory."
