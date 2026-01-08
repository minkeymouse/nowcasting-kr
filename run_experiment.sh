#!/bin/bash
# Training script for DFM model across all three datasets
# Models: DFM with investment, consumption, and production data
# Uses new Hydra nested config structure: model=dfm/{data_type}

set -e

# Get the script directory and set PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

# Activate project's virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

echo "========================================="
echo "Training DFM model for all three datasets"
echo "========================================="
echo ""

# 1. DFM (Dynamic Factor Model) - Investment
echo "========================================="
echo "Dataset 1/3: DFM (Investment)"
echo "========================================="
python -m src.main model=dfm/investment data=investment train=true forecast=false model.max_iter=1

echo ""
echo "========================================="
echo "Dataset 2/3: DFM (Consumption)"
echo "========================================="
python -m src.main model=dfm/consumption data=consumption train=true forecast=false model.max_iter=1

echo ""
echo "========================================="
echo "Dataset 3/3: DFM (Production)"
echo "========================================="
python -m src.main model=dfm/production data=production train=true forecast=false model.max_iter=1

echo ""
echo "========================================="
echo "All DFM models trained successfully!"
echo "========================================="