#!/bin/bash
# Run DFM and DDFM models: Training commands
# Usage: ./run_dfm.sh [--test]
#   --test: Override config for fast testing (e.g., max_iter=5 for DFM, max_epoch=5 for DDFM)

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
    echo "TEST MODE: Using reduced iterations/epochs"
    echo "========================================="
    echo ""
fi

# Set overrides for test mode
if [ "$TEST_MODE" = true ]; then
    DFM_OVERRIDE="model.max_iter=5"
    DDFM_OVERRIDE="model.max_epoch=5"
else
    DFM_OVERRIDE=""
    DDFM_OVERRIDE=""
fi

# =============================================================================
# TRAINING - INVESTMENT
# =============================================================================
echo "========================================="
echo "TRAINING - INVESTMENT"
echo "========================================="
echo ""

echo "Training DFM model (investment)..."
python -m src.main model=dfm/investment data=investment train=true forecast=false $DFM_OVERRIDE

echo ""
echo "Training DDFM model (investment)..."
python -m src.main model=ddfm/investment data=investment train=true forecast=false $DDFM_OVERRIDE

echo ""

# =============================================================================
# TRAINING - PRODUCTION
# =============================================================================
echo "========================================="
echo "TRAINING - PRODUCTION"
echo "========================================="
echo ""

echo "Training DFM model (production)..."
python -m src.main model=dfm/production data=production train=true forecast=false $DFM_OVERRIDE

echo ""
echo "Training DDFM model (production)..."
python -m src.main model=ddfm/production data=production train=true forecast=false $DDFM_OVERRIDE

echo ""
echo "========================================="
echo "All DFM and DDFM models training completed!"
echo "========================================="
echo ""
echo "Models saved to:"
echo "  - Investment: checkpoints/investment/{dfm,ddfm}/"
echo "  - Production: checkpoints/production/{dfm,ddfm}/"
echo ""
