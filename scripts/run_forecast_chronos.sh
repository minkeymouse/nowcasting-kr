#!/bin/bash
# Chronos forecasting script
# 
# This script generates forecasts from trained Chronos models for specified targets.
# Forecasts are saved to predictions/chronos/ directory.
#
# Usage:
#   bash scripts/run_forecast_chronos.sh                    # Forecast all targets (88 weeks)
#   bash scripts/run_forecast_chronos.sh --target KOEQUIPTE # Forecast specific target
#
# Prerequisites:
# - Chronos models must be trained first (run run_train_chronos.sh)
# - Checkpoints must exist in checkpoints/{target}_chronos/model.pkl

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_ROOT"

# Activate virtual environment and ensure python exists
activate_venv
ensure_python

# Parse command-line arguments
TARGET_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --target|-T)
            shift
            if [[ -z "$1" ]]; then
                echo "Error: --target requires a target name"
                exit 1
            fi
            TARGET_FILTER="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--target TARGET]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$PROJECT_ROOT/predictions/chronos"

# Target variables and their config files
declare -A TARGET_CONFIGS=(
    ["KOIPALL.G"]="experiment/production_koipallg_report"
    ["KOEQUIPTE"]="experiment/investment_koequipte_report"
    ["KOWRCCNSE"]="experiment/consumption_kowrccnse_report"
)

# Determine targets to process
if [ -n "$TARGET_FILTER" ]; then
    if [[ -v TARGET_CONFIGS["$TARGET_FILTER"] ]]; then
        TARGETS_TO_PROCESS=("$TARGET_FILTER")
    else
        echo "Error: Unknown target '$TARGET_FILTER'"
        echo "Available targets: ${!TARGET_CONFIGS[@]}"
        exit 1
    fi
else
    TARGETS_TO_PROCESS=("${!TARGET_CONFIGS[@]}")
fi

# Forecast horizon: 88 weeks total (2024-01-01 to 2025-09-01)
# Predictions are made in 24-week chunks with update after each chunk
HORIZON_WEEKS=88

echo "=========================================="
echo "Chronos Forecasting"
echo "=========================================="
echo "Targets: ${TARGETS_TO_PROCESS[@]}"
echo "Horizon: $HORIZON_WEEKS weeks (2024-01-01 to 2025-09-01)"
echo "Output: predictions/chronos/"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Process each target
for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
    MODEL_NAME="chronos"
    CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/model.pkl"
    
    echo "[$TARGET] Processing Chronos forecasts..."
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "[$TARGET] ⚠ Checkpoint not found: $CHECKPOINT_FILE"
        echo "  Run 'bash scripts/run_train_chronos.sh --target $TARGET' to train the model first."
        FAILED+=("${TARGET}_checkpoint_missing")
        continue
    fi
    
    # Create log file for this target
    LOG_FILE="$PROJECT_ROOT/log/${TARGET}_${MODEL_NAME}_forecast_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$PROJECT_ROOT/log"
    
    # Generate forecasts using Python script
    if python -c "
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from src.models import load_model_checkpoint
from src.evalutate.forecast_sktime import forecast
from src.utils import setup_logging, get_project_root
import logging

# Setup logging
project_root = get_project_root()
log_file = Path('$LOG_FILE')
setup_logging(log_dir=log_file.parent, force=True, log_file=log_file)
logger = logging.getLogger(__name__)

target = '$TARGET'
checkpoint_path = Path('$CHECKPOINT_FILE')
horizon_weeks = $HORIZON_WEEKS
predictions_dir = project_root / 'predictions' / 'chronos'

logger.info('='*70)
logger.info(f'Chronos Forecasting for {target}')
logger.info('='*70)
logger.info(f'Checkpoint: {checkpoint_path}')
logger.info(f'Horizon: {horizon_weeks} weeks')

# Load checkpoint
try:
    model, metadata = load_model_checkpoint(checkpoint_path)
    logger.info(f'Loaded model: {type(model).__name__}')
    logger.info(f'Model type: {metadata.get(\"model_type\", \"unknown\")}')
    logger.info(f'Target series: {metadata.get(\"target_series\", \"unknown\")}')
    logger.info(f'Model name: {metadata.get(\"model_name\", \"unknown\")}')
except Exception as e:
    logger.error(f'Failed to load checkpoint: {e}')
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# Generate forecast (returns weekly forecasts)
try:
    logger.info(f'Generating forecast for horizon={horizon_weeks} weeks...')
    forecast_df = forecast(
        checkpoint_path=checkpoint_path,
        horizon=horizon_weeks,
        model_type='chronos'
    )
    
    logger.info(f'Generated forecast shape: {forecast_df.shape}')
    logger.info(f'Forecast index: {forecast_df.index[0]} to {forecast_df.index[-1]}')
    
    # Extract target series forecast
    if isinstance(forecast_df, pd.DataFrame):
        if target in forecast_df.columns:
            target_forecast = forecast_df[target]
        else:
            target_forecast = forecast_df.iloc[:, 0]
    else:
        target_forecast = forecast_df
    
    # Save weekly forecast
    weekly_file = predictions_dir / f'{target}_chronos_weekly.csv'
    target_forecast.to_csv(weekly_file, header=True)
    logger.info(f'Saved weekly forecast to: {weekly_file}')
    
    # Aggregate to monthly
    monthly_forecast = target_forecast.resample('ME').mean()
    monthly_file = predictions_dir / f'{target}_chronos_monthly.csv'
    monthly_forecast.to_csv(monthly_file, header=True)
    logger.info(f'Saved monthly forecast to: {monthly_file}')
    
    logger.info('='*70)
    logger.info(f'✓ Chronos forecasting completed for {target}')
    logger.info('='*70)
    
except Exception as e:
    logger.error(f'Failed to generate forecast: {e}')
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
" 2>&1 | tee "$LOG_FILE"; then
        echo "[$TARGET] ✓ Chronos forecasting completed"
        echo "  Output: predictions/chronos/${TARGET}_chronos_weekly.csv"
        echo "  Output: predictions/chronos/${TARGET}_chronos_monthly.csv"
    else
        echo "[$TARGET] ✗ Chronos forecasting failed"
        echo "  Check log: $LOG_FILE"
        FAILED+=("${TARGET}_forecast")
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "Chronos Forecasting Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All forecasts generated successfully!"
    echo ""
    echo "Generated files:"
    for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
        if [ -f "$PROJECT_ROOT/predictions/chronos/${TARGET}_chronos_weekly.csv" ]; then
            echo "  - predictions/chronos/${TARGET}_chronos_weekly.csv"
        fi
        if [ -f "$PROJECT_ROOT/predictions/chronos/${TARGET}_chronos_monthly.csv" ]; then
            echo "  - predictions/chronos/${TARGET}_chronos_monthly.csv"
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  1. Review forecasts in predictions/chronos/"
    echo "  2. Run result aggregation script to combine all forecasts"
else
    echo "✗ Some forecasts failed:"
    for FAILED_ITEM in "${FAILED[@]}"; do
        echo "  - $FAILED_ITEM"
    done
    exit 1
fi
