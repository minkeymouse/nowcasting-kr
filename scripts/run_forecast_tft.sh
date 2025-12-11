#!/bin/bash
# TFT (Temporal Fusion Transformer) forecasting script
# 
# This script generates forecasts from trained TFT models for specified targets.
# Forecasts are saved to predictions/tft/ directory.
#
# Usage:
#   bash scripts/run_forecast_tft.sh                    # Forecast all targets (88 weeks)
#   bash scripts/run_forecast_tft.sh --target KOEQUIPTE # Forecast specific target
#
# Prerequisites:
# - TFT models must be trained first (run run_train_tft.sh)
# - Checkpoints must exist in checkpoints/{target}_tft/model.pkl

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
mkdir -p "$PROJECT_ROOT/predictions/tft"

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
echo "TFT Forecasting"
echo "=========================================="
echo "Targets: ${TARGETS_TO_PROCESS[@]}"
echo "Horizon: $HORIZON_WEEKS weeks (2024-01-01 to 2025-09-01)"
echo "Output: predictions/tft/"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Process each target
for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
    MODEL_NAME="tft"
    # Check for both .zip (sktime format) and .pkl (custom format)
    CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/model.zip"
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/model.pkl"
    fi
    
    echo "[$TARGET] Processing TFT forecasts..."
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "[$TARGET] ⚠ Checkpoint not found: checkpoints/${TARGET}_${MODEL_NAME}/model.pkl or model.zip"
        echo "  Run 'bash scripts/run_train_tft.sh --target $TARGET' to train the model first."
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
predictions_dir = project_root / 'predictions' / 'tft'

logger.info('='*70)
logger.info(f'TFT Forecasting for {target}')
logger.info('='*70)
logger.info(f'Checkpoint: {checkpoint_path}')
logger.info(f'Horizon: {horizon_weeks} weeks')

# Load checkpoint
try:
    model, metadata = load_model_checkpoint(checkpoint_path)
    logger.info(f'Loaded model: {type(model).__name__}')
    logger.info(f'Model type: {metadata.get(\"model_type\", \"unknown\")}')
except Exception as e:
    logger.error(f'Failed to load checkpoint: {e}')
    sys.exit(1)

# Generate forecast (returns weekly forecasts)
try:
    logger.info(f'Generating forecast for horizon={horizon_weeks} weeks...')
    forecast_df = forecast(
        checkpoint_path=checkpoint_path,
        horizon=horizon_weeks,
        model_type='tft'
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
    
    # Save weekly forecast with date column
    weekly_file = predictions_dir / f'{target}_tft_weekly.csv'
    if isinstance(target_forecast, pd.Series):
        weekly_df = target_forecast.to_frame(name=target)
    else:
        weekly_df = target_forecast
    weekly_df = weekly_df.reset_index()
    weekly_df.columns = ['date', target]
    weekly_df.to_csv(weekly_file, index=False)
    logger.info(f'Saved weekly forecast to: {weekly_file}')
    
    # Aggregate to monthly using pandas resample (sktime-compatible)
    monthly_forecast = target_forecast.resample('ME').mean()
    monthly_file = predictions_dir / f'{target}_tft_monthly.csv'
    if isinstance(monthly_forecast, pd.Series):
        monthly_df = monthly_forecast.to_frame(name=target)
    else:
        monthly_df = monthly_forecast
    monthly_df = monthly_df.reset_index()
    monthly_df.columns = ['date', target]
    monthly_df.to_csv(monthly_file, index=False)
    logger.info(f'Saved monthly forecast to: {monthly_file}')
    
    logger.info('='*70)
    logger.info(f'✓ TFT forecasting completed for {target}')
    logger.info('='*70)
    
    # Return forecast for aggregation
    sys.stdout.write(f'FORECAST_DATA:{target}:{weekly_file}\n')
    
except Exception as e:
    logger.error(f'Failed to generate forecast: {e}')
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
" 2>&1 | tee "$LOG_FILE"; then
        echo "[$TARGET] ✓ TFT forecasting completed"
        echo "  Output: predictions/tft/${TARGET}_tft_weekly.csv"
    else
        echo "[$TARGET] ✗ TFT forecasting failed"
        echo "  Check log: $LOG_FILE"
        FAILED+=("${TARGET}_forecast")
    fi
    
    echo ""
done

# Collect all forecasts for combined file
ALL_FORECASTS=()

# Summary
echo "=========================================="
echo "TFT Forecasting Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All forecasts generated successfully!"
    echo ""
    echo "Combining all forecasts into single file..."
    
    # Combine all weekly forecasts into single DataFrame
    python -c "
import sys
from pathlib import Path
import pandas as pd
from src.utils import get_project_root

project_root = get_project_root()
predictions_dir = project_root / 'predictions' / 'tft'

targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
all_forecasts = {}

for target in targets:
    weekly_file = predictions_dir / f'{target}_tft_weekly.csv'
    if weekly_file.exists():
        df = pd.read_csv(weekly_file, parse_dates=True)
        # Check if 'date' column exists
        if 'date' in df.columns:
            df = df.set_index('date')
        elif df.columns[0] == 'date' or df.columns[0].startswith('Unnamed'):
            # First column is date/index
            df = df.set_index(df.columns[0])
        all_forecasts[target] = df.iloc[:, 0]  # Get first column (the forecast values)

if all_forecasts:
    # Combine into single DataFrame
    combined_df = pd.DataFrame(all_forecasts)
    combined_df.index.name = 'date'
    
    # Save combined weekly forecasts with date column
    combined_file = predictions_dir / 'tft_weekly_forecasts.csv'
    combined_df = combined_df.reset_index()
    combined_df.to_csv(combined_file, index=False)
    print(f'Saved combined weekly forecasts to: {combined_file}')
    print(f'Shape: {combined_df.shape}')
else:
    print('No forecasts to combine')
    sys.exit(1)
" || echo "Warning: Failed to create combined forecasts file"
    
    echo ""
    echo "Creating training_info.json..."
    
    # Create training_info.json with model configuration and metadata
    python -c "
import sys
import json
from pathlib import Path
from datetime import datetime
from src.models import load_model_checkpoint
from src.utils import get_project_root

project_root = get_project_root()
predictions_dir = project_root / 'predictions' / 'tft'
log_dir = project_root / 'log'
predictions_dir.mkdir(parents=True, exist_ok=True)

targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']

# Initialize training info structure
training_info = {
    'model_type': 'tft',
    'forecast_horizon_weeks': 88,
    'forecast_period': {
        'start': '2024-01-01',
        'end': '2025-09-01'
    },
    'targets': {},
    'summary': {
        'total_targets': len(targets),
        'average_training_time_seconds': None,
        'total_training_time_seconds': None
    },
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

def parse_training_time_from_log(log_file):
    \"\"\"Parse training time from log file.\"\"\"
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for training time patterns
            import re
            # Try to find time patterns like \"Training completed in X seconds\"
            patterns = [
                r'training.*?(\d+\.?\d*)\s*seconds',
                r'Training.*?(\d+\.?\d*)\s*seconds',
                r'completed.*?(\d+\.?\d*)\s*seconds'
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    except:
        pass
    return None

# Collect information for each target
for target in targets:
    target_info = {}
    
    checkpoint_path = project_root / 'checkpoints' / f'{target}_tft' / 'model.pkl'
    if checkpoint_path.exists():
        try:
            model, metadata = load_model_checkpoint(checkpoint_path)
            
            # Extract TFT-specific configuration from metadata
            target_info['input_size'] = metadata.get('input_size', 96)
            target_info['hidden_size'] = metadata.get('hidden_size', 64)
            target_info['n_head'] = metadata.get('n_head', 4)
            target_info['dropout'] = metadata.get('dropout', 0.1)
            target_info['learning_rate'] = metadata.get('learning_rate', 0.001)
            target_info['max_epochs'] = metadata.get('max_epochs', 10)
            target_info['batch_size'] = metadata.get('batch_size', 256)
            target_info['scaler_type'] = metadata.get('scaler_type', 'robust')
            target_info['num_covariates'] = metadata.get('num_covariates', 0)
            
            # Get training data length
            training_shape = metadata.get('training_data_shape')
            if training_shape:
                target_info['training_data_length'] = training_shape[0] if isinstance(training_shape, tuple) else training_shape
            else:
                target_info['training_data_length'] = None
            
            target_info['checkpoint_path'] = str(checkpoint_path.relative_to(project_root))
            
        except Exception as e:
            print(f'Warning: Failed to load metadata for {target}: {e}', file=sys.stderr)
    
    # Find log file
    log_files = [f for f in log_dir.glob(f'{target}_tft*.log') if 'forecast' not in f.name]
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        target_info['log_file'] = str(latest_log.relative_to(project_root))
        
        training_time = parse_training_time_from_log(latest_log)
        if training_time:
            target_info['training_time_seconds'] = round(training_time, 2)
            target_info['training_time_minutes'] = round(training_time / 60.0, 4)
    else:
        target_info['log_file'] = None
        target_info['training_time_seconds'] = None
        target_info['training_time_minutes'] = None
    
    training_info['targets'][target] = target_info

# Calculate summary
training_times = [t.get('training_time_seconds') for t in training_info['targets'].values() if t.get('training_time_seconds')]
if training_times:
    avg_time = sum(training_times) / len(training_times)
    total_time = sum(training_times)
    training_info['summary']['average_training_time_seconds'] = round(avg_time, 2)
    training_info['summary']['total_training_time_seconds'] = round(total_time, 2)

# Save training info
info_file = predictions_dir / 'tft_training_info.json'
with open(info_file, 'w') as f:
    json.dump(training_info, f, indent=2)
print(f'Saved training info to: {info_file}')
" || echo "⚠ Failed to create training_info.json"
    
    echo ""
    echo "Generated files:"
    for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
        if [ -f "$PROJECT_ROOT/predictions/tft/${TARGET}_tft_weekly.csv" ]; then
            echo "  - predictions/tft/${TARGET}_tft_weekly.csv"
        fi
        if [ -f "$PROJECT_ROOT/predictions/tft/${TARGET}_tft_monthly.csv" ]; then
            echo "  - predictions/tft/${TARGET}_tft_monthly.csv"
        fi
    done
    if [ -f "$PROJECT_ROOT/predictions/tft/tft_weekly_forecasts.csv" ]; then
        echo "  - predictions/tft/tft_weekly_forecasts.csv"
    fi
    if [ -f "$PROJECT_ROOT/predictions/tft/tft_training_info.json" ]; then
        echo "  - predictions/tft/tft_training_info.json"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Review forecasts in predictions/tft/"
    echo "  2. Run result aggregation script to combine all forecasts"
else
    echo "✗ Some forecasts failed:"
    for FAILED_ITEM in "${FAILED[@]}"; do
        echo "  - $FAILED_ITEM"
    done
    exit 1
fi
