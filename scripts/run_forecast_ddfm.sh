#!/bin/bash
# DDFM forecasting script
# 
# This script generates forecasts from trained DDFM models for specified targets.
# Forecasts are saved to predictions/ddfm/ directory as CSV files.
#
# Usage:
#   bash scripts/run_forecast_ddfm.sh                    # Forecast all targets (88 weeks)
#   bash scripts/run_forecast_ddfm.sh --target KOEQUIPTE # Forecast specific target
#
# Prerequisites:
# - DDFM models must be trained first (run run_train_ddfm.sh)
# - Checkpoints must exist in checkpoints/{target}_ddfm/model.pkl

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
mkdir -p "$PROJECT_ROOT/predictions/ddfm"

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

# Forecast horizon: 88 weeks total (2024-01-01 to 2025-10-31)
# Predictions are made in 24-week chunks with update after each chunk
HORIZON_WEEKS=88

echo "=========================================="
echo "DDFM Forecasting"
echo "=========================================="
echo "Targets: ${TARGETS_TO_PROCESS[@]}"
echo "Horizon: $HORIZON_WEEKS weeks (2024-01-01 to 2025-10-31)"
echo "Output: predictions/ddfm/"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Process each target
for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
    MODEL_NAME="ddfm"
    # Try nested path first (newer format), then fallback to old format
    CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/${TARGET}_${MODEL_NAME}/model.pkl"
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/model.pkl"
    fi
    
    echo "[$TARGET] Processing DDFM forecasts..."
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "[$TARGET] ⚠ Checkpoint not found: $CHECKPOINT_FILE"
        echo "  Run 'bash scripts/run_train_ddfm.sh --target $TARGET' to train the model first."
        FAILED+=("${TARGET}_checkpoint_missing")
        continue
    fi
    
    # Create log file for this target
    LOG_FILE="$PROJECT_ROOT/log/${TARGET}_ddfm_forecast_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$PROJECT_ROOT/log"
    
    # Generate forecasts using Python script with update() pattern
    if python -c "
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from src.models import load_model_checkpoint, forecast_ddfm
from src.utils import (
    setup_logging, get_project_root, resolve_data_path, 
    RECENT_START, RECENT_END, get_config_path
)
from src.train.preprocess import (
    apply_transformations, set_dataframe_frequency, impute_missing_values
)
import logging

# Setup logging
project_root = get_project_root()
log_file = Path('$LOG_FILE')
setup_logging(log_dir=log_file.parent, force=True, log_file=log_file)
logger = logging.getLogger(__name__)

target = '$TARGET'
checkpoint_path = Path('$CHECKPOINT_FILE')
predictions_dir = project_root / 'predictions' / 'ddfm'
predictions_dir.mkdir(parents=True, exist_ok=True)

logger.info('='*70)
logger.info(f'DDFM Forecasting for {target}')
logger.info('='*70)
logger.info(f'Checkpoint: {checkpoint_path}')

# Load checkpoint
try:
    model, metadata = load_model_checkpoint(checkpoint_path)
    logger.info(f'Loaded model: {type(model).__name__}')
    logger.info(f'Model type: {metadata.get(\"model_type\", \"unknown\")}')
except Exception as e:
    logger.error(f'Failed to load checkpoint: {e}')
    sys.exit(1)

# Get clock frequency from model config
clock = getattr(model.config, 'clock', 'w') if hasattr(model, 'config') else 'w'
logger.info(f'Clock frequency: {clock}')

# Get trained series IDs
trained_series_ids = [s.series_id for s in model.config.series] if hasattr(model, 'config') else []
logger.info(f'Trained series: {len(trained_series_ids)} series')

# Load data up to 2023-12-31 for update()
update_end = pd.Timestamp('2023-12-31')
full_data = pd.read_csv(resolve_data_path(), index_col=0, parse_dates=True)
recent_data = full_data[(full_data.index >= RECENT_START) & (full_data.index <= update_end)]
logger.info(f'Loaded recent data: {len(recent_data)} periods ({RECENT_START} to {update_end})')

# Prepare recent_data (same preprocessing as training)
available_series = [s for s in trained_series_ids if s in recent_data.columns]
if target in recent_data.columns and target not in available_series:
    available_series.append(target)

if not available_series:
    logger.error(f'No available series found in recent data')
    sys.exit(1)

selected_data = recent_data[available_series].dropna(how='all')
config_path = str(get_config_path())

# Prepare raw data for update()
# Only apply minimal preprocessing (imputation, frequency setting) if needed
# update() will use model.preprocess (TransformerPipeline) to apply full preprocessing
# This ensures the same preprocessing pipeline as training is used
if selected_data.isnull().sum().sum() > 0:
    selected_data = impute_missing_values(selected_data, model_type='ddfm')
selected_data = set_dataframe_frequency(selected_data)

logger.info(f'Prepared raw data for update(): {selected_data.shape}')

# Get last date from training data
last_date = selected_data.index[-1] if len(selected_data) > 0 else pd.Timestamp('2023-12-31')
logger.info(f'Last date in data: {last_date}')

# Generate forecast from 2024-01-01 to 2025-10-31 (88 weeks)
start_date = pd.Timestamp('2024-01-01')
end_date = pd.Timestamp('2025-10-31')
weeks = pd.date_range(start=start_date, end=end_date, freq='W')
horizon_weeks = len(weeks)
logger.info(f'Generating forecast: {horizon_weeks} weeks ({start_date} to {end_date})')

# Generate forecast using forecast_ddfm function
# Pass raw data - forecast_ddfm() will pass it to update() which handles preprocessing internally
# update() will use model.preprocess (TransformerPipeline) for preprocessing
# base_values are automatically calculated from raw data
try:
    forecast_df = forecast_ddfm(
        model=model,
        horizon=horizon_weeks,
        last_date=last_date,
        y_recent=selected_data,  # Pass raw data - update() will handle preprocessing internally
        target_series=target,
        original_data=full_data  # Keep for backward compatibility if needed
    )
    
    logger.info(f'Generated forecast shape: {forecast_df.shape}')
    logger.info(f'Forecast index: {forecast_df.index[0]} to {forecast_df.index[-1]}')
    
    # Extract target forecast
    if target in forecast_df.columns:
        target_forecast = forecast_df[target]
    else:
        matching_cols = [col for col in forecast_df.columns if target in str(col)]
        if matching_cols:
            target_forecast = forecast_df[matching_cols[0]]
            logger.info(f'Using column {matching_cols[0]} for target {target}')
        else:
            target_forecast = forecast_df.iloc[:, 0]
            logger.warning(f'Target {target} not found, using first column')
    
    # Note: forecast_ddfm() already applies inverse transformation for target series
    # No need to apply inverse transform again here
    logger.info(f'Using forecast values from forecast_ddfm (already inverse transformed)')
    logger.info(f'Target forecast first value: {target_forecast.iloc[0]:.2f}')
    
    # Save weekly forecast
    weekly_file = predictions_dir / f'{target}_ddfm_weekly.csv'
    target_forecast.to_csv(weekly_file, header=True)
    logger.info(f'Saved weekly forecast to: {weekly_file}')
    
    # Aggregate to monthly
    monthly_forecast = target_forecast.resample('ME').mean()
    monthly_file = predictions_dir / f'{target}_ddfm_monthly.csv'
    monthly_forecast.to_csv(monthly_file, header=True)
    logger.info(f'Saved monthly forecast to: {monthly_file}')
    
    logger.info('='*70)
    logger.info(f'✓ DDFM forecasting completed for {target}')
    logger.info('='*70)
    
except Exception as e:
    logger.error(f'Failed to generate forecast: {e}')
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
" 2>&1 | tee "$LOG_FILE"; then
        echo "[$TARGET] ✓ DDFM forecasting completed"
        echo "  Output: predictions/ddfm/${TARGET}_ddfm_weekly.csv"
    else
        echo "[$TARGET] ✗ DDFM forecasting failed"
        echo "  Check log: $LOG_FILE"
        FAILED+=("${TARGET}_forecast")
    fi
    
    echo ""
done

# Create combined weekly forecasts file and training info (like chronos/lstm)
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "Creating combined weekly forecasts file and training info..."
    python -c "
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from src.models import load_model_checkpoint
import sys

project_root = Path('$PROJECT_ROOT')
predictions_dir = project_root / 'predictions' / 'ddfm'
log_dir = project_root / 'log'
targets = ${TARGETS_TO_PROCESS[@]}

# Collect all weekly forecasts
all_forecasts = {}
for target in targets:
    weekly_file = predictions_dir / f'{target}_ddfm_weekly.csv'
    if weekly_file.exists():
        df = pd.read_csv(weekly_file, index_col=0, parse_dates=True)
        all_forecasts[target] = df[target] if target in df.columns else df.iloc[:, 0]

if all_forecasts:
    # Combine into single DataFrame
    combined_df = pd.DataFrame(all_forecasts)
    combined_df.index.name = 'date'
    
    # Save combined file
    combined_file = predictions_dir / 'ddfm_weekly_forecasts.csv'
    combined_df.to_csv(combined_file)
    print(f'✓ Created combined forecast file: {combined_file}')
    print(f'  Shape: {combined_df.shape}')
    print(f'  Columns: {list(combined_df.columns)}')
    
    # Create training info JSON
    training_info = {
        'model_type': 'ddfm',
        'forecast_horizon_weeks': 88,
        'forecast_period': {
            'start': '2024-01-01',
            'end': '2025-10-31'
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
        if not log_file.exists():
            return None
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            matches = re.findall(r'total time=([\d.]+)s', content)
            if matches:
                times = [float(m) for m in matches]
                return max(times)
            return None
        except Exception:
            return None
    
    for target in targets:
        checkpoint_path = project_root / f'checkpoints/{target}_ddfm/{target}_ddfm/model.pkl'
        
        target_info = {
            'model_name': 'ddfm',
            'scaler_type': 'robust',
            'checkpoint_path': str(checkpoint_path.relative_to(project_root))
        }
        
        # Load checkpoint metadata
        if checkpoint_path.exists():
            try:
                model, metadata = load_model_checkpoint(checkpoint_path)
                
                # training_data_index에서 길이 추출
                if 'training_data_index' in metadata and metadata.get('training_data_index'):
                    target_info['training_data_length'] = len(metadata['training_data_index'])
                
                # config에서 DDFM 설정 추출
                if 'config' in metadata and metadata.get('config'):
                    config = metadata['config']
                    if 'experiment' in config and 'model_overrides' in config['experiment']:
                        ddfm_config = config['experiment']['model_overrides'].get('ddfm', {})
                        
                        if 'encoder_layers' in ddfm_config:
                            target_info['encoder_layers'] = ddfm_config['encoder_layers']
                        
                        if 'num_factors' in ddfm_config:
                            target_info['num_factors'] = int(ddfm_config['num_factors'])
                        
                        if 'epochs' in ddfm_config:
                            target_info['epochs'] = int(ddfm_config['epochs'])
                        
                        if 'learning_rate' in ddfm_config:
                            target_info['learning_rate'] = float(ddfm_config['learning_rate'])
                        
                        if 'batch_size' in ddfm_config:
                            target_info['batch_size'] = int(ddfm_config['batch_size'])
                        
                        if 'activation' in ddfm_config:
                            target_info['activation'] = ddfm_config['activation']
            except Exception as e:
                pass
        
        # Find log file
        log_files = [f for f in log_dir.glob(f'{target}_ddfm*.log') if 'forecast' not in f.name]
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            target_info['log_file'] = str(latest_log.relative_to(project_root))
            
            training_time = parse_training_time_from_log(latest_log)
            if training_time:
                target_info['training_time_seconds'] = round(training_time, 2)
                target_info['training_time_minutes'] = round(training_time / 60.0, 4)
        
        training_info['targets'][target] = target_info
    
    # Calculate summary
    training_times = [t.get('training_time_seconds') for t in training_info['targets'].values() if t.get('training_time_seconds')]
    if training_times:
        avg_time = sum(training_times) / len(training_times)
        total_time = sum(training_times)
        training_info['summary']['average_training_time_seconds'] = round(avg_time, 2)
        training_info['summary']['total_training_time_seconds'] = round(total_time, 2)
    
    # Save training info
    info_file = predictions_dir / 'ddfm_training_info.json'
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f'✓ Created training info file: {info_file}')
else:
    print('⚠ No weekly forecasts found to combine')
    sys.exit(1)
" 2>/dev/null || echo "⚠ Failed to create combined forecasts file or training_info.json"
fi

# Summary
echo "=========================================="
echo "DDFM Forecasting Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All forecasts generated successfully!"
    echo ""
    echo "Generated files:"
    for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
        if [ -f "$PROJECT_ROOT/predictions/ddfm/${TARGET}_ddfm_weekly.csv" ]; then
            echo "  - predictions/ddfm/${TARGET}_ddfm_weekly.csv"
            echo "  - predictions/ddfm/${TARGET}_ddfm_monthly.csv"
        fi
    done
    if [ -f "$PROJECT_ROOT/predictions/ddfm/ddfm_weekly_forecasts.csv" ]; then
        echo "  - predictions/ddfm/ddfm_weekly_forecasts.csv"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Review forecasts in predictions/ddfm/"
    echo "  2. Compare with actual values (if available)"
    echo "  3. Use for report generation or further analysis"
else
    echo "✗ Some forecasts failed:"
    for FAILED_ITEM in "${FAILED[@]}"; do
        echo "  - $FAILED_ITEM"
    done
    exit 1
fi
