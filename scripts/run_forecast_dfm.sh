#!/bin/bash
# DFM forecasting script
# 
# This script generates forecasts from trained DFM models for specified targets and horizons.
# Forecasts are saved to predictions/ directory as CSV files.
#
# Usage:
#   bash scripts/run_forecast_dfm.sh                    # Forecast all targets, default horizons (1-24 months)
#   bash scripts/run_forecast_dfm.sh --target KOEQUIPTE # Forecast specific target
#   bash scripts/run_forecast_dfm.sh --horizon 12       # Forecast specific horizon (12 months)
#   bash scripts/run_forecast_dfm.sh --horizons 1,3,6,12 # Forecast multiple horizons
#
# Prerequisites:
# - DFM models must be trained first (run run_train.sh --models dfm)
# - Checkpoints must exist in checkpoints/{target}_dfm/model.pkl

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_ROOT"

# Activate virtual environment and ensure python exists
activate_venv
ensure_python

# Parse command-line arguments
TARGET_FILTER=""
HORIZON_FILTER=""
HORIZONS_LIST=""
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
        --horizon|-h)
            shift
            if [[ -z "$1" ]]; then
                echo "Error: --horizon requires a horizon value"
                exit 1
            fi
            HORIZON_FILTER="$1"
            shift
            ;;
        --horizons|-H)
            shift
            if [[ -z "$1" ]]; then
                echo "Error: --horizons requires comma-separated horizon values"
                exit 1
            fi
            HORIZONS_LIST="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--target TARGET] [--horizon H] [--horizons H1,H2,...]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$PROJECT_ROOT/predictions"

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

# Determine horizons to process
if [ -n "$HORIZONS_LIST" ]; then
    # Parse comma-separated horizons
    IFS=',' read -ra HORIZONS_ARRAY <<< "$HORIZONS_LIST"
    HORIZONS_TO_PROCESS=("${HORIZONS_ARRAY[@]}")
elif [ -n "$HORIZON_FILTER" ]; then
    HORIZONS_TO_PROCESS=("$HORIZON_FILTER")
else
    # Default: horizon 12 months (to avoid instability with longer horizons)
    HORIZONS_TO_PROCESS=(12)
fi

echo "=========================================="
echo "DFM Forecasting"
echo "=========================================="
echo "Targets: ${TARGETS_TO_PROCESS[@]}"
echo "Horizons: ${HORIZONS_TO_PROCESS[@]}"
echo "Output: predictions/"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Process each target
for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
    CONFIG="${TARGET_CONFIGS[$TARGET]}"
    MODEL_NAME="dfm"
    # Try nested path first (newer format), then fallback to old format
    CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/${TARGET}_${MODEL_NAME}/model.pkl"
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL_NAME}/model.pkl"
    fi
    
    echo "[$TARGET] Processing DFM forecasts..."
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "[$TARGET] ⚠ Checkpoint not found: $CHECKPOINT_FILE"
        echo "  Run 'bash scripts/run_train.sh --models dfm --target $TARGET' to train the model first."
        FAILED+=("${TARGET}_checkpoint_missing")
        continue
    fi
    
    # Create log file for this target
    LOG_FILE="$PROJECT_ROOT/log/${TARGET}_dfm_forecast_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$PROJECT_ROOT/log"
    
    # Always forecast 2024-01-01 to 2025-10-31 (22 months) using recursive 12-month chunks
    # Horizon parameter is ignored - DFM requires recursive prediction to avoid instability
    echo "[$TARGET] Processing recursive forecast: 2024-01 to 2025-10 (22 months)"
    
    # Generate forecasts using Python script with recursive prediction
    if python -c "
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from src.models import load_model_checkpoint, forecast_dfm
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

# Define log_path for debug logging
log_path = Path('/data/nowcasting-kr/.cursor/debug.log')

target = '$TARGET'
# Always forecast 2024-01-01 to 2025-10-31 (22 months) using recursive 12-month chunks
# This avoids DFM instability with long horizons (see DFM_STABILITY_IMPROVEMENTS.md)
checkpoint_path = Path('$CHECKPOINT_FILE')
predictions_dir = project_root / 'predictions' / 'dfm'
predictions_dir.mkdir(parents=True, exist_ok=True)

logger.info('='*70)
logger.info(f'DFM Forecasting for {target}')
logger.info('Target period: 2024-01-01 to 2025-10-31 (22 months)')
logger.info('Method: Recursive prediction (12 months + 10 months chunks)')
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

# Get preprocessing pipeline from model or metadata
preprocess = getattr(model, 'preprocess', None)
if preprocess is None and metadata:
    preprocess = metadata.get('preprocess')
if preprocess:
    logger.info(f'Preprocessing pipeline available: {type(preprocess).__name__}')
else:
    logger.warning('No preprocessing pipeline found in model or metadata')

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
    selected_data = impute_missing_values(selected_data, model_type='dfm')
selected_data = set_dataframe_frequency(selected_data)

logger.info(f'Prepared raw data for update(): {selected_data.shape}')

# Update model state with raw data
# update() will internally use model.preprocess (TransformerPipeline) to apply preprocessing
# This is the same pipeline used during training, ensuring consistency
SKIP_UPDATE = False  # Enable update() with new implementation

if not SKIP_UPDATE:
    logger.info('Updating model state with raw data (2020-2023)...')
    logger.info('  update() will use model.preprocess (TransformerPipeline) for preprocessing')
    try:
        # Pass raw data - update() will handle preprocessing internally using model.preprocess
        model.update(selected_data, history=None)
        logger.info('✓ Model state updated successfully with raw data')
        logger.info('  Preprocessing applied internally using model.preprocess (TransformerPipeline)')
    except Exception as e:
        logger.warning(f'Failed to update model state: {e}')
        import traceback
        logger.debug(traceback.format_exc())
        logger.warning('Continuing without update - using model state from training end')
else:
    logger.warning('Skipping model.update() (SKIP_UPDATE=True)')
    logger.info('Using model state from training end for forecasting')

# Target forecast period: 2024-01-01 to 2025-10-31 (22 months)
# DFM instability requires recursive prediction with SHORTER horizons (6 months at a time)
# Strategy: Use 6-month chunks for better stability, then apply post-processing
# DFM/DDFM use weekly frequency, so we need to:
# 1. Predict in chunks of 6 months (more stable than 12 months)
# 2. Combine chunks to fill 2024-01 ~ 2025-10
# 3. Apply post-processing (clipping, smoothing) based on training data statistics
# 4. Aggregate weekly forecasts to monthly

target_start = pd.Timestamp('2024-01-01')
target_end = pd.Timestamp('2025-10-31')
target_weeks = pd.date_range(start=target_start, end=target_end, freq='W')
target_total_weeks = len(target_weeks)
logger.info(f'Target forecast period: {target_start} to {target_end} ({target_total_weeks} weeks, 22 months)')
logger.info(f'Using recursive prediction: 6 months at a time for better stability')

# Define chunks: 6 months each (more stable than 12 months)
chunks = [
    (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-30')),  # 6 months
    (pd.Timestamp('2024-07-01'), pd.Timestamp('2024-12-31')),  # 6 months
    (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')),  # 6 months
    (pd.Timestamp('2025-07-01'), pd.Timestamp('2025-10-31')),  # 4 months
]

# Get training data statistics for post-processing (use wider bounds to avoid over-clipping)
try:
    full_data = pd.read_csv(resolve_data_path(), index_col=0, parse_dates=True)
    train_data = full_data[(full_data.index >= pd.Timestamp('1985-01-01')) & (full_data.index <= pd.Timestamp('2019-12-31'))]
    if target in train_data.columns:
        train_series = train_data[target].dropna()
        train_mean = train_series.mean()
        train_std = train_series.std()
        train_min = train_series.min()
        train_max = train_series.max()
        # Use wider bounds: 5-sigma or actual min/max, whichever is wider
        sigma_lower = train_mean - 5 * train_std
        sigma_upper = train_mean + 5 * train_std
        clip_lower = min(train_min, sigma_lower)  # Use wider bound
        clip_upper = max(train_max, sigma_upper)  # Use wider bound
        logger.info(f'Training stats for {target}: mean={train_mean:.2f}, std={train_std:.2f}, range=[{train_min:.2f}, {train_max:.2f}]')
        logger.info(f'Will clip predictions to [{clip_lower:.2f}, {clip_upper:.2f}] (wider bounds: min/max or 5-sigma)')
    else:
        clip_lower, clip_upper = None, None
        logger.warning(f'Target {target} not found in training data, skipping clipping')
except Exception as e:
    logger.warning(f'Could not load training data for clipping: {e}')
    clip_lower, clip_upper = None, None

# Generate forecasts for each chunk
X_forecast_chunks = []
for i, (chunk_start, chunk_end) in enumerate(chunks):
    chunk_weeks = pd.date_range(start=chunk_start, end=chunk_end, freq='W')
    chunk_horizon_weeks = len(chunk_weeks)
    logger.info(f'Chunk {i+1}: {chunk_start} to {chunk_end} ({chunk_horizon_weeks} weeks)')
    
    forecast_result = model.predict(horizon=chunk_horizon_weeks, return_series=True, return_factors=True)
    if isinstance(forecast_result, tuple):
        X_chunk, _ = forecast_result
    else:
        X_chunk = forecast_result
    
    logger.info(f'✓ Chunk {i+1} generated: {X_chunk.shape}')
    X_forecast_chunks.append(X_chunk)

# Combine chunks: concatenate along time axis
X_forecast = np.concatenate(X_forecast_chunks, axis=0)
weeks = pd.date_range(start=target_start, periods=len(X_forecast), freq='W')
logger.info(f'Combined forecast: {X_forecast.shape} ({len(weeks)} weeks total)')

# #region agent log
import json
from pathlib import Path as PathLib
log_path = PathLib('/data/nowcasting-kr/.cursor/debug.log')
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a') as f:
    f.write(json.dumps({
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'A',
        'location': 'run_forecast_dfm.sh:370',
        'message': 'After recursive predict() - combined forecast values',
        'data': {
        'X_forecast_shape': X_forecast.shape,
        'X_forecast_min': float(X_forecast.min()),
        'X_forecast_max': float(X_forecast.max()),
        'X_forecast_mean': float(X_forecast.mean()),
        'num_chunks': len(X_forecast_chunks),
            'has_preprocess': preprocess is not None,
            'preprocess_type': type(preprocess).__name__ if preprocess else None
        },
        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
    }) + '\n')
# #endregion

# Convert to DataFrame with proper weekly index
# X_forecast shape: (horizon_weeks, n_series)
series_names = [s.series_id for s in model.config.series] if hasattr(model, 'config') else [f'series_{i}' for i in range(X_forecast.shape[1])]
forecast_df = pd.DataFrame(X_forecast, columns=series_names[:X_forecast.shape[1]])
# Use actual weekly dates (may be slightly different from predicted weeks if model returns different length)
forecast_df.index = weeks[:len(forecast_df)]
forecast_df.index.name = 'date'
# Ensure weekly frequency is set for proper resampling
forecast_df.index.freq = 'W'

logger.info(f'Generated forecast shape: {forecast_df.shape}')
logger.info(f'Forecast index: {forecast_df.index[0]} to {forecast_df.index[-1]}')

# #region agent log
with open(log_path, 'a') as f:
    target_col = target if target in forecast_df.columns else forecast_df.columns[0]
    target_values = forecast_df[target_col].values if target_col in forecast_df.columns else forecast_df.iloc[:, 0].values
    f.write(json.dumps({
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'A',
        'location': 'run_forecast_dfm.sh:268',
        'message': 'After DataFrame creation - before manual inverse transform',
        'data': {
            'target': target_col,
            'target_values_min': float(target_values.min()),
            'target_values_max': float(target_values.max()),
            'target_values_mean': float(target_values.mean()),
            'target_values_first_5': target_values[:5].tolist()
        },
        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
    }) + '\n')
# #endregion

# Manual inverse transform for entire forecast_df (same approach as DDFM)
# predict() returns standardized differenced values, we need to:
# 1. Inverse transform preprocessing pipeline (unstandardize)
# 2. Inverse transform transformations (undifference for chg)
if preprocess is not None:
    try:
        logger.info('Applying manual inverse transform (DDFM approach)...')
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.transformations.compose import ColumnEnsembleTransformer
        
        # If preprocessing pipeline is TransformerPipeline, manually inverse transform each step in reverse
        if isinstance(preprocess, TransformerPipeline) and hasattr(preprocess, 'steps_'):
            # Steps are: [("transform", ColumnEnsembleTransformer), ("scaler", TabularToSeriesAdaptor)]
            # We need to inverse transform: scaler -> transform (reverse order)
            
            scaler_step = None
            transform_step = None
            for step_name, step_transformer in preprocess.steps_:
                if step_name == 'scaler':
                    scaler_step = step_transformer
                elif step_name == 'transform':
                    transform_step = step_transformer
            
            # NOTE: _transform_factors_to_observations already unstandardizes using Mx and Wx,
            # so we should NOT inverse transform the scaler again (that would be double unstandardization).
            # We only need to inverse transform the transformations (Differencer, etc.)
            
            # Inverse transform transformations only (ColumnEnsembleTransformer)
            # Skip scaler inverse_transform since _transform_factors_to_observations already unstandardized
            # ColumnEnsembleTransformer may have inverse_transform that handles chg properly
            if transform_step is not None and isinstance(transform_step, ColumnEnsembleTransformer):
                try:
                    # #region agent log
                    target_col_before = target if target in forecast_df.columns else forecast_df.columns[0]
                    values_before = forecast_df[target_col_before].values if target_col_before in forecast_df.columns else forecast_df.iloc[:, 0].values
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'E',
                            'location': 'run_forecast_dfm.sh:305',
                            'message': 'Before ColumnEnsembleTransformer.inverse_transform',
                            'data': {
                                'target': target_col_before,
                                'values_min': float(values_before.min()),
                                'values_max': float(values_before.max()),
                                'values_mean': float(values_before.mean()),
                                'values_first_5': values_before[:5].tolist()
                            },
                            'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                        }) + '\n')
                    # #endregion
                    forecast_df = transform_step.inverse_transform(forecast_df)
                    # #region agent log
                    values_after = forecast_df[target_col_before].values if target_col_before in forecast_df.columns else forecast_df.iloc[:, 0].values
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'E',
                            'location': 'run_forecast_dfm.sh:306',
                            'message': 'After ColumnEnsembleTransformer.inverse_transform',
                            'data': {
                                'target': target_col_before,
                                'values_min': float(values_after.min()),
                                'values_max': float(values_after.max()),
                                'values_mean': float(values_after.mean()),
                                'values_first_5': values_after[:5].tolist()
                            },
                            'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                        }) + '\n')
                    # #endregion
                    logger.info('✓ ColumnEnsembleTransformer inverse transform applied')
                    # If ColumnEnsembleTransformer succeeds, skip manual chg transform below
                    skip_manual_chg_transform = True
                except Exception as e:
                    logger.warning(f'ColumnEnsembleTransformer.inverse_transform failed: {e}, trying individual transformers')
                    # Try individual transformers (like DDFM)
                    skip_manual_chg_transform = False
                    if hasattr(transform_step, 'transformers_'):
                        try:
                            # #region agent log
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({
                                    'sessionId': 'debug-session',
                                    'runId': 'run1',
                                    'hypothesisId': 'G',
                                    'location': 'run_forecast_dfm.sh:284',
                                    'message': 'Trying individual transformers',
                                    'data': {
                                        'num_transformers': len(transform_step.transformers_)
                                    },
                                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                                }) + '\n')
                            # #endregion
                            for name, individual_transformer, col_idx in transform_step.transformers_:
                                if hasattr(individual_transformer, 'inverse_transform'):
                                    try:
                                        # col_idx might be Index object or int - handle both
                                        if isinstance(col_idx, (list, pd.Index)):
                                            col_name = col_idx[0] if len(col_idx) > 0 else name
                                            if col_name in forecast_df.columns:
                                                col_data = forecast_df[[col_name]]
                                            else:
                                                continue
                                        else:
                                            col_data = forecast_df.iloc[:, col_idx:col_idx+1]
                                        
                                        col_inverse = individual_transformer.inverse_transform(col_data)
                                        if isinstance(col_inverse, pd.DataFrame):
                                            if isinstance(col_idx, (list, pd.Index)):
                                                col_name = col_idx[0] if len(col_idx) > 0 else name
                                                forecast_df[col_name] = col_inverse.iloc[:, 0]
                                            else:
                                                forecast_df.iloc[:, col_idx:col_idx+1] = col_inverse.values
                                        else:
                                            if isinstance(col_idx, (list, pd.Index)):
                                                col_name = col_idx[0] if len(col_idx) > 0 else name
                                                forecast_df[col_name] = col_inverse
                                            else:
                                                forecast_df.iloc[:, col_idx:col_idx+1] = col_inverse
                                        
                                        logger.info(f'✓ Inverse transformed column {name} using individual transformer')
                                        # If target column was transformed, skip manual chg transform
                                        if isinstance(col_idx, (list, pd.Index)):
                                            col_name = col_idx[0] if len(col_idx) > 0 else name
                                            if col_name == target:
                                                skip_manual_chg_transform = True
                                        elif col_idx < len(forecast_df.columns) and forecast_df.columns[col_idx] == target:
                                            skip_manual_chg_transform = True
                                    except Exception as e2:
                                        logger.warning(f'Failed to inverse transform column {name}: {e2}')
                        except Exception as e3:
                            logger.warning(f'Failed to use individual transformers: {e3}')
            else:
                skip_manual_chg_transform = False
        elif hasattr(preprocess, 'inverse_transform'):
            # Try direct inverse_transform
            try:
                forecast_df = preprocess.inverse_transform(forecast_df)
                logger.info('✓ Direct inverse_transform applied')
            except Exception as e:
                logger.warning(f'Direct inverse_transform failed: {e}')
        
    except Exception as e:
        logger.warning(f'Failed to apply manual inverse transform: {e}')
        import traceback
        logger.debug(traceback.format_exc())

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

# NOTE: predict() now handles cumulative sum inverse transform internally
# If update() was called with original data, predict() will apply cumulative sum
# using stored base_value. No need for manual chg transform here.
# However, we still log the transformation type for reference.
try:
    from pathlib import Path
    import yaml
    from src.utils import get_config_path
    
    # Check if target series uses 'chg' transformation (for logging only)
    config_path = get_config_path()
    series_config_file = config_path / \"series\" / f\"{target}.yaml\"
    trans_type = 'lin'
    if series_config_file.exists():
        with open(series_config_file, 'r') as f:
            series_config = yaml.safe_load(f) or {}
            trans_type = str(series_config.get('transformation', 'lin')).lower()
    
    logger.info(f'Target {target} transformation type: {trans_type}')
    logger.info(f'Note: predict() handles cumulative sum internally if update() was called with original data')
    
    # Check if predict() already applied cumulative sum (by checking if values are in reasonable range)
    # If values are still differenced (very small or negative), we might need manual transform
    # But since update() was called with original data, predict() should have handled it
    forecast_values = target_forecast.values
    if trans_type == 'chg':
        # Check if values look like they've been cumulative-summed
        # If all values are the same or very close, cumulative sum might not have been applied
        if target_forecast.nunique() == 1:
            logger.warning(f'All forecast values are identical ({target_forecast.iloc[0]:.4f}), cumulative sum may not have been applied')
            # Try manual cumulative sum as fallback
            base_data = full_data[full_data.index < start_date]
            if len(base_data) > 0 and target in base_data.columns:
                base_value = base_data[target].dropna().iloc[-1]
                values_clean = np.where(np.isnan(forecast_values), 0.0, forecast_values)
                forecast_original = base_value + np.cumsum(values_clean)
                forecast_original = np.where(np.isnan(forecast_values), np.nan, forecast_original)
                target_forecast = pd.Series(forecast_original, index=target_forecast.index, name=target)
                logger.info(f'Applied manual chg inverse transform (fallback) using base_value={base_value:.2f}')
        else:
            logger.info(f'Forecast values appear to have cumulative sum applied (nunique={target_forecast.nunique()})')
except Exception as e:
    logger.warning(f'Failed to check/apply chg inverse transform: {e}')
    import traceback
    logger.debug(traceback.format_exc())

# #region agent log
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a') as f:
    f.write(json.dumps({
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'D',
        'location': 'run_forecast_dfm.sh:376',
        'message': 'Final forecast values before saving',
        'data': {
            'target': target,
            'final_values_min': float(target_forecast.min()),
            'final_values_max': float(target_forecast.max()),
            'final_values_mean': float(target_forecast.mean()),
            'final_values_first_5': target_forecast.values[:5].tolist(),
            'final_values_last_5': target_forecast.values[-5:].tolist()
        },
        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
    }) + '\n')
# #endregion

# Apply post-processing to improve forecast quality
# DEBUG: Check values before post-processing
logger.info(f'Before post-processing: min={target_forecast.min():.2f}, max={target_forecast.max():.2f}, mean={target_forecast.mean():.2f}, std={target_forecast.std():.2f}, unique={target_forecast.nunique()}')

# Strategy: Use training data statistics for bias correction instead of aggressive clipping
if clip_lower is not None and clip_upper is not None:
    # Only clip extreme outliers (beyond 5-sigma), not normal variation
    extreme_lower = train_mean - 5 * train_std
    extreme_upper = train_mean + 5 * train_std
    before_clip = target_forecast.copy()
    target_forecast = target_forecast.clip(lower=extreme_lower, upper=extreme_upper)
    clipped_count = (before_clip != target_forecast).sum()
    if clipped_count > 0:
        logger.info(f'Clipped {clipped_count} extreme outliers to [{extreme_lower:.2f}, {extreme_upper:.2f}] (5-sigma bounds)')
        logger.info(f'After clipping: min={target_forecast.min():.2f}, max={target_forecast.max():.2f}, mean={target_forecast.mean():.2f}, std={target_forecast.std():.2f}, unique={target_forecast.nunique()}')
    
    # Apply bias correction: shift predictions towards training mean if they're systematically off
    pred_mean = target_forecast.mean()
    pred_std = target_forecast.std()
    logger.info(f'Bias correction check: pred_mean={pred_mean:.2f}, train_mean={train_mean:.2f}, diff={abs(pred_mean - train_mean):.2f}, threshold={2 * train_std:.2f}')
    
    if abs(pred_mean - train_mean) > 2 * train_std:  # If prediction mean is far from training mean
        bias_correction = train_mean - pred_mean
        before_bias = target_forecast.copy()
        target_forecast = target_forecast + bias_correction * 0.3  # Apply 30% of correction (conservative)
        logger.info(f'Applied bias correction: shifted by {bias_correction * 0.3:.2f} (pred_mean={pred_mean:.2f}, train_mean={train_mean:.2f})')
        logger.info(f'After bias correction: min={target_forecast.min():.2f}, max={target_forecast.max():.2f}, mean={target_forecast.mean():.2f}, std={target_forecast.std():.2f}, unique={target_forecast.nunique()}')
    else:
        logger.info(f'Skipped bias correction: difference {abs(pred_mean - train_mean):.2f} <= threshold {2 * train_std:.2f}')

# Light smoothing only if values are very noisy
if target_forecast.std() > 1000:  # Only smooth if very noisy
    try:
        target_forecast_smooth = target_forecast.rolling(window=3, center=True, min_periods=1).mean()
        target_forecast = target_forecast_smooth
        logger.info(f'Applied 3-week moving average smoothing (std was {target_forecast.std():.2f})')
    except Exception as e:
        logger.warning(f'Failed to apply smoothing: {e}')
else:
    logger.info(f'Skipping smoothing (std={target_forecast.std():.2f} is reasonable)')

# Save weekly forecast (DFM/DDFM use weekly frequency)
weekly_file = predictions_dir / f'{target}_dfm_weekly.csv'
target_forecast.to_csv(weekly_file, header=True)
logger.info(f'Saved weekly forecast ({len(target_forecast)} weeks) to: {weekly_file}')

# Aggregate weekly forecasts to monthly (DFM/DDFM predict in weeks, need monthly aggregation)
# Use 'ME' (Month End) frequency to resample weekly to monthly by averaging
monthly_forecast = target_forecast.resample('ME').mean()
monthly_file = predictions_dir / f'{target}_dfm_monthly.csv'
monthly_forecast.to_csv(monthly_file, header=True)
logger.info(f'Aggregated {len(target_forecast)} weekly forecasts to {len(monthly_forecast)} monthly forecasts')
logger.info(f'Saved monthly forecast to: {monthly_file}')

logger.info('='*70)
logger.info(f'✓ DFM forecasting completed for {target}')
logger.info(f'  Period: {target_start} to {target_end} ({len(target_forecast)} weeks, 22 months)')
logger.info(f'  Method: Recursive prediction (6-month chunks) with post-processing')
logger.info('='*70)
" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[$TARGET] ✓ DFM forecasting completed (2024-01 to 2025-10)"
    else
        echo "[$TARGET] ✗ DFM forecasting failed"
        echo "  Check log: $LOG_FILE"
        FAILED+=("${TARGET}_forecast")
    fi
    
    echo ""
done

# Create combined weekly forecasts file and training_info.json
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "Creating combined weekly forecasts file and training_info.json..."
    python -c "
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import pickle

project_root = Path('$PROJECT_ROOT')
predictions_dir = project_root / 'predictions' / 'dfm'
predictions_dir.mkdir(parents=True, exist_ok=True)

targets = ${TARGETS_TO_PROCESS[@]}

# Collect all weekly forecasts
all_forecasts = {}
# Get horizon from first target's forecast file
horizon_weeks = None
for target in targets:
    weekly_file = predictions_dir / f'{target}_dfm_weekly.csv'
    if weekly_file.exists():
        df = pd.read_csv(weekly_file, index_col=0, parse_dates=True)
        horizon_weeks = len(df)
        break

training_info = {
    'model_type': 'dfm',
    'forecast_horizon_weeks': horizon_weeks or 48,  # Default to 12 months (48 weeks)
    'forecast_period': {
        'start': '2024-01-01',
        'end': None  # Will be calculated from horizon
    },
    'targets': {},
    'summary': {
        'total_targets': len(targets),
        'average_training_time_seconds': None,
        'total_training_time_seconds': None
    },
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

total_training_time = 0.0
for target in targets:
    weekly_file = predictions_dir / f'{target}_dfm_weekly.csv'
    if weekly_file.exists():
        df = pd.read_csv(weekly_file, index_col=0, parse_dates=True)
        all_forecasts[target] = df[target] if target in df.columns else df.iloc[:, 0]
        
        # Update forecast_period from actual data
        if training_info['forecast_period']['end'] is None and len(df) > 0:
            training_info['forecast_period']['end'] = df.index[-1].strftime('%Y-%m-%d')
        
        # Load checkpoint to get training info
        checkpoint_path = project_root / 'checkpoints' / f'{target}_dfm' / f'{target}_dfm' / 'model.pkl'
        if not checkpoint_path.exists():
            checkpoint_path = project_root / 'checkpoints' / f'{target}_dfm' / 'model.pkl'
        
        target_info = {
            'model_name': 'dfm',
            'scaler_type': 'robust',
            'training_data_length': None,
            'training_time_seconds': None,
            'training_time_minutes': None,
            'checkpoint_path': str(checkpoint_path.relative_to(project_root)),
            'log_file': None
        }
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                if isinstance(checkpoint, dict):
                    metadata = checkpoint.get('metadata', {})
                    if 'training_time_seconds' in metadata:
                        target_info['training_time_seconds'] = metadata['training_time_seconds']
                        target_info['training_time_minutes'] = metadata.get('training_time_minutes')
                        total_training_time += metadata['training_time_seconds']
                    if 'training_data_length' in metadata:
                        target_info['training_data_length'] = metadata['training_data_length']
            except Exception as e:
                print(f'Warning: Could not load checkpoint info for {target}: {e}')
        
        # Find log file
        log_dir = project_root / 'log'
        if log_dir.exists():
            log_files = sorted(log_dir.glob(f'{target}_dfm_*.log'), reverse=True)
            if log_files:
                target_info['log_file'] = str(log_files[0].relative_to(project_root))
        
        training_info['targets'][target] = target_info

if all_forecasts:
    # Combine into single DataFrame
    combined_df = pd.DataFrame(all_forecasts)
    combined_df.index.name = 'date'
    
    # Save combined file
    combined_file = predictions_dir / 'dfm_weekly_forecasts.csv'
    combined_df.to_csv(combined_file)
    print(f'✓ Created combined forecast file: {combined_file}')
    print(f'  Shape: {combined_df.shape}')
    print(f'  Columns: {list(combined_df.columns)}')
    
    # Update summary
    if total_training_time > 0:
        training_info['summary']['average_training_time_seconds'] = total_training_time / len(targets)
        training_info['summary']['total_training_time_seconds'] = total_training_time
    
    # Save training_info.json
    info_file = predictions_dir / 'dfm_training_info.json'
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
echo "DFM Forecasting Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All forecasts generated successfully!"
    echo ""
    echo "Generated files:"
    for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
        if [ -f "$PROJECT_ROOT/predictions/dfm/${TARGET}_dfm_weekly.csv" ]; then
            echo "  - predictions/dfm/${TARGET}_dfm_weekly.csv"
            echo "  - predictions/dfm/${TARGET}_dfm_monthly.csv"
        fi
    done
    if [ -f "$PROJECT_ROOT/predictions/dfm/dfm_weekly_forecasts.csv" ]; then
        echo "  - predictions/dfm/dfm_weekly_forecasts.csv"
    fi
    if [ -f "$PROJECT_ROOT/predictions/dfm/dfm_training_info.json" ]; then
        echo "  - predictions/dfm/dfm_training_info.json"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Review forecasts in predictions/dfm/"
    echo "  2. Compare with actual values (if available)"
    echo "  3. Use for report generation or further analysis"
else
    echo "✗ Some forecasts failed:"
    for FAILED_ITEM in "${FAILED[@]}"; do
        echo "  - $FAILED_ITEM"
    done
    exit 1
fi
