#!/bin/bash
# LSTM training script
# 
# This script trains LSTM models for specified targets.
# Models are saved to checkpoints/ directory.
#
# Usage:
#   bash scripts/run_train_lstm.sh                    # Train all targets
#   bash scripts/run_train_lstm.sh --target KOEQUIPTE  # Train specific target
#   bash scripts/run_train_lstm.sh --test             # Test mode (1 epoch)

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_ROOT"

# Activate virtual environment and ensure python exists
activate_venv
ensure_python

# Parse command-line arguments
TEST_MODE=0
TARGET_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test|-t)
            TEST_MODE=1
            shift
            ;;
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
            echo "Usage: $0 [--test] [--target TARGET]"
            exit 1
            ;;
    esac
done

# Create directories
if [ $TEST_MODE -eq 1 ]; then
    CHECKPOINT_BASE="checkpoints_test"
    echo "=========================================="
    echo "TEST MODE: Running 1 epoch training"
    echo "=========================================="
else
    CHECKPOINT_BASE="checkpoints"
    echo "=========================================="
    echo "LSTM Training"
    echo "=========================================="
fi
mkdir -p "$PROJECT_ROOT/$CHECKPOINT_BASE"
mkdir -p "$PROJECT_ROOT/log"

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

MODEL="lstm"

echo "Targets: ${TARGETS_TO_PROCESS[@]}"
echo "Model: $MODEL"
echo "Checkpoints will be saved to: $CHECKPOINT_BASE/"
echo "Logs will be saved to: log/"
if [ $TEST_MODE -eq 1 ]; then
    echo "Training: 1 epoch (TEST MODE)"
else
    echo "Training: Normal mode (config defaults)"
fi
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Train each target
for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
    CONFIG="${TARGET_CONFIGS[$TARGET]}"
    
    echo "----------------------------------------"
    echo "[$TARGET] Training LSTM model"
    echo "Config: $CONFIG"
    echo "----------------------------------------"
    
    # Check if checkpoint already exists (skip check in test mode)
    CHECKPOINT_DIR="$PROJECT_ROOT/$CHECKPOINT_BASE/${TARGET}_${MODEL}"
    MODEL_FILE="${CHECKPOINT_DIR}/model.pkl"
    
    if [ $TEST_MODE -eq 0 ] && [ -f "$MODEL_FILE" ]; then
        echo "[$TARGET] ⏭ LSTM checkpoint already exists at $MODEL_FILE, skipping..."
        echo ""
        continue
    fi
    
    # Clean up old logs for this model before training
    python -c "
from pathlib import Path
from src.utils import cleanup_logs
log_dir = Path('$PROJECT_ROOT/log')
cleanup_logs(log_dir, keep_count=2)
" 2>/dev/null || true
    
    # Create log file for this target
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$PROJECT_ROOT/log/${TARGET}_${MODEL}_${TIMESTAMP}.log"
    
    # Build Hydra override arguments
    HYDRA_OVERRIDES=(
        "--config-name" "$CONFIG"
        "experiment.model=$MODEL"
        "experiment.checkpoint_dir=$CHECKPOINT_BASE"
        "+experiment.log_dir=log"
        "hydra.run.dir=."
        "hydra.output_subdir=null"
        "experiment.command=train"
    )
    
    # Add test mode overrides (1 epoch for LSTM)
    if [ $TEST_MODE -eq 1 ]; then
        HYDRA_OVERRIDES+=("experiment.model_overrides.lstm.epochs=1")
        echo "[$TARGET] Using test mode: 1 epoch"
    fi
    
    # Run training
    if python -m src.train.train_common "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$LOG_FILE"; then
        if [ -f "$MODEL_FILE" ]; then
            echo "[$TARGET] ✓ LSTM training completed"
            echo "  Checkpoint: $MODEL_FILE"
            echo "  Log: $LOG_FILE"
        else
            echo "[$TARGET] ✗ LSTM training completed but checkpoint not found"
            echo "  Log: $LOG_FILE"
            FAILED+=("${TARGET}_checkpoint_missing")
        fi
    else
        echo "[$TARGET] ✗ LSTM training failed"
        echo "  Log: $LOG_FILE"
        FAILED+=("${TARGET}_training")
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "LSTM Training Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All LSTM models trained successfully!"
    echo ""
    echo "Generated checkpoints:"
    for TARGET in "${TARGETS_TO_PROCESS[@]}"; do
        if [ -f "$PROJECT_ROOT/$CHECKPOINT_BASE/${TARGET}_${MODEL}/model.pkl" ]; then
            echo "  - $CHECKPOINT_BASE/${TARGET}_${MODEL}/model.pkl"
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  1. Review training logs in log/"
    echo "  2. Run forecasts: bash scripts/run_forecast_lstm.sh"
    echo "  3. Evaluate models using evaluation scripts"
else
    echo "✗ Some training failed:"
    for FAILED_ITEM in "${FAILED[@]}"; do
        echo "  - $FAILED_ITEM"
    done
    exit 1
fi
