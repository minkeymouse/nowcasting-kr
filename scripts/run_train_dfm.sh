#!/bin/bash
# DFM training script
# 
# This script trains DFM models for specified targets.
# Models are saved to checkpoints/{target}_dfm/model.pkl
#
# Usage:
#   bash scripts/run_train_dfm.sh                    # Train all targets
#   bash scripts/run_train_dfm.sh --target KOEQUIPTE # Train specific target
#   bash scripts/run_train_dfm.sh --test             # Test mode (reduced iterations)

# Don't exit on error - continue with other targets
# set -e  # Exit on error

# Parse command-line arguments
TEST_MODE=0
TARGETS_FILTER=()
FORCE_RETRAIN=0
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
            if [[ "$1" == *","* ]]; then
                IFS=',' read -ra TARGETS_FILTER <<< "$1"
            else
                TARGETS_FILTER=("$1")
            fi
            shift
            ;;
        --force|-f)
            FORCE_RETRAIN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--target TARGET1,TARGET2,...] [--force]"
            exit 1
            ;;
    esac
done

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_ROOT"

# Graceful shutdown handling
CLEANUP_PIDS=()
SHUTDOWN_REQUESTED=0

cleanup() {
    if [ $SHUTDOWN_REQUESTED -eq 0 ]; then
        SHUTDOWN_REQUESTED=1
        echo ""
        echo "=========================================="
        echo "Shutdown requested (Ctrl+C). Cleaning up..."
        echo "=========================================="
        
        # Kill all tracked background processes
        for pid in "${CLEANUP_PIDS[@]}"; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Terminating process $pid..."
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Kill the current Python process and its children
        PYTHON_PIDS=$(pgrep -f "train_common.*dfm" 2>/dev/null || true)
        if [ -n "$PYTHON_PIDS" ]; then
            echo "Terminating Python training processes..."
            echo "$PYTHON_PIDS" | xargs kill -TERM 2>/dev/null || true
        fi
        
        # Also kill any child processes of this script
        CHILD_PIDS=$(pgrep -P $$ 2>/dev/null || true)
        if [ -n "$CHILD_PIDS" ]; then
            echo "Terminating child processes..."
            echo "$CHILD_PIDS" | xargs kill -TERM 2>/dev/null || true
        fi
        
        # Wait a bit for graceful shutdown
        sleep 2
        
        # Force kill if still running
        REMAINING=$(pgrep -f "train_common.*dfm" 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            echo "Force killing remaining processes..."
            echo "$REMAINING" | xargs kill -KILL 2>/dev/null || true
        fi
        
        # Kill process group as last resort
        kill -TERM -$$ 2>/dev/null || true
        sleep 1
        kill -KILL -$$ 2>/dev/null || true
        
        echo "Cleanup completed."
        exit 130
    fi
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Activate virtual environment and ensure python exists
activate_venv
ensure_python

# Create directories
if [ $TEST_MODE -eq 1 ]; then
    CHECKPOINT_BASE="checkpoints_test"
    echo "=========================================="
    echo "TEST MODE: Running with reduced iterations"
    echo "=========================================="
else
    CHECKPOINT_BASE="checkpoints"
    echo "=========================================="
    echo "Starting DFM training"
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

echo "Targets: ${!TARGET_CONFIGS[@]}"
echo "Model: dfm"
echo "Checkpoints will be saved to: $CHECKPOINT_BASE/"
echo "Logs will be saved to: log/"
if [ $TEST_MODE -eq 1 ]; then
    echo "Training: Reduced iterations (TEST MODE)"
else
    echo "Training: Normal iterations"
fi
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Filter targets if specified
TARGETS_TO_TRAIN=("${!TARGET_CONFIGS[@]}")
if [ ${#TARGETS_FILTER[@]} -gt 0 ]; then
    TARGETS_TO_TRAIN=()
    for TARGET_FILTER in "${TARGETS_FILTER[@]}"; do
        if [[ -v TARGET_CONFIGS["$TARGET_FILTER"] ]]; then
            TARGETS_TO_TRAIN+=("$TARGET_FILTER")
        else
            echo "Warning: Target '$TARGET_FILTER' not found in TARGET_CONFIGS, skipping..."
        fi
    done
    if [ ${#TARGETS_TO_TRAIN[@]} -eq 0 ]; then
        echo "Error: No valid targets specified"
        exit 1
    fi
    echo "Filtering to specific targets: ${TARGETS_TO_TRAIN[@]}"
fi

# Train DFM for each target
for TARGET in "${TARGETS_TO_TRAIN[@]}"; do
    
    CONFIG="${TARGET_CONFIGS[$TARGET]}"
    echo "----------------------------------------"
    echo "Training DFM for target: $TARGET"
    echo "Config: $CONFIG"
    echo "----------------------------------------"
    
    # Check if model checkpoint already exists (skip check in test mode or if force retrain)
    CHECKPOINT_DIR="$PROJECT_ROOT/$CHECKPOINT_BASE/${TARGET}_dfm"
    MODEL_FILE="${CHECKPOINT_DIR}/model.pkl"
    
    if [ $TEST_MODE -eq 0 ] && [ $FORCE_RETRAIN -eq 0 ] && [ -f "$MODEL_FILE" ]; then
        echo "[$TARGET] ⏭ DFM checkpoint already exists at $MODEL_FILE, skipping..."
        echo "  Use --force to retrain anyway"
        continue
    fi
    
    if [ $FORCE_RETRAIN -eq 1 ] && [ -f "$MODEL_FILE" ]; then
        echo "[$TARGET] 🔄 Force retraining (existing checkpoint will be overwritten)"
    fi
    
    # Clean up old logs for this model before training
    python -c "
from pathlib import Path
from src.utils import cleanup_logs
log_dir = Path('$PROJECT_ROOT/log')
cleanup_logs(log_dir, keep_count=2)
" 2>/dev/null || true
    
    # Run training with checkpoint_dir and log_dir override
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$PROJECT_ROOT/log/${TARGET}_dfm_${TIMESTAMP}.log"
    
    # Build Hydra override arguments
    HYDRA_OVERRIDES=(
        "--config-name" "$CONFIG"
        "experiment.model=dfm"
        "experiment.checkpoint_dir=$CHECKPOINT_BASE"
        "+experiment.log_dir=log"
        "hydra.run.dir=."
        "hydra.output_subdir=null"
        "experiment.command=train"
    )
    
    # Add test mode overrides (reduced iterations for DFM)
    if [ $TEST_MODE -eq 1 ]; then
        HYDRA_OVERRIDES+=("experiment.model_overrides.dfm.max_iter=10")
        HYDRA_OVERRIDES+=("experiment.model_overrides.dfm.threshold=1e-3")
    fi
    
    # Run training and capture output
    set -o pipefail
    python -m src.train.train_common "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    set +o pipefail
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$TARGET] ✓ DFM completed successfully"
        echo "  Checkpoint: $MODEL_FILE"
    else
        if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
            echo "[$TARGET] ⚠ DFM interrupted by user"
            break
        else
            echo "[$TARGET] ✗ DFM failed (exit code: $EXIT_CODE)"
            echo "  Check log: $LOG_FILE"
            FAILED+=("${TARGET}_dfm")
        fi
    fi
    
    echo ""
done

# Clear trap before final report
trap - SIGINT SIGTERM

echo "=========================================="
if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
    echo "Training interrupted by user"
else
    echo "DFM training completed"
fi
echo "=========================================="

# Report results
if [ ${#FAILED[@]} -eq 0 ]; then
    if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
        echo "⚠ Training was interrupted. Some models may be incomplete."
        exit 130
    else
        echo "✓ All DFM models trained successfully!"
        echo ""
        echo "Checkpoints saved to: $CHECKPOINT_BASE/"
        echo "  - ${TARGETS_TO_TRAIN[@]/%/_dfm/model.pkl}"
        exit 0
    fi
else
    echo "✗ Some DFM models failed:"
    for FAILED_MODEL in "${FAILED[@]}"; do
        echo "  - $FAILED_MODEL"
    done
    exit 1
fi
