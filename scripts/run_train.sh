#!/bin/bash
# Training script for nowcasting models
# Trains all models (ARIMA, VAR, DFM, DDFM) for all target variables
# Saves checkpoints to checkpoints/ and logs to log/

# Don't exit on error - continue with other models
# set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
        # Find the Python process running train.py
        PYTHON_PIDS=$(pgrep -f "train.py.*$CONFIG.*$MODEL" 2>/dev/null || true)
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
        REMAINING=$(pgrep -f "train.py.*$CONFIG.*$MODEL" 2>/dev/null || true)
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

# Activate virtual environment if it exists (check in project root)
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: .venv"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated virtual environment: venv"
fi

# Use python3 if python is not available
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3
    else
        echo "Error: python or python3 not found"
        exit 1
    fi
fi

# Create directories (in project root)
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/log"

# Target variables and their config files
declare -A TARGETS=(
    ["KOIPALL.G"]="experiment/production_koipallg_report"
    ["KOEQUIPTE"]="experiment/investment_koequipte_report"
    ["KOWRCCNSE"]="experiment/consumption_kowrccnse_report"
)

# Models to train
# Train all four models: ARIMA, VAR, DFM, DDFM
MODELS=("arima" "var" "dfm" "ddfm")

echo "=========================================="
echo "Starting training for all models"
echo "=========================================="
echo "Targets: ${!TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Checkpoints will be saved to: checkpoints/"
echo "Logs will be saved to: log/"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# Train each model for each target
for TARGET in "${!TARGETS[@]}"; do
    
    CONFIG="${TARGETS[$TARGET]}"
    echo "----------------------------------------"
    echo "Training models for target: $TARGET"
    echo "Config: $CONFIG"
    echo "----------------------------------------"
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "[$TARGET] Training $MODEL..."
        
        # Check if model checkpoint already exists
        CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL}"
        MODEL_FILE="${CHECKPOINT_DIR}/model.pkl"
        
        if [ -f "$MODEL_FILE" ]; then
            echo "[$TARGET] ⏭ $MODEL checkpoint already exists at $MODEL_FILE, skipping..."
            continue
        fi
        
        # Run training with checkpoint_dir and log_dir override
        # Use --config-name flag for Hydra to properly parse config name
        # Override experiment.model, experiment.checkpoint_dir, and experiment.log_dir
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$PROJECT_ROOT/log/${TARGET}_${MODEL}_${TIMESTAMP}.log"
        
        # Run training and capture output
        # Use exec to ensure signal handling works properly
        if python "$PROJECT_ROOT/src/train.py" \
            --config-name "$CONFIG" \
            experiment.model="$MODEL" \
            experiment.checkpoint_dir=checkpoints \
            +experiment.log_dir=log \
            experiment.command=train 2>&1 | tee "$LOG_FILE"; then
            echo "[$TARGET] ✓ $MODEL completed successfully"
        else
            EXIT_CODE=${PIPESTATUS[0]}
            if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
                echo "[$TARGET] ⚠ $MODEL interrupted by user"
                break 2  # Break out of both loops
            else
                echo "[$TARGET] ✗ $MODEL failed (exit code: $EXIT_CODE)"
                FAILED+=("${TARGET}_${MODEL}")
            fi
        fi
    done
    
    echo ""
done

# Clear trap before final report
trap - SIGINT SIGTERM

echo "=========================================="
if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
    echo "Training interrupted by user"
else
    echo "Training completed"
fi
echo "=========================================="

# Report results
if [ ${#FAILED[@]} -eq 0 ]; then
    if [ $SHUTDOWN_REQUESTED -eq 1 ]; then
        echo "⚠ Training was interrupted. Some models may be incomplete."
        exit 130
    else
        echo "✓ All models trained successfully!"
        exit 0
    fi
else
    echo "✗ Some models failed:"
    for FAILED_MODEL in "${FAILED[@]}"; do
        echo "  - $FAILED_MODEL"
    done
    exit 1
fi

