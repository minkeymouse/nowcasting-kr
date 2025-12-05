#!/bin/bash
# High-priority experiments for RESULTS_NEEDED.md
# Run all experiments for model comparison
# Maximum 3 parallel background processes to avoid OOM

# Don't exit on error - continue even if some models fail
set +e

# Kill any existing training/experiment processes
echo "Checking for existing processes..."
pkill -f "python.*training" 2>/dev/null
pkill -f "python.*experiment" 2>/dev/null
sleep 2

# Check if any experiments are already running
RUNNING_COUNT=$(ps aux | grep -E "python.*training.*compare" | grep -v grep | wc -l)
if [ "$RUNNING_COUNT" -gt 0 ]; then
    echo "Warning: $RUNNING_COUNT experiment(s) already running. Skipping new experiments."
    exit 0
fi

# Activate virtual environment
source .venv/bin/activate

# Target series
TARGETS=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")

# Models to compare (9 models as per RESULTS_NEEDED.md)
# Note: Some models may not be implemented yet - they will be skipped if config not found
MODELS=("arima" "var" "vecm" "dfm" "ddfm" "xgboost" "lightgbm" "deepar" "tft")

# Horizons
HORIZONS=(1 7 28)

# Maximum parallel processes
MAX_PARALLEL=3

echo "=========================================="
echo "Running High-Priority Experiments"
echo "=========================================="
echo "Target Series: ${TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Horizons: ${HORIZONS[@]}"
echo "Max Parallel: $MAX_PARALLEL"
echo "=========================================="
echo ""

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Function to run experiment in background
run_experiment() {
    local target=$1
    echo ""
    echo "=========================================="
    echo "Starting: $target"
    echo "=========================================="
    echo ""
    
    python3 src/training.py compare \
        --target-series "$target" \
        --models "${MODELS[@]}" \
        --horizons "${HORIZONS[@]}" \
        > "outputs/comparisons/${target}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$target] ✓ Completed"
    else
        echo "[$target] ⚠ Completed with errors (check log)"
    fi
}

# Run comparison for each target series with parallel limit
for target in "${TARGETS[@]}"; do
    wait_for_slot
    run_experiment "$target" &
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all experiments to complete..."
wait

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""

# Aggregate results according to RESULTS_NEEDED.md structure
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="
echo ""

python3 -m src.results_aggregator

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Individual comparisons: outputs/comparisons/"
echo "  - Aggregated results: outputs/experiments/"
echo ""

