#!/bin/bash
# High-priority experiments for RESULTS_NEEDED.md
# Run all experiments for model comparison
# Maximum 5 parallel background processes to avoid OOM

# Don't exit on error - continue even if some models fail
set +e

# Kill any existing training/experiment processes
echo "Checking for existing processes..."
RUNNING_COUNT=$(ps aux | grep -E "python.*train.*compare" | grep -v grep | wc -l)
if [ "$RUNNING_COUNT" -gt 0 ]; then
    echo "Found $RUNNING_COUNT running experiment(s). Terminating..."
    pkill -f "python.*train" 2>/dev/null
    pkill -f "python.*infer" 2>/dev/null
    pkill -f "python.*experiment" 2>/dev/null
    sleep 3
    
    # Force kill if still running
    REMAINING=$(ps aux | grep -E "python.*train.*compare" | grep -v grep | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        echo "Force killing remaining processes..."
        pkill -9 -f "python.*train" 2>/dev/null
        pkill -9 -f "python.*infer" 2>/dev/null
        pkill -9 -f "python.*experiment" 2>/dev/null
        sleep 2
    fi
    echo "Existing experiments terminated."
else
    echo "No existing experiments found."
fi

# Activate virtual environment
source .venv/bin/activate

# Create output directories
mkdir -p outputs/comparisons
mkdir -p outputs/experiments

# Function to clean up old results for a target series
cleanup_old_results() {
    local target=$1
    local comparisons_dir="outputs/comparisons"
    
    if [ ! -d "$comparisons_dir" ]; then
        return
    fi
    
    # Find all result directories for this target (matching pattern: target_YYYYMMDD_HHMMSS)
    local result_dirs=$(find "$comparisons_dir" -maxdepth 1 -type d -name "${target}_*" 2>/dev/null | sort)
    
    if [ -z "$result_dirs" ]; then
        return
    fi
    
    # Count directories
    local count=$(echo "$result_dirs" | wc -l)
    
    if [ "$count" -le 1 ]; then
        # Only one or zero directories, nothing to clean
        return
    fi
    
    # Keep the latest (last in sorted list) and remove all others
    local latest_dir=$(echo "$result_dirs" | tail -1)
    
    echo "Cleaning up old results for $target (keeping latest, removing $((count-1)) old result(s))..."
    echo "$result_dirs" | while IFS= read -r dir; do
        if [ -n "$dir" ] && [ "$dir" != "$latest_dir" ]; then
            echo "  Removing: $dir"
            rm -rf "$dir" 2>/dev/null || true
        fi
    done
    
    # Also clean up old log files for this target (keep only the latest)
    local log_files=$(find "$comparisons_dir" -maxdepth 1 -type f -name "${target}_*.log" 2>/dev/null | sort)
    local log_count=$(echo "$log_files" | wc -l)
    
    if [ "$log_count" -gt 1 ]; then
        local latest_log=$(echo "$log_files" | tail -1)
        
        echo "Cleaning up old log files for $target (keeping latest, removing $((log_count-1)) old log(s))..."
        echo "$log_files" | while IFS= read -r log; do
            if [ -n "$log" ] && [ "$log" != "$latest_log" ]; then
                echo "  Removing: $log"
                rm -f "$log" 2>/dev/null || true
            fi
        done
    fi
}

# Target series
TARGETS=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")

# Models to compare (as per experiment config files)
# Note: Actual models are read from experiment config files
MODELS=("arima" "var" "dfm" "ddfm")

# Horizons (all 3 as per RESULTS_NEEDED.md)
HORIZONS=(1 7 28)

# Maximum parallel processes
MAX_PARALLEL=5

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

# Function to map target series to config name
get_config_name() {
    local target=$1
    case "$target" in
        "KOGDP...D")
            echo "experiment/kogdp_report"
            ;;
        "KOCNPER.D")
            echo "experiment/kocnper_report"
            ;;
        "KOGFCF..D")
            echo "experiment/kogfcf_report"
            ;;
        *)
            # Fallback: try to construct config name from target
            local config_name=$(echo "$target" | tr '[:upper:]' '[:lower:]' | sed 's/\.\.\.//g' | sed 's/\.//g')
            echo "experiment/${config_name}_report"
            ;;
    esac
}

# Function to run experiment in background
run_experiment() {
    local target=$1
    local config_name=$(get_config_name "$target")
    echo ""
    echo "=========================================="
    echo "Starting: $target"
    echo "Config: $config_name"
    echo "=========================================="
    echo ""
    
    python3 src/train.py compare \
        --config-name "$config_name" \
        > "outputs/comparisons/${target}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$target] ✓ Completed"
    else
        echo "[$target] ⚠ Completed with errors (check log)"
    fi
}

# Clean up old results before running new experiments
echo "=========================================="
echo "Cleaning up old experiment results"
echo "=========================================="
for target in "${TARGETS[@]}"; do
    cleanup_old_results "$target"
done
echo ""

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

python3 -m src.eval.aggregator

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Experiment logs: outputs/comparisons/*.log"
echo "  - Individual comparisons: outputs/comparisons/"
echo "  - Aggregated results: outputs/experiments/"
echo ""
echo "To view latest logs:"
echo "  ls -lht outputs/comparisons/*.log | head -5"
echo ""

