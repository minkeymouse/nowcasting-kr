#!/bin/bash
# High-priority experiments for RESULTS_NEEDED.md
# Run all experiments for model comparison
# Maximum 5 parallel background processes to avoid OOM

# Don't exit on error - continue even if some models fail
set +e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

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

# Target series (define before validation)
TARGETS=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")

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

# Validate environment before starting
validate_environment() {
    echo "=========================================="
    echo "Validating Environment"
    echo "=========================================="
    
    # Check virtual environment
    if [ ! -d ".venv" ]; then
        echo "✗ Error: Virtual environment (.venv) not found"
        echo "  Create it with: python3 -m venv .venv"
        return 1
    fi
    echo "✓ Virtual environment exists"
    
    # Activate virtual environment
    source .venv/bin/activate || {
        echo "✗ Error: Failed to activate virtual environment"
        return 1
    }
    echo "✓ Virtual environment activated"
    
    # Check data file
    if [ ! -f "data/sample_data.csv" ]; then
        echo "✗ Error: Data file (data/sample_data.csv) not found"
        return 1
    fi
    echo "✓ Data file exists"
    
    # Check config files
    local missing_configs=0
    for target in "${TARGETS[@]}"; do
        local config_name=$(get_config_name "$target")
        local config_file="config/${config_name}.yaml"
        if [ ! -f "$config_file" ]; then
            echo "✗ Error: Config file not found: $config_file"
            missing_configs=$((missing_configs + 1))
        fi
    done
    
    if [ $missing_configs -gt 0 ]; then
        echo "✗ Error: $missing_configs config file(s) missing"
        return 1
    fi
    echo "✓ All config files exist"
    
    # Check Python dependencies (basic check)
    if ! python3 -c "import hydra" 2>/dev/null; then
        echo "⚠ Warning: hydra not installed (may be needed)"
    fi
    
    if ! python3 -c "import sktime" 2>/dev/null; then
        echo "⚠ Warning: sktime not installed (required for ARIMA/VAR)"
    fi
    
    # Check dfm-python (try to import)
    if ! python3 -c "import sys; sys.path.insert(0, 'dfm-python/src'); import dfm_python" 2>/dev/null; then
        echo "⚠ Warning: dfm-python not found in path (required for DFM/DDFM)"
    fi
    
    echo "✓ Environment validation complete"
    echo ""
    return 0
}

# Activate virtual environment and validate
if ! validate_environment; then
    echo "Environment validation failed. Please fix the issues above."
    exit 1
fi

# Create output directories
mkdir -p outputs/comparisons
mkdir -p outputs/experiments

# Function to check if experiment is already completed
is_experiment_complete() {
    local target=$1
    local comparisons_dir="outputs/comparisons"
    
    if [ ! -d "$comparisons_dir" ]; then
        return 1  # Not complete (directory doesn't exist)
    fi
    
    # Find all result directories for this target (matching pattern: target_YYYYMMDD_HHMMSS)
    local result_dirs=$(find "$comparisons_dir" -maxdepth 1 -type d -name "${target}_*" 2>/dev/null | sort)
    
    if [ -z "$result_dirs" ]; then
        return 1  # Not complete (no result directories)
    fi
    
    # Check the latest result directory for completion
    local latest_dir=$(echo "$result_dirs" | tail -1)
    
    # Check if comparison_results.json exists (indicates successful completion)
    if [ -f "${latest_dir}/comparison_results.json" ]; then
        return 0  # Complete
    else
        return 1  # Not complete (missing result file)
    fi
}

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

# Models to compare (as per experiment config files)
# Note: Actual models are read from experiment config files
MODELS=("arima" "var" "dfm" "ddfm")

# Horizons (all 3 as per RESULTS_NEEDED.md)
HORIZONS=(1 7 28)

# Maximum parallel processes
MAX_PARALLEL=5

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

echo "=========================================="
echo "Running High-Priority Experiments"
echo "=========================================="
echo "Target Series: ${TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Horizons: ${HORIZONS[@]}"
echo "Max Parallel: $MAX_PARALLEL"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Function to run experiment in background
run_experiment() {
    local target=$1
    local config_name=$(get_config_name "$target")
    local log_file="outputs/comparisons/${target}_$(date +%Y%m%d_%H%M%S).log"
    local start_time=$(date +%s)
    
    echo ""
    echo "=========================================="
    echo "Starting: $target"
    echo "Config: $config_name"
    echo "Log: $log_file"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Run experiment with timeout (24 hours max per experiment)
    timeout 86400 python3 src/train.py compare \
        --config-name "$config_name" \
        > "$log_file" 2>&1
    
    EXIT_CODE=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$target] ✓ Completed in ${hours}h ${minutes}m"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$target] ✗ Timeout after 24 hours (check log: $log_file)"
    else
        echo "[$target] ⚠ Completed with errors (exit code: $EXIT_CODE, check log: $log_file)"
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

# Check for completed experiments and skip them
echo "=========================================="
echo "Checking for completed experiments"
echo "=========================================="
TARGETS_TO_RUN=()
for target in "${TARGETS[@]}"; do
    if is_experiment_complete "$target"; then
        echo "✓ $target: Already complete (skipping)"
    else
        echo "→ $target: Needs to run"
        TARGETS_TO_RUN+=("$target")
    fi
done
echo ""

if [ ${#TARGETS_TO_RUN[@]} -eq 0 ]; then
    echo "All experiments are already complete!"
    echo "To re-run experiments, delete the result directories in outputs/comparisons/"
    exit 0
fi

echo "Running experiments for: ${TARGETS_TO_RUN[@]}"
echo ""

# Run comparison for each target series with parallel limit (only incomplete ones)
for target in "${TARGETS_TO_RUN[@]}"; do
    wait_for_slot
    run_experiment "$target" &
done

# Wait for all background jobs to complete with progress monitoring
echo ""
echo "Waiting for all experiments to complete..."
echo ""

# Monitor progress
MONITOR_INTERVAL=60  # Check every minute
while [ $(jobs -r | wc -l) -gt 0 ]; do
    running=$(jobs -r | wc -l)
    completed=$((${#TARGETS_TO_RUN[@]} - running))
    echo "[$(date '+%H:%M:%S')] Progress: $completed/${#TARGETS_TO_RUN[@]} completed, $running running..."
    sleep $MONITOR_INTERVAL
done

wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Aggregate results according to RESULTS_NEEDED.md structure
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="
echo ""

if python3 -c "from src.eval import main_aggregator; main_aggregator()" 2>&1; then
    echo ""
    echo "✓ Aggregation completed successfully"
else
    echo ""
    echo "⚠ Warning: Aggregation had errors (results may still be available)"
fi

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo ""

# Summary of results
echo "Results Summary:"
echo "  - Experiment logs: outputs/comparisons/*.log"
echo "  - Individual comparisons: outputs/comparisons/"
echo "  - Aggregated results: outputs/experiments/"
echo ""

# Count results
local result_dirs=$(find outputs/comparisons -maxdepth 1 -type d -name "KOGDP*" -o -name "KOCNPER*" -o -name "KOGFCF*" 2>/dev/null | wc -l)
local log_files=$(find outputs/comparisons -maxdepth 1 -type f -name "*.log" 2>/dev/null | wc -l)

echo "Generated:"
echo "  - $result_dirs comparison directory(ies)"
echo "  - $log_files log file(s)"
echo ""

if [ -f "outputs/experiments/aggregated_results.csv" ]; then
    local csv_lines=$(wc -l < outputs/experiments/aggregated_results.csv)
    echo "  - Aggregated CSV with $csv_lines line(s)"
fi

echo ""
echo "To view latest logs:"
echo "  ls -lht outputs/comparisons/*.log | head -5"
echo ""
echo "To view results:"
echo "  ls -lht outputs/comparisons/ | head -10"
echo ""

