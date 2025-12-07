#!/bin/bash
# Run all experiments for model comparison
# 3 targets × 4 models × 3 horizons = 36 combinations
# Maximum 5 parallel background processes to avoid OOM

# Don't exit on error - continue even if some models fail
set +e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Kill any existing training/experiment processes
echo "=========================================="
echo "Checking for existing processes..."
echo "=========================================="

# Save current script PID and parent PID to exclude from killing
CURRENT_PID=$$
PARENT_PID=$PPID
SCRIPT_NAME=$(basename "$0")

# Find all related processes - ONLY Python training processes, NOT shell scripts
PATTERNS=(
    "python.*train.*compare"
    "python.*src/train.py"
    "python.*train.py.*compare"
    "python.*infer"
    "python.*experiment"
    "hydra.*train"
)

FOUND_PIDS=()
for pattern in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
        # Exclude current process and parent process
        if [ -n "$pid" ] && [ "$pid" != "$CURRENT_PID" ] && [ "$pid" != "$PARENT_PID" ]; then
            # Verify it's actually a Python process, not a shell script
            local proc_cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "")
            if [[ "$proc_cmd" == *"python"* ]] && [[ "$proc_cmd" != *"bash"* ]] && [[ "$proc_cmd" != *"$SCRIPT_NAME"* ]]; then
                FOUND_PIDS+=("$pid")
            fi
        fi
    done < <(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')
done

# Remove duplicates
UNIQUE_PIDS=($(printf '%s\n' "${FOUND_PIDS[@]}" | sort -u))

if [ ${#UNIQUE_PIDS[@]} -gt 0 ]; then
    echo "Found ${#UNIQUE_PIDS[@]} running experiment process(es): ${UNIQUE_PIDS[*]}"
    echo "Terminating processes..."
    
    # First, try graceful termination (SIGTERM)
    for pid in "${UNIQUE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Sending SIGTERM to PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait for processes to terminate
    sleep 5
    
    # Check remaining processes and force kill if needed
    REMAINING_PIDS=()
    for pid in "${UNIQUE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            REMAINING_PIDS+=("$pid")
        fi
    done
    
    if [ ${#REMAINING_PIDS[@]} -gt 0 ]; then
        echo "Force killing ${#REMAINING_PIDS[@]} remaining process(es): ${REMAINING_PIDS[*]}"
        for pid in "${REMAINING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Sending SIGKILL to PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 2
    fi
    
    # Final check - use pkill as backup (only for Python processes, not shell scripts)
    # NOTE: We don't use pkill for run_experiment.sh to avoid killing ourselves
    for pattern in "${PATTERNS[@]}"; do
        # Only kill Python processes matching the pattern
        pkill -9 -f "$pattern" 2>/dev/null || true
    done
    
    echo "✓ Existing experiments terminated."
else
    echo "✓ No existing experiments found."
fi
echo ""

# Target series (define before validation)
TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")

# Function to map target series to config name
get_config_name() {
    local target=$1
    case "$target" in
        "KOEQUIPTE")
            echo "experiment/koequipte_report"
            ;;
        "KOWRCCNSE")
            echo "experiment/kowrccnse_report"
            ;;
        "KOIPALL.G")
            echo "experiment/koipallg_report"
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
    
    # Check data file (try both possible names)
    if [ -f "data/data.csv" ]; then
        DATA_FILE="data/data.csv"
        echo "✓ Data file exists: $DATA_FILE"
    elif [ -f "data/sample_data.csv" ]; then
        DATA_FILE="data/sample_data.csv"
        echo "✓ Data file exists: $DATA_FILE"
    else
        echo "✗ Error: Data file not found (checked: data/data.csv, data/sample_data.csv)"
        return 1
    fi
    
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

# Function to check if experiment is already completed with valid results
# If models_filter is provided, only checks if those specific models are complete
is_experiment_complete() {
    local target=$1
    local models_filter="${2:-}"  # Optional: comma-separated list of models to check
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
    
    # Check if comparison_results.json exists (indicates completion)
    if [ ! -f "${latest_dir}/comparison_results.json" ]; then
        return 1  # Not complete (missing result file)
    fi
    
    # Check if requested models (or all models if no filter) have valid results
    if command -v python3 >/dev/null 2>&1; then
        # Use Python to check for valid results
        local has_valid=$(python3 <<EOF
import json
import sys

models_to_check = "${models_filter}".split() if "${models_filter}" else None

try:
    with open("${latest_dir}/comparison_results.json", 'r') as f:
        results = json.load(f)
    
    if 'results' not in results:
        print("0")
        sys.exit(0)
    
    # If models filter is provided, check ALL requested models are complete
    if models_to_check and len(models_to_check) > 0 and models_to_check[0]:
        models_to_check = [m.lower().strip() for m in models_to_check]
        all_complete = True
        missing_models = []
        
        for model_name in models_to_check:
            model_result = results['results'].get(model_name, {})
            status = model_result.get('status', '')
            
            # Must be 'completed' status
            if status != 'completed':
                all_complete = False
                missing_models.append(f"{model_name}(status:{status})")
                continue
            
            # Must have metrics
            if 'metrics' not in model_result:
                all_complete = False
                missing_models.append(f"{model_name}(no_metrics)")
                continue
            
            metrics = model_result.get('metrics', {})
            forecast_metrics = metrics.get('forecast_metrics', {})
            
            # Must have valid results for at least one horizon
            model_has_valid = False
            for horizon, horizon_metrics in forecast_metrics.items():
                n_valid = horizon_metrics.get('n_valid', 0)
                if n_valid > 0:
                    model_has_valid = True
                    break
            
            if not model_has_valid:
                all_complete = False
                missing_models.append(f"{model_name}(no_valid_results)")
        
        if not all_complete:
            # Debug output (will be ignored, but helps debugging)
            sys.stderr.write(f"Incomplete models: {', '.join(missing_models)}\\n")
            print("0")
            sys.exit(0)
        
        print("1")
        sys.exit(0)
    else:
        # No filter: require ALL 4 models (arima, var, dfm, ddfm) to be complete
        required_models = ['arima', 'var', 'dfm', 'ddfm']
        all_complete = True
        
        for model_name in required_models:
            model_result = results['results'].get(model_name, {})
            status = model_result.get('status', '')
            
            if status != 'completed':
                all_complete = False
                break
            
            if 'metrics' not in model_result:
                all_complete = False
                break
            
            metrics = model_result.get('metrics', {})
            forecast_metrics = metrics.get('forecast_metrics', {})
            
            model_has_valid = False
            for horizon, horizon_metrics in forecast_metrics.items():
                n_valid = horizon_metrics.get('n_valid', 0)
                if n_valid > 0:
                    model_has_valid = True
                    break
            
            if not model_has_valid:
                all_complete = False
                break
        
        print("1" if all_complete else "0")
        sys.exit(0)
except Exception as e:
    # If parsing fails, consider it incomplete
    sys.stderr.write(f"Error checking results: {e}\\n")
    print("0")
    sys.exit(0)
EOF
)
        if [ "$has_valid" = "1" ]; then
            return 0  # Complete with valid results
        else
            return 1  # Not complete (no valid results)
        fi
    else
        # Fallback: if Python not available, assume incomplete (safer)
        return 1  # Not complete (cannot verify)
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
# Can be overridden via MODELS environment variable (e.g., MODELS="arima var" bash run_experiment.sh)
if [ -n "$MODELS" ]; then
    # Parse space-separated models from environment variable
    read -ra MODELS_ARRAY <<< "$MODELS"
    MODELS=("${MODELS_ARRAY[@]}")
else
    MODELS=("arima" "var" "dfm" "ddfm")
fi

# Horizons (1 to 30 days) - all horizons will be evaluated and averaged
HORIZONS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

# Maximum parallel processes
MAX_PARALLEL=5

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

echo "=========================================="
echo "Running Experiments"
echo "=========================================="
echo "Target Series: ${TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Horizons: ${HORIZONS[@]} (1-30 days, will be averaged per model-target pair)"
echo "Max Parallel: $MAX_PARALLEL"
if [ -n "$MODELS" ]; then
    echo "Note: Models filtered via MODELS environment variable"
fi
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
    # Use venv Python explicitly since background processes don't inherit venv activation
    # If MODELS is set, filter to only run those models
    local cmd_args=("--config-name" "$config_name")
    if [ -n "$MODELS" ]; then
        # Convert MODELS array to space-separated string for --models flag
        local models_str="${MODELS[*]}"
        cmd_args+=("--models" $models_str)
    fi
    
    timeout 86400 .venv/bin/python3 src/train.py compare "${cmd_args[@]}" \
        > "$log_file" 2>&1
    
    EXIT_CODE=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    # Verify results actually exist and are valid
    local result_dir=$(find outputs/comparisons -maxdepth 1 -type d -name "${target}_*" 2>/dev/null | sort | tail -1)
    local has_valid_results=0
    
    if [ -n "$result_dir" ] && [ -f "${result_dir}/comparison_results.json" ]; then
        # Quick check: verify JSON is valid and has results
        if python3 -c "import json; data = json.load(open('${result_dir}/comparison_results.json')); exit(0 if 'results' in data else 1)" 2>/dev/null; then
            has_valid_results=1
        fi
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        if [ $has_valid_results -eq 1 ]; then
            echo "[$target] ✓ Completed in ${hours}h ${minutes}m (results verified)"
        else
            echo "[$target] ⚠ Completed but no valid results found (exit code: $EXIT_CODE, check log: $log_file)"
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$target] ✗ Timeout after 24 hours (check log: $log_file)"
    else
        echo "[$target] ✗ Failed (exit code: $EXIT_CODE, check log: $log_file)"
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
# An experiment is considered complete if all requested models have valid results
# (status='completed' and at least one horizon has n_valid > 0)
# Experiments with unavailable results will be re-run
echo "=========================================="
echo "Checking for completed experiments"
echo "=========================================="
TARGETS_TO_RUN=()
# Convert MODELS array to space-separated string for is_experiment_complete
# If MODELS is not set, check for all 4 models (arima var dfm ddfm)
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS_STR="arima var dfm ddfm"
    MODELS_DISPLAY="all 4 models (arima var dfm ddfm)"
else
    MODELS_STR="${MODELS[*]}"
    MODELS_DISPLAY="${MODELS[*]}"
fi

for target in "${TARGETS[@]}"; do
    if is_experiment_complete "$target" "$MODELS_STR"; then
        echo "✓ $target: Already complete for $MODELS_DISPLAY (skipping)"
    else
        echo "→ $target: Needs to run for $MODELS_DISPLAY"
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

# Aggregate results
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="
echo ""

if .venv/bin/python3 -c "from src.eval import main_aggregator; main_aggregator()" 2>&1; then
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
result_dirs=$(find outputs/comparisons -maxdepth 1 -type d -name "KOEQUIPTE*" -o -name "KOWRCCNSE*" -o -name "KOIPALL.G*" 2>/dev/null | wc -l)
log_files=$(find outputs/comparisons -maxdepth 1 -type f -name "*.log" 2>/dev/null | wc -l)

echo "Generated:"
echo "  - $result_dirs comparison directory(ies)"
echo "  - $log_files log file(s)"
echo ""

if [ -f "outputs/experiments/aggregated_results.csv" ]; then
    csv_lines=$(wc -l < outputs/experiments/aggregated_results.csv)
    echo "  - Aggregated CSV with $csv_lines line(s)"
fi

echo ""
echo "To view latest logs:"
echo "  ls -lht outputs/comparisons/*.log | head -5"
echo ""
echo "To view results:"
echo "  ls -lht outputs/comparisons/ | head -10"
echo ""

