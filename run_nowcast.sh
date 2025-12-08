#!/bin/bash
# Nowcasting script - runs nowcasting for DFM and DDFM models only
# Loads models from checkpoint/ directory and generates nowcasts
# Period: 2024-01 to 2025-10 (22 months)
# Time points: 4 weeks ago, 1 week ago

# Don't exit on error - continue even if some models fail
set +e

# Graceful shutdown handler
cleanup_on_exit() {
    echo ""
    echo "=========================================="
    echo "Received interrupt signal. Cleaning up..."
    echo "=========================================="
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 130
}

# Set up signal handlers for graceful shutdown
trap cleanup_on_exit SIGINT SIGTERM

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ============================================================================
# Argument Parsing
# ============================================================================

TEST_MODE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test|-t)
            TEST_MODE=1
            shift
            ;;
        --jobs|-j)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test, -t              Run in test mode (validate setup without running)"
            echo "  --jobs, -j N            Maximum parallel jobs (default: number of CPU cores)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "This script runs nowcasting for DFM and DDFM models only."
            echo "Nowcast with 4 weeks ago and 1 week ago (2024-01 to 2025-10, 22 months)"
            echo ""
            echo "Results are saved to:"
            echo "  - outputs/backtest/{target}_{model}_backtest.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Target series
TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")

# Models: Only DFM and DDFM support nowcasting
MODELS=("dfm" "ddfm")


# Function to map target series to config name
get_config_name() {
    local target=$1
    case "$target" in
        "KOEQUIPTE")
            echo "experiment/investment_koequipte_report"
            ;;
        "KOWRCCNSE")
            echo "experiment/consumption_kowrccnse_report"
            ;;
        "KOIPALL.G")
            echo "experiment/production_koipallg_report"
            ;;
        *)
            local config_name=$(echo "$target" | tr '[:upper:]' '[:lower:]' | sed 's/\.\.\.//g' | sed 's/\.//g')
            echo "experiment/${config_name}_report"
            ;;
    esac
}

# Function to check if checkpoint exists
checkpoint_exists() {
    local target=$1
    local model=$2
    local checkpoint_file="checkpoint/${target}_${model}/model.pkl"
    if [ -f "$checkpoint_file" ] && [ -s "$checkpoint_file" ]; then
        return 0  # Exists
    else
        return 1  # Missing
    fi
}

# Note: Results are always overwritten (no existence checks)

# ============================================================================
# Test Mode Function
# ============================================================================

run_test_mode() {
    echo "=========================================="
    echo "TEST MODE: Validating Setup"
    echo "=========================================="
    echo ""
    
    local test_passed=0
    local test_failed=0
    
    # Test 1: Virtual environment
    echo "[TEST 1] Checking virtual environment..."
    if [ ! -d ".venv" ]; then
        echo "  ✗ FAIL: Virtual environment (.venv) not found"
        test_failed=$((test_failed + 1))
    else
        echo "  ✓ PASS: Virtual environment exists"
        test_passed=$((test_passed + 1))
        
        if source .venv/bin/activate 2>/dev/null; then
            echo "  ✓ PASS: Virtual environment can be activated"
            test_passed=$((test_passed + 1))
        else
            echo "  ✗ FAIL: Failed to activate virtual environment"
            test_failed=$((test_failed + 1))
        fi
    fi
    echo ""
    
    # Test 2: Python dependencies
    echo "[TEST 2] Checking Python dependencies..."
    source .venv/bin/activate 2>/dev/null || true
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import pandas, numpy, hydra, omegaconf, sktime" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ PASS: Core dependencies available"
            test_passed=$((test_passed + 1))
        else
            echo "  ✗ FAIL: Missing core dependencies"
            test_failed=$((test_failed + 1))
        fi
    else
        echo "  ✗ FAIL: python3 not found"
        test_failed=$((test_failed + 1))
    fi
    echo ""
    
    # Test 3: Config files
    echo "[TEST 3] Checking config files..."
    local missing_configs=0
    for target in "${TARGETS[@]}"; do
        local config_name=$(get_config_name "$target")
        local config_file="config/${config_name}.yaml"
        if [ -f "$config_file" ]; then
            echo "  ✓ PASS: Config file exists: $config_file"
            test_passed=$((test_passed + 1))
        else
            echo "  ✗ FAIL: Config file not found: $config_file"
            missing_configs=$((missing_configs + 1))
            test_failed=$((test_failed + 1))
        fi
    done
    echo ""
    
    # Test 4: Checkpoint files (DFM and DDFM only)
    echo "[TEST 4] Checking checkpoint files for models: ${MODELS[@]}..."
    if [ ! -d "checkpoint" ]; then
        echo "  ✗ FAIL: checkpoint/ directory not found"
        test_failed=$((test_failed + 1))
    else
        echo "  ✓ PASS: checkpoint/ directory exists"
        test_passed=$((test_passed + 1))
        
        local missing_checkpoints=0
        local found_checkpoints=0
        for target in "${TARGETS[@]}"; do
            for model in "${MODELS[@]}"; do
                if checkpoint_exists "$target" "$model"; then
                    echo "  ✓ PASS: Checkpoint exists: ${target}_${model}"
                    found_checkpoints=$((found_checkpoints + 1))
                    test_passed=$((test_passed + 1))
                else
                    echo "  ✗ FAIL: Checkpoint missing: ${target}_${model}"
                    missing_checkpoints=$((missing_checkpoints + 1))
                    test_failed=$((test_failed + 1))
                fi
            done
        done
        
        if [ $missing_checkpoints -eq 0 ]; then
            echo "  ✓ PASS: All required checkpoints exist ($found_checkpoints checkpoints)"
        else
            echo "  ⚠ WARN: $missing_checkpoints checkpoint(s) missing"
        fi
    fi
    echo ""
    
    # Test 5: Output directories
    echo "[TEST 5] Checking output directories..."
    local dirs=("outputs/nowcast" "outputs/backtest")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            if mkdir -p "$dir" 2>/dev/null; then
                echo "  ✓ PASS: Can create $dir/ directory"
                test_passed=$((test_passed + 1))
                rmdir "$dir" 2>/dev/null || true
            else
                echo "  ✗ FAIL: Cannot create $dir/ directory"
                test_failed=$((test_failed + 1))
            fi
        else
            if [ -w "$dir" ]; then
                echo "  ✓ PASS: $dir/ is writable"
                test_passed=$((test_passed + 1))
            else
                echo "  ✗ FAIL: $dir/ is not writable"
                test_failed=$((test_failed + 1))
            fi
        fi
    done
    echo ""
    
    # Summary
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Tests passed: $test_passed"
    echo "Tests failed: $test_failed"
    echo ""
    
    if [ $test_failed -eq 0 ]; then
        echo "✓ All tests passed! Setup is ready."
        return 0
    else
        echo "✗ Some tests failed. Please fix the issues above."
        return 1
    fi
}

# ============================================================================
# Main Script Logic
# ============================================================================

# If test mode, run tests and exit
if [ "$TEST_MODE" -eq 1 ]; then
    if run_test_mode; then
        exit 0
    else
        exit 1
    fi
fi

# Validate environment
validate_environment() {
    echo "=========================================="
    echo "Validating Environment"
    echo "=========================================="
    
    if [ ! -d ".venv" ]; then
        echo "✗ Error: Virtual environment (.venv) not found"
        return 1
    fi
    echo "✓ Virtual environment exists"
    
    source .venv/bin/activate || {
        echo "✗ Error: Failed to activate virtual environment"
        return 1
    }
    echo "✓ Virtual environment activated"
    
    if [ ! -d "checkpoint" ]; then
        echo "✗ Error: checkpoint/ directory not found. Please run run_train.sh first."
        return 1
    fi
    echo "✓ Checkpoint directory exists"
    
    echo "✓ Environment validation complete"
    echo ""
    return 0
}

# Activate virtual environment and validate
if ! validate_environment; then
    echo "Environment validation failed. Please fix the issues above."
    exit 1
fi

# Validate checkpoints
validate_checkpoints() {
    echo "=========================================="
    echo "Validating Checkpoints"
    echo "=========================================="
    
    local missing_count=0
    local found_count=0
    local missing_list=()
    
    for target in "${TARGETS[@]}"; do
        for model in "${MODELS[@]}"; do
            if checkpoint_exists "$target" "$model"; then
                echo "✓ ${target}_${model}: Checkpoint exists"
                found_count=$((found_count + 1))
            else
                echo "✗ ${target}_${model}: Checkpoint missing"
                missing_count=$((missing_count + 1))
                missing_list+=("${target}_${model}")
            fi
        done
    done
    
    echo ""
    echo "Checkpoint Summary:"
    echo "  Found: $found_count"
    echo "  Missing: $missing_count"
    
    if [ $missing_count -gt 0 ]; then
        echo ""
        echo "⚠ Warning: $missing_count checkpoint(s) missing:"
        for missing in "${missing_list[@]}"; do
            echo "    - $missing"
        done
        echo ""
        echo "Nowcasting will skip missing models."
        if [ -t 0 ]; then
            read -p "Continue with available models? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled."
                exit 1
            fi
        else
            echo "Non-interactive mode: Continuing with available models..."
        fi
    fi
    
    echo ""
    return 0
}

validate_checkpoints

# Ensure output directories exist
ensure_output_dirs() {
    mkdir -p outputs/nowcast
    mkdir -p outputs/backtest
    mkdir -p log
}

ensure_output_dirs

# ============================================================================
# Parallel Execution Configuration
# ============================================================================

# Maximum number of parallel jobs (GPU-aware default, capped at 3)
if [ -z "${MAX_PARALLEL_JOBS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        MAX_PARALLEL_JOBS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        [ "$MAX_PARALLEL_JOBS" -lt 1 ] && MAX_PARALLEL_JOBS=1
    else
        MAX_PARALLEL_JOBS=1
    fi
fi
[ "$MAX_PARALLEL_JOBS" -gt 3 ] && MAX_PARALLEL_JOBS=3

# ============================================================================
# Nowcasting (4 weeks ago, 1 week ago)
# ============================================================================
echo "=========================================="
echo "Nowcasting"
echo "=========================================="
echo "Period: 2024-01 to 2025-10 (22 months)"
echo "Time points: 4 weeks ago, 1 week ago"
echo "Target Series: ${TARGETS[@]}"
echo "Models: ${MODELS[@]} (DFM and DDFM only)"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
    
SUCCESSFUL_NOWCASTS=()
FAILED_NOWCASTS=()
TOTAL_COMBINATIONS=$((${#TARGETS[@]} * ${#MODELS[@]}))
NOWCAST_PIDS=()
NOWCAST_PID_MAP=()
current_combination=0
    
    # Function to run nowcast for a single target-model combination
    run_nowcast_combination() {
        local target=$1
        local model=$2
        local combination_num=$3
        local config_name=$(get_config_name "$target")
        
        # Always overwrite existing results
        mkdir -p outputs/nowcast || {
            echo "✗ Error: Cannot create outputs/nowcast/ directory"
            return 1
        }
        local log_file="log/${target}_${model}_nowcast.log"
        
        # Run nowcast and capture exit code
        timeout 86400 .venv/bin/python3 src/train.py nowcast \
            --config-name "$config_name" \
            --model "$model" \
            --checkpoint-dir "checkpoint" \
            --nowcast-start "2024-01-01" \
            --nowcast-end "2025-10-31" \
            > "$log_file" 2>&1
        local EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$combination_num/$TOTAL_COMBINATIONS] [$target-$model] ✓ Nowcast completed"
            return 0
        elif [ $EXIT_CODE -eq 124 ]; then
            echo "[$combination_num/$TOTAL_COMBINATIONS] [$target-$model] ✗ Timeout after 24 hours"
            return 1
        else
            echo "[$combination_num/$TOTAL_COMBINATIONS] [$target-$model] ✗ Failed (exit code: $EXIT_CODE)"
            return 1
        fi
    }
    
# Build list of tasks to run
TASKS=()
for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Skip if checkpoint doesn't exist
        if ! checkpoint_exists "$target" "$model"; then
            echo "Skipping ${target}_${model} (checkpoint missing)"
            continue
        fi
        TASKS+=("$target|$model")
    done
done

# Run nowcasts in parallel with job control
for task in "${TASKS[@]}"; do
    IFS='|' read -r target model <<< "$task"
    current_combination=$((current_combination + 1))
    
    # Wait for available slot if we've reached max parallel jobs
    while [ ${#NOWCAST_PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; do
        for pid in "${NOWCAST_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, wait for it and get result
                wait "$pid"
                exit_code=$?
                task_for_pid="${NOWCAST_PID_MAP[$pid]}"
                IFS='|' read -r t m <<< "$task_for_pid"
                if [ $exit_code -eq 0 ]; then
                    SUCCESSFUL_NOWCASTS+=("${t}_${m}")
                else
                    FAILED_NOWCASTS+=("${t}_${m}")
                fi
                # Remove from arrays
                NEW_PIDS=()
                for p in "${NOWCAST_PIDS[@]}"; do
                    [ "$p" != "$pid" ] && NEW_PIDS+=("$p")
                done
                NOWCAST_PIDS=("${NEW_PIDS[@]}")
                unset NOWCAST_PID_MAP[$pid]
                break
            fi
        done
        sleep 0.5
    done
    
    # Start new job in background
    (run_nowcast_combination "$target" "$model" "$current_combination") &
    pid=$!
    NOWCAST_PIDS+=("$pid")
    NOWCAST_PID_MAP[$pid]="$target|$model"
done

# Wait for all remaining jobs
for pid in "${NOWCAST_PIDS[@]}"; do
    # Wait for process (works even if already finished)
    wait "$pid" 2>/dev/null
    exit_code=$?
    task_for_pid="${NOWCAST_PID_MAP[$pid]}"
    if [ -n "$task_for_pid" ]; then
        IFS='|' read -r t m <<< "$task_for_pid"
        if [ $exit_code -eq 0 ]; then
            SUCCESSFUL_NOWCASTS+=("${t}_${m}")
        else
            FAILED_NOWCASTS+=("${t}_${m}")
        fi
    fi
done

echo ""
echo "Nowcast Summary:"
if [ ${#SUCCESSFUL_NOWCASTS[@]} -gt 0 ]; then
    echo "  ✓ Successful: ${SUCCESSFUL_NOWCASTS[@]}"
fi
if [ ${#FAILED_NOWCASTS[@]} -gt 0 ]; then
    echo "  ✗ Failed: ${FAILED_NOWCASTS[@]}"
fi
echo ""

# ============================================================================
# Final Summary
# ============================================================================

echo "=========================================="
echo "Final Summary"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved to:"
echo "  - Nowcast results: outputs/backtest/{target}_{model}_backtest.json"
echo ""
echo "Note: All results are in JSON/CSV format. Tables and plots are generated separately."
echo ""

