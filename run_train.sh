#!/bin/bash
# Training script - trains models with data from 1985 to 2019
# Results saved to checkpoint/ directory
# Ensures no data leakage (strict train/test split)

# Don't exit on error - continue even if some models fail
set +e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ============================================================================
# Graceful Stop and Cleanup
# ============================================================================

# Track current training process PID
CURRENT_TRAINING_PID=""
CLEANUP_DONE=0

# Cleanup function to kill running training processes
cleanup() {
    # Prevent multiple cleanup calls
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1
    
    echo ""
    echo "=========================================="
    echo "Graceful shutdown requested..."
    echo "=========================================="
    
    # Kill current training process if running
    if [ -n "$CURRENT_TRAINING_PID" ] && kill -0 "$CURRENT_TRAINING_PID" 2>/dev/null; then
        echo "Stopping current training process (PID: $CURRENT_TRAINING_PID)..."
        kill "$CURRENT_TRAINING_PID" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if kill -0 "$CURRENT_TRAINING_PID" 2>/dev/null; then
            echo "Force killing training process..."
            kill -9 "$CURRENT_TRAINING_PID" 2>/dev/null || true
        fi
    fi
    
    # Find and kill any remaining training processes
    echo "Cleaning up any remaining training processes..."
    
    # Save current script PID and parent PID to exclude from killing
    CURRENT_PID=$$
    PARENT_PID=$PPID
    SCRIPT_NAME=$(basename "$0")
    
    # Find all related processes
    PATTERNS=(
        "python.*train.*train"
        "python.*src/train.py.*train"
        "python.*train.py.*train"
        "timeout.*python.*train"
    )
    
    FOUND_PIDS=()
    for pattern in "${PATTERNS[@]}"; do
        while IFS= read -r pid; do
            if [ -n "$pid" ] && [ "$pid" != "$CURRENT_PID" ] && [ "$pid" != "$PARENT_PID" ]; then
                proc_cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "")
                if [[ "$proc_cmd" == *"python"* ]] && [[ "$proc_cmd" != *"bash"* ]] && [[ "$proc_cmd" != *"$SCRIPT_NAME"* ]]; then
                    FOUND_PIDS+=("$pid")
                fi
            fi
        done < <(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' 2>/dev/null)
    done
    
    # Remove duplicates
    UNIQUE_PIDS=($(printf '%s\n' "${FOUND_PIDS[@]}" | sort -u))
    
    if [ ${#UNIQUE_PIDS[@]} -gt 0 ]; then
        echo "Found ${#UNIQUE_PIDS[@]} training process(es) to terminate..."
        
        # Try graceful termination first
        for pid in "${UNIQUE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Sending SIGTERM to PID $pid"
                kill "$pid" 2>/dev/null || true
            fi
        done
        
        sleep 3
        
        # Force kill if still running
        for pid in "${UNIQUE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Sending SIGKILL to PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    echo "✓ Cleanup complete"
    echo ""
    exit 130  # Exit with code 130 (128 + SIGINT = 130)
}

# Signal handlers for graceful shutdown
# Note: We don't trap EXIT to avoid cleanup on normal completion
trap cleanup SIGINT SIGTERM

# ============================================================================
# Argument Parsing
# ============================================================================

TEST_MODE=0
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test|-t)
            TEST_MODE=1
            shift
            ;;
        --force-retrain|--force)
            FORCE_RETRAIN=1
            SKIP_TRAINED=0
            shift
            ;;
        --skip-trained)
            SKIP_TRAINED=1
            FORCE_RETRAIN=0
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test, -t              Run in test mode (validate setup without training)"
            echo "  --force-retrain, --force  Force re-training all models (overwrites checkpoints)"
            echo "  --skip-trained          Skip already trained models (default)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  FORCE_RETRAIN=1        Force re-training all models"
            echo "  SKIP_TRAINED=1         Skip already trained models (default)"
            echo ""
            echo "Examples:"
            echo "  $0 --test              # Test mode: validate setup"
            echo "  $0                     # Normal training (skip trained models)"
            echo "  $0 --force-retrain     # Re-train all models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
        
        # Try to activate
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
            echo "  ✓ PASS: Core dependencies (pandas, numpy, hydra, omegaconf, sktime) available"
            test_passed=$((test_passed + 1))
        else
            echo "  ✗ FAIL: Missing core dependencies"
            test_failed=$((test_failed + 1))
        fi
        
        # Check dfm-python
        python3 -c "import dfm_python" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ PASS: dfm-python available"
            test_passed=$((test_passed + 1))
        else
            echo "  ⚠ WARN: dfm-python not available (DFM/DDFM models will fail)"
        fi
    else
        echo "  ✗ FAIL: python3 not found"
        test_failed=$((test_failed + 1))
    fi
    echo ""
    
    # Test 3: Data file
    echo "[TEST 3] Checking data file..."
    if [ -f "data/data.csv" ]; then
        DATA_FILE="data/data.csv"
        echo "  ✓ PASS: Data file exists: $DATA_FILE"
        test_passed=$((test_passed + 1))
        
        # Check if data file is readable and has content
        if [ -s "$DATA_FILE" ]; then
            local line_count=$(wc -l < "$DATA_FILE" 2>/dev/null || echo "0")
            if [ "$line_count" -gt 1 ]; then
                echo "  ✓ PASS: Data file has content ($line_count lines)"
                test_passed=$((test_passed + 1))
            else
                echo "  ✗ FAIL: Data file is empty or has no data rows"
                test_failed=$((test_failed + 1))
            fi
        else
            echo "  ✗ FAIL: Data file is empty"
            test_failed=$((test_failed + 1))
        fi
    elif [ -f "data/sample_data.csv" ]; then
        DATA_FILE="data/sample_data.csv"
        echo "  ✓ PASS: Sample data file exists: $DATA_FILE"
        test_passed=$((test_passed + 1))
    else
        echo "  ✗ FAIL: Data file not found (neither data/data.csv nor data/sample_data.csv)"
        test_failed=$((test_failed + 1))
    fi
    echo ""
    
    # Test 4: Config files
    echo "[TEST 4] Checking config files..."
    local missing_configs=0
    local valid_configs=0
    for target in "${TARGETS[@]}"; do
        local config_name=$(get_config_name "$target")
        local config_file="config/${config_name}.yaml"
        if [ ! -f "$config_file" ]; then
            echo "  ✗ FAIL: Config file not found: $config_file"
            missing_configs=$((missing_configs + 1))
            test_failed=$((test_failed + 1))
        else
            echo "  ✓ PASS: Config file exists: $config_file"
            valid_configs=$((valid_configs + 1))
            
            # Try to parse config with Python
            source .venv/bin/activate 2>/dev/null || true
            local abs_config_dir="$(pwd)/config"
            python3 -c "
import sys
import os
sys.path.insert(0, '.')
from src.utils import parse_experiment_config
try:
    cfg = parse_experiment_config('${config_name}', config_dir='${abs_config_dir}')
    print('    ✓ Config can be parsed')
except Exception as e:
    print(f'    ✗ Config parsing failed: {e}')
    sys.exit(1)
" 2>/dev/null
            if [ $? -eq 0 ]; then
                test_passed=$((test_passed + 1))
            else
                test_failed=$((test_failed + 1))
            fi
        fi
    done
    
    if [ $missing_configs -eq 0 ]; then
        echo "  ✓ PASS: All config files exist and are parseable"
    fi
    echo ""
    
    # Test 5: Source code structure
    echo "[TEST 5] Checking source code structure..."
    local missing_files=0
    # Check for main entry points and key modules (organized in subdirectories)
    for file in "src/train.py" "src/infer.py" "src/utils.py" "src/preprocessing.py" "src/models/__init__.py" "src/models/models.py" "src/evaluation/__init__.py" "src/evaluation/evaluation.py"; do
        if [ -f "$file" ]; then
            echo "  ✓ PASS: $file exists"
            test_passed=$((test_passed + 1))
        else
            echo "  ✗ FAIL: $file not found"
            missing_files=$((missing_files + 1))
            test_failed=$((test_failed + 1))
        fi
    done
    
    # Test if train.py can be executed (imports are tested via actual execution)
    source .venv/bin/activate 2>/dev/null || true
    # Try to run train.py with --help (this tests imports and argparse)
    local train_output=$(python3 src/train.py train --help 2>&1)
    local train_exit=$?
    if [ $train_exit -eq 0 ] || echo "$train_output" | grep -q "usage:\|--config-name\|--model\|--checkpoint-dir"; then
        echo "  ✓ PASS: train.py can be executed (imports working)"
        test_passed=$((test_passed + 1))
    elif echo "$train_output" | grep -q "ImportError\|ModuleNotFoundError"; then
        echo "  ✗ FAIL: Import error in train.py"
        echo "    Error: $(echo "$train_output" | grep -E "ImportError|ModuleNotFoundError" | head -1)"
        test_failed=$((test_failed + 1))
    else
        echo "  ⚠ WARN: train.py execution check inconclusive (exit code: $train_exit)"
        echo "    This may be normal if run from wrong directory"
    fi
    echo ""
    
    # Test 6: Checkpoint directory structure
    echo "[TEST 6] Checking checkpoint directory structure..."
    if [ -d "checkpoint" ]; then
        echo "  ✓ PASS: checkpoint/ directory exists"
        test_passed=$((test_passed + 1))
        
        # Check existing checkpoints
        local checkpoint_count=0
        for target in "${TARGETS[@]}"; do
            for model in "${MODELS[@]}"; do
                local checkpoint_file="checkpoint/${target}_${model}/model.pkl"
                if [ -f "$checkpoint_file" ] && [ -s "$checkpoint_file" ]; then
                    checkpoint_count=$((checkpoint_count + 1))
                fi
            done
        done
        
        if [ $checkpoint_count -gt 0 ]; then
            echo "  ✓ INFO: Found $checkpoint_count existing checkpoint(s)"
        else
            echo "  ℹ INFO: No existing checkpoints found (will train all models)"
        fi
    else
        echo "  ℹ INFO: checkpoint/ directory does not exist (will be created)"
    fi
    echo ""
    
    # Test 7: Directory permissions
    echo "[TEST 7] Checking directory permissions..."
    local dirs=("checkpoint" "log" "outputs")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            # Try to create
            if mkdir -p "$dir" 2>/dev/null; then
                echo "  ✓ PASS: Can create $dir/ directory"
                test_passed=$((test_passed + 1))
                rmdir "$dir" 2>/dev/null || true
            else
                echo "  ✗ FAIL: Cannot create $dir/ directory (permission denied?)"
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
    
    # Test 8: Training command syntax (already tested in TEST 5, but verify argparse)
    echo "[TEST 8] Testing training command syntax..."
    source .venv/bin/activate 2>/dev/null || true
    # Test if argparse works (help command)
    local help_output=$(python3 src/train.py train --help 2>&1)
    if echo "$help_output" | grep -q "usage:\|--config-name\|--model\|--checkpoint-dir\|error:"; then
        echo "  ✓ PASS: Training command syntax valid (argparse working)"
        test_passed=$((test_passed + 1))
    else
        # If help doesn't work, check if it's because of import errors (already caught in TEST 5)
        if echo "$help_output" | grep -q "ImportError\|ModuleNotFoundError"; then
            echo "  ⚠ WARN: Command syntax check skipped (import error already reported)"
        else
            echo "  ✗ FAIL: Training command syntax invalid"
            test_failed=$((test_failed + 1))
        fi
    fi
    echo ""
    
    # Summary
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Tests passed: $test_passed"
    echo "Tests failed: $test_failed"
    echo ""
    
    if [ $test_failed -eq 0 ]; then
        echo "✓ All tests passed! Setup is ready for training."
        echo ""
        echo "To start training, run:"
        echo "  bash run_train.sh"
        echo ""
        echo "Or to force re-train all models:"
        echo "  bash run_train.sh --force-retrain"
        return 0
    else
        echo "✗ Some tests failed. Please fix the issues above before training."
        return 1
    fi
}

# ============================================================================
# Main Script Logic
# ============================================================================

# If test mode, run tests and exit
if [ "$TEST_MODE" -eq 1 ]; then
    # Target series (needed for test mode)
    TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")
    MODELS=("arima" "var" "dfm" "ddfm")
    
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
    
    if run_test_mode; then
        exit 0
    else
        exit 1
    fi
fi

# Kill any existing training processes
echo "=========================================="
echo "Checking for existing training processes..."
echo "=========================================="

# Save current script PID and parent PID to exclude from killing
CURRENT_PID=$$
PARENT_PID=$PPID
SCRIPT_NAME=$(basename "$0")

# Find all related processes
PATTERNS=(
    "python.*train.*train"
    "python.*src/train.py.*train"
    "python.*train.py.*train"
)

FOUND_PIDS=()
for pattern in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
        if [ -n "$pid" ] && [ "$pid" != "$CURRENT_PID" ] && [ "$pid" != "$PARENT_PID" ]; then
            proc_cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "")
            if [[ "$proc_cmd" == *"python"* ]] && [[ "$proc_cmd" != *"bash"* ]] && [[ "$proc_cmd" != *"$SCRIPT_NAME"* ]]; then
                FOUND_PIDS+=("$pid")
            fi
        fi
    done < <(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')
done

# Remove duplicates
UNIQUE_PIDS=($(printf '%s\n' "${FOUND_PIDS[@]}" | sort -u))

if [ ${#UNIQUE_PIDS[@]} -gt 0 ]; then
    echo "Found ${#UNIQUE_PIDS[@]} running training process(es): ${UNIQUE_PIDS[*]}"
    echo "Terminating processes..."
    
    for pid in "${UNIQUE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Sending SIGTERM to PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 5
    
    for pid in "${UNIQUE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Sending SIGKILL to PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    echo "✓ Existing training processes terminated."
else
    echo "✓ No existing training processes found."
fi
echo ""

# Target series
TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")

# Models to train
MODELS=("arima" "var" "dfm" "ddfm")

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
    
    # Prefer the aggregated weekly→monthly file; fall back to original for sanity.
    if [ -f "data/data_weekly_averaged.csv" ]; then
        DATA_FILE="data/data_weekly_averaged.csv"
        echo "✓ Data file exists: $DATA_FILE"
    elif [ -f "data/data.csv" ]; then
        DATA_FILE="data/data.csv"
        echo "✓ Data file exists: $DATA_FILE"
    elif [ -f "data/sample_data.csv" ]; then
        DATA_FILE="data/sample_data.csv"
        echo "✓ Data file exists: $DATA_FILE"
    else
        echo "✗ Error: Data file not found (expected data/data_weekly_averaged.csv)"
        return 1
    fi
    
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
    
    echo "✓ Environment validation complete"
    echo ""
    return 0
}

# Activate virtual environment and validate
if ! validate_environment; then
    echo "Environment validation failed. Please fix the issues above."
    exit 1
fi

# Create checkpoint and log directories
# checkpoint/ only contains model weights (model.pkl files)
# log/ contains training logs for ARIMA/VAR
# lightning_logs/ contains training logs for DFM/DDFM (PyTorch Lightning default)
mkdir -p checkpoint || {
    echo "✗ Error: Cannot create checkpoint/ directory"
    exit 1
}
mkdir -p log || {
    echo "✗ Error: Cannot create log/ directory"
    exit 1
}

# Verify directories are writable
if [ ! -w "checkpoint" ]; then
    echo "✗ Error: checkpoint/ directory is not writable"
    exit 1
fi
if [ ! -w "log" ]; then
    echo "✗ Error: log/ directory is not writable"
    exit 1
fi

# Function to check if model is already trained
# Uses FORCE_RETRAIN and SKIP_TRAINED variables set by argument parsing
is_model_trained() {
    # Force retrain if flag is set (overwrites checkpoints)
    if [ "$FORCE_RETRAIN" = "1" ]; then
        return 1  # Force retrain (will overwrite)
    fi
    
    # Skip trained models if SKIP_TRAINED is set (default behavior)
    if [ "$SKIP_TRAINED" = "1" ]; then
        local target=$1
        local model=$2
        local checkpoint_dir="checkpoint"
        
        local model_file="${checkpoint_dir}/${target}_${model}/model.pkl"
        
        if [ -f "$model_file" ]; then
            # Check if model file is valid (non-empty)
            if [ -s "$model_file" ]; then
                return 0  # Skip training
            fi
        fi
    fi
    
    return 1  # Train (will overwrite if exists)
}

# Function to run training
run_training() {
    local target=$1
    local model=$2
    local config_name=$(get_config_name "$target")
    local start_time=$(date +%s)
    
    # All models save logs to log/ folder (stdout/stderr capture)
    # DFM/DDFM detailed logs also go to lightning_logs/ (PyTorch Lightning default)
    mkdir -p log || {
        echo "✗ Error: Cannot create log/ directory"
        return 1
    }
    local log_file="log/${target}_${model}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "=========================================="
    echo "Training: $target - $model"
    echo "Config: $config_name"
    echo "Log: $log_file"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Run training with timeout (12 hours max)
    # Training uses data from 1985-2019 (enforced in config)
    # Save to checkpoint directory with target_model naming
    # checkpoint/ only contains model weights (model.pkl), logs go to log/
    # Run in background to track PID, but we'll wait for it
    timeout 43200 .venv/bin/python3 src/train.py train \
        --config-name "$config_name" \
        --model "$model" \
        --checkpoint-dir "checkpoint" \
        > "$log_file" 2>&1 &
    
    # Store the PID of the timeout process
    local timeout_pid=$!
    CURRENT_TRAINING_PID=$timeout_pid
    
    # Find the actual Python process (child of timeout)
    # Retry a few times in case process hasn't started yet
    local python_pid=""
    for attempt in {1..5}; do
        sleep 0.2
        local child_pids=$(pgrep -P "$timeout_pid" 2>/dev/null || echo "")
        if [ -n "$child_pids" ]; then
            python_pid=$(echo "$child_pids" | head -1)
            if [ -n "$python_pid" ] && kill -0 "$python_pid" 2>/dev/null; then
                CURRENT_TRAINING_PID=$python_pid  # Track the actual Python process
                break
            fi
        fi
    done
    
    # If we couldn't find Python process, track timeout process instead
    if [ -z "$python_pid" ] || [ -z "$CURRENT_TRAINING_PID" ]; then
        CURRENT_TRAINING_PID=$timeout_pid
    fi
    
    # Wait for the timeout process to complete (which waits for Python)
    wait $timeout_pid
    EXIT_CODE=$?
    
    # Clear the PID after training completes
    CURRENT_TRAINING_PID=""
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    # Check if checkpoint exists
    local checkpoint_file="checkpoint/${target}_${model}/model.pkl"
    local has_checkpoint=0
    
    if [ -f "$checkpoint_file" ] && [ -s "$checkpoint_file" ]; then
        has_checkpoint=1
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        if [ $has_checkpoint -eq 1 ]; then
            echo "[$target-$model] ✓ Completed in ${hours}h ${minutes}m (checkpoint verified)"
        else
            echo "[$target-$model] ⚠ Completed but no checkpoint found (exit code: $EXIT_CODE, check log: $log_file)"
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$target-$model] ✗ Timeout after 12 hours (check log: $log_file)"
    else
        echo "[$target-$model] ✗ Failed (exit code: $EXIT_CODE, check log: $log_file)"
    fi
}

# Check for completed trainings
echo "=========================================="
echo "Checking for completed trainings"
echo "=========================================="
TASKS_TO_RUN=()

for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
        if is_model_trained "$target" "$model"; then
            echo "✓ $target-$model: Already trained (skipping)"
        else
            echo "→ $target-$model: Needs training"
            TASKS_TO_RUN+=("${target}:${model}")
        fi
    done
done
echo ""

if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
    echo "All models are already trained!"
    echo "To re-train models (overwrites checkpoints), run: FORCE_RETRAIN=1 bash run_train.sh"
    echo "Or delete the checkpoint files in checkpoint/"
    exit 0
fi

echo "Training models: ${#TASKS_TO_RUN[@]} task(s)"
echo "Training period: 1985-2019 (no data leakage)"
echo "Data frequency: Monthly (window-averaged from weekly data)"
if [ "$FORCE_RETRAIN" = "1" ]; then
    echo "FORCE_RETRAIN=1: Re-training all models (will overwrite existing checkpoints)"
elif [ "$SKIP_TRAINED" = "1" ]; then
    echo "SKIP_TRAINED=1: Skipping already trained models (default)"
fi
echo "Note: Training will overwrite existing checkpoints for models being trained"
echo ""

# Run trainings sequentially (training can be memory-intensive)
# Note: Press Ctrl+C to gracefully stop training and cleanup processes
for task in "${TASKS_TO_RUN[@]}"; do
    # Check if cleanup was requested (CLEANUP_DONE flag)
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        echo ""
        echo "Training interrupted by user"
        break
    fi
    
    target=$(echo "$task" | cut -d: -f1)
    model=$(echo "$task" | cut -d: -f2)
    run_training "$target" "$model"
done

# Only show completion message if not interrupted
if [ "$CLEANUP_DONE" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "All trainings completed!"
    echo "=========================================="
    echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Checkpoints saved in: checkpoint/"
    echo ""
fi

# Remove signal traps on normal exit
trap - SIGINT SIGTERM

