#!/bin/bash
# Comprehensive test script - verifies all targets and models before full run
# Tests all 4 targets × 4 models with horizon 1 for quick verification
# Can be customized via environment variables

set +e  # Don't exit on error - continue testing other models/targets

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "✗ Error: Virtual environment (.venv) not found"
    echo "  Create it with: python3 -m venv .venv"
    exit 1
fi
source .venv/bin/activate

# All targets to test
ALL_TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G" "KOMPRI30G")

# Targets to test (can be overridden via TARGETS environment variable)
# Format: TARGETS="KOEQUIPTE KOWRCCNSE" bash run_test_experiment.sh
if [ -n "$TARGETS" ]; then
    read -ra TARGETS_ARRAY <<< "$TARGETS"
    TEST_TARGETS=("${TARGETS_ARRAY[@]}")
else
    TEST_TARGETS=("${ALL_TARGETS[@]}")  # Test all targets by default
fi

# Function to map target to config name
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
        "KOMPRI30G")
            echo "experiment/kompri30g_report"
            ;;
        *)
            echo "experiment/test_experiment"
            ;;
    esac
}

# Models to test (can be overridden via MODELS environment variable)
if [ -n "$MODELS" ]; then
    read -ra MODELS_ARRAY <<< "$MODELS"
    TEST_MODELS=("${MODELS_ARRAY[@]}")
else
    TEST_MODELS=("arima" "var" "dfm" "ddfm")  # Test all models
fi

# Horizon for testing (horizon 1 is fastest)
TEST_HORIZON=1

# Maximum parallel targets to test (to avoid resource exhaustion)
MAX_PARALLEL_TARGETS=2

# Results tracking file
RESULTS_FILE="/tmp/test_experiment_results_$$.json"

# Initialize results file
cat > "$RESULTS_FILE" <<EOF
{
  "total_tests": 0,
  "passed_tests": 0,
  "failed_tests": 0,
  "targets": {},
  "models": {}
}
EOF

# Validate environment
echo "=========================================="
echo "Validating Test Environment"
echo "=========================================="

# Check data file (try both possible names)
if [ -f "data/data.csv" ]; then
    DATA_FILE="data/data.csv"
    echo "✓ Data file exists: $DATA_FILE"
elif [ -f "data/sample_data.csv" ]; then
    DATA_FILE="data/sample_data.csv"
    echo "✓ Data file exists: $DATA_FILE"
else
    echo "✗ Error: Data file not found (checked: data/data.csv, data/sample_data.csv)"
    exit 1
fi

# Validate config files
missing_configs=0
for target in "${TEST_TARGETS[@]}"; do
    config_name=$(get_config_name "$target")
    config_file="config/${config_name}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "✗ Error: Config file not found: $config_file"
        missing_configs=$((missing_configs + 1))
    else
        echo "✓ Config exists: $config_file (target: $target)"
    fi
done

if [ $missing_configs -gt 0 ]; then
    echo "✗ Error: $missing_configs config file(s) missing"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Configuration"
echo "=========================================="
echo "Targets to test: ${TEST_TARGETS[@]}"
echo "Models to test: ${TEST_MODELS[@]}"
echo "Horizon: $TEST_HORIZON"
echo "Max parallel targets: $MAX_PARALLEL_TARGETS"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Create output directory
mkdir -p outputs/comparisons

# Function to update results in JSON file
update_result() {
    local target=$1
    local model=$2
    local status=$3  # "passed", "warning", or "failed"
    local n_valid=${4:-0}
    
    python3 <<PYTHON_SCRIPT
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
except:
    results = {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "targets": {}, "models": {}}

results["total_tests"] = results.get("total_tests", 0) + 1

if "$status" == "passed" or "$status" == "warning":
    results["passed_tests"] = results.get("passed_tests", 0) + 1
else:
    results["failed_tests"] = results.get("failed_tests", 0) + 1

# Update target status
if "$target" not in results["targets"]:
    results["targets"]["$target"] = {"passed": 0, "failed": 0, "total": 0}

results["targets"]["$target"]["total"] += 1
if "$status" == "passed" or "$status" == "warning":
    results["targets"]["$target"]["passed"] += 1
else:
    results["targets"]["$target"]["failed"] += 1

# Update model status
model_key = "${target}_${model}"
results["models"][model_key] = {
    "target": "$target",
    "model": "$model",
    "status": "$status",
    "n_valid": $n_valid
}

with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
PYTHON_SCRIPT
}

# Function to test a single model for a target
test_model() {
    local target=$1
    local model=$2
    local config_name=$(get_config_name "$target")
    
    local log_file="outputs/comparisons/test_${target}_${model}_h${TEST_HORIZON}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "[$target/$model] Starting test..."
    echo "  Config: $config_name"
    echo "  Log: $log_file"
    
    # Set timeout based on model (DDFM takes longer)
    local timeout_seconds=900  # 15 minutes default
    if [ "$model" = "ddfm" ]; then
        timeout_seconds=1800  # 30 minutes for DDFM
    elif [ "$model" = "dfm" ]; then
        timeout_seconds=1200  # 20 minutes for DFM
    fi
    
    # Run test
    timeout $timeout_seconds python3 src/train.py compare \
        --config-name "$config_name" \
        --models "$model" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Check if n_valid > 0 in results
        local result_file=$(find outputs/comparisons -name "${target}_*" -type d | sort -r | head -1)/comparison_results.json
        local n_valid=0
        
        if [ -f "$result_file" ]; then
            n_valid=$(python3 -c "
import json
try:
    with open('$result_file', 'r') as f:
        data = json.load(f)
    result = data.get('results', {}).get('$model', {})
    metrics = result.get('metrics', {})
    forecast_metrics = metrics.get('forecast_metrics', {})
    n_valid = forecast_metrics.get('$TEST_HORIZON', {}).get('n_valid', 0)
    print(n_valid)
except:
    print(0)
" 2>/dev/null || echo "0")
        fi
        
        if [ "$n_valid" -gt 0 ]; then
            echo "[$target/$model] ✓ PASSED (n_valid=$n_valid)"
            update_result "$target" "$model" "passed" "$n_valid"
            return 0
        else
            echo "[$target/$model] ⚠ PASSED but n_valid=0 (check log: $log_file)"
            update_result "$target" "$model" "warning" "0"
            return 0
        fi
    elif [ $exit_code -eq 124 ]; then
        echo "[$target/$model] ✗ TIMEOUT (${timeout_seconds}s limit, check log: $log_file)"
        update_result "$target" "$model" "failed" "0"
        return 1
    else
        echo "[$target/$model] ✗ FAILED (exit code: $exit_code, check log: $log_file)"
        update_result "$target" "$model" "failed" "0"
        return 1
    fi
}

# Function to test all models for a target
test_target() {
    local target=$1
    echo ""
    echo "=========================================="
    echo "Testing Target: $target"
    echo "=========================================="
    
    local target_passed=0
    local target_failed=0
    
    for model in "${TEST_MODELS[@]}"; do
        if test_model "$target" "$model"; then
            target_passed=$((target_passed + 1))
        else
            target_failed=$((target_failed + 1))
        fi
    done
    
    if [ $target_failed -eq 0 ]; then
        echo ""
        echo "[$target] ✓ All models passed ($target_passed/${#TEST_MODELS[@]})"
    else
        echo ""
        echo "[$target] ✗ Some models failed ($target_passed passed, $target_failed failed)"
    fi
}

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_TARGETS ]; do
        sleep 5
    done
}

# Run tests
echo "=========================================="
echo "Running Tests"
echo "=========================================="
echo ""

# Test targets (with parallel limit)
for target in "${TEST_TARGETS[@]}"; do
    wait_for_slot
    test_target "$target" &
done

# Wait for all background jobs
echo ""
echo "Waiting for all tests to complete..."
echo ""

# Monitor progress
MONITOR_INTERVAL=30  # Check every 30 seconds
while [ $(jobs -r | wc -l) -gt 0 ]; do
    running=$(jobs -r | wc -l)
    completed=$((${#TEST_TARGETS[@]} - running))
    echo "[$(date '+%H:%M:%S')] Progress: $completed/${#TEST_TARGETS[@]} targets completed, $running running..."
    sleep $MONITOR_INTERVAL
done

wait

# Load final results
FINAL_RESULTS=$(python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    print(json.dumps(json.load(f)))
" 2>/dev/null || echo '{"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "targets": {}, "models": {}}')

TOTAL_TESTS=$(echo "$FINAL_RESULTS" | python3 -c "import json, sys; print(json.load(sys.stdin).get('total_tests', 0))")
PASSED_TESTS=$(echo "$FINAL_RESULTS" | python3 -c "import json, sys; print(json.load(sys.stdin).get('passed_tests', 0))")
FAILED_TESTS=$(echo "$FINAL_RESULTS" | python3 -c "import json, sys; print(json.load(sys.stdin).get('failed_tests', 0))")

# Final Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Overall Results:"
echo "  Total tests: $TOTAL_TESTS"
echo "  Passed: $PASSED_TESTS"
echo "  Failed: $FAILED_TESTS"
echo ""

echo "Target Results:"
for target in "${TEST_TARGETS[@]}"; do
    target_status=$(echo "$FINAL_RESULTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
target = data.get('targets', {}).get('$target', {})
passed = target.get('passed', 0)
failed = target.get('failed', 0)
total = target.get('total', 0)
if failed == 0 and total > 0:
    print('passed')
elif passed > 0:
    print('partial')
else:
    print('failed')
")
    if [ "$target_status" = "passed" ]; then
        echo "  ✓ $target: PASSED"
    elif [ "$target_status" = "partial" ]; then
        echo "  ⚠ $target: PARTIAL (some models passed)"
    else
        echo "  ✗ $target: FAILED"
    fi
done
echo ""

echo "Model Results (by target):"
for target in "${TEST_TARGETS[@]}"; do
    echo "  $target:"
    for model in "${TEST_MODELS[@]}"; do
        model_status=$(echo "$FINAL_RESULTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
model_key = '${target}_${model}'
model_data = data.get('models', {}).get(model_key, {})
status = model_data.get('status', 'unknown')
n_valid = model_data.get('n_valid', 0)
print(f'{status}:{n_valid}')
" 2>/dev/null || echo "unknown:0")
        status_part=$(echo "$model_status" | cut -d: -f1)
        n_valid=$(echo "$model_status" | cut -d: -f2)
        case "$status_part" in
            "passed")
                echo "    ✓ $model: PASSED (n_valid=$n_valid)"
                ;;
            "warning")
                echo "    ⚠ $model: PASSED (n_valid=0)"
                ;;
            "failed")
                echo "    ✗ $model: FAILED"
                ;;
            *)
                echo "    ? $model: NOT TESTED"
                ;;
        esac
    done
done
echo ""

# Clean up results file
rm -f "$RESULTS_FILE"

# Determine overall status
if [ $FAILED_TESTS -eq 0 ] && [ $TOTAL_TESTS -gt 0 ]; then
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
    echo ""
    echo "Ready for full experiment run:"
    echo "  ./run_experiment.sh"
    echo ""
    exit 0
else
    echo "=========================================="
    echo "✗ Some tests failed"
    echo "=========================================="
    echo ""
    echo "Review failed tests before running full experiment:"
    echo "  - Check logs in: outputs/comparisons/test_*.log"
    echo "  - Fix issues and re-run: ./run_test_experiment.sh"
    echo ""
    echo "To test specific targets/models:"
    echo "  TARGETS=\"KOEQUIPTE\" MODELS=\"arima var\" ./run_test_experiment.sh"
    echo ""
    exit 1
fi
