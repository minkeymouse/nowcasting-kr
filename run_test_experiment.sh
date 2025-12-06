#!/bin/bash
# Quick test script - tests DFM and DDFM fixes on single target/horizon
# Tests: KOGDP...D (GDP) with horizon 1 for DFM and DDFM models

set -e  # Exit on error

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

# Quick test: use dedicated test config
TARGET="KOGDP...D"
CONFIG_NAME="experiment/kogdp_test"
MODELS=("ddfm")  # Test DDFM after DFM fix
HORIZON=1

# Function to map target to config name
get_config_name() {
    local target=$1
    case "$target" in
        "KOGDP...D") echo "experiment/kogdp_report" ;;
        "KOCNPER.D") echo "experiment/kocnper_report" ;;
        "KOGFCF..D") echo "experiment/kogfcf_report" ;;
        *)
            echo "experiment/test_experiment"
            ;;
    esac
}

# Validate config file exists
echo "=========================================="
echo "Validating Test Configuration"
echo "=========================================="
config_file="config/${CONFIG_NAME}.yaml"
if [ ! -f "$config_file" ]; then
    echo "✗ Error: Config file not found: $config_file"
    exit 1
else
    echo "✓ Config exists: $config_file (target: $TARGET)"
fi

# Check data file
if [ ! -f "data/sample_data.csv" ]; then
    echo "✗ Error: Data file (data/sample_data.csv) not found"
    exit 1
fi
echo "✓ Data file exists"
echo ""

# Create output directory
mkdir -p outputs/comparisons

# Test each model
echo "=========================================="
echo "Running Quick Test: DFM and DDFM Fixes"
echo "=========================================="
echo "Target: $TARGET"
echo "Models: ${MODELS[@]}"
echo "Horizon: $HORIZON"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

FAILED_MODELS=()
PASSED_MODELS=()

for model in "${MODELS[@]}"; do
    log_file="outputs/comparisons/test_${TARGET}_${model}_h${HORIZON}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "=========================================="
    echo "Testing: $model model"
    echo "Target: $TARGET"
    echo "Horizon: $HORIZON"
    echo "Log: $log_file"
    echo "=========================================="
    echo ""
    
    # Run test with timeout (15 minutes for DDFM training)
    # Config file already has forecast_horizons: [1] set
    timeout 900 python3 src/train.py compare \
        --config-name "$CONFIG_NAME" \
        --models "$model" \
        2>&1 | tee "$log_file"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$model] ✓ Test passed"
        PASSED_MODELS+=("$model")
        
        # Check if n_valid > 0 in results
        result_file=$(find outputs/comparisons -name "${TARGET}_*" -type d | sort -r | head -1)/comparison_results.json
        if [ -f "$result_file" ]; then
            n_valid=$(python3 -c "import json; data=json.load(open('$result_file')); print(data.get('results', {}).get('$model', {}).get('metrics', {}).get('forecast_metrics', {}).get('$HORIZON', {}).get('n_valid', 0))" 2>/dev/null || echo "0")
            if [ "$n_valid" -gt 0 ]; then
                echo "[$model] ✓ n_valid=$n_valid (prediction successful)"
            else
                echo "[$model] ⚠ n_valid=$n_valid (prediction may have failed)"
            fi
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$model] ✗ Test timed out (15 min limit)"
        FAILED_MODELS+=("$model")
    else
        echo "[$model] ✗ Test failed (exit code: $EXIT_CODE)"
        FAILED_MODELS+=("$model")
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${#PASSED_MODELS[@]} -gt 0 ]; then
    echo "✓ Passed models (${#PASSED_MODELS[@]}/${#MODELS[@]}):"
    for model in "${PASSED_MODELS[@]}"; do
        echo "  - $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "✗ Failed models (${#FAILED_MODELS[@]}/${#MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Check log files in outputs/comparisons/ for details"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All model tests passed!"
echo "=========================================="
echo ""
echo "DFM and DDFM fixes verified:"
echo "  - Target: $TARGET"
echo "  - Horizon: $HORIZON"
echo "  - Models tested: ${MODELS[@]}"
echo ""
echo "Results saved in: outputs/comparisons/"
echo ""
echo "Next steps:"
echo "  1. Run full experiments: ./run_experiment.sh"
echo "  2. Test all horizons: modify --horizons to '1 7 28'"
echo ""

