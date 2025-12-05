#!/bin/bash
# Test experiment script - verifies all three target series experiments
# Tests: KOGDP...D (GDP), KOCNPER.D (Private Consumption), KOGFCF..D (Gross Fixed Capital Formation)
# As per nowcasting-report structure: 3 targets × 4 models × 3 horizons

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

# Three target series as per nowcasting-report
TARGETS=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")
CONFIG_NAMES=("experiment/kogdp_report" "experiment/kocnper_report" "experiment/kogfcf_report")

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

# Validate config files exist
echo "=========================================="
echo "Validating Test Configuration"
echo "=========================================="
MISSING_CONFIGS=0
for target in "${TARGETS[@]}"; do
    config_name=$(get_config_name "$target")
    config_file="config/${config_name}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "✗ Error: Config file not found: $config_file"
        MISSING_CONFIGS=$((MISSING_CONFIGS + 1))
    else
        echo "✓ Config exists: $config_file (target: $target)"
    fi
done

if [ $MISSING_CONFIGS -gt 0 ]; then
    echo "✗ Error: $MISSING_CONFIGS config file(s) missing"
    exit 1
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

# Test each target series
echo "=========================================="
echo "Running Test Experiments for All Targets"
echo "=========================================="
echo "Targets: ${TARGETS[@]}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

FAILED_TARGETS=()
PASSED_TARGETS=()

for i in "${!TARGETS[@]}"; do
    target="${TARGETS[$i]}"
    config_name="${CONFIG_NAMES[$i]}"
    log_file="outputs/comparisons/test_${target}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "=========================================="
    echo "Testing: $target"
    echo "Config: $config_name"
    echo "Log: $log_file"
    echo "=========================================="
    echo ""
    
    # Run test with timeout (30 minutes per target for quick validation)
    timeout 1800 python3 src/train.py compare \
        --config-name "$config_name" \
        2>&1 | tee "$log_file"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$target] ✓ Test passed"
        PASSED_TARGETS+=("$target")
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$target] ✗ Test timed out (30 min limit)"
        FAILED_TARGETS+=("$target")
    else
        echo "[$target] ✗ Test failed (exit code: $EXIT_CODE)"
        FAILED_TARGETS+=("$target")
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${#PASSED_TARGETS[@]} -gt 0 ]; then
    echo "✓ Passed targets (${#PASSED_TARGETS[@]}/${#TARGETS[@]}):"
    for target in "${PASSED_TARGETS[@]}"; do
        echo "  - $target"
    done
fi

if [ ${#FAILED_TARGETS[@]} -gt 0 ]; then
    echo "✗ Failed targets (${#FAILED_TARGETS[@]}/${#TARGETS[@]}):"
    for target in "${FAILED_TARGETS[@]}"; do
        echo "  - $target"
    done
    echo ""
    echo "Check log files in outputs/comparisons/ for details"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All test experiments passed!"
echo "=========================================="
echo ""
echo "All three target series are available and working:"
echo "  - KOGDP...D (GDP)"
echo "  - KOCNPER.D (Private Consumption)"
echo "  - KOGFCF..D (Gross Fixed Capital Formation)"
echo ""
echo "Results saved in: outputs/comparisons/"
echo ""
echo "Next steps:"
echo "  1. Run full experiments: ./run_experiment.sh"
echo "  2. Check results align with nowcasting-report structure"
echo ""

