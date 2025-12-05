#!/bin/bash
# Test experiment script using test_experiment.yaml
# Runs all models specified in the test config for quick validation

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Config name (experiment config file)
CONFIG_NAME="experiment/test_experiment"

echo "=========================================="
echo "Running Test Experiment"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "=========================================="
echo ""

# Run comparison using the test experiment config
# The config contains all necessary information: models, target_series, horizons, etc.
python3 src/train.py compare \
    --config-name "$CONFIG_NAME" \
    2>&1 | tee "outputs/comparisons/test_experiment_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test experiment completed successfully"
    echo "=========================================="
    echo ""
    echo "Results saved in: outputs/comparisons/"
else
    echo ""
    echo "=========================================="
    echo "✗ Test experiment failed (exit code: $EXIT_CODE)"
    echo "=========================================="
    echo ""
    echo "Check the log file for details"
    exit $EXIT_CODE
fi

