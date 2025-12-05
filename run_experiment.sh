#!/bin/bash
# High-priority experiments for RESULTS_NEEDED.md
# Run all experiments for model comparison

# Don't exit on error - continue even if some models fail
set +e

# Activate virtual environment
source .venv/bin/activate

# Target series
TARGETS=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")

# Models to compare (9 models as per RESULTS_NEEDED.md)
# Note: Some models may not be implemented yet - they will be skipped if config not found
MODELS=("arima" "var" "vecm" "dfm" "ddfm" "xgboost" "lightgbm" "deepar" "tft")

# Horizons
HORIZONS=(1 7 28)

echo "=========================================="
echo "Running High-Priority Experiments"
echo "=========================================="
echo "Target Series: ${TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Horizons: ${HORIZONS[@]}"
echo "=========================================="
echo ""

# Run comparison for each target series
for target in "${TARGETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing: $target"
    echo "=========================================="
    echo ""
    
    python3 src/training.py compare \
        --target-series "$target" \
        --models "${MODELS[@]}" \
        --horizons "${HORIZONS[@]}"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Completed: $target"
    else
        echo ""
        echo "⚠ Completed with errors: $target (check output above)"
    fi
    echo ""
done

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

