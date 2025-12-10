#!/bin/bash
# Forecasting and report generation script
# 
# This script:
# 1. Evaluates trained models on test data (2024-2025) and generates comparison results
# 2. Aggregates results into CSV format
# 3. Generates plots for the report
# 4. Generates LaTeX tables for the report
#
# Prerequisites:
# - Models must be trained first (run run_train.sh)
# - Checkpoints must exist in checkpoints/{target}_{model}/model.pkl

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: .venv"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated virtual environment: venv"
fi

# Use python3 if python is not available
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3
    else
        echo "Error: python or python3 not found"
        exit 1
    fi
fi

# Create output directories
mkdir -p "$PROJECT_ROOT/outputs/comparisons"
mkdir -p "$PROJECT_ROOT/outputs/experiments"
mkdir -p "$PROJECT_ROOT/nowcasting-report/images"
mkdir -p "$PROJECT_ROOT/nowcasting-report/tables"

# Target variables and their config files
declare -A TARGETS=(
    ["KOIPALL.G"]="experiment/production_koipallg_report"
    ["KOEQUIPTE"]="experiment/investment_koequipte_report"
    ["KOWRCCNSE"]="experiment/consumption_kowrccnse_report"
)

# Models to evaluate (dfm, arima, var)
MODELS=("dfm" "arima" "var")

echo "=========================================="
echo "Forecasting and Report Generation"
echo "=========================================="
echo "Targets: ${!TARGETS[@]}"
echo "Models: ${MODELS[@]}"
echo "=========================================="
echo ""

# Track failures
FAILED=()

# ============================================================================
# Step 1: Evaluate models and generate comparison results
# ============================================================================
echo "Step 1: Evaluating models on test data (2024-2025)..."
echo "------------------------------------------------------"

for TARGET in "${!TARGETS[@]}"; do
    CONFIG="${TARGETS[$TARGET]}"
    echo ""
    echo "[$TARGET] Evaluating models..."
    
    # Check if any checkpoints exist for this target
    has_checkpoints=false
    for MODEL in "${MODELS[@]}"; do
        CHECKPOINT_FILE="$PROJECT_ROOT/checkpoints/${TARGET}_${MODEL}/model.pkl"
        if [ -f "$CHECKPOINT_FILE" ]; then
            has_checkpoints=true
            break
        fi
    done
    
    if [ "$has_checkpoints" = false ]; then
        echo "[$TARGET] ⚠ No checkpoints found. Skipping evaluation."
        echo "  Run 'bash scripts/run_train.sh' to train models first."
        continue
    fi
    
    # Run comparison using train.py with command=compare
    # Convert MODELS array to Hydra list format: [arima,var]
    # Use IFS to join array elements with comma
    MODELS_STR=$(IFS=,; echo "${MODELS[*]}")
    if python "$PROJECT_ROOT/src/train.py" \
        --config-name "$CONFIG" \
        experiment.command=compare \
        experiment.models="[${MODELS_STR}]" \
        experiment.checkpoint_dir=checkpoints \
        +experiment.log_dir=log 2>&1 | tee "$PROJECT_ROOT/log/${TARGET}_compare_$(date +%Y%m%d_%H%M%S).log"; then
        echo "[$TARGET] ✓ Comparison completed"
    else
        echo "[$TARGET] ✗ Comparison failed"
        FAILED+=("${TARGET}_compare")
    fi
done

echo ""
echo "Step 1 complete!"
echo ""

# ============================================================================
# Step 2: Aggregate results
# ============================================================================
echo "Step 2: Aggregating results..."
echo "------------------------------------------------------"

if python "$PROJECT_ROOT/src/aggregate_results.py"; then
    echo "✓ Results aggregated successfully"
    echo "  Output: outputs/experiments/aggregated_results.csv"
else
    echo "✗ Result aggregation failed"
    FAILED+=("aggregate_results")
fi

echo ""

# ============================================================================
# Step 3: Generate plots
# ============================================================================
echo "Step 3: Generating forecast plots..."
echo "------------------------------------------------------"

if python "$PROJECT_ROOT/nowcasting-report/code/plot_forecasts.py"; then
    echo "✓ Plots generated successfully"
    echo "  Output: nowcasting-report/images/"
else
    echo "✗ Plot generation failed"
    FAILED+=("plot_forecasts")
fi

echo ""

# ============================================================================
# Step 4: Generate LaTeX tables
# ============================================================================
echo "Step 4: Generating LaTeX tables..."
echo "------------------------------------------------------"

if python "$PROJECT_ROOT/nowcasting-report/code/table_forecasts.py"; then
    echo "✓ Tables generated successfully"
    echo "  Output: nowcasting-report/tables/"
else
    echo "✗ Table generation failed"
    FAILED+=("table_forecasts")
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "Forecast and Report Generation Complete"
echo "=========================================="

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All steps completed successfully!"
    echo ""
    echo "Generated files:"
    echo "  - outputs/comparisons/{target}/comparison_results.json"
    echo "  - outputs/experiments/aggregated_results.csv"
    echo "  - nowcasting-report/images/*.png"
    echo "  - nowcasting-report/tables/*.tex"
    echo ""
    echo "Next steps:"
    echo "  1. Review generated plots in nowcasting-report/images/"
    echo "  2. Review generated tables in nowcasting-report/tables/"
    echo "  3. Compile LaTeX report (if needed)"
else
    echo "✗ Some steps failed:"
    for FAILED_STEP in "${FAILED[@]}"; do
        echo "  - $FAILED_STEP"
    done
    exit 1
fi
