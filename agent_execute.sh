#!/bin/bash
# Agent execute script - automatically checks what experiments are needed and runs them
# This is the ONLY entry point for running experiments (called by cursor-headless.sh Step 1)
# Supports: train, forecast, backtest, all (or auto-detect if no argument)

# Don't exit on error - continue even if some steps fail
set +e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Configuration
TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")
MODELS=("arima" "var" "dfm" "ddfm")

# ============================================================================
# Helper Functions
# ============================================================================

checkpoint_exists() {
    local target=$1
    local model=$2
    local checkpoint_file="checkpoint/${target}_${model}/model.pkl"
    if [[ -f "$checkpoint_file" ]] && [[ -s "$checkpoint_file" ]]; then
        return 0  # Exists
    else
        return 1  # Missing
    fi
}

check_training_needed() {
    local all_trained=1
    for target in "${TARGETS[@]}"; do
        for model in "${MODELS[@]}"; do
            if ! checkpoint_exists "$target" "$model"; then
                all_trained=0
                return 1
            fi
        done
    done
    return 0  # All trained
}

check_forecasting_needed() {
    if [[ ! -f "outputs/experiments/aggregated_results.csv" ]]; then
        return 1  # Needed
    fi
    # Check if file has valid data (more than header)
    local line_count
    line_count=$(wc -l < "outputs/experiments/aggregated_results.csv" 2>/dev/null || echo "0")
    if [[ $line_count -le 1 ]]; then
        return 1  # Needed (empty file)
    fi
    return 0  # Not needed
}

check_backtesting_needed() {
    for target in "${TARGETS[@]}"; do
        for model in "${MODELS[@]}"; do
            local backtest_file="outputs/backtest/${target}_${model}_backtest.json"
            if [[ ! -f "$backtest_file" ]]; then
                return 1  # Needed
            fi
            # Check if JSON is valid and has completed results
            # For DFM/DDFM: check for 'status': 'completed' and 'results_by_timepoint'
            # For ARIMA/VAR: check for 'status': 'not_supported' (expected for non-DFM models)
            if ! python3 -c "import json; data = json.load(open('${backtest_file}')); status = data.get('status', ''); exit(0 if status == 'completed' or (status == 'not_supported' and '${model}' in ['arima', 'var']) else 1)" 2>/dev/null; then
                return 1  # Needed (invalid or incomplete results)
            fi
        done
    done
    return 0  # Not needed
}

# ============================================================================
# Argument Parsing
# ============================================================================

MODE="${1:-auto}"

case "$MODE" in
    train|forecast|backtest|all)
        # Valid mode
        ;;
    auto)
        # Auto-detect mode (default)
        ;;
    --help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  train      Run training only (run_train.sh)"
        echo "  forecast   Run forecasting only (run_forecast.sh or run_nowcast.sh --skip-nowcast)"
        echo "  backtest   Run backtesting only (run_backtest.sh or run_nowcast.sh --skip-forecast)"
        echo "  all        Run all in sequence (train → forecast → backtest)"
        echo "  auto       Auto-detect what's needed and run (default)"
        echo "  (no arg)   Same as 'auto'"
        echo ""
        echo "This script is the ONLY entry point for running experiments."
        echo "It automatically checks what's needed and runs only missing experiments."
        echo ""
        echo "Results are saved to:"
        echo "  - checkpoint/{target}_{model}/model.pkl (training)"
        echo "  - outputs/comparisons/{target}/comparison_results.json (forecasting)"
        echo "  - outputs/experiments/aggregated_results.csv (forecasting aggregated)"
        echo "  - outputs/backtest/{target}_{model}_backtest.json (backtesting)"
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# ============================================================================
# Environment Validation
# ============================================================================

validate_environment() {
    if [[ ! -d ".venv" ]]; then
        echo "✗ Error: Virtual environment (.venv) not found"
        return 1
    fi
    
    source .venv/bin/activate || {
        echo "✗ Error: Failed to activate virtual environment"
        return 1
    }
    
    # Check if required scripts exist
    local missing_scripts=0
    for script in "run_train.sh" "run_nowcast.sh"; do
        if [[ ! -f "$script" ]]; then
            echo "✗ Error: Required script not found: $script"
            missing_scripts=$((missing_scripts + 1))
        fi
    done
    
    if [[ $missing_scripts -gt 0 ]]; then
        return 1
    fi
    
    return 0
}

if ! validate_environment; then
    echo "Environment validation failed. Please fix the issues above."
    exit 1
fi

# ============================================================================
# Main Logic
# ============================================================================

echo "=========================================="
echo "Agent Execute Script"
echo "=========================================="
echo "Mode: $MODE"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Determine what to run
NEEDS_TRAIN=0
NEEDS_FORECAST=0
NEEDS_BACKTEST=0

if [[ "$MODE" == "auto" ]]; then
    echo "Auto-detecting what experiments are needed..."
    echo ""
    
    # Check training
    if check_training_needed; then
        echo "✓ Training appears complete: All models exist in checkpoint/"
    else
        echo "→ Training needed: Some models missing in checkpoint/"
        NEEDS_TRAIN=1
    fi
    
    # Check forecasting
    if check_forecasting_needed; then
        echo "✓ Forecasting appears complete: aggregated_results.csv exists with data"
    else
        echo "→ Forecasting needed: aggregated_results.csv missing or empty"
        NEEDS_FORECAST=1
    fi
    
    # Check backtesting
    if check_backtesting_needed; then
        echo "✓ Backtesting appears complete: All backtest results exist"
    else
        echo "→ Backtesting needed: Some backtest results missing in outputs/backtest/"
        NEEDS_BACKTEST=1
    fi
    
    echo ""
    
    if [[ $NEEDS_TRAIN -eq 0 ]] && [[ $NEEDS_FORECAST -eq 0 ]] && [[ $NEEDS_BACKTEST -eq 0 ]]; then
        echo "✓ All experiments appear complete. Nothing to run."
        exit 0
    fi
elif [[ "$MODE" == "train" ]]; then
    NEEDS_TRAIN=1
elif [[ "$MODE" == "forecast" ]]; then
    NEEDS_FORECAST=1
elif [[ "$MODE" == "backtest" ]]; then
    NEEDS_BACKTEST=1
elif [[ "$MODE" == "all" ]]; then
    NEEDS_TRAIN=1
    NEEDS_FORECAST=1
    NEEDS_BACKTEST=1
fi

# Run training
if [[ $NEEDS_TRAIN -eq 1 ]]; then
    echo "=========================================="
    echo "Running Training"
    echo "=========================================="
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    if bash run_train.sh; then
        echo ""
        echo "✓ Training completed"
    else
        echo ""
        echo "⚠ Training had errors (check logs in log/ directory)"
    fi
    echo ""
fi

# Run forecasting
if [[ $NEEDS_FORECAST -eq 1 ]]; then
    echo "=========================================="
    echo "Running Forecasting"
    echo "=========================================="
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Use run_nowcast.sh with --skip-nowcast to run only forecasting
    if bash run_nowcast.sh --skip-nowcast; then
        echo ""
        echo "✓ Forecasting completed"
    else
        echo ""
        echo "⚠ Forecasting had errors (check logs in outputs/forecast/ directory)"
    fi
    echo ""
fi

# Run backtesting
if [[ $NEEDS_BACKTEST -eq 1 ]]; then
    echo "=========================================="
    echo "Running Backtesting (Nowcasting)"
    echo "=========================================="
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Use run_nowcast.sh with --skip-forecast to run only backtesting
    if bash run_nowcast.sh --skip-forecast; then
        echo ""
        echo "✓ Backtesting completed"
    else
        echo ""
        echo "⚠ Backtesting had errors (check logs in outputs/nowcast/ directory)"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [[ $NEEDS_TRAIN -eq 1 ]]; then
    echo "Training: $(if check_training_needed; then echo "✓ Complete"; else echo "⚠ May have issues (check checkpoint/)"; fi)"
fi

if [[ $NEEDS_FORECAST -eq 1 ]]; then
    echo "Forecasting: $(if check_forecasting_needed; then echo "✓ Complete"; else echo "⚠ May have issues (check outputs/experiments/)"; fi)"
fi

if [[ $NEEDS_BACKTEST -eq 1 ]]; then
    echo "Backtesting: $(if check_backtesting_needed; then echo "✓ Complete"; else echo "⚠ May have issues (check outputs/backtest/)"; fi)"
fi

echo ""
echo "Results saved to:"
echo "  - Training: checkpoint/{target}_{model}/model.pkl"
echo "  - Forecasting: outputs/comparisons/ and outputs/experiments/aggregated_results.csv"
echo "  - Backtesting: outputs/backtest/{target}_{model}_backtest.json"
echo ""

