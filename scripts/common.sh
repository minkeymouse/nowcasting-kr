#!/bin/bash
# Common functions for training, forecasting, and nowcasting scripts

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Target series
TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")

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
    local checkpoint_file="$PROJECT_ROOT/checkpoints/${target}_${model}/model.pkl"
    [ -f "$checkpoint_file" ] && [ -s "$checkpoint_file" ]
}

# Function to activate virtual environment
activate_venv() {
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
        echo "Activated virtual environment: .venv"
    elif [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        echo "Activated virtual environment: venv"
    fi
}

# Function to ensure python command exists
ensure_python() {
    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            alias python=python3
        else
            echo "Error: python or python3 not found"
            exit 1
        fi
    fi
}

# Validate environment
validate_environment() {
    [ -d ".venv" ] || { echo "✗ Error: Virtual environment (.venv) not found"; return 1; }
    source .venv/bin/activate || { echo "✗ Error: Failed to activate virtual environment"; return 1; }
    mkdir -p checkpoint || { echo "✗ Error: Cannot create checkpoint/ directory"; return 1; }
    return 0
}

