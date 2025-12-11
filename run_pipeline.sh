#!/bin/bash

# Pipeline script to run train -> forecast sequentially for each model
# Exit on error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPT_DIR}/scripts"

echo "Starting pipeline: train -> forecast for all models"
echo "=================================================="

# Define models in order
models=("dfm" "ddfm" "lstm" "tft" "chronos")

# Run train -> forecast for each model
for model in "${models[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing model: ${model}"
    echo "=================================================="
    
    # Run training
    echo "Running training for ${model}..."
    bash "${SCRIPTS_DIR}/run_train_${model}.sh"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for ${model}"
        exit 1
    fi
    
    echo "Training completed for ${model}"
    
    # Run forecasting
    echo "Running forecasting for ${model}..."
    bash "${SCRIPTS_DIR}/run_forecast_${model}.sh"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Forecasting failed for ${model}"
        exit 1
    fi
    
    echo "Forecasting completed for ${model}"
done

echo ""
echo "=================================================="
echo "Pipeline completed successfully!"
echo "=================================================="

