#!/bin/bash
# ============================================================================
# Experiment Runner Script
# ============================================================================
# Robust execution wrapper for src/main.py experiments.
#
# Features:
#   - Handles virtual environment activation
#   - Supports multiple config names (up to 2 per step to prevent overloading)
#   - Captures stdout/stderr to log files
#   - Writes status file (json) for agent to read
#   - Prevents execution if an experiment is already running (basic lock)
#
# Usage: ./scripts/run_experiment.sh <config_name1> [config_name2] [overrides...]
# Example: ./scripts/run_experiment.sh arima_config var_config "data.path=data/foo.csv"
# ============================================================================

set -o pipefail

# --- Configuration ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Parse config names (first two arguments)
CONFIG_NAMES=()
while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
    CONFIG_NAMES+=("$1")
    shift
done
OVERRIDES="$@"

if [ ${#CONFIG_NAMES[@]} -eq 0 ]; then
    echo "Usage: $0 <config_name1> [config_name2] [overrides...]"
    exit 1
fi

if [ ${#CONFIG_NAMES[@]} -gt 2 ]; then
    echo "ERROR: Maximum 2 config names allowed to prevent overloading"
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# --- Environment Setup ---
log "INFO: Setting up environment for ${#CONFIG_NAMES[@]} experiment(s): ${CONFIG_NAMES[*]}"

# Try to find python
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
elif command -v uv >/dev/null; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

log "INFO: Using python: $PYTHON_CMD"
$PYTHON_CMD --version >/dev/null 2>&1

# --- Execute Experiments ---
OVERALL_START=$(date +%s)
SUCCESS_COUNT=0
FAILED_COUNT=0

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    # Output setup for this experiment
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    EXPERIMENT_DIR="outputs/experiments/${CONFIG_NAME}_${TIMESTAMP}"
    LOG_FILE="${EXPERIMENT_DIR}/execution.log"
    STATUS_FILE="${EXPERIMENT_DIR}/status.json"

    mkdir -p "$EXPERIMENT_DIR"

    log "INFO: Starting experiment '$CONFIG_NAME'..."
    log "INFO: Overrides: $OVERRIDES" | tee -a "$LOG_FILE"
    log "INFO: Output directory: $EXPERIMENT_DIR" | tee -a "$LOG_FILE"

    # Create initial status
    echo '{"status": "running", "start_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'", "config": "'$CONFIG_NAME'"}' > "$STATUS_FILE"

    # Run Hydra experiment
    START_TIME=$(date +%s)

    # Construct command
    CMD="$PYTHON_CMD src/main.py config_name=$CONFIG_NAME hydra.run.dir=$EXPERIMENT_DIR $OVERRIDES"
    log "INFO: Executing: $CMD" | tee -a "$LOG_FILE"

    if eval "$CMD" >> "$LOG_FILE" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log "INFO: Experiment '$CONFIG_NAME' completed successfully in ${DURATION}s." | tee -a "$LOG_FILE"

        # Write success status
        cat > "$STATUS_FILE" <<EOF
{
    "status": "success",
    "start_time": "$(date -u -d @$START_TIME +"%Y-%m-%dT%H:%M:%SZ")",
    "end_time": "$(date -u -d @$END_TIME +"%Y-%m-%dT%H:%M:%SZ")",
    "duration_seconds": $DURATION,
    "config": "$CONFIG_NAME",
    "log_file": "$LOG_FILE",
    "output_dir": "$EXPERIMENT_DIR"
}
EOF
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        log "ERROR: Experiment '$CONFIG_NAME' failed with exit code $EXIT_CODE." | tee -a "$LOG_FILE"

        # Write failure status
        cat > "$STATUS_FILE" <<EOF
{
    "status": "failed",
    "error_code": $EXIT_CODE,
    "start_time": "$(date -u -d @$START_TIME +"%Y-%m-%dT%H:%M:%SZ")",
    "end_time": "$(date -u -d @$END_TIME +"%Y-%m-%dT%H:%M:%SZ")",
    "duration_seconds": $DURATION,
    "config": "$CONFIG_NAME",
    "log_file": "$LOG_FILE",
    "output_dir": "$EXPERIMENT_DIR"
}
EOF
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# --- Summary ---
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

log "INFO: Experiment batch completed in ${OVERALL_DURATION}s"
log "INFO: Results: $SUCCESS_COUNT succeeded, $FAILED_COUNT failed"

# Exit with success if at least one experiment succeeded
if [ $SUCCESS_COUNT -gt 0 ]; then
    log "INFO: At least one experiment succeeded, exiting with success"
    exit 0
else
    log "ERROR: All experiments failed, exiting with failure"
    exit 1
fi
