#!/bin/bash
set -e
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Configuration
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CMD_TIMEOUT="${CMD_TIMEOUT:-1800}"  # 30 minutes for agent steps
EXPERIMENT_TIMEOUT="${EXPERIMENT_TIMEOUT:-7200}"  # 2 hours for experiments

# Directories
RUN_DIR=".cursor/run"
CHAT_FILE="${RUN_DIR}/chat_id.txt"
REPORT_DIR="nowcasting-report"
EXPERIMENT_SCRIPT="run_experiment.sh"
OUTPUTS_DIR="outputs"

# Files
STATUS_FILE="STATUS.md"
EXPERIMENT_LOG="${RUN_DIR}/experiment_status.log"
EXPERIMENT_PIDS="${RUN_DIR}/experiment_pids.txt"

# State
CANCELLED=0
CURRENT_ITERATION=0
CURRENT_STEP=0
AGENT_PID=0
MAX_CONCURRENT_EXPERIMENTS=5
mkdir -p "$RUN_DIR"

# State file for graceful shutdown recovery
STATE_FILE="${RUN_DIR}/workflow_state.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() { log "INFO: $@"; }
log_warn() { log "WARN: $@"; }
log_error() { log "ERROR: $@"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [iterations]

Runs agent workflow for Nowcasting Report and Experiments:
1. Study the report (nowcasting-report directory)
2. Prepare experiments (check/update config files)
3. Run experiments in background (max 5 concurrent)
4. Monitor experiment results
5. Update report with results (tables, figures, text)

Arguments:
  iterations   Number of workflow iterations (default: ${MAX_ITERATIONS})

Environment:
  VENV_PATH           Virtualenv path (default: .venv)
  PYTHON_BIN          Python executable (default: python3)
  CMD_TIMEOUT         Timeout for agent steps in seconds (default: 1800)
  EXPERIMENT_TIMEOUT  Timeout for experiments in seconds (default: 7200)
EOF
}

if [ -n "${1:-}" ]; then
    if echo "$1" | grep -qE '^[0-9]+$' && [ "$1" -ge 1 ]; then
        MAX_ITERATIONS="$1"
    else
        echo "Invalid iterations: $1" >&2
        usage
        exit 1
    fi
fi

save_state() {
    cat > "$STATE_FILE" <<EOF
CURRENT_ITERATION=${CURRENT_ITERATION}
CURRENT_STEP=${CURRENT_STEP}
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
EOF
}

graceful_shutdown() {
    if [ "$CANCELLED" -eq 1 ]; then
        log_warn "Force shutdown requested"
        kill -9 $$ 2>/dev/null || true
        exit 130
    fi
    CANCELLED=1
    
    log_warn "=========================================="
    log_warn "Graceful shutdown requested"
    log_warn "=========================================="
    log_warn "Current iteration: ${CURRENT_ITERATION}/${MAX_ITERATIONS}"
    log_warn "Current step: ${CURRENT_STEP}"
    
    save_state
    log_info "State saved to ${STATE_FILE}"
    
    # Kill any running agent processes
    if [ "$AGENT_PID" -gt 0 ]; then
        log_info "Terminating agent process (PID: ${AGENT_PID})"
        kill "$AGENT_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$AGENT_PID" 2>/dev/null || true
    fi
    
    # Kill running experiments
    kill_running_experiments
    
    # Kill any cursor-agent processes
    pkill -P $$ cursor-agent 2>/dev/null || true
    pkill -P $$ timeout 2>/dev/null || true
    kill -- -$$ 2>/dev/null || true
    
    log_warn "Shutdown complete. Use Ctrl+C again to force exit."
    exit 130
}

# Trap signals for graceful shutdown
trap graceful_shutdown INT TERM

activate_venv() {
    local act="${VENV_PATH}/bin/activate"
    if [ -f "$act" ]; then
        # shellcheck source=/dev/null
        . "$act"
        log_info "Activated virtualenv"
    fi
}

require_tool() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        log_error "Required tool not found: $name"
        exit 1
    fi
}

create_chat() {
    require_tool cursor-agent
    local out
    out=$(cursor-agent create-chat 2>&1 || true)
    local id
    id=$(echo "$out" | head -1)
    if [ -z "$id" ]; then
        log_error "Failed to create chat"
        return 1
    fi
    echo -n "$id" > "$CHAT_FILE"
    echo "$id"
}

get_chat_id() {
    if [ -s "$CHAT_FILE" ]; then
        cat "$CHAT_FILE"
    else
        create_chat
    fi
}

run_agent() {
    local prompt="$1"
    local chat_id
    chat_id="$(get_chat_id)" || return 1

    log_info "Running agent (chat: ${chat_id})"
    
    if [ "$CANCELLED" -eq 1 ]; then
        return 130
    fi
    
    timeout "${CMD_TIMEOUT}s" cursor-agent agent \
        --resume "$chat_id" \
        --model auto \
        --print \
        --output-format text \
        --approve-mcps \
        --force \
        "$prompt" 2>&1 || {
        local exit_code=$?
        if [ "$CANCELLED" -eq 1 ]; then
            return 130
        fi
        return $exit_code
    }
}

count_running_experiments() {
    ps aux | grep -E "python.*train.*compare|run_experiment.sh" | grep -v grep | wc -l
}

kill_running_experiments() {
    local count=$(count_running_experiments)
    if [ "$count" -gt 0 ]; then
        log_info "Terminating ${count} running experiment(s)..."
        pkill -f "python.*train.*compare" 2>/dev/null || true
        pkill -f "run_experiment.sh" 2>/dev/null || true
        sleep 3
        pkill -9 -f "python.*train.*compare" 2>/dev/null || true
        pkill -9 -f "run_experiment.sh" 2>/dev/null || true
    fi
}

cleanup_old_experiment_results() {
    local comparisons_dir="outputs/comparisons"
    
    if [ ! -d "$comparisons_dir" ]; then
        return
    fi
    
    # Target series to clean up
    local targets=("KOGDP...D" "KOCNPER.D" "KOGFCF..D")
    
    for target in "${targets[@]}"; do
        # Find all result directories for this target
        local result_dirs=$(find "$comparisons_dir" -maxdepth 1 -type d -name "${target}_*" 2>/dev/null | sort)
        
        if [ -z "$result_dirs" ]; then
            continue
        fi
        
        # Count directories
        local count=$(echo "$result_dirs" | wc -l)
        
        if [ "$count" -le 1 ]; then
            continue
        fi
        
        # Keep the latest and remove all others
        local latest_dir=$(echo "$result_dirs" | tail -1)
        
        log_info "Cleaning up old results for $target (keeping latest, removing $((count-1)) old result(s))..."
        echo "$result_dirs" | while IFS= read -r dir; do
            if [ -n "$dir" ] && [ "$dir" != "$latest_dir" ]; then
                rm -rf "$dir" 2>/dev/null || true
            fi
        done
        
        # Clean up old log files
        local log_files=$(find "$comparisons_dir" -maxdepth 1 -type f -name "${target}_*.log" 2>/dev/null | sort)
        local log_count=$(echo "$log_files" | wc -l)
        
        if [ "$log_count" -gt 1 ]; then
            local latest_log=$(echo "$log_files" | tail -1)
            
            echo "$log_files" | while IFS= read -r log; do
                if [ -n "$log" ] && [ "$log" != "$latest_log" ]; then
                    rm -f "$log" 2>/dev/null || true
                fi
            done
        fi
    done
}

wait_for_experiment_slot() {
    while [ $(count_running_experiments) -ge $MAX_CONCURRENT_EXPERIMENTS ]; do
        log_info "Waiting for experiment slot (${MAX_CONCURRENT_EXPERIMENTS} max concurrent)..."
        sleep 30
    done
}

run_experiment_background() {
    if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
        log_error "Experiment script not found: $EXPERIMENT_SCRIPT"
        return 1
    fi
    
    wait_for_experiment_slot
    
    log_info "Starting experiment in background..."
    chmod +x "$EXPERIMENT_SCRIPT"
    
    # Run experiment in background and log PID
    nohup bash "$EXPERIMENT_SCRIPT" >> "${RUN_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    local pid=$!
    echo "$pid" >> "$EXPERIMENT_PIDS"
    
    log_info "Experiment started (PID: $pid)"
    log_info "Current running experiments: $(count_running_experiments)/${MAX_CONCURRENT_EXPERIMENTS}"
    
    return 0
}

check_experiment_status() {
    local running=$(count_running_experiments)
    local completed=0
    local failed=0
    
    # Check for completed experiments (look for comparison results)
    if [ -d "$OUTPUTS_DIR/comparisons" ]; then
        completed=$(find "$OUTPUTS_DIR/comparisons" -name "comparison_results.json" -newer "$EXPERIMENT_LOG" 2>/dev/null | wc -l)
    fi
    
    # Check for failed experiments (look for error logs)
    if [ -d "$OUTPUTS_DIR/comparisons" ]; then
        failed=$(find "$OUTPUTS_DIR/comparisons" -name "*.log" -exec grep -l "Failed\|Error\|Traceback" {} \; 2>/dev/null | wc -l)
    fi
    
    echo "Running: $running, Completed: $completed, Failed: $failed"
}

workflow_step() {
    local step_num="$1"
    local step_name="$2"
    local prompt="$3"

    log_info "=========================================="
    log_info "Step ${step_num}: ${step_name}"
    log_info "=========================================="
    
    CURRENT_STEP=$step_num
    save_state

    run_agent "$prompt"
}

main() {
    log_info "=========================================="
    log_info "Nowcasting Report & Experiment Workflow"
    log_info "=========================================="
    log_info "Max iterations: ${MAX_ITERATIONS}"
    log_info "Report directory: ${REPORT_DIR}"
    log_info "Experiment script: ${EXPERIMENT_SCRIPT}"
    log_info "Max concurrent experiments: ${MAX_CONCURRENT_EXPERIMENTS}"
    log_info "Experiment timeout: ${EXPERIMENT_TIMEOUT}s"
    log_info ""
    log_info "Workflow Steps per Iteration:"
    log_info "  1. Study the report (analyze nowcasting-report contents)"
    log_info "  2. Prepare experiments (check/update config files)"
    log_info "  3. Run experiments in background (max 5 concurrent)"
    log_info "  4. Monitor experiment results"
    log_info "  5. Update report with results (tables, figures, text)"
    log_info ""
    log_info "Press Ctrl+C to gracefully shutdown"
    log_info "State will be saved for recovery"
    log_info ""

    # Check for saved state
    if [ -f "$STATE_FILE" ]; then
        log_info "Found saved state file: ${STATE_FILE}"
        # shellcheck source=/dev/null
        . "$STATE_FILE" 2>/dev/null || true
    fi
    
    # Verify report directory exists
    if [ ! -d "$REPORT_DIR" ]; then
        log_error "Report directory not found: $REPORT_DIR"
        exit 1
    fi
    
    # Verify experiment script exists
    if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
        log_error "Experiment script not found: $EXPERIMENT_SCRIPT"
        exit 1
    fi
    
    # Initialize experiment tracking
    > "$EXPERIMENT_PIDS"
    > "$EXPERIMENT_LOG"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Workflow started" >> "$EXPERIMENT_LOG"

    iter=1
    while [ "$iter" -le "$MAX_ITERATIONS" ] && [ "$CANCELLED" -eq 0 ]; do
        CURRENT_ITERATION=$iter
        log_info ""
        log_info "=========================================="
        log_info "Iteration ${iter}/${MAX_ITERATIONS}"
        log_info "=========================================="
        
        save_state

        # Step 1: Study the report
        workflow_step 1 "Study the Report" \
            "Study the nowcasting report in ${REPORT_DIR}:
- Read main.tex to understand the report structure
- Review all content files in ${REPORT_DIR}/contents/:
  - 1_introduction.tex: Introduction and motivation
  - 2_literature_review.tex or 2_dfm_modeling.tex: Literature review and DFM theory
  - 3_theoretical_background.tex or 3_high_frequency.tex: Theoretical background
  - 4_method_and_experiment.tex or 4_deep_learning.tex: Methodology and experiments
  - 5_result.tex: Results section
  - 6_discussion.tex: Discussion
  - 7_conclusion.tex: Conclusion
- Review existing tables in ${REPORT_DIR}/tables/:
  - Check what metrics are reported
  - Understand table structure and format
- Review existing figures in ${REPORT_DIR}/images/:
  - Check what visualizations are included
  - Understand figure requirements
- Review references.bib for cited papers
- Identify:
  - What experiments are described in the report
  - What results are missing or need updating
  - What tables/figures need to be generated
  - What sections need revision based on new experiments
- Document findings in ${RUN_DIR}/report_analysis_${iter}.md"

        # Step 2: Prepare experiments
        workflow_step 2 "Prepare Experiments" \
            "Prepare experiments based on report analysis:
- Review experiment config files in config/experiment/:
  - kogdp_report.yaml, kogfcf_report.yaml, kocnper_report.yaml
  - test_experiment.yaml
- Check if configs match what's described in the report
- Verify model configurations (arima, var, dfm, ddfm)
- Verify target series (KOGDP...D, KOCNPER.D, KOGFCF..D)
- Verify forecast horizons ([1, 7, 28])
- Check model_overrides for fast testing parameters
- Update configs if needed to match report requirements
- Verify run_experiment.sh is configured correctly:
  - MAX_PARALLEL=5 for concurrent experiments
  - Correct target series and models
- Document experiment plan in ${RUN_DIR}/experiment_plan_${iter}.md:
  - List experiments to run
  - Expected outputs
  - Required metrics and tables"

        # Step 3: Run experiments in background
        log_info "=========================================="
        log_info "Step 3: Run Experiments in Background"
        log_info "=========================================="
        CURRENT_STEP=3
        save_state
        
        # Check if experiments are already running
        local running=$(count_running_experiments)
        if [ "$running" -gt 0 ]; then
            log_info "Found ${running} experiment(s) already running"
            log_info "Waiting for them to complete or terminate..."
            kill_running_experiments
            sleep 5
        fi
        
        # Clean up old experiment results before starting new ones
        log_info "Cleaning up old experiment results (keeping only latest for each target)..."
        cleanup_old_experiment_results
        
        # Start experiments in background
        log_info "Starting experiments with max ${MAX_CONCURRENT_EXPERIMENTS} concurrent..."
        if run_experiment_background; then
            log_info "Experiments started successfully"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Experiments started (iteration ${iter})" >> "$EXPERIMENT_LOG"
        else
            log_error "Failed to start experiments"
        fi

        # Step 4: Monitor experiment results
        workflow_step 4 "Monitor Experiment Results" \
            "Monitor and analyze experiment results:
- Wait for experiments to complete (check running processes)
- Monitor experiment logs in ${OUTPUTS_DIR}/comparisons/:
  - Check for completion: *.log files
  - Check for results: */comparison_results.json
  - Check for errors: grep for 'Failed', 'Error', 'Traceback'
- Check experiment status:
  - Running: ps aux | grep 'python.*train.*compare'
  - Completed: find outputs/comparisons -name 'comparison_results.json'
  - Failed: find outputs/comparisons -name '*.log' | xargs grep -l 'Failed'
- Review comparison results:
  - Read comparison_results.json files
  - Check metrics: sMSE, sMAE, sRMSE for each horizon
  - Identify best models per horizon
  - Check for any model failures
- Review aggregated results:
  - Check outputs/experiments/ for aggregated tables
  - Run: python3 -m src.eval.aggregator (if not already run)
- Generate summary in ${RUN_DIR}/experiment_results_${iter}.md:
  - List completed experiments
  - List failed experiments (if any)
  - Summary of metrics per model and horizon
  - Best model per target series and horizon
  - Any issues or anomalies"

        # Step 5: Update report with results
        workflow_step 5 "Update Report with Results" \
            "Update the nowcasting report with experiment results:
- Based on experiment results from step 4, update the report:
- Update tables in ${REPORT_DIR}/tables/:
  - tab_overall_metrics.tex: Overall metrics across all models
  - tab_overall_metrics_by_horizon.tex: Metrics by forecast horizon
  - tab_overall_metrics_by_target.tex: Metrics by target series
  - tab_nowcasting_metrics.tex: Nowcasting-specific metrics
  - Use data from outputs/experiments/ aggregated results
  - Format tables in LaTeX using appropriate packages (booktabs, etc.)
- Update or generate figures in ${REPORT_DIR}/images/:
  - model_comparison.png: Compare model performance
  - forecast_vs_actual.png: Forecast vs actual plots
  - accuracy_heatmap.png: Heatmap of metrics
  - horizon_trend.png: Performance trends across horizons
  - Use code/plot.py or create new plotting scripts
  - Save figures in appropriate format for LaTeX
- Update content files:
  - 5_result.tex: Add/update results section with new metrics
  - 4_method_and_experiment.tex: Update experiment description if needed
  - 6_discussion.tex: Discuss new findings
  - Ensure results match the tables and figures
- Update references if new papers are cited
- Verify LaTeX compilation:
  - Check that all tables/figures are referenced correctly
  - Ensure no missing references or undefined citations
  - Test compilation: cd ${REPORT_DIR} && pdflatex main.tex (if pdflatex available)
- Document changes in ${RUN_DIR}/report_updates_${iter}.md:
  - List updated sections
  - List new tables/figures
  - Note any issues or remaining work"

        if [ "$CANCELLED" -eq 1 ]; then
            break
        fi
        
        log_info "Iteration ${iter} completed"
        
        # Check experiment status
        local status=$(check_experiment_status)
        log_info "Experiment status: $status"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Iteration ${iter} completed - $status" >> "$EXPERIMENT_LOG"
        
        iter=$((iter+1))
        
        # Wait a bit before next iteration
        if [ "$iter" -le "$MAX_ITERATIONS" ]; then
            log_info "Waiting 60 seconds before next iteration..."
            sleep 60
        fi
    done

    if [ "$CANCELLED" -eq 1 ]; then
        log_warn "Workflow interrupted at iteration ${CURRENT_ITERATION}, step ${CURRENT_STEP}"
        log_info "State saved to ${STATE_FILE}"
        exit 130
    fi

    log_info ""
    log_info "Workflow completed (${MAX_ITERATIONS} iterations)"
    
    # Final experiment status
    local final_status=$(check_experiment_status)
    log_info "Final experiment status: $final_status"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Workflow completed - $final_status" >> "$EXPERIMENT_LOG"
    
    # Clean up state file on successful completion
    rm -f "$STATE_FILE"
}

main
