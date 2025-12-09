#!/usr/bin/env bash
# Cursor headless workflow runner for nowcasting-kr
# Implements the steps described in WORKFLOW.md using cursor-agent
#
# REPORT-FOCUSED WORKFLOW:
# - Report writing and improvement (primary focus)
# - DDFM metrics improvement research (continuous)
# - Experiment execution via agent_execute.sh (run_train.sh, run_nowcast.sh, run_forecast.sh as needed)
# - Table and plot generation from experiment results
# - LaTeX PDF compilation (LaTeX installed, compilation optional)
# - Automatic commit and push to remote origin (submodules pushed every 10 iterations)
# - Legacy code cleanup (dfm-python updated to latest version)

# Don't exit on error - pipeline should persist until user cancels
set +e
set -u  # Still fail on undefined variables
set -o pipefail  # But catch pipe errors without exiting

# Configuration
readonly REPO_ROOT="/data/nowcasting-kr"
readonly VENV_PATH="${REPO_ROOT}/.venv/bin/activate"
readonly LINE_LIMIT=1000
readonly MAX_RETRIES="${MAX_RETRIES:-3}"
readonly RETRY_DELAY="${RETRY_DELAY:-5}"
readonly LOG_DIR="${REPO_ROOT}/.cursor-logs"
readonly BACKUP_DIR="${LOG_DIR}/backups"
readonly STEP_STATUS_FILE="${REPO_ROOT}/.cursor-step-status"
readonly TARGETS=("KOEQUIPTE" "KOWRCCNSE" "KOIPALL.G")
readonly MODELS=("ARIMA" "VAR" "DFM" "DDFM")
readonly HORIZONS=(1 7 28)
readonly TOTAL_COMBINATIONS=36

ITERATION="${ITERATION:-0}"

# Utility functions
usage() {
  cat <<'EOF'
Usage: cursor-headless.sh <number_of_iterations>

Runs the specified number of iterations (each iteration runs steps 1-9).

Arguments:
  number_of_iterations  Number of iterations to run (1, 2, 3, ...)

Steps in each iteration (report-focused workflow):
  1   Run experiments if needed via @agent_execute.sh (checks what's needed and runs train/forecast/backtest/all)
  2   Report writing: Analyze results and write/update report sections (check FEEDBACK.md)
  3   Report writing: Continue report sections and plan DDFM metrics improvements (check FEEDBACK.md)
  4   Report writing: Generate tables and plots, update report with results (check FEEDBACK.md)
  5   DDFM metrics research: Plan and implement DDFM metrics improvements (check FEEDBACK.md)
  6   Report writing: Continue report sections and incorporate DDFM improvements (check FEEDBACK.md)
  7   Report writing: Finalize report sections, compile PDF, verify content (check FEEDBACK.md)
  8   Summarize iteration; refresh CONTEXT/STATUS/ISSUES
  9   Stage, commit & push to origin (push submodules every 2nd iteration for user review)

Environment:
  COMMIT_MSG - Commit message for step 9 (default: "Iteration N")
  MAX_RETRIES - Number of retry attempts on failure (default: 3)
  RETRY_DELAY - Delay between retries in seconds (default: 5)

Examples:
  ./cursor-headless.sh 1  # Run 1 iteration
  ./cursor-headless.sh 2  # Run 2 iterations
EOF
}

# Log error but don't exit - pipeline should continue
die() { 
  log_error "$*"
  return 1
}

log() {
  local level="$1"
  shift
  local msg="$*"
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  local log_file="${LOG_DIR}/iteration-${ITERATION}.log"
  mkdir -p "$LOG_DIR"
  echo "[$timestamp] [$level] $msg" | tee -a "$log_file"
}

log_info() { log "INFO" "$@"; }
log_error() { log "ERROR" "$@"; }
log_warn() { log "WARN" "$@"; }

# Step tracking
mark_step_completed() {
  local step="$1"
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  mkdir -p "$(dirname "$STEP_STATUS_FILE")"
  echo "$step|$timestamp|${ITERATION}" >> "$STEP_STATUS_FILE"
  log_info "Marked step $step as completed at $timestamp (iteration ${ITERATION})"
}

show_step_history() {
  log_info "Step completion history:"
  if [[ -f "$STEP_STATUS_FILE" ]]; then
    local count=0
    while IFS='|' read -r step timestamp iteration; do
      count=$((count + 1))
      log_info "  Step $step completed at $timestamp (iteration $iteration)"
    done < <(tail -n 10 "$STEP_STATUS_FILE" 2>/dev/null)
    [[ $count -eq 0 ]] && log_info "  No steps completed yet"
  else
    log_info "  No steps completed yet"
  fi
}

# Environment validation
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "missing required command: $1"
    return 1
  fi
  return 0
}

ensure_env() {
  require_cmd cursor-agent
  require_cmd bash
}

activate_venv() {
  if [[ ! -f "$VENV_PATH" ]]; then
    log_error "virtualenv not found at $VENV_PATH"
    return 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_PATH" || {
    log_error "Failed to activate virtualenv"
    return 1
  }
  return 0
}

assert_file() {
  if [[ ! -f "$1" ]]; then
    log_error "required file not found: $1"
    return 1
  fi
  return 0
}

guard_line_limit() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    log_warn "file missing for line limit check: $f"
    return 0  # Don't fail if file doesn't exist
  fi
  local lines
  lines=$(wc -l < "$f")
  if [[ "$lines" -gt "$LINE_LIMIT" ]]; then
    log_error "File $f exceeds ${LINE_LIMIT} lines ($lines lines) - but continuing"
    return 1  # Return error but don't exit
  fi
  return 0
}

backup_file_if_exists() {
  local f="$1"
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_DIR"
    local basename
    basename=$(basename "$f")
    local backup="${BACKUP_DIR}/${basename}.backup.$(date +%s)"
    cp "$f" "$backup"
    log_info "Backed up $f to $backup"
  fi
}

latest_output_dir() {
  [[ ! -d "${REPO_ROOT}/outputs" ]] && return 1
  ls -1td "${REPO_ROOT}/outputs"/*/ 2>/dev/null | head -n1 | sed 's|/$||' || return 1
}

# DFM/DDFM package check
check_dfm_package() {
  if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    if ! python3 -c "import sys; sys.path.insert(0, 'dfm-python/src'); import dfm_python" 2>/dev/null; then
      log_warn "dfm-python package not available - DFM/DDFM experiments will fail"
      log_warn "Install dependencies: cd dfm-python && pip install -e ."
      return 1
    else
      log_info "✓ dfm-python package available"
      return 0
    fi
  fi
  return 1
}

check_dfm_results() {
  local latest_dir
  latest_dir=$(latest_output_dir 2>/dev/null || echo "")
  if [[ -n "$latest_dir" ]] && [[ -d "$latest_dir" ]] && [[ -f "${latest_dir}/comparison_results.json" ]]; then
    if python3 -c "import json, sys; data = json.load(open('${latest_dir}/comparison_results.json')); dfm_status = data['results'].get('dfm', {}).get('status'); ddfm_status = data['results'].get('ddfm', {}).get('status'); sys.exit(0 if dfm_status == 'failed' or ddfm_status == 'failed' else 1)" 2>/dev/null; then
      log_warn "⚠ DFM/DDFM experiments failed - Check comparison_results.json for error details"
      log_warn "⚠ Only ARIMA/VAR results available (18/${TOTAL_COMBINATIONS} combinations appear complete)"
      return 1
    fi
  fi
  return 0
}

# Check which experiments are needed
check_what_experiments_needed() {
  local needs_train=0
  local needs_forecast=0
  local needs_backtest=0
  
  # Check if training is needed
  local all_trained=1
  for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
      local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
      local checkpoint_file="${REPO_ROOT}/checkpoint/${target}_${model_lower}/model.pkl"
      if [[ ! -f "$checkpoint_file" ]] || [[ ! -s "$checkpoint_file" ]]; then
        all_trained=0
        break 2
      fi
    done
  done
  
  if [[ $all_trained -eq 0 ]]; then
    needs_train=1
    log_info "→ Training needed: Some models missing in checkpoint/"
  else
    log_info "✓ Training appears complete: All models exist in checkpoint/ (may need verification)"
  fi
  
  # Check if forecasting is needed
  if [[ ! -f "${REPO_ROOT}/outputs/experiments/aggregated_results.csv" ]]; then
    needs_forecast=1
    log_info "→ Forecasting needed: aggregated_results.csv not found"
  else
    # Check if file has valid data (more than header)
    local line_count
    line_count=$(wc -l < "${REPO_ROOT}/outputs/experiments/aggregated_results.csv" 2>/dev/null || echo "0")
    if [[ $line_count -le 1 ]]; then
      needs_forecast=1
      log_info "→ Forecasting needed: aggregated_results.csv is empty"
    else
      log_info "✓ Forecasting appears complete: aggregated_results.csv exists with data (may need verification)"
    fi
  fi
  
  # Check if backtesting is needed
  local all_backtested=1
  for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
      local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
      local backtest_file="${REPO_ROOT}/outputs/backtest/${target}_${model_lower}_backtest.json"
      if [[ ! -f "$backtest_file" ]]; then
        all_backtested=0
        break 2
      fi
      # Check if JSON is valid and has results
      if ! python3 -c "import json; data = json.load(open('${backtest_file}')); exit(0 if 'results' in data and len(data.get('results', {})) > 0 else 1)" 2>/dev/null; then
        all_backtested=0
        break 2
      fi
    done
  done
  
  if [[ $all_backtested -eq 0 ]]; then
    needs_backtest=1
    log_info "→ Backtesting needed: Some backtest results missing in outputs/backtest/"
  else
    log_info "✓ Backtesting appears complete: All backtest results exist (may need verification)"
  fi
  
  echo "$needs_train $needs_forecast $needs_backtest"
}

# Workflow context
workflow_context() {
  cat <<'EOF'
# RULES
- DO NOT CREATE FILES or MARKDOWNS
- DO NOT DELETE FILES
- ONLY WORK under modifying existing codes
- NEVER HALLUCINATE. Only use references at references.bib
- src/ should contain maximum 15 files including __init__.py
- CONTEXT.md, STATUS.md, ISSUES.md MUST BE UNDER 1000 lines
- Try to improve incrementally. Prioritize tasks and work on them one by one
- CRITICAL: DO NOT claim "complete", "verified", "resolved", "no issues", "production ready", "done", or "everything works" unless you actually FIXED or IMPROVED something
- CRITICAL: If you say something is "done", you must have actually MADE CHANGES to code or files
- CRITICAL: Always identify REAL problems and FIX them, don't just document that "everything is fine"
- CRITICAL: There's ALWAYS room for improvement - never claim things are "complete" or "finished"
- CRITICAL: If checkpoint/ is empty, models are NOT trained - this is a REAL problem that needs fixing
- CRITICAL: If outputs/backtest/ doesn't exist, nowcasting is NOT done - this is a REAL problem
- CRITICAL: If you find issues, FIX them in code, don't just say "verified" or "documented"
- CRITICAL: Always acknowledge limitations and areas for improvement, even after fixing issues

# GOAL
- Write complete report (under 15 pages) comparing 4 models (ARIMA, VAR, DFM, DDFM) on 3 targets
- Improve DDFM metrics through continuous research and experimentation
- Clean up legacy code (dfm-python updated to latest version, remove deprecated code)

# EXPERIMENT CONFIGURATION
- Targets: 3 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G)
- Models: 4 (ARIMA, VAR, DFM, DDFM)
- Forecasting Horizons: 22 months (2024-01 to 2025-10)
- Nowcasting: 22 months (2024-01 to 2025-10), 2 time points (4 weeks, 1 week before)
- Series configs: All series use block: null
- Data file: data/data.csv
- Training period: 1985-01-01 to 2019-12-31 (no data leakage)

# REPORT STRUCTURE
- 7 sections: Introduction, Methodology, Results (Forecasting, Nowcasting, Performance), Discussion, Issues
- Target: Under 15 pages
- Tables: 3 required tables (dataset/params, forecasting results, nowcasting backtest)
- Plots: 4 required plots (forecast vs actual, accuracy heatmap, performance trend, nowcasting comparison)

# DESIRED OUTPUT (from WORKFLOW.md)
## Tables (3 required):
1. Dataset details and model parameters (ARIMA, VAR, DFM, DDFM model and training params)
2. Standardized MSE/MAE for (target, model, horizon) pairs - 36 rows
3. DFM/DDFM backtest results for 2024-2025 by month (train 1985-2019, nowcast Jan 2024-Oct 2025)

## Images (4 required):
1. Forecast vs actual plots: 3 plots (one per target), 34 months total (12 historical + 22 forecasts)
2. Accuracy heatmap: 4 models × 3 targets (standardized RMSE)
3. Performance trend: Performance by forecasting horizon (1-22 months)
4. Nowcasting comparison: 3 pairs of plots (4 weeks vs 1 week before) for each target

# WORKFLOW PRIORITY (REPORT-FOCUSED):
1. FIRST: Run experiments if needed (use @agent_execute.sh to decide which to run: train/forecast/backtest/all)
2. THEN: Generate tables and plots from experiment results as specified in WORKFLOW.md
3. THEN: Write/update report sections using generated tables and plots - ensure all theoretically correct details are documented
4. CONTINUOUS: Research and improve DDFM metrics (evaluate, experiment, iterate)
5. CLEANUP: Remove legacy code (dfm-python updated to latest version, remove deprecated/unused code)
6. OPTIONAL: Check LaTeX PDF compilation if needed (LaTeX is installed, but compilation is optional)
7. COMMIT & PUSH: Ensure all changes are committed and pushed to remote origin

# EXPERIMENT EXECUTION RULES:
- Step 1 automatically checks what experiments are needed and runs @agent_execute.sh
- Step 1 checks: checkpoint/ (training), outputs/experiments/aggregated_results.csv (forecasting), outputs/backtest/ (backtesting)
- Step 1 automatically executes: bash agent_execute.sh (auto-detects what's needed)
- agent_execute.sh calls: run_train.sh, run_nowcast.sh, run_forecast.sh as needed
- Agent ONLY modifies code - Agent MUST NOT execute any scripts
- Agent MUST NOT directly execute run_train.sh, run_nowcast.sh, run_forecast.sh, or agent_execute.sh

# FOCUS AREAS:
- Report writing: Write and update report sections with results, analysis, and discussion
- DDFM metrics improvement: Research and improve DDFM evaluation metrics through experimentation
- Table and plot generation: Generate all required tables and plots from experiment results
- Legacy code cleanup: Remove deprecated/unused code (dfm-python updated to latest version)
- Experiment execution: Run experiments via agent_execute.sh when needed (train/forecast/backtest)

# EXPERIMENT SCRIPTS (ONLY EXECUTED BY cursor-headless.sh Step 1):
- @agent_execute.sh: ONLY entry point for running experiments. Executed automatically by Step 1.
  - bash agent_execute.sh (auto)    # Auto-detect what's needed and run (default)
  - bash agent_execute.sh train      # Run training (run_train.sh) - saves to checkpoint/
  - bash agent_execute.sh forecast   # Run forecasting (run_forecast.sh) - loads from checkpoint/
  - bash agent_execute.sh backtest   # Run backtesting (run_nowcast.sh) - nowcasting with time points
  - bash agent_execute.sh all      # Run all in sequence
- agent_execute.sh calls: run_train.sh, run_nowcast.sh, run_forecast.sh as needed
- Step 1 automatically checks checkpoint/, outputs/experiments/, outputs/backtest/ to determine what's needed
- Agent ONLY modifies code - Agent MUST NOT execute any scripts (run_train.sh, run_nowcast.sh, run_forecast.sh, agent_execute.sh)

# RESOURCES
- CONTEXT.md: Context offloading for persistence
- STATUS.md: Track progress and status for next iteration
- ISSUES.md: Track resolved issues and next steps (keep under 1000 lines)
- FEEDBACK.md: User feedback file - check regularly, incorporate into improvements
- src/: Engine for running experiments (max 15 files)
- dfm-python/: Core DFM/DDFM package
- nowcasting-report/: LaTeX report submodule
- checkpoint/: Trained models (checkpoint/{target}_{model}/model.pkl)
- outputs/comparisons/: Per-target forecasting results
- outputs/experiments/: Aggregated forecasting results (aggregated_results.csv)
- outputs/backtest/: Nowcasting backtest results ({target}_{model}_backtest.json)
- config/: Hydra YAML configs

# FEEDBACK PROCESS
- User reviews report every 10 iterations (when submodules are pushed)
- User provides feedback in FEEDBACK.md
- Check FEEDBACK.md regularly and incorporate feedback into improvements

EOF
}

# Prompt builders
build_report_priority_prompt() {
  cat <<'EOF'
REPORT-FOCUSED PRIORITY:
1. REPORT WRITING: ACTUALLY WRITE report sections:
   - Write/update report sections in nowcasting-report/contents/ based on experiment results
   - Ensure all required tables and plots are generated and referenced in report
   - Check FEEDBACK.md for user feedback and incorporate into report
   - Don't just say "should be written" - actually WRITE the content
   - After generating tables/plots, UPDATE the report sections to reference them

2. DDFM METRICS IMPROVEMENT: RESEARCH AND IMPROVE:
   - Analyze DDFM performance metrics (sMSE, sMAE, sRMSE) in evaluation results
   - Research ways to improve DDFM metrics through code improvements
   - Experiment with different configurations, hyperparameters, or training procedures
   - Document findings and improvements in report
   - Use agent_execute.sh to run experiments when needed to test improvements

3. TABLE/PLOT GENERATION: MUST regenerate after any result changes:
   - After Step 1 runs experiments, regenerate tables and plots
   - After updating report sections, verify tables/plots are up to date
   - Use: python3 nowcasting-report/code/table.py and python3 nowcasting-report/code/plot.py
   - Verify outputs exist in nowcasting-report/tables/ and nowcasting-report/images/

4. LEGACY CODE CLEANUP: REMOVE deprecated/unused code:
   - dfm-python is updated to latest version - remove any deprecated code
   - Remove unused imports, functions, or modules
   - Clean up any temporary fixes or workarounds
   - Ensure code follows latest dfm-python patterns

5. EXPERIMENT EXECUTION: Run experiments when needed:
   - Use agent_execute.sh to run experiments (train/forecast/backtest) when needed
   - Check checkpoint/, outputs/experiments/, outputs/backtest/ to determine what's needed
   - Agent MUST NOT directly execute scripts - only modify code

CRITICAL: DO NOT use words like "complete", "verified", "resolved", "no issues", "production ready", "done", "everything works", or "all fixed" unless you actually FIXED or IMPROVED something in code or files. Always acknowledge there's room for improvement. Never claim things are "done" or "finished" - there's always more to improve.
EOF
}

build_priority_order_prompt() {
  cat <<'EOF'
PRIORITY ORDER (REPORT-FOCUSED): 
1) Run experiments if needed (Step 1 automatically runs via @agent_execute.sh - checks checkpoint/, outputs/experiments/, outputs/backtest/ to determine what's needed)
2) Generate required tables as specified in WORKFLOW.md (3 tables: dataset/params, standardized MSE/MAE for forecasting, nowcasting backtest monthly) from outputs/experiments/aggregated_results.csv, outputs/comparisons/, and outputs/backtest/
3) Generate required plots as specified in WORKFLOW.md (4 types: forecast vs actual per target, accuracy heatmap, performance trend, nowcasting comparison) using nowcasting-report/code/plot.py
4) Write/update report sections using generated tables and plots - ensure all theoretically correct details are documented as specified in WORKFLOW.md
5) Research and improve DDFM metrics - analyze results, experiment with improvements, document findings
6) Remove legacy code - clean up deprecated/unused code (dfm-python updated to latest version)
7) Update LaTeX tables in nowcasting-report/tables/ with actual results
8) OPTIONAL: Check LaTeX PDF compilation if needed (LaTeX is installed: cd nowcasting-report && ./compile.sh) - verify page count <15, check for errors. All build artifacts are saved to compiled/ directory.
9) COMMIT & PUSH: Ensure all changes are committed and pushed to remote origin

CRITICAL: Tables and plots MUST be regenerated whenever experiment results are updated. Use:
- python3 nowcasting-report/code/table.py (generates all LaTeX tables)
- python3 nowcasting-report/code/plot.py (generates all plots)
These scripts automatically check for available data and generate tables/plots accordingly.
EOF
}

build_common_prompt_suffix() {
  cat <<'EOF'
Check @FEEDBACK.md for user feedback and incorporate feedback items. When planning tasks, consider which experiments are needed and which appear complete (check outputs/ directory - verify actual state, don't assume). If new experiments are needed, update run_experiment.sh to include only missing experiments. Focus on incremental tasks and continuous improvement. Keep ISSUES.md under 1000 lines. Do not create new files. Remember: there's always room for improvement - never claim things are "done" or "complete".
EOF
}

# Cursor agent execution
run_cursor_with_retry() {
  local cmd_type="$1"
  shift
  local prompt="$*"
  local attempt=1
  local exit_code=0
  
  if ! command -v cursor-agent >/dev/null 2>&1; then
    log_error "cursor-agent command not found"
    return 127
  fi
  
  while [[ $attempt -le $MAX_RETRIES ]]; do
    log_info "cursor-agent attempt $attempt/$MAX_RETRIES ($cmd_type)"
    
    local full_prompt
    full_prompt=$(workflow_context)
    full_prompt+="$prompt"
    
    local cmd_args=()
    case "$cmd_type" in
      text)
        cmd_args=(cursor-agent -p "$full_prompt")
        ;;
      force)
        cmd_args=(cursor-agent -p --force "$full_prompt")
        ;;
      stream)
        cmd_args=(cursor-agent -p --force --output-format stream-json --stream-partial-output "$full_prompt")
        ;;
      *)
        die "unknown cursor command type: $cmd_type"
        ;;
    esac
    
    if "${cmd_args[@]}"; then
      log_info "cursor-agent succeeded on attempt $attempt"
      return 0
    else
      exit_code=$?
      log_warn "cursor-agent failed on attempt $attempt with exit code $exit_code"
      [[ $attempt -lt $MAX_RETRIES ]] && log_info "Retrying in ${RETRY_DELAY} seconds..." && sleep "$RETRY_DELAY"
      attempt=$((attempt + 1))
    fi
  done
  
  log_error "cursor-agent failed after $MAX_RETRIES attempts (final exit code: $exit_code)"
  return $exit_code
}

cursor_text() { run_cursor_with_retry "text" "$@"; }
cursor_force() { run_cursor_with_retry "force" "$@"; }
cursor_stream() { run_cursor_with_retry "stream" "$@"; }

# Prerequisites validation
validate_prerequisites() {
  local step="$1"
  log_info "Validating prerequisites for step $step"
  
  # Check disk space
  if command -v df >/dev/null 2>&1; then
    local avail_kb
    avail_kb=$(df "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || echo "")
    if [[ -n "$avail_kb" ]] && [[ "$avail_kb" =~ ^[0-9]+$ ]] && [[ $avail_kb -lt 1048576 ]]; then
      log_warn "Low disk space: ${avail_kb}KB available (need at least 1GB)"
    fi
  fi
  
  case "$step" in
    1)
      assert_file "${REPO_ROOT}/agent_execute.sh" || log_error "agent_execute.sh not found - this is the ONLY entry point for experiments"
      assert_file "${REPO_ROOT}/run_train.sh" || log_warn "run_train.sh not found"
      assert_file "${REPO_ROOT}/run_forecast.sh" || log_warn "run_forecast.sh not found"
      assert_file "${REPO_ROOT}/run_nowcast.sh" || log_warn "run_nowcast.sh not found"
      check_dfm_package || true
      ;;
    3|4|5|6|7|8)
      assert_file "${REPO_ROOT}/ISSUES.md" || log_warn "ISSUES.md not found, step may fail"
      ;;
    4)
      if [[ ! -d "${REPO_ROOT}/outputs" ]]; then
        log_error "outputs/ directory not found"
        return 1
      fi
      ;;
    9)
      if ! require_cmd git; then
        log_error "git command not found"
        return 1
      fi
      if [[ ! -d "${REPO_ROOT}/.git" ]]; then
        log_error "not a git repository"
        return 1
      fi
      ;;
  esac
  return 0
}

# Git operations
safe_git_add() {
  cd "$REPO_ROOT"
  git add -u
  for f in STATUS.md ISSUES.md CONTEXT.md; do
    if [[ -f "$f" ]] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      log_info "Added untracked file: $f"
    fi
  done
}

safe_git_commit() {
  cd "$REPO_ROOT"
  local commit_msg="$1"
  
  if git diff --quiet --cached && git diff --quiet; then
    log_info "No changes to commit"
    return 0
  fi
  
  if ! git diff --quiet; then
    log_info "Uncommitted changes detected, staging them"
    safe_git_add
  fi
  
  if git diff --quiet --cached; then
    log_warn "No staged changes after git add, nothing to commit"
    return 0
  fi
  
  if git commit -m "$commit_msg"; then
    log_info "Committed changes: $commit_msg"
    return 0
  else
    log_error "Git commit failed"
    return 1
  fi
}

safe_git_submodule_push() {
  cd "$REPO_ROOT" || {
    log_error "Failed to cd to $REPO_ROOT for submodule push"
    return 1
  }
  
  log_info "Pushing submodules to origin main (iteration ${ITERATION})"
  local failed=0
  
  # Push nowcasting-report submodule (primary report that user reviews)
  if [[ -d "nowcasting-report/.git" ]]; then
    log_info "Pushing nowcasting-report submodule..."
    cd "nowcasting-report" || {
      log_error "Failed to cd to nowcasting-report"
      failed=1
    }
    if git push origin main 2>&1; then
      log_info "✓ nowcasting-report pushed successfully"
    else
      log_warn "⚠ nowcasting-report push had errors"
      failed=1
    fi
    cd "$REPO_ROOT" || {
      log_error "Failed to return to $REPO_ROOT"
      return 1
    }
  else
    log_warn "nowcasting-report/.git not found - skipping push"
  fi
  
  # Optionally push dfm-python submodule (less critical)
  if [[ -d "dfm-python/.git" ]]; then
    log_info "Pushing dfm-python submodule..."
    cd "dfm-python" || {
      log_warn "Failed to cd to dfm-python - continuing"
      return $failed
    }
    if git push origin main 2>&1; then
      log_info "✓ dfm-python pushed successfully"
    else
      log_warn "⚠ dfm-python push had errors (non-critical)"
      # Don't mark as failed for dfm-python - it's less critical
    fi
    cd "$REPO_ROOT" || {
      log_error "Failed to return to $REPO_ROOT"
      return 1
    }
  fi
  
  [[ $failed -eq 1 ]] && log_warn "Some submodule pushes had errors, but continuing"
  return $failed
}

# Table and plot generation functions
generate_tables_and_plots() {
  log_info "Generating tables and plots from experiment results"
  activate_venv
  cd "$REPO_ROOT" || {
    log_error "Failed to cd to $REPO_ROOT"
    return 1
  }
  
  local tables_generated=0
  local plots_generated=0
  
  # Generate tables
  log_info "Generating LaTeX tables..."
  if python3 nowcasting-report/code/table_forecasts.py 2>&1 && python3 nowcasting-report/code/table_nowcasts.py 2>&1; then
    log_info "✓ Tables generated successfully"
    tables_generated=1
  else
    log_warn "⚠ Table generation had errors (may be due to missing data)"
  fi
  
  # Generate plots
  log_info "Generating plots..."
  if python3 nowcasting-report/code/plot_forecasts.py 2>&1 && python3 nowcasting-report/code/plot_nowcasts.py 2>&1; then
    log_info "✓ Plots generated successfully"
    plots_generated=1
  else
    log_warn "⚠ Plot generation had errors (may be due to missing data)"
  fi
  
  # Verify outputs
  if [[ $tables_generated -eq 1 ]]; then
    local table_count=0
    if [[ -d "nowcasting-report/tables" ]]; then
      table_count=$(find nowcasting-report/tables -name "*.tex" 2>/dev/null | wc -l)
      log_info "Found $table_count LaTeX table(s) in nowcasting-report/tables/"
    fi
  fi
  
  if [[ $plots_generated -eq 1 ]]; then
    local plot_count=0
    if [[ -d "nowcasting-report/images" ]]; then
      plot_count=$(find nowcasting-report/images -name "*.png" 2>/dev/null | wc -l)
      log_info "Found $plot_count plot(s) in nowcasting-report/images/"
    fi
  fi
  
  return 0
}

# Step implementations
step1_run_experiment() {
  ensure_env
  log_info "Starting step 1: Run experiments (automatic execution via agent_execute.sh)"
  validate_prerequisites 1
  activate_venv
  cd "$REPO_ROOT"
  
  # Check if agent_execute.sh exists
  if [[ ! -f "${REPO_ROOT}/agent_execute.sh" ]]; then
    log_error "agent_execute.sh not found - this is the ONLY entry point for experiments"
    return 1
  fi
  
  if ! check_dfm_package; then
    log_warn "⚠ WARNING: dfm-python package not available - DFM/DDFM experiments will fail"
    log_warn "⚠ To fix: Install dfm-python dependencies: cd dfm-python && pip install -e ."
  fi
  
  # agent_execute.sh automatically checks what's needed and runs only missing experiments
  # No parameters needed - it handles everything automatically
  log_info "Running agent_execute.sh (automatically checks and runs only missing experiments)"
  if bash agent_execute.sh; then
    log_info "Step 1 completed successfully"
    
    # Regenerate tables and plots after experiments complete
    log_info "Regenerating tables and plots after experiments..."
    generate_tables_and_plots || log_warn "Table/plot generation had errors, but continuing"
    
    mark_step_completed 1
    return 0
  else
    log_error "Step 1 failed, but continuing to next step"
    return 1
  fi
}

step2_report_writing() {
  ensure_env
  log_info "Starting step 2: Report writing - Analyze results and write/update report sections"
  validate_prerequisites 2
  cd "$REPO_ROOT"
  
  local out_dir
  out_dir=$(latest_output_dir 2>/dev/null || echo "")
  
  local prompt="Write and update report sections in @nowcasting-report/contents/ based on experiment results. Check @STATUS.md and @ISSUES.md for current state and pending tasks. Check @FEEDBACK.md for any user feedback that needs to be addressed. Focus on turning latest forecast/nowcast outputs into text, tables, and plots, and flag any missing runs needed for the next iteration. "
  prompt+="$(build_report_priority_prompt) "
  
  if [[ -n "$out_dir" ]] && [[ -d "$out_dir" ]]; then
    prompt+="Study the experiment run output in ${out_dir} (latest run). Analyze results and write/update report sections accordingly. If tables/plots are missing, GENERATE them using nowcasting-report/code/table.py and nowcasting-report/code/plot.py. If report sections are incomplete, WRITE them. Don't just say 'should be written' - actually WRITE the content. Check which experiments appear to be completed by examining checkpoint/, outputs/comparisons/, outputs/backtest/ directory structure and log files - but verify actual state, don't assume. "
  fi
  
  prompt+="CRITICAL: Agent ONLY modifies code. Experiment execution is handled automatically by Step 1 via @agent_execute.sh. "
  prompt+="Agent MUST NOT execute any scripts (run_train.sh, run_nowcast.sh, run_forecast.sh, agent_execute.sh). "
  prompt+="If results are missing (checkpoint/ empty, outputs/experiments/ or outputs/backtest/ missing), list them explicitly in ISSUES.md for the next iteration to run. "
  prompt+="Step 1 automatically checks checkpoint/, outputs/experiments/, outputs/backtest/ and runs needed experiments. "
  prompt+="CRITICAL: DO NOT claim things are 'complete', 'verified', 'resolved', or 'no issues' unless you actually FIXED or IMPROVED something. "
  prompt+="If tables/plots are missing, GENERATE them. If report sections are incomplete, WRITE them. Don't just document - actually CREATE the content. "
  prompt+="Current experiment configuration: ${#TARGETS[@]} targets × ${#MODELS[@]} models. Forecasting: 22 horizons (2024-01 to 2025-10). Nowcasting: 22 months × 2 time points. Report structure: 7 sections (Introduction, Methodology, Results, Discussion, Issues) - target under 15 pages. REQUIRED OUTPUT: 3 tables and 4 plot types (including nowcasting comparison). WORKFLOW PRIORITY: 1) Generate tables and plots from outputs/, 2) Write/update report sections, 3) Research DDFM metrics improvements, 4) Clean up legacy code. The report focuses on comparing 4 models across 3 targets for both forecasting and nowcasting. If FEEDBACK.md contains user feedback, prioritize addressing those items. Focus on REPORT WRITING - actually write the content."
  
  if cursor_text "$prompt"; then
    log_info "Step 2 completed successfully"
    mark_step_completed 2
    return 0
  else
    log_error "Step 2 failed, but continuing to next step"
    return 1
  fi
}

step3_report_continue() {
  ensure_env
  log_info "Starting step 3: Report writing - Continue report sections and plan DDFM metrics improvements"
  validate_prerequisites 3
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  
  local prompt="Continue writing report sections in @nowcasting-report/contents/. Plan DDFM metrics improvements based on analysis of experiment results. Update ISSUES.md with concrete, actionable plan for DDFM metrics research. "
  prompt+="$(build_report_priority_prompt) "
  prompt+="$(build_priority_order_prompt) "
  prompt+="CRITICAL: If plan includes running experiments, agent MUST use @agent_execute.sh (ONLY entry point). Agent decides which to run based on checkpoint/, outputs/comparisons/, outputs/backtest/ status. Agent MUST NOT directly execute run_train.sh, run_nowcast.sh, or run_forecast.sh. Explicitly list missing forecast/nowcast artifacts (aggregated_results.csv slices, comparison_results, backtest JSON) so next iteration can run them. "
  prompt+="Focus on: 1) Writing report sections with results and analysis, 2) Planning DDFM metrics improvements, 3) Generating tables/plots if missing, 4) Calling out which forecasts/nowcasts are missing so Step 1 can run them. "
  prompt+="$(build_common_prompt_suffix)"
  
  if cursor_force "$prompt"; then
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    log_info "Step 3 completed successfully"
    mark_step_completed 3
    return 0
  else
    log_error "Step 3 failed, but continuing to next step"
    return 1
  fi
}

step4_report_tables_plots() {
  ensure_env
  log_info "Starting step 4: Report writing - Generate tables and plots, update report with results"
  validate_prerequisites 4 || log_warn "Prerequisites validation failed, but continuing"
  cd "$REPO_ROOT" || {
    log_error "Failed to cd to $REPO_ROOT"
    return 1
  }
  
  local out_dir
  out_dir=$(latest_output_dir)
  if [[ -z "$out_dir" ]] || [[ ! -d "$out_dir" ]]; then
    log_warn "no outputs/ directory found - step may fail"
    # Continue anyway - agent can still work without outputs
  fi
  
  log_info "Generating tables and plots from: $out_dir"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  
  # Generate tables and plots first
  log_info "Regenerating tables and plots..."
  generate_tables_and_plots || log_warn "Table/plot generation had errors, but continuing"
  
  local prompt="Generate tables and plots from experiment results, then update report sections in @nowcasting-report/contents/ to reference them. CRITICAL: 1) Ensure all required tables are generated using nowcasting-report/code/table.py from outputs/experiments/aggregated_results.csv, outputs/comparisons/, and outputs/backtest/. 2) Ensure all required plots are generated using nowcasting-report/code/plot.py. 3) Update report sections to reference generated tables and plots. 4) Check checkpoint/, outputs/comparisons/, outputs/backtest/ to see which experiments appear complete - but verify actual state, don't assume. If experiments are missing, note them in ISSUES.md - Step 1 will automatically run them in the next iteration. Agent ONLY modifies code - Agent MUST NOT execute any scripts. If there are errors or issues, FIX them in code, don't just document. Check @FEEDBACK.md for any user feedback related to results. Update STATUS.md, ISSUES.md, and CONTEXT.md if necessary. Keep all files under ${LINE_LIMIT} lines. Do not create new files. CRITICAL: DO NOT claim 'complete', 'verified', 'resolved', 'no issues', or 'done' unless you actually FIXED something in code. Always acknowledge there's room for improvement."
  
  if cursor_force "$prompt"; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
    
    log_info "Step 4 completed successfully"
    mark_step_completed 4
    return 0
  else
    log_error "Step 4 failed, but continuing to next step"
    return 1
  fi
}

step5_ddfm_metrics_research() {
  ensure_env
  log_info "Starting step 5: DDFM metrics research - Plan and implement DDFM metrics improvements"
  validate_prerequisites 5
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  
  local prompt="Research and improve DDFM metrics. Analyze DDFM performance in experiment results (sMSE, sMAE, sRMSE). Plan and implement improvements to DDFM metrics through code changes, configuration adjustments, or training procedure improvements. Document findings and improvements in report. "
  prompt+="$(build_report_priority_prompt) "
  prompt+="$(build_priority_order_prompt) "
  prompt+="CRITICAL: Focus on DDFM metrics improvement research. Analyze results, experiment with improvements, document findings. If experiments are needed to test improvements, note them in ISSUES.md - Step 1 will handle experiment execution automatically via @agent_execute.sh. Agent ONLY modifies code - Agent MUST NOT execute any scripts. If there are improvement points in DDFM metrics calculation or evaluation, FIX them in code. If there are improvement points in DDFM training or configuration, FIX them in code. Document improvements in report. CRITICAL: DO NOT claim 'complete', 'verified', 'resolved', or 'no issues' unless you actually FIXED something. If you find problems, FIX them, don't just document that 'everything is fine'. "
  prompt+="$(build_common_prompt_suffix)"
  
  if cursor_force "$prompt"; then
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    log_info "Step 5 completed successfully"
    mark_step_completed 5
    return 0
  else
    log_error "Step 5 failed, but continuing to next step"
    return 1
  fi
}

step6_report_continue() {
  ensure_env
  log_info "Starting step 6: Report writing - Continue report sections and incorporate DDFM improvements"
  validate_prerequisites 6
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  
  local prompt="Continue writing report sections in @nowcasting-report/contents/. Incorporate DDFM metrics improvements into report. Execute the active plan items in ISSUES.md following the priority order. CRITICAL PRIORITIES: 1) WRITE/UPDATE REPORT SECTIONS: If report sections are incomplete, WRITE them. Don't just verify - actually CREATE the content. 2) INCORPORATE DDFM IMPROVEMENTS: Document DDFM metrics improvements and findings in report. 3) ENSURE REPORT DOCUMENTATION: If tables/plots are missing, GENERATE them using nowcasting-report/code/table.py and nowcasting-report/code/plot.py. 4) REMOVE LEGACY CODE: Clean up deprecated/unused code (dfm-python updated to latest version). "
  prompt+="CRITICAL: Agent ONLY modifies code - Agent MUST NOT execute any scripts (run_train.sh, run_nowcast.sh, run_forecast.sh, agent_execute.sh). If experiments are missing, note them in ISSUES.md - Step 1 will automatically run them in the next iteration. "
  prompt+="CRITICAL: DO NOT claim 'complete', 'verified', 'resolved', or 'no issues' unless you actually FIXED something in code or files. If you find problems, FIX them. If tables/plots are missing, GENERATE them. If report is incomplete, WRITE it. Don't just document that 'everything is fine' - actually MAKE CHANGES. "
  prompt+="$(build_priority_order_prompt) "
  prompt+="Check @FEEDBACK.md for user feedback and incorporate feedback into updates. Apply code/report updates needed. After code changes, ensure tables and plots are regenerated. OPTIONAL: If LaTeX is available, check PDF compilation (cd nowcasting-report && ./compile.sh) to verify report compiles correctly. All build artifacts are saved to compiled/ directory. Focus on incremental improvements and FIX as many problems as possible. Do not create new files. Use existing files only and keep STATUS.md, ISSUES.md, and CONTEXT.md under ${LINE_LIMIT} lines."
  
  if cursor_stream "$prompt"; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
    
    # Generate tables and plots after agent work
    log_info "Regenerating tables and plots after code changes..."
    generate_tables_and_plots || log_warn "Table/plot generation had errors, but continuing"
    
    log_info "Step 6 completed successfully"
    mark_step_completed 6
    return 0
  else
    log_error "Step 6 failed, but continuing to next step"
    return 1
  fi
}

step7_report_finalize() {
  ensure_env
  log_info "Starting step 7: Report writing - Finalize report sections, compile PDF, verify content"
  validate_prerequisites 7
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  
  local prompt="Finalize report sections in @nowcasting-report/contents/. Compile PDF and verify content. Continue remaining items from ISSUES.md following the priority order. CRITICAL PRIORITIES: 1) FINALIZE REPORT SECTIONS: Complete any unfinished report sections, ensure all tables and plots are referenced correctly. 2) COMPILE PDF: If LaTeX is available, compile PDF (cd nowcasting-report && ./compile.sh) to verify report compiles correctly. 3) VERIFY CONTENT: Check that all required tables, plots, and sections are present and correct. 4) REMOVE LEGACY CODE: Clean up deprecated/unused code (dfm-python updated to latest version). "
  prompt+="CRITICAL: Agent ONLY modifies code - Agent MUST NOT execute any scripts (run_train.sh, run_nowcast.sh, run_forecast.sh, agent_execute.sh). If experiments are missing, note them in ISSUES.md - Step 1 will automatically run them in the next iteration. "
  prompt+="CRITICAL: DO NOT claim 'complete', 'verified', 'resolved', or 'no issues' unless you actually FIXED something in code or files. If you find problems, FIX them. If tables/plots are missing, GENERATE them. If report is incomplete, WRITE it. Don't just document that 'everything is fine' - actually MAKE CHANGES. "
  prompt+="$(build_priority_order_prompt) "
  prompt+="Check @FEEDBACK.md for any new user feedback and incorporate into ongoing work. After code changes, ensure tables and plots are regenerated. OPTIONAL: If LaTeX is available and report sections are updated, check PDF compilation to verify changes compile correctly. Focus on incremental improvements and FIX as many problems as possible. Do not create new files. Keep STATUS.md, ISSUES.md, and CONTEXT.md under ${LINE_LIMIT} lines."
  
  if cursor_stream "$prompt"; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
    
    # Generate tables and plots after agent work
    log_info "Regenerating tables and plots after code changes..."
    generate_tables_and_plots || log_warn "Table/plot generation had errors, but continuing"
    
    log_info "Step 7 completed successfully"
    mark_step_completed 7
    return 0
  else
    log_error "Step 7 failed, but continuing to next step"
    return 1
  fi
}

step8_summarize_iteration() {
  ensure_env
  log_info "Starting step 8: Summarize iteration"
  validate_prerequisites 8
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  
  local prompt="Identify the work done in this iteration. Be HONEST about what's actually done vs what's not done. Update STATUS.md and ISSUES.md for the next iteration. Only mark issues as addressed if you actually attempted to FIX them in code - but acknowledge there may still be improvements needed. Remove old addressed issues to keep file under ${LINE_LIMIT} lines. Update experiment status in STATUS.md (appears complete/pending combinations) - be ACCURATE, don't claim things are complete when they're not. Document findings: report writing status, DDFM metrics improvement research status, table/plot generation status, legacy code cleanup status. CRITICAL: DO NOT claim 'complete', 'verified', 'resolved', 'no issues', 'done', or 'production ready' unless you actually FIXED or IMPROVED something. Always acknowledge there's room for improvement. If checkpoint/ is empty, say models are NOT trained. If outputs/backtest/ doesn't exist, say nowcasting is NOT done. Be HONEST about the actual state. Note: Changes will be committed and pushed to remote origin in step 9. Submodules are pushed every 10 iterations - user will review report and provide feedback in FEEDBACK.md. LaTeX PDF compilation is optional (LaTeX is installed) - can be checked if needed, but focus on ensuring all changes are committed and pushed. Next iteration will start fresh so you need to leave the proper context for next iteration. Keep each file under ${LINE_LIMIT} lines. Do not create new files."
  
  if cursor_force "$prompt"; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
    log_info "Step 8 completed successfully"
    mark_step_completed 8
    return 0
  else
    log_error "Step 8 failed, but continuing to next step"
    return 1
  fi
}

safe_git_push() {
  cd "$REPO_ROOT" || {
    log_error "Failed to cd to $REPO_ROOT for push"
    return 1
  }
  
  # Get current branch name
  local current_branch
  current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
  
  log_info "Pushing main repository to origin ${current_branch} (iteration ${ITERATION})"
  
  if git push origin "${current_branch}" 2>&1; then
    log_info "✓ Main repository pushed successfully to origin ${current_branch}"
    return 0
  else
    log_warn "⚠ Main repository push had errors (may need manual push or remote not configured)"
    return 1
  fi
}

step9_commit() {
  ensure_env
  log_info "Starting step 9: Commit and push changes (iteration ${ITERATION})"
  validate_prerequisites 9 || log_warn "Prerequisites validation failed, but continuing"
  cd "$REPO_ROOT" || {
    log_error "Failed to cd to $REPO_ROOT"
    return 1
  }
  
  if [[ -z "${COMMIT_MSG:-}" ]]; then
    log_error "COMMIT_MSG must be set for step 9 - using default"
    COMMIT_MSG="Iteration ${ITERATION}"
  fi
  
  guard_line_limit "${REPO_ROOT}/STATUS.md"
  guard_line_limit "${REPO_ROOT}/ISSUES.md"
  guard_line_limit "${REPO_ROOT}/CONTEXT.md"
  
  if safe_git_commit "$COMMIT_MSG"; then
    # Push main repository to origin
    log_info "Pushing main repository to origin..."
    safe_git_push || log_warn "Main repository push had errors but continuing"
    
    # Push submodules every 10 iterations (iterations 10, 20, 30, ...)
    if [[ "${ITERATION:-0}" =~ ^[0-9]+$ ]] && (( ITERATION % 10 == 0 )); then
      log_info "Pushing submodules to origin main (iteration ${ITERATION} - user will review and provide feedback)"
      if safe_git_submodule_push; then
        log_info "✓ Submodules pushed successfully"
      else
        log_warn "⚠ Submodule push had errors but continuing"
      fi
      log_info "Note: User will review report in nowcasting-report/ and provide feedback in FEEDBACK.md"
    else
      log_info "Skipping submodule push (iteration ${ITERATION} - push happens every 10 iterations)"
    fi
    log_info "Step 9 completed successfully"
    mark_step_completed 9
    return 0
  else
    log_error "Step 9 git commit failed, but iteration completed"
    # Still try to push main repository and submodules
    log_info "Attempting to push main repository despite commit failure (iteration ${ITERATION})"
    safe_git_push || log_warn "Main repository push had errors but continuing"
    
    if [[ "${ITERATION:-0}" =~ ^[0-9]+$ ]] && (( ITERATION % 10 == 0 )); then
      log_info "Attempting submodule push despite commit failure (iteration ${ITERATION})"
      safe_git_submodule_push || log_warn "Submodule push had errors but continuing"
    fi
    return 1
  fi
}

# Cleanup
cleanup_old_backups() {
  [[ ! -d "$BACKUP_DIR" ]] && return 0
  for f in STATUS.md ISSUES.md CONTEXT.md; do
    ls -t "${BACKUP_DIR}/${f}.backup."* 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true
  done
}

# Step runner
run_step() {
  local step="$1"
  log_info "-----------------------------------------"
  log_info "Running step $step of iteration ${ITERATION}"
  log_info "-----------------------------------------"
  
  local step_result=0
  case "$step" in
    1) step1_run_experiment || step_result=1 ;;
    2) step2_report_writing || step_result=1 ;;
    3) step3_report_continue || step_result=1 ;;
    4) step4_report_tables_plots || step_result=1 ;;
    5) step5_ddfm_metrics_research || step_result=1 ;;
    6) step6_report_continue || step_result=1 ;;
    7) step7_report_finalize || step_result=1 ;;
    8) step8_summarize_iteration || step_result=1 ;;
    9) step9_commit || step_result=1 ;;
    *)
      log_error "Invalid step: $step"
      step_result=1
      ;;
  esac
  
  if [[ $step_result -eq 1 ]]; then
    log_warn "Step $step had errors, but pipeline continues"
  fi
  
  # Always return 0 to ensure pipeline continues
  return 0
}

# Main
main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1  # Only exit on usage error
  fi
  
  local num_iters="$1"
  if ! [[ "$num_iters" =~ ^[0-9]+$ ]] || [[ "$num_iters" -lt 1 ]]; then
    log_error "Invalid number of iterations: $num_iters (must be >= 1)"
    usage
    exit 1  # Only exit on invalid input
  fi
  
  mkdir -p "$LOG_DIR"
  
  log_info "========================================="
  log_info "Running ${num_iters} iteration(s) (iterations 1-${num_iters})"
  log_info "Each iteration runs steps 1-9"
  log_info "Timestamp: $(date)"
  log_info "========================================="
  
  show_step_history
  
  for iter in $(seq 1 "$num_iters"); do
    ITERATION="$iter"
    [[ -z "${COMMIT_MSG:-}" ]] && COMMIT_MSG="Iteration ${iter}"
    
    log_info "========================================="
    log_info "Starting iteration ${iter} (steps 1-9)"
    log_info "COMMIT_MSG: ${COMMIT_MSG}"
    log_info "========================================="
    
    (( iter % 10 == 0 )) && cleanup_old_backups
    
    for step in $(seq 1 9); do
      run_step "$step"
    done
    
    log_info "========================================="
    log_info "Iteration ${iter} completed successfully"
    log_info "========================================="
    
    unset COMMIT_MSG
  done
  
  log_info "========================================="
  log_info "All ${num_iters} iteration(s) completed successfully"
  log_info "Iterations 1-${num_iters} finished"
  log_info "========================================="
}

main "$@"
