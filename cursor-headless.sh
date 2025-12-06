#!/usr/bin/env bash
# Cursor headless workflow runner for nowcasting-kr
# Implements the steps described in WORKFLOW.md using cursor-agent in print mode.
# No new files are created; existing files are updated in-place when needed.

set -euo pipefail

REPO_ROOT="/data/nowcasting-kr"
VENV_PATH="${REPO_ROOT}/.venv/bin/activate"
LINE_LIMIT=1000
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"  # seconds
LOG_DIR="${REPO_ROOT}/.cursor-logs"
BACKUP_DIR="${LOG_DIR}/backups"
STEP_STATUS_FILE="${REPO_ROOT}/.cursor-step-status"
ITERATION="${ITERATION:-0}"

usage() {
  cat <<'EOF'
Usage: cursor-headless.sh <number_of_iterations>

Runs the specified number of iterations (each iteration runs steps 1-9).

Arguments:
  number_of_iterations  Number of iterations to run (e.g., 1, 2, 3, ...)
                        Runs iterations 1 through N sequentially

Steps in each iteration (match WORKFLOW.md):
  1   Run experiment (@run_experiment.sh)
  2   Inspect codebases (src/, dfm-python/, nowcasting-report/) - fresh start, check FEEDBACK.md
  3   Work on the plan from step 2 (incorporate FEEDBACK.md)
  4   Analyze results; update STATUS/ISSUES/CONTEXT
  5   Plan improvements for dfm-python & report (check FEEDBACK.md)
  6   Execute improvement plan (code/report updates, incorporate FEEDBACK.md)
  7   Continue remaining plan items (check FEEDBACK.md)
  8   Summarize iteration; refresh CONTEXT/STATUS/ISSUES
  9   Stage & commit (push submodules every 2nd iteration for user review)

Environment:
  COMMIT_MSG - Commit message for step 9 (default: "Iteration N")
  
Optional environment variables:
  MAX_RETRIES - Number of retry attempts on failure (default: 3)
  RETRY_DELAY - Delay between retries in seconds (default: 5)

Examples:
  ./cursor-headless.sh 1  # Run 1 iteration (iteration 1)
  ./cursor-headless.sh 2  # Run 2 iterations (iteration 1, 2)
  ./cursor-headless.sh 5  # Run 5 iterations (iteration 1, 2, 3, 4, 5)
EOF
}

die() { echo "error: $*" >&2; exit 1; }

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

# Step tracking functions
mark_step_completed() {
  local step="$1"
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  mkdir -p "$(dirname "$STEP_STATUS_FILE")"
  echo "$step|$timestamp|${ITERATION}" >> "$STEP_STATUS_FILE"
  log_info "Marked step $step as completed at $timestamp (iteration ${ITERATION})"
}

get_step_status() {
  local step="$1"
  if [[ ! -f "$STEP_STATUS_FILE" ]]; then
    return 1
  fi
  grep "^${step}|" "$STEP_STATUS_FILE" | tail -n1
}


show_step_history() {
  log_info "Step completion history:"
  if [[ -f "$STEP_STATUS_FILE" ]]; then
    local count=0
    while IFS='|' read -r step timestamp iteration; do
      count=$((count + 1))
      log_info "  Step $step completed at $timestamp (iteration $iteration)"
    done < <(tail -n 10 "$STEP_STATUS_FILE" 2>/dev/null)
    if [[ $count -eq 0 ]]; then
      log_info "  No steps completed yet"
    fi
  else
    log_info "  No steps completed yet"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

ensure_env() {
  require_cmd cursor-agent
  require_cmd bash
}

activate_venv() {
  [[ -f "$VENV_PATH" ]] || die "virtualenv not found at $VENV_PATH"
  # shellcheck disable=SC1090
  source "$VENV_PATH"
}

assert_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

guard_line_limit() {
  local f="$1"
  [[ -f "$f" ]] || die "file missing for line limit check: $f"
  local lines
  lines=$(wc -l < "$f")
  if [[ "$lines" -gt "$LINE_LIMIT" ]]; then
    log_error "File $f exceeds ${LINE_LIMIT} lines ($lines lines)"
    die "$f exceeds ${LINE_LIMIT} lines ($lines)"
  fi
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
  # Portable version using ls -lt instead of GNU find -printf
  if [[ ! -d "${REPO_ROOT}/outputs" ]]; then
    return 1
  fi
  # Use ls -lt to sort by modification time, get most recent directory
  ls -1td "${REPO_ROOT}/outputs"/*/ 2>/dev/null | head -n1 | sed 's|/$||' || return 1
}

workflow_context() {
  cat <<'EOF'
# RULES
- DO NOT CREATE FILES or MARKDOWNS
- DO NOT DELETE FILES
- ONLY WORK under modifying existing codes.
- NEVER HALLUCINATE. Only use the references at references.bib when writing the report.
- When need to add new information from knowledgebase, add the citation at references.bib
- src/ should contain maximum 15 files including __init__.py file. If necessary, restructure and consolidate.
- CONTEXT.md, STATUS.md, ISSUES.md MUST BE UNDER 1000 lines.
- Try to improve incrementally. Do not try to do everything at once. Try to prioritize the tasks and work on them one by one.

# GOAL
- Write the Complete report (under 15 pages) in @nowcasting-report/ comparing 4 models (ARIMA, VAR, DFM, DDFM) on 3 targets (KOEQUIPTE, KOWRCCNSE, KOIPALL.G)
- Finalize the package @dfm-python/ with clean code pattern, consistent and generic naming

# EXPERIMENT CONFIGURATION
- Targets: 3 (KOEQUIPTE - Equipment Investment, KOWRCCNSE - Wholesale/Retail Sales, KOIPALL.G - Industrial Production)
- Models: 4 (ARIMA, VAR, DFM, DDFM)
- Horizons: 3 (1, 7, 28 days)
- Total: 36 combinations (3 × 4 × 3)
- Series configs: All series use block: null (only global block for DFM/DDFM)
- Data file: data/data.csv

# REPORT STRUCTURE
- 6 sections: Introduction, Methodology, Production Model (KOIPALL.G), Investment Model (KOEQUIPTE), Consumption Model (KOWRCCNSE), Conclusion
- Target: Under 15 pages (condensed from previous 20-30 page target)
- Tables: tab_overall_metrics, tab_by_target, tab_by_horizon
- Focus: Compare ARIMA, VAR, DFM, DDFM models across 3 targets

# DESIRED OUTPUT (from WORKFLOW.md)
## Tables (3 required):
1. Table with dataset details and model parameters (ARIMA, VAR, DFM, DDFM model and training params)
2. Table with standardized MSE and standardized MAE for (target, model, horizon) pairs - 36 rows (3 targets × 4 models × 3 horizons)
3. Table with DFM and DDFM backtest results for 2024-2025 by month (train 1985-2019, nowcast Jan 2024-Oct 2025, sMSE/sMAE per month)

## Images (3 required):
1. Forecast vs actual plots: 3 plots (one per target), each showing 60 months total - 30 months historical (single original line) + 30 months forecasts (5 lines: original, ARIMA, VAR, DFM, DDFM). X-axis: monthly timestamp, Y-axis: target series.
2. Accuracy heatmap: 4 models × 3 targets (standardized RMSE)
3. Performance trend: Performance by forecasting horizon (1, 7, 28 days)

# WORKFLOW PRIORITY:
1. FIRST: Generate tables and plots from experiment results (outputs/comparisons/, outputs/experiments/)
2. THEN: Build/update report sections using the generated tables and plots
3. Use nowcasting-report/code/plot.py to generate images, update LaTeX tables in nowcasting-report/tables/

# RESOURCES
- CONTEXT.md: Use this file for context offloading for persistence if necessary.
- STATUS.md: Use this file to track the progress and leave the status for next iteration on updates.
- ISSUES.md: Track resolved issues and next steps. Keep file under 1000 lines. Mark resolved issues clearly.
- FEEDBACK.md: User feedback file - Check this file regularly (especially after every 2 iterations when submodules are pushed). User will provide feedback on the report content, structure, and quality. Incorporate feedback into improvements.
- src/ : engine for running the experiment. This module provides wrapper for @sktime and @dfm-python packages with preprocessing - training - inference. Maximum 15 files including __init__.py.
- dfm-python/ : Core DFM/DDFM package - finalized with clean code patterns, consistent naming, legacy code cleaned up.
- nowcasting-report/ : LaTeX report submodule - Structure ready, need to populate with experiment results. User reviews report every 2 iterations and provides feedback in FEEDBACK.md.
- nowcasting-report/code/plot.py : Code for creating plots used in the paper based on the results in outputs/ directory. Images should be created at nowcasting-report/images/*.png and used in the report properly.
- nowcasting-report/tables/ : LaTeX table files that need to be populated with actual results
- neo4j mcp : knowledgebase containing references. NEVER hallucinate.
- outputs/ : directory containing experiment results from @run_experiment.sh
- outputs/comparisons/ : Per-target comparison results (comparison_results.json, comparison_table.csv)
- outputs/experiments/ : Aggregated results (aggregated_results.csv - 36 rows)
- config/ : Hydra YAML configs in config/experiment/ (3 target configs: koequipte_report, kowrccnse_report, koipallg_report), config/model/, config/series/ (all series have block: null)
- run_test_experiment.sh : Test script to verify all targets and models before full run
- DDFM_COMPARISON.md : Comparison of original ddfm implementation and dfm-python

# FEEDBACK PROCESS
- User reviews the report in nowcasting-report/ submodule every 2 iterations (when submodules are pushed to origin main)
- User provides feedback in FEEDBACK.md file
- Check FEEDBACK.md regularly and incorporate feedback into report improvements
- Focus on: report clarity, content accuracy, structure, missing information, formatting issues

EOF
}

run_cursor_with_retry() {
  local cmd_type="$1"
  shift
  local prompt="$*"
  local attempt=1
  local exit_code=0
  
  # Verify cursor-agent is available before retrying
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
      if [[ $attempt -lt $MAX_RETRIES ]]; then
        log_info "Retrying in ${RETRY_DELAY} seconds..."
        sleep "$RETRY_DELAY"
      fi
      attempt=$((attempt + 1))
    fi
  done
  
  log_error "cursor-agent failed after $MAX_RETRIES attempts (final exit code: $exit_code)"
  return $exit_code
}

cursor_text() {
  run_cursor_with_retry "text" "$@"
}

cursor_force() {
  run_cursor_with_retry "force" "$@"
}

cursor_stream() {
  run_cursor_with_retry "stream" "$@"
}

validate_prerequisites() {
  local step="$1"
  log_info "Validating prerequisites for step $step"
  
  # Check disk space (at least 1GB free)
  if command -v df >/dev/null 2>&1; then
    local avail_kb
    avail_kb=$(df "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || echo "")
    if [[ -n "$avail_kb" ]] && [[ "$avail_kb" =~ ^[0-9]+$ ]] && [[ $avail_kb -lt 1048576 ]]; then
      log_warn "Low disk space: ${avail_kb}KB available (need at least 1GB)"
    fi
  fi
  
  # Validate required files exist
  case "$step" in
    1)
      assert_file "${REPO_ROOT}/run_experiment.sh"
      ;;
    3|4|5|6|7|8)
      assert_file "${REPO_ROOT}/ISSUES.md"
      ;;
    4)
      [[ -d "${REPO_ROOT}/outputs" ]] || die "outputs/ directory not found"
      ;;
    9)
      require_cmd git
      [[ -d "${REPO_ROOT}/.git" ]] || die "not a git repository"
      ;;
  esac
}

safe_git_add() {
  cd "$REPO_ROOT"
  # Only add tracked files and specific markdown files, not everything
  git add -u  # Update tracked files
  # Add specific files if they exist and are untracked
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
  
  # Check if there are no changes to commit (both staged and unstaged)
  if git diff --quiet --cached && git diff --quiet; then
    log_info "No changes to commit"
    return 0
  fi
  
  # Check for uncommitted changes and stage them
  if ! git diff --quiet; then
    log_info "Uncommitted changes detected, staging them"
    safe_git_add
  fi
  
  # Check again if we have anything staged after adding
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
  cd "$REPO_ROOT"
  log_info "Pushing submodules to origin main (iteration ${ITERATION})"
  local failed=0
  
  # Push each submodule individually to handle errors gracefully
  if [ -d "nowcasting-report/.git" ]; then
    log_info "Pushing nowcasting-report submodule..."
    cd nowcasting-report
    if git push origin main 2>&1; then
      log_info "✓ nowcasting-report pushed successfully"
    else
      log_warn "⚠ nowcasting-report push had errors"
      failed=1
    fi
    cd "$REPO_ROOT"
  fi
  
  if [ -d "dfm-python/.git" ]; then
    log_info "Pushing dfm-python submodule..."
    cd dfm-python
    if git push origin main 2>&1; then
      log_info "✓ dfm-python pushed successfully"
    else
      log_warn "⚠ dfm-python push had errors"
      failed=1
    fi
    cd "$REPO_ROOT"
  fi
  
  if [[ $failed -eq 1 ]]; then
    log_warn "Some submodule pushes had errors, but continuing"
    return 1
  fi
  return 0
}

step1_run_experiment() {
  ensure_env
  log_info "Starting step 1: Run experiment"
  validate_prerequisites 1
  activate_venv
  cd "$REPO_ROOT"
  log_info "Note: run_experiment.sh runs experiments for 3 targets (KOEQUIPTE, KOWRCCNSE, KOIPALL.G). Check outputs/ directory for existing results before running new experiments."
  log_info "For incremental testing, use MODELS filter: MODELS=\"dfm\" bash run_experiment.sh or MODELS=\"ddfm\" bash run_experiment.sh"
  log_info "For verification, use run_test_experiment.sh first to verify all targets and models work correctly."
  log_info "Current configuration: 3 targets × 4 models × 3 horizons = 36 combinations"
  if bash run_experiment.sh; then
    log_info "Step 1 completed successfully"
    mark_step_completed 1
    return 0
  else
    log_error "Step 1 failed, but continuing to next step"
    return 1
  fi
}

step2_inspect_code() {
  ensure_env
  log_info "Starting step 2: Inspect code (fresh new start)"
  validate_prerequisites 2
  cd "$REPO_ROOT"
  local out_dir
  out_dir=$(latest_output_dir 2>/dev/null || echo "")
  local prompt="Inspect the @src/ @dfm-python/ and @nowcasting-report/ to understand the project structure, components, and data flow. Check @STATUS.md and @ISSUES.md for current state and pending tasks. Check @FEEDBACK.md for any user feedback that needs to be addressed."
  if [[ -n "$out_dir" ]] && [[ -d "$out_dir" ]]; then
    prompt+=" Study the experiment run output in ${out_dir} (latest run) and plan how to generate tables/plots and update the @nowcasting-report with results. Check which experiments have already been completed by examining the outputs/ directory structure and log files."
  fi
  prompt+=" Current experiment configuration: 3 targets (KOEQUIPTE, KOWRCCNSE, KOIPALL.G) × 4 models (ARIMA, VAR, DFM, DDFM) × 3 horizons (1, 7, 28) = 36 combinations. Report structure: 6 sections (Introduction, Methodology, Production Model, Investment Model, Consumption Model, Conclusion) - target under 15 pages. REQUIRED OUTPUT: 3 tables (dataset/params, standardized MSE/MAE 36 rows, DFM/DDFM backtest monthly) and 3 image types (forecast vs actual per target, accuracy heatmap, performance trend). WORKFLOW: First generate tables and plots from outputs/, then build report sections. The report focuses on comparing 4 models across 3 targets organized by economic sector (Production: KOIPALL.G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE). If FEEDBACK.md contains user feedback, prioritize addressing those items. This is a fresh new start - provide a comprehensive understanding of the project."
  if cursor_text "$prompt"; then
    log_info "Step 2 completed successfully"
    mark_step_completed 2
    return 0
  else
    log_error "Step 2 failed, but continuing to next step"
    return 1
  fi
}

step3_plan_from_inspection() {
  ensure_env
  log_info "Starting step 3: Plan from inspection"
  validate_prerequisites 3
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  if cursor_force "Work on the plan from step 2. Based on the inspection results, create or update a concrete, actionable plan in ISSUES.md. Check @FEEDBACK.md for any user feedback and incorporate feedback items into the plan. PRIORITY ORDER: 1) Generate required tables (3 tables: dataset/params, standardized MSE/MAE 36 rows, DFM/DDFM backtest monthly) from outputs/experiments/aggregated_results.csv and outputs/comparisons/, 2) Generate required plots (3 types: forecast vs actual per target, accuracy heatmap, performance trend) using nowcasting-report/code/plot.py, 3) Update LaTeX tables in nowcasting-report/tables/ with actual results, 4) Build/update report sections using generated tables and plots. When planning tasks, consider which experiments are needed for the report and which are already complete (check outputs/ directory). If new experiments are needed, note that run_experiment.sh should be updated in later steps to include only missing experiments. Focus on incremental tasks that can be completed step by step. Keep ISSUES.md under ${LINE_LIMIT} lines. Do not create new files."; then
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    log_info "Step 3 completed successfully"
    mark_step_completed 3
    return 0
  else
    log_error "Step 3 failed, but continuing to next step"
    return 1
  fi
}

step4_analyze_results() {
  ensure_env
  log_info "Starting step 4: Analyze results"
  validate_prerequisites 4
  cd "$REPO_ROOT"
  local out_dir
  out_dir=$(latest_output_dir)
  if [[ -z "$out_dir" ]] || [[ ! -d "$out_dir" ]]; then
    log_error "No valid outputs directory found"
    die "no outputs/ directory found"
  fi
  log_info "Analyzing results in: $out_dir"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  if cursor_force "Analyze the results in ${out_dir}. If there are errors or issues, update them in STATUS.md and ISSUES.md and inspect what happened. If there's something wrong with the numbers, also update them in STATUS.md and think about what happened. Mark resolved issues clearly in ISSUES.md (use ✅ RESOLVED status). Check @FEEDBACK.md for any user feedback related to results. Update STATUS.md, ISSUES.md, and CONTEXT.md if necessary. Keep all files under ${LINE_LIMIT} lines. Do not create new files."; then
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

step5_plan_improvements() {
  ensure_env
  log_info "Starting step 5: Plan improvements"
  validate_prerequisites 5
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  if cursor_force "Plan how to improve the dfm-python package and nowcasting-report paper. The report structure is 6 sections (Introduction, Methodology, Production Model, Investment Model, Consumption Model, Conclusion) targeting under 15 pages. Focus on comparing 4 models (ARIMA, VAR, DFM, DDFM) across 3 targets (KOEQUIPTE, KOWRCCNSE, KOIPALL.G). Check @FEEDBACK.md for user feedback and prioritize addressing feedback items in the plan. If there are improvement points in the codes, such as numerical stability, convergence issues, theoretically wrong implementation (refer to knowledgebase and legacy clone repos if needed), include the improvements on them in the plan. If there are improvement points in the report, such as hallucination, lack of detail, redundancy, unnatural flow, include the improvements in the plan. If there are improvement points in the code quality such as redundancies, non-generic naming in dfm-python, inefficient logic, monkey patch, temporal fixes, include them in the plan. Note: Legacy code cleanup is completed. dfm-python is finalized with consistent naming and clean patterns. If there are any new experiments needed for the report or extensions, changes in experiment, include them in the plan. IMPORTANT: When planning new experiments, also update run_experiment.sh to include only experiments that are not already complete. Check outputs/ directory to see which experiments have already been run and exclude them from run_experiment.sh. This ensures each iteration only runs missing experiments needed to complete the report. Do not make the plan too long. Leave the tasks at ISSUES.md and work incrementally. Plan with manageable tasks. Keep ISSUES.md under ${LINE_LIMIT} lines. Do not create new files."; then
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    log_info "Step 5 completed successfully"
    mark_step_completed 5
    return 0
  else
    log_error "Step 5 failed, but continuing to next step"
    return 1
  fi
}

step6_execute_plan() {
  ensure_env
  log_info "Starting step 6: Execute plan"
  validate_prerequisites 6
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  if cursor_stream "Work on the plan. Execute the active plan items in ISSUES.md following the priority order: FIRST generate tables and plots, THEN build report. Check @FEEDBACK.md for user feedback and incorporate feedback into updates. PRIORITY 1: Generate required tables from outputs/experiments/aggregated_results.csv and outputs/comparisons/ - create 3 tables (dataset/params table, standardized MSE/MAE table with 36 rows, DFM/DDFM backtest monthly table). PRIORITY 2: Generate required plots using nowcasting-report/code/plot.py - create 3 image types (forecast vs actual per target showing 60 months, accuracy heatmap 4×3, performance trend by horizon). PRIORITY 3: Update LaTeX tables in nowcasting-report/tables/ with actual results from generated tables. PRIORITY 4: Build/update report sections (nowcasting-report/contents/) using the generated tables and plots. Apply code/report updates needed for dfm-python and nowcasting-report. IMPORTANT: If the plan includes new experiments needed for the report, update run_experiment.sh to include only experiments that are not already complete. Check outputs/ directory to identify which experiments have already been run (look for existing result directories and log files). Modify run_experiment.sh to exclude completed experiments and only run missing ones. This ensures each iteration progressively completes the report without re-running unnecessary experiments. Focus on incremental improvements and prioritize tasks. Do not create new files. Use existing files only and keep STATUS.md, ISSUES.md, and CONTEXT.md under ${LINE_LIMIT} lines."; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
    log_info "Step 6 completed successfully"
    mark_step_completed 6
    return 0
  else
    log_error "Step 6 failed, but continuing to next step"
    return 1
  fi
}

step7_continue_plan() {
  ensure_env
  log_info "Starting step 7: Continue plan"
  validate_prerequisites 7
  cd "$REPO_ROOT"
  backup_file_if_exists "${REPO_ROOT}/STATUS.md"
  backup_file_if_exists "${REPO_ROOT}/ISSUES.md"
  backup_file_if_exists "${REPO_ROOT}/CONTEXT.md"
  if cursor_stream "Keep working on the plan with any unfinished tasks. Continue remaining items from ISSUES.md following the priority order: tables/plots first, then report. Check @FEEDBACK.md for any new user feedback and incorporate into ongoing work. Continue generating required tables (3 tables: dataset/params, standardized MSE/MAE 36 rows, DFM/DDFM backtest monthly) and plots (3 types: forecast vs actual per target, accuracy heatmap, performance trend) if not yet complete. Then continue updating LaTeX tables and report sections. IMPORTANT: If there are remaining experiments needed for the report, update run_experiment.sh to include only experiments that are not already complete. Check outputs/ directory to see which experiments have already been run and exclude them. This ensures the next iteration (step 1) only runs missing experiments. Focus on incremental improvements and complete as many tasks as possible. Do not create new files. Keep STATUS.md, ISSUES.md, and CONTEXT.md under ${LINE_LIMIT} lines."; then
    guard_line_limit "${REPO_ROOT}/STATUS.md"
    guard_line_limit "${REPO_ROOT}/ISSUES.md"
    guard_line_limit "${REPO_ROOT}/CONTEXT.md"
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
  if cursor_force "Identify the work done in this iteration. Identify what's done, what's not done. Update STATUS.md and ISSUES.md for the next iteration. Mark resolved issues clearly in ISSUES.md. Remove old resolved issues to keep file under ${LINE_LIMIT} lines. Update experiment status in STATUS.md (completed/pending combinations). Note: Submodules are pushed every 2 iterations - user will review report and provide feedback in FEEDBACK.md. Next iteration will start fresh so you need to leave the proper context for next iteration. Keep each file under ${LINE_LIMIT} lines. Do not create new files."; then
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

step9_commit() {
  ensure_env
  log_info "Starting step 9: Commit changes (iteration ${ITERATION})"
  validate_prerequisites 9
  cd "$REPO_ROOT"
  [[ -n "${COMMIT_MSG:-}" ]] || die "COMMIT_MSG must be set for step 9"
  
  guard_line_limit "${REPO_ROOT}/STATUS.md"
  guard_line_limit "${REPO_ROOT}/ISSUES.md"
  guard_line_limit "${REPO_ROOT}/CONTEXT.md"
  
  if safe_git_commit "$COMMIT_MSG"; then
    # Push submodules every 2 iterations (user reviews report and provides feedback)
    if [[ "${ITERATION:-0}" =~ ^[0-9]+$ ]] && (( ITERATION % 2 == 0 )); then
      log_info "Pushing submodules to origin main (iteration ${ITERATION} - user will review and provide feedback)"
      safe_git_submodule_push || log_warn "Submodule push had errors but continuing"
      log_info "Note: User will review report in nowcasting-report/ and provide feedback in FEEDBACK.md"
    fi
    log_info "Step 9 completed successfully"
    mark_step_completed 9
    return 0
  else
    log_error "Step 9 failed, but iteration completed"
    return 1
  fi
}

cleanup_old_backups() {
  # Keep only last 10 backups of each file
  if [[ ! -d "$BACKUP_DIR" ]]; then
    return 0
  fi
  for f in STATUS.md ISSUES.md CONTEXT.md; do
    ls -t "${BACKUP_DIR}/${f}.backup."* 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true
  done
}

run_step() {
  local step="$1"
  log_info "-----------------------------------------"
  log_info "Running step $step of iteration ${ITERATION}"
  log_info "-----------------------------------------"
  
  local step_result=0
  case "$step" in
    1) step1_run_experiment || step_result=1 ;;
    2) step2_inspect_code || step_result=1 ;;
    3) step3_plan_from_inspection || step_result=1 ;;
    4) step4_analyze_results || step_result=1 ;;
    5) step5_plan_improvements || step_result=1 ;;
    6) step6_execute_plan || step_result=1 ;;
    7) step7_continue_plan || step_result=1 ;;
    8) step8_summarize_iteration || step_result=1 ;;
    9) step9_commit || step_result=1 ;;
    *) die "Invalid step: $step" ;;
  esac
  
  if [[ $step_result -eq 1 ]]; then
    log_warn "Step $step had errors, but pipeline continues"
  fi
  
  return $step_result
}

main() {
  [[ $# -eq 1 ]] || { usage; exit 1; }
  
  # Parse number of iterations
  local num_iters="$1"
  
  if ! [[ "$num_iters" =~ ^[0-9]+$ ]] || [[ "$num_iters" -lt 1 ]]; then
    die "Invalid number of iterations: $num_iters (must be >= 1)"
  fi
  
  mkdir -p "$LOG_DIR"
  
  log_info "========================================="
  log_info "Running ${num_iters} iteration(s) (iterations 1-${num_iters})"
  log_info "Each iteration runs steps 1-9"
  log_info "Timestamp: $(date)"
  log_info "========================================="
  
  # Show step completion history
  show_step_history
  
  # Run iterations
  for iter in $(seq 1 "$num_iters"); do
    ITERATION="$iter"
    
    # Set COMMIT_MSG for this iteration
    if [[ -z "${COMMIT_MSG:-}" ]]; then
      COMMIT_MSG="Iteration ${iter}"
    fi
    
    log_info "========================================="
    log_info "Starting iteration ${iter} (steps 1-9)"
    log_info "COMMIT_MSG: ${COMMIT_MSG}"
    log_info "========================================="
    
    # Cleanup old backups periodically (every 10 iterations)
    if (( iter % 10 == 0 )); then
      cleanup_old_backups
    fi
    
    # Run all steps 1-9
    for step in $(seq 1 9); do
      run_step "$step"
    done
    
    log_info "========================================="
    log_info "Iteration ${iter} completed successfully"
    log_info "========================================="
    
    # Reset COMMIT_MSG for next iteration
    unset COMMIT_MSG
  done
  
  log_info "========================================="
  log_info "All ${num_iters} iteration(s) completed successfully"
  log_info "Iterations 1-${num_iters} finished"
  log_info "========================================="
}

main "$@"
