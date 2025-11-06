#!/bin/bash
# DFM Workflow Script - Train and Nowcast
# This script runs train_dfm.py and nowcasting_dfm.py to populate the database
# Designed for headless CLI execution (e.g., overnight runs)

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file
LOG_FILE="${SCRIPT_DIR}/dfm_workflow.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$TIMESTAMP] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$TIMESTAMP] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$TIMESTAMP] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found. Please create it first."
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source .venv/bin/activate

# Load environment variables
if [ -f ".env.local" ]; then
    log "Loading environment variables from .env.local..."
    export $(cat .env.local | grep -v '^#' | grep -v '^$' | xargs)
else
    log_warning ".env.local not found. Proceeding without it."
fi

# Configuration
CONFIG_PATH="src/spec/001_initial_spec.csv"
MAX_ITERATIONS=100  # Limit iterations for quick test
TIMEOUT_TRAIN=1800  # 30 minutes for training
TIMEOUT_NOWCAST=600  # 10 minutes for nowcasting

log "=========================================="
log "DFM Workflow Script Started"
log "=========================================="
log "Config: $CONFIG_PATH"
log "Max iterations: $MAX_ITERATIONS"
log "=========================================="

# Step 1: Train DFM Model
log ""
log "Step 1: Training DFM Model..."
log "----------------------------------------"

MODEL_FILE="ResDFM.pkl"
if [ -f "$MODEL_FILE" ]; then
    log_warning "Model file exists: $MODEL_FILE"
    log "Skipping training. Delete $MODEL_FILE to retrain."
else
    log "Running train_dfm.py..."
    
    if timeout $TIMEOUT_TRAIN python3 scripts/train_dfm.py \
        "model.config_path=$CONFIG_PATH" \
        "data.use_database=true" \
        "data.strict_mode=false" \
        "dfm.max_iter=$MAX_ITERATIONS" \
        >> "$LOG_FILE" 2>&1; then
        
        if [ -f "$MODEL_FILE" ]; then
            MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            log_success "Model trained successfully: $MODEL_FILE ($MODEL_SIZE)"
        else
            log_error "Training completed but model file not found"
            exit 1
        fi
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            log_error "Training timed out after $TIMEOUT_TRAIN seconds"
        else
            log_error "Training failed with exit code $EXIT_CODE"
            log "Check $LOG_FILE for details"
        fi
        exit 1
    fi
fi

# Step 2: Verify blocks are saved
log ""
log "Step 2: Verifying blocks in database..."
log "----------------------------------------"

python3 << 'PYTHON_EOF'
import sys
from adapters.database import _get_db_client

try:
    client = _get_db_client()
    result = client.table('blocks').select('*').eq('config_name', '001-initial-spec').limit(1).execute()
    print(f"✓ Blocks in database: {len(result.data)} records")
    if len(result.data) == 0:
        print("⚠ Warning: No blocks found. Blocks should be saved during training.")
except Exception as e:
    print(f"⚠ Could not verify blocks: {e}")
PYTHON_EOF

# Step 3: Run Nowcasting
log ""
log "Step 3: Running Nowcasting..."
log "----------------------------------------"

if timeout $TIMEOUT_NOWCAST python3 scripts/nowcast_dfm.py \
    "model.config_path=$CONFIG_PATH" \
    "data.use_database=true" \
    "data.strict_mode=false" \
    >> "$LOG_FILE" 2>&1; then
    
    log_success "Nowcasting completed successfully"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        log_error "Nowcasting timed out after $TIMEOUT_NOWCAST seconds"
    else
        log_error "Nowcasting failed with exit code $EXIT_CODE"
        log "Check $LOG_FILE for details"
    fi
    exit 1
fi

# Step 4: Verify database updates
log ""
log "Step 4: Verifying database updates..."
log "----------------------------------------"

python3 << 'PYTHON_EOF'
import sys
from adapters.database import _get_db_client

try:
    client = _get_db_client()
    
    # Check forecasts
    result = client.table('forecasts').select('series_id, forecast_date, forecast_value, created_at').order('created_at', desc=True).limit(10).execute()
    print(f"✓ Forecasts in database: {len(result.data)} records")
    
    if result.data:
        print("  Latest forecasts:")
        for f in result.data[:5]:
            print(f"    - {f.get('series_id')}: {f.get('forecast_value')} on {f.get('forecast_date')}")
    else:
        print("  ⚠ Warning: No forecasts found in database")
    
    # Check factors (if table exists)
    try:
        result = client.table('factors').select('*').order('created_at', desc=True).limit(5).execute()
        print(f"✓ Factors in database: {len(result.data)} records")
    except Exception:
        print("  ℹ Factors table not checked (may not exist)")
    
    # Check blocks
    result = client.table('blocks').select('*').eq('config_name', '001-initial-spec').limit(1).execute()
    print(f"✓ Blocks: {len(result.data)} records for 001-initial-spec")
    
except Exception as e:
    print(f"✗ Database verification failed: {e}")
    sys.exit(1)
PYTHON_EOF

# Final summary
log ""
log "=========================================="
log_success "DFM Workflow Completed Successfully"
log "=========================================="
log "Log file: $LOG_FILE"
log "Model file: $MODEL_FILE"
log "=========================================="

exit 0

