# DFM Workflow Script

## Overview

`run_dfm_workflow.sh` is a headless CLI script that runs the complete DFM workflow:
1. Train DFM model (`train_dfm.py`)
2. Run nowcasting (`nowcast_dfm.py`)
3. Verify database updates

## Usage

### Basic execution
```bash
./run_dfm_workflow.sh
```

### For overnight/headless runs
```bash
# Run in background with nohup
nohup ./run_dfm_workflow.sh > dfm_workflow.out 2>&1 &

# Or use screen/tmux
screen -S dfm_workflow
./run_dfm_workflow.sh
# Press Ctrl+A then D to detach
```

### Check progress
```bash
# View log file
tail -f dfm_workflow.log

# Check if still running
ps aux | grep run_dfm_workflow
```

## Configuration

The script uses:
- Config: `src/spec/001_initial_spec.csv`
- Max iterations: 100 (for quick test, full runs via GitHub Actions)
- Timeout: 30 min (train), 10 min (nowcast)

## Requirements

- Virtual environment (`.venv`) must exist
- Environment variables (`.env.local`) for database connection
- Database must have vintage data (run `ingest_api.py` first if needed)

## Output

- Log file: `dfm_workflow.log`
- Model file: `ResDFM.pkl`
- Database updates: `forecasts`, `blocks`, `factors` tables

## Error Handling

The script exits on any error. Check `dfm_workflow.log` for details.
