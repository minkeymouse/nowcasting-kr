#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the FastAPI app on port 2020
echo "Starting Nowcasting API on port 2020..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 2020 --reload

