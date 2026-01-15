#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "========================================="
echo "DDFM - PRODUCTION (short-term)"
echo "========================================="
python -m src.main model=ddfm/production data=production train=true experiment=short_term model.max_epoch=50

echo ""
echo "========================================="
echo "DDFM - INVESTMENT (short-term)"
echo "========================================="
python -m src.main model=ddfm/investment data=investment train=true experiment=short_term model.max_epoch=50

echo ""
echo "========================================="
echo "Done. Results saved to outputs/short_term/"
echo "========================================="