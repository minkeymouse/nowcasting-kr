#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "========================================="
echo "DFM - PRODUCTION (train, 1 iteration - DEBUG)"
echo "========================================="
python3 -m src.main model=dfm/production data=production train=true forecast=false model.max_iter=1

echo ""
echo "========================================="
echo "DFM - PRODUCTION (short-term experiment)"
echo "========================================="
python3 -m src.main model=dfm/production data=production train=false forecast=false experiment=short_term

echo ""
echo "========================================="
echo "DFM - PRODUCTION (long-term experiment)"
echo "========================================="
python3 -m src.main model=dfm/production data=production train=false forecast=false experiment=long_term


echo ""
echo "========================================="
echo "DFM - INVESTMENT (short-term experiment)"
echo "========================================="
python3 -m src.main model=dfm/investment data=investment train=false forecast=false experiment=short_term

echo ""
echo "========================================="
echo "DFM - INVESTMENT (long-term experiment)"
echo "========================================="
python3 -m src.main model=dfm/investment data=investment train=false forecast=false experiment=long_term

echo ""
echo "========================================="
echo "Done. Results saved to outputs/{short_term,long_term}/"
echo "========================================="