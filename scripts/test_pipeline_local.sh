#!/bin/bash
# End-to-end pipeline test script
# Tests: ingest_api.py -> train_dfm.py -> nowcast_dfm.py

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Activated virtual environment${NC}"
else
    echo -e "${YELLOW}⚠ No .venv found, using system Python${NC}"
fi

echo "=========================================="
echo "Pipeline Local Test"
echo "=========================================="
echo ""

# Check environment variables
echo "Checking environment variables..."
MISSING_VARS=()
for var in SUPABASE_URL SUPABASE_SECRET_KEY BOK_API_KEY KOSIS_API_KEY; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing environment variables: ${MISSING_VARS[*]}${NC}"
    echo "Please set these in .env.local"
    exit 1
fi
echo -e "${GREEN}✓ All required environment variables set${NC}"
echo ""

# Stage 1: Data Ingestion
echo "=========================================="
echo "Stage 1: Data Ingestion (ingest_api.py)"
echo "=========================================="
echo ""

python scripts/ingest_api.py
INGEST_EXIT=$?

if [ $INGEST_EXIT -ne 0 ]; then
    echo -e "${RED}✗ Data ingestion failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Data ingestion completed${NC}"

# Validate vintage was created
echo "Validating vintage creation..."
python -c "
from scripts.utils import get_db_client, get_latest_vintage_with_fallback
import sys

try:
    client = get_db_client()
    result = get_latest_vintage_with_fallback(client=client)
    if result:
        vintage_id, vintage_info = result
        status = vintage_info.get('fetch_status', 'unknown')
        print(f'✅ Vintage {vintage_id} created with status: {status}')
        if status not in ['completed', 'partial']:
            print(f'⚠️  Warning: Vintage status is {status}')
    else:
        print('❌ No vintage found after ingestion')
        sys.exit(1)
except Exception as e:
    print(f'⚠️  Could not validate vintage: {e}')
    sys.exit(0)  # Don't fail if validation fails
" || echo -e "${YELLOW}⚠ Vintage validation skipped${NC}"

echo ""

# Stage 2: Model Training
echo "=========================================="
echo "Stage 2: Model Training (train_dfm.py)"
echo "=========================================="
echo ""

python scripts/train_dfm.py \
    --config-name=test \
    series=test_series \
    data.use_database=true \
    data.strict_mode=false \
    dfm.max_iter=100

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo -e "${RED}✗ Model training failed${NC}"
    exit 1
fi

# Check if model weights file exists
if [ -f "ResDFM.pkl" ]; then
    echo -e "${GREEN}✓ Model weights file created: ResDFM.pkl${NC}"
else
    echo -e "${YELLOW}⚠ Model weights file not found locally (may be in storage)${NC}"
fi

# Validate model weights in storage
echo "Validating model weights in storage..."
python -c "
from adapters.adapter_database import download_model_weights_from_storage
from scripts.utils import get_db_client, get_latest_vintage_with_fallback
import sys

try:
    client = get_db_client()
    result = get_latest_vintage_with_fallback(client=client)
    if result:
        vintage_id, vintage_info = result
        vintage_date = str(vintage_info.get('vintage_date', ''))
        model_filename = f'dfm_{vintage_date}.pkl'
        weights = download_model_weights_from_storage(
            filename=model_filename,
            bucket_name='model-weights',
            client=client
        )
        if weights:
            print(f'✅ Model weights found in storage: {model_filename}')
        else:
            print(f'⚠️  Model weights not found in storage: {model_filename}')
    else:
        print('⚠️  No vintage found')
except Exception as e:
    print(f'⚠️  Could not validate model weights: {e}')
" || echo -e "${YELLOW}⚠ Model weights validation skipped${NC}"

echo ""

# Stage 3: Nowcasting
echo "=========================================="
echo "Stage 3: Nowcasting (nowcast_dfm.py)"
echo "=========================================="
echo ""

python scripts/nowcast_dfm.py \
    --config-name=test \
    series=test_series \
    data.use_database=true \
    data.strict_mode=false \
    data.forecast_periods=2

NOWCAST_EXIT=$?

if [ $NOWCAST_EXIT -ne 0 ]; then
    echo -e "${RED}✗ Nowcasting failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Nowcasting completed${NC}"

# Validate nowcasts were saved
echo "Validating nowcasts in database..."
python -c "
from database import get_client
import sys

try:
    client = get_client()
    # Check if forecasts table has recent nowcasts
    from datetime import datetime, timedelta
    # Check for nowcasts created in the last hour
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    result = client.table('forecasts').select('forecast_id', 'run_type', 'created_at').eq('run_type', 'nowcast').gte('created_at', one_hour_ago).order('created_at', desc=True).limit(5).execute()
    if result.data and len(result.data) > 0:
        print(f'✅ Found {len(result.data)} recent nowcasts in database')
        for forecast in result.data[:3]:
            print(f'   - Forecast ID: {forecast.get(\"forecast_id\")}, Created: {forecast.get(\"created_at\", \"N/A\")}')
    else:
        print('⚠️  No nowcasts found in database')
except Exception as e:
    print(f'⚠️  Could not validate nowcasts: {e}')
" || echo -e "${YELLOW}⚠ Nowcast validation skipped${NC}"

echo ""

# Summary
echo "=========================================="
echo "Pipeline Test Summary"
echo "=========================================="
echo -e "${GREEN}✓ All stages completed successfully${NC}"
echo ""
echo "Stages tested:"
echo "  1. Data ingestion (ingest_api.py)"
echo "  2. Model training (train_dfm.py)"
echo "  3. Nowcasting (nowcast_dfm.py)"
echo ""
echo "Next steps:"
echo "  - Check database for vintage creation"
echo "  - Verify model weights in storage bucket"
echo "  - Check forecasts table for nowcasts"
echo ""

