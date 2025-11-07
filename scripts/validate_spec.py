"""Spec file validation script.

This script validates the spec CSV file and tests API endpoints for each series.
It identifies which series have API errors and suggests corrections.

Usage:
    python scripts/validate_spec.py [--spec-file path/to/spec.csv] [--use-db]
"""

import sys
import logging
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_locations = [
    project_root / '.env.local',
    Path('/home/minkeymouse/Nowcasting') / '.env.local',
    Path.home() / '.env.local',
    Path('.env.local'),
]

env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        logger.info(f"✅ Loaded environment from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    logger.warning("⚠️  .env.local not found in standard locations")

from database.db_utils import (
    initialize_api_clients,
    fetch_series_data,
    RateLimiter,
)
from database.operations import generate_series_id
from adapters.adapter_database import (
    download_spec_csv_from_storage,
    get_latest_spec_csv_filename,
)
from scripts.utils import get_db_client


def load_spec_file(spec_path: Optional[Path] = None, use_db: bool = False) -> pd.DataFrame:
    """Load spec CSV file from local path or database storage."""
    if use_db:
        try:
            client = get_db_client()
            spec_filename = get_latest_spec_csv_filename(bucket_name="spec", client=client)
            if spec_filename:
                logger.info(f"Loading spec from database storage: {spec_filename}")
                csv_content = download_spec_csv_from_storage(
                    filename=spec_filename,
                    bucket_name="spec",
                    client=client
                )
                if csv_content:
                    import io
                    df = pd.read_csv(io.BytesIO(csv_content))
                    logger.info(f"✅ Loaded spec CSV from database: {len(df)} series")
                    return df
        except Exception as e:
            logger.warning(f"Failed to load from database: {e}, trying local file...")
    
    # Fallback to local file
    if spec_path is None:
        spec_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
    
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    
    df = pd.read_csv(spec_path)
    logger.info(f"✅ Loaded spec CSV from local file: {len(df)} series")
    return df


def validate_series(
    row: pd.Series,
    bok_client: Optional[Any],
    kosis_client: Optional[Any],
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """Validate a single series by testing API call.
    
    Returns:
        Dict with keys: series_id, success, error_message, data_points
    """
    series_id = generate_series_id(
        row.get('api_source', ''),
        row.get('data_code', ''),
        row.get('item_id', '')
    )
    
    result = {
        'series_id': series_id,
        'series_name': row.get('series_name', ''),
        'api_source': row.get('api_source', ''),
        'data_code': row.get('data_code', ''),
        'item_id': row.get('item_id', ''),
        'frequency': row.get('frequency', ''),
        'success': False,
        'error_message': None,
        'data_points': 0,
    }
    
    # Check required fields
    if not row.get('api_source'):
        result['error_message'] = "Missing api_source"
        return result
    
    if not row.get('data_code'):
        result['error_message'] = "Missing data_code"
        return result
    
    # Get appropriate API client
    api_client = None
    if row.get('api_source') == 'BOK' and bok_client:
        api_client = bok_client
        rate_limiter.wait_if_needed('BOK')
    elif row.get('api_source') == 'KOSIS' and kosis_client:
        api_client = kosis_client
        rate_limiter.wait_if_needed('KOSIS')
    else:
        result['error_message'] = f"No API client available for {row.get('api_source')}"
        return result
    
    # Test API call (fetch_series_data handles date range internally)
    try:
        df_data = fetch_series_data(
            series_id=series_id,
            api_code=row.get('data_code'),
            api_client=api_client,
            source=row.get('api_source'),
            frequency=row.get('frequency', 'm'),
            start_date=None,  # Will default to 10 years back
            end_date=None     # Will default to today
        )
        
        if df_data is None or df_data.empty:
            result['error_message'] = "API returned no data"
            return result
        
        result['success'] = True
        result['data_points'] = len(df_data)
        result['date_range'] = f"{df_data['date'].min()} to {df_data['date'].max()}"
        
    except Exception as e:
        error_msg = str(e)
        if 'rate limit' in error_msg.lower() or '602' in error_msg:
            result['error_message'] = f"Rate limit error: {error_msg[:100]}"
        elif 'invalid' in error_msg.lower() or 'not found' in error_msg.lower():
            result['error_message'] = f"Invalid API code/parameters: {error_msg[:100]}"
        else:
            result['error_message'] = f"API error: {error_msg[:100]}"
    
    return result


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate spec CSV file and test API endpoints'
    )
    parser.add_argument(
        '--spec-file',
        type=str,
        default=None,
        help='Path to spec CSV file (default: src/spec/001_initial_spec.csv)'
    )
    parser.add_argument(
        '--use-db',
        action='store_true',
        help='Load spec from database storage bucket instead of local file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of series to validate (for testing)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Spec File Validation")
    print("=" * 80)
    print()
    
    # Load spec file
    try:
        spec_path = Path(args.spec_file) if args.spec_file else None
        df = load_spec_file(spec_path=spec_path, use_db=args.use_db)
    except Exception as e:
        logger.error(f"Failed to load spec file: {e}")
        sys.exit(1)
    
    # Generate series_id for each row
    df['series_id'] = df.apply(
        lambda row: generate_series_id(
            row.get('api_source', ''),
            row.get('data_code', ''),
            row.get('item_id', '')
        ),
        axis=1
    )
    
    print(f"📄 Loaded {len(df)} series from spec file")
    print()
    
    # Initialize API clients
    print("Initializing API clients...")
    bok_client, kosis_client = initialize_api_clients()
    
    if not bok_client and not kosis_client:
        logger.error("No API clients available. Check API keys.")
        print("❌ Error: No API clients available")
        sys.exit(1)
    
    if bok_client:
        print("✅ BOK API client initialized")
    if kosis_client:
        print("✅ KOSIS API client initialized")
    print()
    
    # Validate each series
    print("=" * 80)
    print("Validating Series")
    print("=" * 80)
    print()
    
    rate_limiter = RateLimiter(bok_delay=0.6, kosis_delay=0.5)
    results = []
    
    # Limit series if specified
    df_to_validate = df.head(args.limit) if args.limit else df
    
    for idx, row in df_to_validate.iterrows():
        print(f"[{idx+1}/{len(df_to_validate)}] {row.get('series_id', 'unknown')}: {row.get('series_name', '')[:50]}...")
        result = validate_series(row, bok_client, kosis_client, rate_limiter)
        results.append(result)
        
        if result['success']:
            print(f"   ✅ Success: {result['data_points']} data points ({result.get('date_range', 'N/A')})")
        else:
            print(f"   ❌ Failed: {result['error_message']}")
        print()
    
    # Summary
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print()
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total series validated: {len(results)}")
    print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print()
    
    if failed:
        print("Failed Series:")
        print("-" * 80)
        for r in failed:
            print(f"  {r['series_id']}: {r['series_name'][:50]}")
            print(f"    API: {r['api_source']}, Code: {r['data_code']}, Item: {r.get('item_id', 'N/A')}")
            print(f"    Error: {r['error_message']}")
            print()
        
        print("=" * 80)
        print("Suggestions for Failed Series:")
        print("=" * 80)
        print()
        print("1. Check API documentation for correct data_code and item_id")
        print("2. For KOSIS, verify item_code1 (objL1) and item_code2 (itmId) parameters")
        print("3. Test API endpoints manually with curl:")
        print()
        print("   BOK:")
        print("   curl 'https://ecos.bok.or.kr/api/StatisticSearch/{AUTH_KEY}/json/kr/1/1000/{STAT_CODE}/{FREQ}/{START}/{END}'")
        print()
        print("   KOSIS:")
        print("   curl 'https://kosis.kr/openapi/statisticsData.do?method=getList&apiKey={API_KEY}&format=json&jsonVD=Y&userStatsId={STAT_CODE}&prdSe={FREQ}&startPrdDe={START}&endPrdDe={END}&objL1={ITEM_CODE1}&itmId={ITEM_CODE2}'")
        print()
        print("4. Update spec CSV with correct values")
        print("5. Upload corrected spec to database storage bucket 'spec'")
        print()
    
    # Exit with error code if any failures
    if failed:
        sys.exit(1)
    else:
        print("✅ All series validated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

