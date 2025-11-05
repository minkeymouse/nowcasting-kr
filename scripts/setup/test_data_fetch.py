"""Test script to verify data fetching from CSV specification.

This script tests fetching a few series from migrations/001_initial_spec.csv
to verify API connectivity and data format before running full initialization.

Usage:
    python scripts/setup/test_data_fetch.py
    python scripts/setup/test_data_fetch.py --limit 5
"""

import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_paths = [
    project_root / '.env.local',
    project_root.parent / '.env.local',
    Path.home() / 'Nowcasting' / '.env.local',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

import pandas as pd

from database.settings import BOKAPIConfig, KOSISAPIConfig
from services.api.bok_client import BOKAPIClient
from services.api.kosis_client import KOSISAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_series_data(
    series_id: str,
    api_code: str,
    api_client: any,
    source: str,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch data for a single series from API."""
    # Map frequency to API format
    freq_map = {'q': 'Q', 'm': 'M', 'd': 'D', 'a': 'A', 'sa': 'S'}
    api_freq = freq_map.get(frequency.lower(), frequency.upper())
    
    # Set date range if not provided
    if not end_date:
        end_date_obj = datetime.now()
        if api_freq == 'Q':
            q = (end_date_obj.month - 1) // 3 + 1
            end_date = f"{end_date_obj.year}Q{q}"
        elif api_freq == 'M':
            end_date = end_date_obj.strftime('%Y%m')
        elif api_freq == 'D':
            end_date = end_date_obj.strftime('%Y%m%d')
        else:
            end_date = end_date_obj.strftime('%Y')
    
    if not start_date:
        start_date_obj = datetime.now() - timedelta(days=365*2)  # 2 years for testing
        if api_freq == 'Q':
            q = (start_date_obj.month - 1) // 3 + 1
            start_date = f"{start_date_obj.year}Q{q}"
        elif api_freq == 'M':
            start_date = start_date_obj.strftime('%Y%m')
        elif api_freq == 'D':
            start_date = start_date_obj.strftime('%Y%m%d')
        else:
            start_date = start_date_obj.strftime('%Y')
    
    try:
        # Fetch data
        data_result = api_client.fetch_statistic_data(
            stat_code=api_code,
            frequency=api_freq,
            start_date=start_date,
            end_date=end_date,
            item_code1='?',
            item_code2='?',
            item_code3='?',
            item_code4='?'
        )
        
        if 'StatisticSearch' not in data_result:
            logger.warning(f"No data returned for {series_id}")
            return pd.DataFrame()
        
        rows = data_result['StatisticSearch'].get('row', [])
        if not rows:
            logger.warning(f"Empty data for {series_id}")
            return pd.DataFrame()
        
        # Parse into DataFrame
        data_list = []
        for row in rows:
            time_str = row.get('TIME', '')
            value = row.get('DATA_VALUE')
            
            if value is None:
                continue
            
            # Parse date based on frequency
            try:
                if api_freq == 'Q':
                    # Format: 2023Q1
                    year, q = time_str.split('Q')
                    month = int(q) * 3
                    date_obj = datetime(int(year), month, 1).date()
                elif api_freq == 'M':
                    # Format: 202301
                    date_obj = datetime.strptime(time_str, '%Y%m').date()
                elif api_freq == 'D':
                    # Format: 20230101
                    date_obj = datetime.strptime(time_str, '%Y%m%d').date()
                else:
                    # Annual or other
                    date_obj = datetime(int(time_str), 12, 31).date()
                
                data_list.append({
                    'date': date_obj,
                    'value': float(value),
                    'series_id': series_id
                })
            except Exception as e:
                logger.warning(f"Failed to parse date {time_str} for {series_id}: {e}")
                continue
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {e}")
        return pd.DataFrame()


def main():
    """Test data fetching from CSV."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data fetching from CSV specification')
    parser.add_argument(
        '--csv-file',
        type=str,
        default='migrations/001_initial_spec.csv',
        help='Path to CSV specification file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of series to test (default: 5)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Testing Data Fetching from CSV Specification")
    logger.info("=" * 80)
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Testing {args.limit} series")
    logger.info("=" * 80)
    logger.info("")
    
    # Load CSV
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    df_csv = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df_csv)} series")
    logger.info("")
    
    # Initialize API clients
    bok_key = os.getenv('BOK_API_KEY')
    kosis_key = os.getenv('KOSIS_API_KEY')
    
    bok_client = None
    if bok_key:
        bok_config = BOKAPIConfig(auth_key=bok_key)
        bok_client = BOKAPIClient(bok_config)
        logger.info("✓ BOK API client initialized")
    else:
        logger.warning("⚠ BOK_API_KEY not set")
    
    kosis_client = None
    if kosis_key:
        kosis_config = KOSISAPIConfig(api_key=kosis_key)
        kosis_client = KOSISAPIClient(kosis_config)
        logger.info("✓ KOSIS API client initialized")
    else:
        logger.warning("⚠ KOSIS_API_KEY not set")
    
    logger.info("")
    
    # Test a few series
    test_series = df_csv.head(args.limit)
    
    stats = {
        'total': len(test_series),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    for idx, row in test_series.iterrows():
        series_id = row['series_id']
        series_name = row['series_name']
        api_code = row['api_code']
        api_source = row['api_source']
        frequency = row['frequency']
        transformation = row['transformation']
        
        logger.info(f"[{idx + 1}/{len(test_series)}] {series_id}: {series_name}")
        logger.info(f"  API: {api_source}, Code: {api_code}, Freq: {frequency}, Transform: {transformation}")
        
        try:
            # Get appropriate API client
            api_client = None
            if api_source == 'BOK' and bok_client:
                api_client = bok_client
            elif api_source == 'KOSIS' and kosis_client:
                api_client = kosis_client
            else:
                logger.warning(f"  ⚠ No API client available for {api_source}")
                stats['skipped'] += 1
                continue
            
            # Fetch data
            logger.info(f"  Fetching data...")
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=api_client,
                source=api_source,
                frequency=frequency
            )
            
            if df_data.empty:
                logger.warning(f"  ⚠ No data fetched")
                stats['skipped'] += 1
                continue
            
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            logger.info(f"    Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            logger.info(f"    Value range: {df_data['value'].min():.2f} to {df_data['value'].max():.2f}")
            logger.info(f"    Sample values: {df_data['value'].head(3).tolist()}")
            
            stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            stats['failed'] += 1
            stats['errors'].append(f"{series_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"Total tested: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    
    if stats['errors']:
        logger.warning("Errors encountered:")
        for error in stats['errors']:
            logger.warning(f"  - {error}")
    
    logger.info("=" * 80)
    
    if stats['successful'] > 0:
        logger.info("✓ Test passed! Data fetching works correctly.")
    else:
        logger.error("❌ Test failed! No data was successfully fetched.")
        sys.exit(1)


if __name__ == '__main__':
    main()

