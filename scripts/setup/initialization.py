"""Consolidated initialization script for DFM database setup.

This script:
1. Loads CSV specification file (migrations/001_initial_spec.csv) to get selected series
2. Fetches data from APIs (BOK/KOSIS) for those series
3. Updates Supabase database with:
   - Series metadata
   - Observations data
   - Model configuration with block assignments
   - Creates a vintage

Usage:
    python scripts/setup/initialization.py
    python scripts/setup/initialization.py --csv-file migrations/001_initial_spec.csv
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
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
import numpy as np

from database import (
    get_client,
    create_vintage,
    update_vintage_status,
    create_ingestion_job,
    update_ingestion_job,
    upsert_series,
    insert_observations_from_dataframe,
    save_model_config,
    get_source_id,
    get_latest_vintage_id,
)
from database.models import SeriesModel
from database.settings import BOKAPIConfig, KOSISAPIConfig
from services.api.bok_client import BOKAPIClient
from services.api.kosis_client import KOSISAPIClient
# Import load_config_from_csv inside main to avoid Hydra initialization issues

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# No longer needed - CSV provides api_code and api_source directly


def fetch_series_data(
    series_id: str,
    api_code: str,
    api_client: Any,
    source: str,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch data for a single series from API.
    
    Parameters
    ----------
    series_id : str
        Series ID (e.g., BOK_200Y001)
    api_code : str
        API code for the statistic (e.g., 200Y001)
    api_client : Any
        API client instance (BOKAPIClient or KOSISAPIClient)
    source : str
        Source code (BOK or KOSIS)
    frequency : str
        Frequency code (q, m, d, etc.)
    start_date : str, optional
        Start date in API format
    end_date : str, optional
        End date in API format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, value, series_id
    """
    
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
        start_date_obj = datetime.now() - timedelta(days=365*10)  # 10 years
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


def main() -> None:
    """Main initialization workflow."""
    parser = argparse.ArgumentParser(
        description='Initialize DFM database with data from CSV specification'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='migrations/001_initial_spec.csv',
        help='Path to CSV specification file (default: migrations/001_initial_spec.csv)'
    )
    parser.add_argument(
        '--vintage-date',
        type=str,
        help='Vintage date (YYYY-MM-DD). Defaults to today.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - fetch data but do not save to database'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DFM Database Initialization")
    logger.info("=" * 80)
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    logger.info()
    
    # Load model configuration from CSV
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        from src.nowcasting.data_loader import load_config_from_csv
        model_cfg = load_config_from_csv(csv_path)
        logger.info(f"Loaded model config: {len(model_cfg.series)} series")
        logger.info(f"Block names: {model_cfg.block_names}")
    except Exception as e:
        logger.error(f"Failed to load model config from CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Determine vintage date
    if args.vintage_date:
        vintage_date = datetime.strptime(args.vintage_date, '%Y-%m-%d').date()
    else:
        vintage_date = date.today()
    
    dry_run = args.dry_run
    
    logger.info(f"Vintage date: {vintage_date}")
    logger.info()
    
    # Check if vintage already exists
    client = get_client()
    existing_vintage_id = get_latest_vintage_id(vintage_date=vintage_date, client=client)
    if existing_vintage_id:
        logger.warning(f"Vintage {vintage_date} already exists (ID: {existing_vintage_id})")
        if not dry_run:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted")
                sys.exit(0)
    
    # Create vintage
    if not dry_run:
        vintage = create_vintage(
            vintage_date=vintage_date,
            country='KR',
            client=client
        )
        vintage_id = vintage['vintage_id']
        logger.info(f"Created vintage: {vintage_id}")
        
        # Create ingestion job
        job = create_ingestion_job(
            vintage_id=vintage_id,
            github_run_id=None,
            client=client
        )
        job_id = job['job_id']
        logger.info(f"Created ingestion job: {job_id}")
    else:
        vintage_id = None
        job_id = None
        logger.info("Dry run: Skipping vintage/job creation")
    
    logger.info()
    
    # Initialize API clients
    bok_key = os.getenv('BOK_API_KEY')
    kosis_key = os.getenv('KOSIS_API_KEY')
    
    bok_client = None
    if bok_key:
        bok_config = BOKAPIConfig(auth_key=bok_key)
        bok_client = BOKAPIClient(bok_config)
        logger.info("✓ BOK API client initialized")
    
    kosis_client = None
    if kosis_key:
        kosis_config = KOSISAPIConfig(api_key=kosis_key)
        kosis_client = KOSISAPIClient(kosis_config)
        logger.info("✓ KOSIS API client initialized")
    
    logger.info()
    
    # Get source IDs
    try:
        bok_source_id = get_source_id('BOK', client=client) if not dry_run else None
        kosis_source_id = get_source_id('KOSIS', client=client) if not dry_run else None
    except ValueError as e:
        logger.error(f"Source not found: {e}")
        logger.error("Run initial setup to create data sources first")
        sys.exit(1)
    
    # Process each series
    logger.info(f"Processing {len(model_cfg.series)} series...")
    logger.info()
    
    all_observations = []
    stats = {
        'total': len(model_cfg.series),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    # Read CSV to get api_code and api_source (not in ModelConfig)
    csv_df = pd.read_csv(csv_path)
    csv_dict = csv_df.set_index('series_id').to_dict('index')
    
    for i, series_cfg in enumerate(model_cfg.series, 1):
        series_id = series_cfg.series_id
        series_name = series_cfg.series_name
        frequency = series_cfg.frequency
        transformation = series_cfg.transformation
        units = getattr(series_cfg, 'units', None)
        category = getattr(series_cfg, 'category', None)
        
        logger.info(f"[{i}/{len(model_cfg.series)}] {series_id}: {series_name}")
        
        try:
            # Get api_code and api_source from CSV
            if series_id not in csv_dict:
                logger.warning(f"  ⚠ Series {series_id} not found in CSV")
                stats['skipped'] += 1
                continue
            
            csv_row = csv_dict[series_id]
            api_code = csv_row.get('api_code', '')
            api_source = csv_row.get('api_source', '')
            
            if not api_code or not api_source:
                logger.warning(f"  ⚠ Missing api_code or api_source for {series_id}")
                stats['skipped'] += 1
                continue
            
            # Get appropriate API client
            api_client = None
            source_id = None
            if api_source == 'BOK' and bok_client:
                api_client = bok_client
                source_id = bok_source_id
            elif api_source == 'KOSIS' and kosis_client:
                api_client = kosis_client
                source_id = kosis_source_id
            else:
                logger.warning(f"  ⚠ No API client available for {api_source}")
                stats['skipped'] += 1
                continue
            
            # Fetch data
            logger.info(f"  Fetching data from {api_source} API (code: {api_code})...")
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=api_client,
                source=api_source,
                frequency=frequency
            )
            
            if df_data.empty:
                logger.warning(f"  ⚠ No data fetched for {series_id}")
                stats['skipped'] += 1
                continue
            
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            
            # Create/update series metadata
            if not dry_run:
                series_model = SeriesModel(
                    series_id=series_id,
                    series_name=series_name,
                    frequency=frequency,
                    transformation=transformation,
                    units=units,
                    category=category,
                    api_source=api_source,
                    api_code=api_code,
                    is_active=True
                )
                upsert_series(series_model, client=client)
                logger.info(f"  ✓ Updated series metadata")
            
            # Add to observations list
            df_data['vintage_id'] = vintage_id
            df_data['job_id'] = job_id
            all_observations.append(df_data)
            
            stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {series_id}: {e}")
            stats['failed'] += 1
            stats['errors'].append(f"{series_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        logger.info()
    
    # Insert all observations
    if all_observations and not dry_run:
        logger.info("Inserting observations into database...")
        df_obs = pd.concat(all_observations, ignore_index=True)
        
        result = insert_observations_from_dataframe(
            df=df_obs,
            vintage_id=vintage_id,
            job_id=job_id,
            client=client
        )
        
        logger.info(f"✓ Inserted {len(df_obs)} observations")
    
    # Save model configuration
    if not dry_run:
        logger.info()
        logger.info("Saving model configuration to database...")
        
        # Use CSV filename as config name (without extension)
        config_name = csv_path.stem.replace('_', '-')  # e.g., 001-initial-spec
        
        # Extract block assignments from ModelConfig
        block_names = model_cfg.block_names
        block_records = []
        
        for series_cfg in model_cfg.series:
            blocks = getattr(series_cfg, 'blocks', None)
            if blocks:
                for block_idx, block_name in enumerate(block_names):
                    if blocks[block_idx] == 1:
                        block_records.append({
                            'series_id': series_cfg.series_id,
                            'block_name': block_name,
                            'block_index': block_idx
                        })
        
        # Convert to config_json format
        config_json = {
            'block_names': block_names,
            'series': [
                {
                    'series_id': s.series_id,
                    'series_name': s.series_name,
                    'frequency': s.frequency,
                    'transformation': s.transformation,
                    'units': getattr(s, 'units', None),
                    'category': getattr(s, 'category', None),
                    'blocks': getattr(s, 'blocks', None)
                }
                for s in model_cfg.series
            ]
        }
        
        # Save model config
        config_result = save_model_config(
            config_name=config_name,
            config_json=config_json,
            block_names=block_names,
            description=f"DFM model configuration loaded from CSV: {csv_path.name}",
            country='KR',
            client=client
        )
        
        config_id = config_result['config_id']
        logger.info(f"✓ Saved model configuration: {config_name} (ID: {config_id})")
        
        # Save block assignments
        if block_records:
            from database.operations import TABLES
            
            # Delete existing assignments and insert new ones
            client.table(TABLES['model_block_assignments']).delete().eq('config_id', config_id).execute()
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(block_records), batch_size):
                batch = [
                    {**rec, 'config_id': config_id}
                    for rec in block_records[i:i + batch_size]
                ]
                client.table(TABLES['model_block_assignments']).insert(batch).execute()
            
            logger.info(f"✓ Saved {len(block_records)} block assignments")
    
    # Update vintage status
    if not dry_run and vintage_id:
        update_vintage_status(
            vintage_id=vintage_id,
            status='completed',
            client=client
        )
        update_ingestion_job(
            job_id=job_id,
            status='completed',
            successful_series=stats['successful'],
            failed_series=stats['failed'],
            total_series=stats['total'],
            client=client
        )
    
    # Summary
    logger.info()
    logger.info("=" * 80)
    logger.info("Initialization Summary")
    logger.info("=" * 80)
    if not dry_run:
        logger.info(f"Vintage ID: {vintage_id}")
        logger.info(f"Job ID: {job_id}")
    logger.info(f"Total series: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    
    if stats['errors']:
        logger.warning("Errors encountered:")
        for error in stats['errors']:
            logger.warning(f"  - {error}")
    
    logger.info("=" * 80)
    
    if stats['failed'] > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == '__main__':
    main()

