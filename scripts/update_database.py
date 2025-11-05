"""Database update script for DFM forecasting system.

This script handles two scenarios:
1. New spec files (002_*.csv) - Checks if series already exist, only adds missing ones
2. Incremental updates - Fetches only new observations (after latest date in DB)

Usage:
    # Update with new spec file
    python scripts/update_database.py --spec-file src/spec/002_new_spec.csv
    
    # Incremental update (fetch new data for all existing series)
    python scripts/update_database.py --incremental
    
    # Update specific series only
    python scripts/update_database.py --series-ids BOK_200Y001 BOK_200Y002
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
import os
import time

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (script is in scripts/ directory)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"✅ Loaded environment from: {env_path}")
else:
    logger.warning("⚠️  .env.local not found at project root")

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
    get_latest_observation_date,
    check_series_exists,
    get_observations,
)
from database.models import SeriesModel
from database.settings import BOKAPIConfig, KOSISAPIConfig
from services.api.bok_client import BOKAPIClient
from services.api.kosis_client import KOSISAPIClient

# Import fetch_series_data from initialization
from initialization import fetch_series_data


def load_model_config_from_csv(csv_path: Path):
    """Load model configuration from CSV file."""
    from src.nowcasting.config_loader import load_model_config_from_csv
    return load_model_config_from_csv(csv_path)


def process_new_spec_file(
    csv_path: Path,
    client: Any,
    bok_client: Optional[BOKAPIClient] = None,
    kosis_client: Optional[KOSISAPIClient] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Process a new spec file (002_*.csv).
    
    Only adds series that don't already exist in the database.
    If series exists, skips it entirely (no updates).
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV spec file
    client : Any
        Supabase client
    bok_client : BOKAPIClient, optional
        BOK API client
    kosis_client : KOSISAPIClient, optional
        KOSIS API client
    dry_run : bool
        If True, don't actually update database
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the update
    """
    logger.info(f"Processing new spec file: {csv_path}")
    print(f"📋 Processing new spec file: {csv_path}")
    
    # Load CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Spec file not found: {csv_path}")
    
    model_cfg = load_model_config_from_csv(csv_path)
    csv_df = pd.read_csv(csv_path)
    csv_dict = csv_df.set_index('series_id').to_dict('index')
    
    # Get source IDs
    bok_source_id = get_source_id('BOK', client) if bok_client else None
    kosis_source_id = get_source_id('KOSIS', client) if kosis_client else None
    
    # Create vintage for this update
    vintage_date = date.today()
    vintage_id = None
    if not dry_run:
        try:
            vintage_result = create_vintage(vintage_date=vintage_date, country='KR', client=client)
            vintage_id = vintage_result['vintage_id']
        except Exception as e:
            # Vintage might already exist
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                logger.info(f"Vintage for {vintage_date} already exists, retrieving...")
                latest_vintage = get_latest_vintage_id(vintage_date=vintage_date, client=client)
                if latest_vintage:
                    vintage_id = latest_vintage
            else:
                raise
    
    # Create ingestion job
    job_id = None
    if not dry_run:
        try:
            job_result = create_ingestion_job(
                vintage_date=vintage_date,
                github_run_id=None,
                client=client
            )
            job_id = job_result['job_id']
        except Exception as e:
            logger.warning(f"Could not create ingestion job: {e}")
    
    stats = {
        'total': len(model_cfg.series),
        'new_series': 0,
        'existing_series': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'observations_inserted': 0,
        'errors': []
    }
    
    # Rate limiting
    last_api_call_time = {}
    min_delay_seconds = 0.6  # BOK: 300 calls per 3 minutes
    
    all_observations = []
    
    for i, series_cfg in enumerate(model_cfg.series, 1):
        series_id = series_cfg.series_id
        series_name = series_cfg.series_name
        frequency = series_cfg.frequency
        transformation = series_cfg.transformation
        
        print(f"\n[{i}/{len(model_cfg.series)}] {series_id}")
        print(f"   Name: {series_name[:70]}...")
        logger.info(f"[{i}/{len(model_cfg.series)}] {series_id}: {series_name}")
        
        try:
            # Check if series already exists
            if check_series_exists(series_id, client):
                logger.info(f"  ✅ Series {series_id} already exists - skipping (no updates)")
                print(f"   ✅ Series already exists - skipping")
                stats['existing_series'] += 1
                stats['skipped'] += 1
                continue
            
            # Series doesn't exist - proceed with full fetch and insert
            stats['new_series'] += 1
            
            # Get api_code and api_source from CSV
            if series_id not in csv_dict:
                logger.warning(f"  ⚠ Series {series_id} not found in CSV")
                stats['skipped'] += 1
                continue
            
            csv_row = csv_dict[series_id]
            api_code = csv_row.get('api_code')
            api_source = csv_row.get('api_source')
            
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
                # Rate limiting
                current_time = time.time()
                if api_source in last_api_call_time:
                    time_since_last = current_time - last_api_call_time[api_source]
                    if time_since_last < min_delay_seconds:
                        sleep_time = min_delay_seconds - time_since_last
                        time.sleep(sleep_time)
                last_api_call_time[api_source] = time.time()
            elif api_source == 'KOSIS' and kosis_client:
                api_client = kosis_client
                source_id = kosis_source_id
                # Rate limiting
                current_time = time.time()
                if api_source in last_api_call_time:
                    time_since_last = current_time - last_api_call_time[api_source]
                    if time_since_last < 0.5:
                        sleep_time = 0.5 - time_since_last
                        time.sleep(sleep_time)
                last_api_call_time[api_source] = time.time()
            else:
                logger.warning(f"  ⚠ No API client available for {api_source}")
                stats['skipped'] += 1
                continue
            
            # Fetch data (full history for new series)
            print(f"   🌐 Fetching data from {api_source} API...")
            logger.info(f"  Fetching data from {api_source} API (code: {api_code})...")
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=api_client,
                source=api_source,
                frequency=frequency,
                start_date=None,  # Fetch full history
                end_date=None
            )
            
            if df_data.empty:
                logger.warning(f"  ⚠ No data fetched for {series_id}")
                print(f"   ⚠️  No data fetched")
                stats['skipped'] += 1
                continue
            
            # Prepare series metadata
            series_model = SeriesModel(
                series_id=series_id,
                series_name=series_name,
                frequency=frequency,
                transformation=transformation,
                units=getattr(series_cfg, 'units', None),
                category=getattr(series_cfg, 'category', None),
                source_id=source_id
            )
            
            # Insert series metadata
            if not dry_run:
                upsert_series(series_model, client=client)
                logger.info(f"  ✅ Inserted series metadata for {series_id}")
            
            # Prepare observations
            obs_df = pd.DataFrame({
                'series_id': series_id,
                'date': pd.to_datetime(df_data['date']),
                'value': df_data['value']
            })
            
            # Deduplicate
            obs_df = obs_df.drop_duplicates(subset=['series_id', 'date'], keep='first')
            
            if not obs_df.empty:
                all_observations.append(obs_df)
                logger.info(f"  ✅ Collected {len(obs_df)} observations for {series_id}")
                print(f"   ✅ Collected {len(obs_df)} observations")
            
            stats['successful'] += 1
            
        except Exception as e:
            error_msg = f"{series_id}: {str(e)[:100]}"
            logger.error(f"  ❌ Error processing {series_id}: {e}", exc_info=True)
            print(f"   ❌ Error: {str(e)[:50]}...")
            stats['failed'] += 1
            stats['errors'].append(error_msg)
    
    # Batch insert all observations
    if all_observations and not dry_run:
        print(f"\n📦 Inserting {len(all_observations)} series observations...")
        logger.info(f"Batch inserting observations for {len(all_observations)} series...")
        
        combined_obs = pd.concat(all_observations, ignore_index=True)
        combined_obs['vintage_id'] = vintage_id
        
        # Deduplicate again before insert
        combined_obs = combined_obs.drop_duplicates(subset=['series_id', 'vintage_id', 'date'], keep='first')
        
        total_inserted = insert_observations_from_dataframe(combined_obs, client=client)
        stats['observations_inserted'] = total_inserted
        logger.info(f"✅ Inserted {total_inserted} observations")
        print(f"   ✅ Inserted {total_inserted} observations")
        
        # Update job status
        if job_id:
            update_ingestion_job(job_id, status='completed', client=client)
    
    # Save model config if new series were added
    if stats['new_series'] > 0 and not dry_run:
        print(f"\n💾 Saving model configuration...")
        logger.info("Saving model configuration...")
        
        block_assignments = {}
        for series_cfg in model_cfg.series:
            series_id = series_cfg.series_id
            if series_id in csv_dict:
                row = csv_dict[series_id]
                block_assignments[series_id] = {
                    'Global': int(row.get('Global', 0)),
                    'Consumption': int(row.get('Consumption', 0)),
                    'Investment': int(row.get('Investment', 0)),
                    'External': int(row.get('External', 0))
                }
        
        save_model_config(
            config_name=model_cfg.config_name,
            country=model_cfg.country,
            series_block_assignments=block_assignments,
            client=client
        )
        logger.info("✅ Saved model configuration")
        print(f"   ✅ Saved model configuration")
    
    return stats


def incremental_update(
    series_ids: Optional[List[str]] = None,
    client: Any = None,
    bok_client: Optional[BOKAPIClient] = None,
    kosis_client: Optional[KOSISAPIClient] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Perform incremental update - fetch only new observations.
    
    For each series, fetches data from the latest observation date forward.
    If series doesn't exist, fetches full history.
    
    Parameters
    ----------
    series_ids : List[str], optional
        Specific series IDs to update (if None, updates all existing series)
    client : Any
        Supabase client
    bok_client : BOKAPIClient, optional
        BOK API client
    kosis_client : KOSISAPIClient, optional
        KOSIS API client
    dry_run : bool
        If True, don't actually update database
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the update
    """
    logger.info("Starting incremental update...")
    print("🔄 Starting incremental update...")
    
    # Get all existing series or filter by series_ids
    from database import list_series
    all_series = list_series(client=client)
    
    if series_ids:
        # Filter to requested series
        series_to_update = [s for s in all_series if s['series_id'] in series_ids]
        missing = set(series_ids) - {s['series_id'] for s in series_to_update}
        if missing:
            logger.warning(f"Series not found in database: {missing}")
            print(f"   ⚠️  Series not found: {missing}")
    else:
        series_to_update = all_series
    
    if not series_to_update:
        logger.warning("No series to update")
        print("   ⚠️  No series to update")
        return {
            'total': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'observations_inserted': 0,
            'errors': []
        }
    
    # Get source IDs
    bok_source_id = get_source_id('BOK', client) if bok_client else None
    kosis_source_id = get_source_id('KOSIS', client) if kosis_client else None
    
    # Create vintage for this update
    vintage_date = date.today()
    vintage_id = None
    if not dry_run:
        try:
            vintage_result = create_vintage(vintage_date=vintage_date, country='KR', client=client)
            vintage_id = vintage_result['vintage_id']
        except Exception as e:
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                logger.info(f"Vintage for {vintage_date} already exists, retrieving...")
                latest_vintage = get_latest_vintage_id(vintage_date=vintage_date, client=client)
                if latest_vintage:
                    vintage_id = latest_vintage
            else:
                raise
    
    # Create ingestion job
    job_id = None
    if not dry_run:
        try:
            job_result = create_ingestion_job(
                vintage_date=vintage_date,
                github_run_id=None,
                client=client
            )
            job_id = job_result['job_id']
        except Exception as e:
            logger.warning(f"Could not create ingestion job: {e}")
    
    stats = {
        'total': len(series_to_update),
        'updated': 0,
        'skipped': 0,
        'failed': 0,
        'observations_inserted': 0,
        'errors': []
    }
    
    # Rate limiting
    last_api_call_time = {}
    min_delay_seconds = 0.6
    
    all_observations = []
    
    # Load CSV to get API codes (we need this for fetching)
    csv_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
    csv_dict = {}
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_dict = csv_df.set_index('series_id').to_dict('index')
    
    for i, series in enumerate(series_to_update, 1):
        series_id = series['series_id']
        frequency = series.get('frequency', 'm')
        api_source = None
        api_code = None
        
        # Try to get API info from CSV
        if series_id in csv_dict:
            api_code = csv_dict[series_id].get('api_code')
            api_source = csv_dict[series_id].get('api_source')
        else:
            # Try to infer from series_id
            if series_id.startswith('BOK_'):
                api_source = 'BOK'
                api_code = series_id.replace('BOK_', '')
            elif series_id.startswith('KOSIS_'):
                api_source = 'KOSIS'
                api_code = series_id.replace('KOSIS_', '').split('_')[-1]
        
        print(f"\n[{i}/{len(series_to_update)}] {series_id}")
        logger.info(f"[{i}/{len(series_to_update)}] {series_id}")
        
        try:
            # Get latest observation date
            latest_date = get_latest_observation_date(series_id, vintage_id=None, client=client)
            
            if latest_date:
                # Fetch only new data (from latest_date + 1 period forward)
                # Calculate start_date based on frequency
                if frequency == 'q':
                    # Add one quarter
                    q = (latest_date.month - 1) // 3 + 1
                    if q == 4:
                        start_date = f"{latest_date.year + 1}Q1"
                    else:
                        start_date = f"{latest_date.year}Q{q + 1}"
                elif frequency == 'm':
                    # Add one month
                    if latest_date.month == 12:
                        start_date_obj = date(latest_date.year + 1, 1, 1)
                    else:
                        start_date_obj = date(latest_date.year, latest_date.month + 1, 1)
                    start_date = start_date_obj.strftime('%Y%m')
                elif frequency == 'd':
                    # Add one day
                    start_date_obj = latest_date + timedelta(days=1)
                    start_date = start_date_obj.strftime('%Y%m%d')
                else:
                    # Default: add one year
                    start_date_obj = date(latest_date.year + 1, 1, 1)
                    start_date = start_date_obj.strftime('%Y')
                
                print(f"   📅 Latest observation: {latest_date}, fetching from {start_date}")
                logger.info(f"  Latest observation: {latest_date}, fetching from {start_date}")
            else:
                # No existing data - fetch full history
                start_date = None
                print(f"   📅 No existing data - fetching full history")
                logger.info(f"  No existing data - fetching full history")
            
            if not api_source or not api_code:
                logger.warning(f"  ⚠ Could not determine API source/code for {series_id}")
                stats['skipped'] += 1
                continue
            
            # Get appropriate API client
            api_client = None
            if api_source == 'BOK' and bok_client:
                api_client = bok_client
                # Rate limiting
                current_time = time.time()
                if api_source in last_api_call_time:
                    time_since_last = current_time - last_api_call_time[api_source]
                    if time_since_last < min_delay_seconds:
                        sleep_time = min_delay_seconds - time_since_last
                        time.sleep(sleep_time)
                last_api_call_time[api_source] = time.time()
            elif api_source == 'KOSIS' and kosis_client:
                api_client = kosis_client
                # Rate limiting
                current_time = time.time()
                if api_source in last_api_call_time:
                    time_since_last = current_time - last_api_call_time[api_source]
                    if time_since_last < 0.5:
                        sleep_time = 0.5 - time_since_last
                        time.sleep(sleep_time)
                last_api_call_time[api_source] = time.time()
            else:
                logger.warning(f"  ⚠ No API client available for {api_source}")
                stats['skipped'] += 1
                continue
            
            # Fetch data
            print(f"   🌐 Fetching new data from {api_source} API...")
            logger.info(f"  Fetching new data from {api_source} API...")
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=api_client,
                source=api_source,
                frequency=frequency,
                start_date=start_date,
                end_date=None
            )
            
            if df_data.empty:
                logger.info(f"  ✅ No new data available for {series_id}")
                print(f"   ✅ No new data available")
                stats['skipped'] += 1
                continue
            
            # Prepare observations
            obs_df = pd.DataFrame({
                'series_id': series_id,
                'date': pd.to_datetime(df_data['date']),
                'value': df_data['value']
            })
            
            # Deduplicate
            obs_df = obs_df.drop_duplicates(subset=['series_id', 'date'], keep='first')
            
            if not obs_df.empty:
                all_observations.append(obs_df)
                logger.info(f"  ✅ Collected {len(obs_df)} new observations for {series_id}")
                print(f"   ✅ Collected {len(obs_df)} new observations")
            
            stats['updated'] += 1
            
        except Exception as e:
            error_msg = f"{series_id}: {str(e)[:100]}"
            logger.error(f"  ❌ Error updating {series_id}: {e}", exc_info=True)
            print(f"   ❌ Error: {str(e)[:50]}...")
            stats['failed'] += 1
            stats['errors'].append(error_msg)
    
    # Batch insert all observations
    if all_observations and not dry_run:
        print(f"\n📦 Inserting {len(all_observations)} series observations...")
        logger.info(f"Batch inserting observations for {len(all_observations)} series...")
        
        combined_obs = pd.concat(all_observations, ignore_index=True)
        combined_obs['vintage_id'] = vintage_id
        
        # Deduplicate again before insert
        combined_obs = combined_obs.drop_duplicates(subset=['series_id', 'vintage_id', 'date'], keep='first')
        
        total_inserted = insert_observations_from_dataframe(combined_obs, client=client)
        stats['observations_inserted'] = total_inserted
        logger.info(f"✅ Inserted {total_inserted} observations")
        print(f"   ✅ Inserted {total_inserted} observations")
        
        # Update job status
        if job_id:
            update_ingestion_job(job_id, status='completed', client=client)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Update database with new spec files or incremental data')
    parser.add_argument('--spec-file', type=str, help='Path to new spec CSV file (e.g., 002_new_spec.csv)')
    parser.add_argument('--incremental', action='store_true', help='Perform incremental update (fetch new data)')
    parser.add_argument('--series-ids', nargs='+', help='Specific series IDs to update (for incremental mode)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (don\'t update database)')
    
    args = parser.parse_args()
    
    if not args.spec_file and not args.incremental:
        parser.error("Must specify either --spec-file or --incremental")
    
    if args.spec_file and args.incremental:
        parser.error("Cannot specify both --spec-file and --incremental")
    
    # Initialize clients
    client = get_client()
    
    bok_client = None
    bok_key = os.getenv('BOK_API_KEY')
    if bok_key:
        bok_config = BOKAPIConfig(auth_key=bok_key)
        bok_client = BOKAPIClient(bok_config)
        logger.info("✅ BOK API client initialized")
    else:
        logger.warning("⚠️  BOK_API_KEY not set")
    
    kosis_client = None
    kosis_key = os.getenv('KOSIS_API_KEY')
    if kosis_key:
        kosis_config = KOSISAPIConfig(api_key=kosis_key)
        kosis_client = KOSISAPIClient(kosis_config)
        logger.info("✅ KOSIS API client initialized")
    else:
        logger.warning("⚠️  KOSIS_API_KEY not set")
    
    # Process based on mode
    if args.spec_file:
        csv_path = Path(args.spec_file)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        
        stats = process_new_spec_file(
            csv_path=csv_path,
            client=client,
            bok_client=bok_client,
            kosis_client=kosis_client,
            dry_run=args.dry_run
        )
        
        print("\n" + "=" * 80)
        print("📊 Update Summary (New Spec File)")
        print("=" * 80)
        print(f"   Total series: {stats['total']}")
        print(f"   ✅ New series added: {stats['new_series']}")
        print(f"   ⏭️  Existing series skipped: {stats['existing_series']}")
        print(f"   ✅ Successful: {stats['successful']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   📈 Observations inserted: {stats['observations_inserted']}")
        
    elif args.incremental:
        stats = incremental_update(
            series_ids=args.series_ids,
            client=client,
            bok_client=bok_client,
            kosis_client=kosis_client,
            dry_run=args.dry_run
        )
        
        print("\n" + "=" * 80)
        print("📊 Update Summary (Incremental)")
        print("=" * 80)
        print(f"   Total series: {stats['total']}")
        print(f"   ✅ Updated: {stats['updated']}")
        print(f"   ⏭️  Skipped (no new data): {stats['skipped']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   📈 Observations inserted: {stats['observations_inserted']}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:
            print(f"   - {error}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more")
    
    print("=" * 80)
    
    if stats.get('failed', 0) > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == '__main__':
    main()

