"""Consolidated initialization script for macroeconomic forecasting database setup.

This script:
1. Loads CSV specification file (src/spec/001_initial_spec.csv) to get selected series
2. Fetches data from APIs (BOK/KOSIS) for those series
3. Updates Supabase database with:
   - Series metadata
   - Observations data
   - Model configuration with block assignments
   - Creates a vintage

Usage:
    python initialization.py
    python initialization.py --csv-file src/spec/001_initial_spec.csv
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv
import os

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (script is at project root)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"✅ Loaded environment from: {env_path}")
else:
    logger.warning("⚠️  .env.local not found at project root")

import pandas as pd

from database import (
    get_client,
    update_vintage_status,
    insert_observations_from_dataframe,
)
from database.db_utils import (
    RateLimiter,
    initialize_api_clients,
    load_series_from_csv,
    get_block_names_from_csv,
    process_series,
    print_statistics_summary,
)
from database import ensure_vintage_and_job, finalize_ingestion_job


def main() -> None:
    """Main initialization workflow."""
    parser = argparse.ArgumentParser(
        description='Initialize macroeconomic forecasting database with data from CSV specification'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='src/spec/001_initial_spec.csv',
        help='Path to CSV specification file (default: src/spec/001_initial_spec.csv)'
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
    
    print("\n" + "=" * 80)
    print("🚀 MACROECONOMIC FORECASTING DATABASE INITIALIZATION")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("Macroeconomic Forecasting Database Initialization")
    logger.info("=" * 80)
    print(f"📄 CSV file: {args.csv_file}")
    print(f"🧪 Dry run: {args.dry_run}")
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Dry run: {args.dry_run}")
    print("=" * 80)
    logger.info("=" * 80)
    print()
    
    # Load model configuration from CSV
    print("\n📂 Loading CSV specification file...")
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"   ✅ Found CSV file: {csv_path}")
    try:
        print("   📖 Parsing CSV configuration...")
        series_list = load_series_from_csv(csv_path)
        block_names = get_block_names_from_csv(csv_path)
        print(f"   ✅ Loaded CSV: {len(series_list)} series")
        print(f"   📊 Block names: {', '.join(block_names)}")
        logger.info(f"Loaded CSV: {len(series_list)} series")
        logger.info(f"Block names: {block_names}")
    except Exception as e:
        print(f"❌ Failed to load model config: {e}")
        logger.error(f"Failed to load model config from CSV: {e}", exc_info=True)
        sys.exit(1)
    
    # Determine vintage date
    print("\n📅 Determining vintage date...")
    if args.vintage_date:
        vintage_date = datetime.strptime(args.vintage_date, '%Y-%m-%d').date()
        print(f"   ✅ Using provided vintage date: {vintage_date}")
    else:
        vintage_date = date.today()
        print(f"   ✅ Using today's date: {vintage_date}")
    
    dry_run = args.dry_run
    
    logger.info(f"Vintage date: {vintage_date}")
    print()
    
    # Create vintage
    print("\n📦 Creating vintage...")
    client = get_client()
    import os
    github_run_id = os.getenv('GITHUB_RUN_ID', f'manual-{vintage_date.isoformat()}')
    vintage_id = ensure_vintage_and_job(
        vintage_date=vintage_date,
        client=client,
        dry_run=dry_run,
        github_run_id=github_run_id
    )
    if vintage_id:
        print(f"   ✅ Vintage: {vintage_id}")
    if not dry_run and not vintage_id:
        vintage_id = None
        print("   🧪 Dry run: Skipping vintage creation")
        logger.info("Dry run: Skipping vintage creation")
    
    print()
    
    # Initialize API clients
    print("\n🔧 Initializing API clients...")
    bok_client, kosis_client = initialize_api_clients()
    if bok_client:
        print("   ✅ BOK API client initialized")
    else:
        print("   ⚠️  BOK API client not available")
    if kosis_client:
        print("   ✅ KOSIS API client initialized")
    else:
        print("   ⚠️  KOSIS API client not available")
    
    print()
    
    # Source codes (no longer need database lookup)
    bok_source_code = 'BOK'
    kosis_source_code = 'KOSIS'
    
    # Process each series
    print("\n" + "=" * 80)
    print(f"🔄 PROCESSING {len(series_list)} SERIES")
    print("=" * 80)
    logger.info(f"Processing {len(series_list)} series...")
    print()
    
    # Rate limiting
    rate_limiter = RateLimiter(bok_delay=0.6, kosis_delay=0.5)
    
    all_observations = []
    stats = {
        'total': len(series_list),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    for i, series_cfg in enumerate(series_list, 1):
        series_id = series_cfg['series_id']
        series_name = series_cfg['series_name']
        
        print(f"\n[{i}/{len(series_list)}] {series_id}")
        print(f"   Name: {series_name[:70]}...")
        print(f"   Frequency: {series_cfg['frequency']}, Transformation: {series_cfg['transformation']}")
        logger.info(f"[{i}/{len(series_list)}] {series_id}: {series_name}")
        
        try:
            df_data, success, error_msg = process_series(
                series_cfg=series_cfg,
                bok_client=bok_client,
                kosis_client=kosis_client,
                rate_limiter=rate_limiter,
                vintage_id=vintage_id,
                client=client,
                dry_run=dry_run,
                github_run_id=github_run_id
            )
            
            if not success:
                if error_msg:
                    print(f"   ⚠️  {error_msg}")
                    logger.warning(f"  ⚠ {error_msg}")
                stats['skipped'] += 1
                continue
            
            print(f"   ✅ Fetched {len(df_data)} data points")
            if len(df_data) > 0:
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            
            all_observations.append(df_data)
            stats['successful'] += 1
            print(f"   ✅ Series {series_id} processed successfully")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            logger.error(f"  ❌ Error processing {series_id}: {e}", exc_info=True)
            stats['failed'] += 1
            stats['errors'].append(f"{series_id}: {str(e)}")
        
        print()
    
    # Insert all observations
    print("\n" + "=" * 80)
    print("💾 INSERTING OBSERVATIONS INTO DATABASE")
    print("=" * 80)
    if all_observations and not dry_run:
        print(f"📊 Preparing {len(all_observations)} series for batch insertion...")
        logger.info("Inserting observations into database...")
        df_obs = pd.concat(all_observations, ignore_index=True)
        
        # Deduplicate observations (same as test_initialization.py)
        df_obs = df_obs.drop_duplicates(subset=['series_id', 'vintage_id', 'date'], keep='first')
        print(f"   Total observations: {len(df_obs)} (after deduplication)")
        print("   💾 Inserting into database...")
        
        result = insert_observations_from_dataframe(
            df=df_obs,
            vintage_id=vintage_id,
            github_run_id=github_run_id,
            client=client
        )
        
        print(f"   ✅ Successfully inserted {len(df_obs)} observations")
        logger.info(f"✓ Inserted {len(df_obs)} observations")
    elif all_observations and dry_run:
        df_obs = pd.concat(all_observations, ignore_index=True)
        print(f"🧪 Dry run: Would insert {len(df_obs)} observations")
    else:
        print("⚠️  No observations to insert")
    
    # Model configuration is now loaded directly from CSV, no need to save to database
    
    # Update vintage status
    print("\n" + "=" * 80)
    print("📝 UPDATING STATUS")
    print("=" * 80)
    if not dry_run and vintage_id:
        print("   📅 Updating vintage status to 'completed'...")
        update_vintage_status(
            vintage_id=vintage_id,
            status='completed',
            client=client
        )
        print("   ✅ Vintage status updated")
        
        print("   📋 Finalizing vintage...")
        finalize_ingestion_job(
            vintage_id=vintage_id,
            status='completed',
            successful_series=stats['successful'],
            failed_series=stats['failed'],
            total_series=stats['total'],
            client=client
        )
        print("   ✅ Vintage finalized")
    else:
        print("🧪 Dry run: Skipping status updates")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ INITIALIZATION SUMMARY")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("Initialization Summary")
    logger.info("=" * 80)
    
    print_statistics_summary(stats, vintage_id if not dry_run else None)
    
    logger.info("=" * 80)
    
    if stats['failed'] > 0 and not args.dry_run:
        print("\n❌ Exiting with error code due to failed series")
        sys.exit(1)
    else:
        print("\n✅ Initialization completed successfully!")


if __name__ == '__main__':
    main()

