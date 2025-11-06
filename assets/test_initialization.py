#!/usr/bin/env python3
"""
Test script for API data stream initialization.
Tests data flow with a small sample (2-3 series) to verify everything works.
"""
import sys
import logging
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
import os

# Add project root to path (script is in assets/, but project root is parent)
project_root = Path(__file__).resolve().parent.parent  # Go up from assets/ to project root
# Ensure we use the correct path (worktree)
import os
os.chdir(str(project_root))
# Insert project root at the beginning, but keep other paths
sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / '.env.local'
if not env_path.exists():
    print("⚠️  .env.local not found at project root")
else:
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from database.client import get_client
    from database.operations import (
        insert_observations_from_dataframe,
        update_vintage_status,
        ensure_vintage_and_job,
        finalize_ingestion_job
    )
    from database.db_utils import (
        load_series_from_csv,
        initialize_api_clients,
        process_series,
        print_statistics_summary,
        RateLimiter,
    )
    
    logger.info("=" * 80)
    logger.info("Testing API Data Stream Initialization")
    logger.info("=" * 80)
    print("\n🚀 Starting test initialization...")
    print("=" * 80)
    
    # Initialize database client
    print("\n📡 Connecting to database...")
    client = get_client()
    logger.info("✅ Database client initialized")
    print("✅ Database connection established")
    
    # Load test series from CSV spec file for comprehensive testing
    import pandas as pd
    spec_csv_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
    
    if spec_csv_path.exists():
        print(f"📋 Loading series from: {spec_csv_path}")
        logger.info(f"Loading series from CSV: {spec_csv_path}")
        test_series = load_series_from_csv(spec_csv_path)
        print(f"   ✅ Loaded {len(test_series)} test series from CSV")
        logger.info(f"Loaded {len(test_series)} test series from CSV")
    else:
        # Fallback to hardcoded test data if CSV not found
        logger.warning(f"CSV spec not found at {spec_csv_path}, using hardcoded test data")
        print(f"⚠️  CSV spec not found, using hardcoded test data")
        test_series = [
            {
                'series_id': 'BOK_200Y109_10201',
                'series_name': '2.1.2.2.3. 국내총생산에 대한 지출(원계열, 명목, 분기 및 연간)',
                'frequency': 'q',
                'transformation': 'pch',
                'category': 'GDP',
                'units': 'PCT',
                'data_code': '200Y109',
                'item_id': '10201',
                'api_source': 'BOK',
                'blocks': {'Global': 1, 'Consumption': 0, 'Investment': 0, 'External': 0}
            },
            {
                'series_id': 'BOK_200Y105_1400',
                'series_name': '2.1.2.1.3. 경제활동별 GDP 및 GNI(원계열, 명목, 분기 및 연간)',
                'frequency': 'q',
                'transformation': 'pch',
                'category': 'GDP',
                'units': 'PCT',
                'data_code': '200Y105',
                'item_id': '1400',
                'api_source': 'BOK',
                'blocks': {'Global': 1, 'Consumption': 0, 'Investment': 0, 'External': 0}
            }
        ]
    
    print(f"\n📊 Test Configuration:")
    print(f"   Series count: {len(test_series)}")
    logger.info(f"\n📊 Testing with {len(test_series)} series:")
    for i, s in enumerate(test_series, 1):
        print(f"   {i}. {s['series_id']}: {s['series_name'][:50]}...")
        logger.info(f"   - {s['series_id']}: {s['series_name'][:50]}...")
    
    # Step 1: Initialize API clients
    print("\n" + "=" * 80)
    print("STEP 1: Initializing API Clients")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Initializing API clients")
    logger.info("-" * 80)
    
    bok_client, kosis_client = initialize_api_clients()
    if bok_client:
        print("   ✅ BOK API client initialized")
    else:
        print("   ⚠️  BOK API client not available")
    if kosis_client:
        print("   ✅ KOSIS API client initialized")
    else:
        print("   ⚠️  KOSIS API client not available")
    
    # Step 3: Test fetching data for one series
    print("\n" + "=" * 80)
    print("STEP 3: Fetching and Processing Data")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Testing data fetch (limited to last 4 periods)")
    logger.info("-" * 80)
    
    vintage_date = date.today()
    vintage_id = None  # Will be set when creating vintage
    print(f"📅 Vintage date: {vintage_date}")
    processed_series = []
    all_observations = []  # Collect all observations for batch insertion
    
    # Statistics tracking
    stats = {
        'total': len(test_series),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'by_frequency': {'q': 0, 'm': 0, 'd': 0, 'a': 0},
        'by_source': {'BOK': 0, 'KOSIS': 0},
        'observations_inserted': 0,
        'errors': []
    }
    
    # Create vintage (matching initialization.py)
    print("\n📦 Creating vintage...")
    import os
    github_run_id = os.getenv('GITHUB_RUN_ID', f'test-{vintage_date.isoformat()}')
    vintage_id = ensure_vintage_and_job(
        vintage_date=vintage_date,
        client=client,
        dry_run=False,
        github_run_id=github_run_id
    )
    # Handle tuple return (legacy compatibility)
    if isinstance(vintage_id, tuple):
        vintage_id = vintage_id[0] if vintage_id[0] else None
    
    if not vintage_id:
        print(f"   ❌ Failed to create vintage - cannot continue")
        logger.error(f"   ❌ Failed to create vintage - cannot continue")
        sys.exit(1)
    
    print(f"   ✅ Vintage ID: {vintage_id}")
    logger.info(f"   ✅ Vintage ID: {vintage_id}")
    
    # Rate limiting
    rate_limiter = RateLimiter(bok_delay=0.6, kosis_delay=0.5)
    
    print(f"\n🔄 Processing {len(test_series)} series...")
    print(f"   Testing frequencies: {set(s['frequency'] for s in test_series)}")
    print(f"   Testing sources: {set(s['api_source'] for s in test_series)}")
    logger.info(f"Processing {len(test_series)} series")
    logger.info(f"Frequencies: {set(s['frequency'] for s in test_series)}")
    logger.info(f"Sources: {set(s['api_source'] for s in test_series)}")
    
    for idx, series_cfg in enumerate(test_series, 1):
        series_id = series_cfg['series_id']
        series_name = series_cfg['series_name']
        
        print(f"\n[{idx}/{len(test_series)}] {series_id}")
        print(f"   Name: {series_name[:60]}...")
        logger.info(f"[{idx}/{len(test_series)}] {series_id}: {series_name}")
        
        try:
            # Ensure vintage_id is an integer, not tuple
            actual_vintage_id = vintage_id
            if isinstance(vintage_id, tuple):
                actual_vintage_id = vintage_id[0] if vintage_id[0] else None
            
            if actual_vintage_id is None:
                logger.warning(f"⚠️  Skipping {series_id} - vintage_id is None")
                stats['skipped'] += 1
                continue
            
            df_data, success, error_msg = process_series(
                series_cfg=series_cfg,
                bok_client=bok_client,
                kosis_client=kosis_client,
                rate_limiter=rate_limiter,
                vintage_id=actual_vintage_id,
                client=client,
                dry_run=False,
                github_run_id=github_run_id
            )
            
            if not success:
                if error_msg:
                    print(f"   ⚠️  {error_msg}")
                    logger.warning(f"  ⚠ {error_msg}")
                stats['skipped'] += 1
                continue
            
            # Limit data for testing (use first 12 rows)
            max_test_rows = 12
            df_test = df_data.head(max_test_rows).copy()
            df_test = df_test.drop_duplicates(subset=['series_id', 'date'], keep='first')
            
            print(f"   ✅ Fetched {len(df_data)} data points (using {len(df_test)} for test)")
            if len(df_test) > 0:
                print(f"      Date range: {df_test['date'].min()} to {df_test['date'].max()}")
            logger.info(f"  ✓ Fetched {len(df_data)} data points (using {len(df_test)} for test)")
            
            all_observations.append(df_test)
            processed_series.append(series_id)
            stats['successful'] += 1
            stats['by_frequency'][series_cfg['frequency']] = stats['by_frequency'].get(series_cfg['frequency'], 0) + 1
            stats['by_source'][series_cfg['api_source']] = stats['by_source'].get(series_cfg['api_source'], 0) + 1
            print(f"   ✅ Successfully processed {series_id}")
            
        except Exception as e:
            error_msg = str(e)
            stats['failed'] += 1
            stats['errors'].append({
                'series_id': series_id,
                'error': error_msg
            })
            logger.error(f"   ❌ Error processing {series_id}: {e}", exc_info=True)
            print(f"   ❌ Error: {error_msg[:100]}")
            continue
    
    # Step 4: Batch insert all observations (matching initialization.py)
    print("\n" + "=" * 80)
    print("STEP 4: Batch Inserting Observations")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 4: Batch inserting observations")
    logger.info("-" * 80)
    
    if all_observations:
        print(f"📊 Preparing {len(all_observations)} series for batch insertion...")
        logger.info(f"Preparing {len(all_observations)} series for batch insertion")
        import pandas as pd
        df_obs = pd.concat(all_observations, ignore_index=True)
        print(f"   Total observations: {len(df_obs)}")
        print("   💾 Inserting into database...")
        logger.info(f"Total observations: {len(df_obs)}")
        
        try:
            # Handle tuple return (legacy compatibility)
            actual_vintage_id = vintage_id
            if isinstance(vintage_id, tuple):
                actual_vintage_id = vintage_id[0] if vintage_id[0] else None
            
            inserted_count = insert_observations_from_dataframe(
                df=df_obs,
                vintage_id=actual_vintage_id,
                github_run_id=github_run_id,
                client=client
            )
            stats['observations_inserted'] = inserted_count
            print(f"   ✅ Successfully inserted {inserted_count} observations")
        except Exception as e:
            logger.error(f"❌ Failed to insert observations: {e}")
            print(f"   ❌ Failed to insert observations: {e}")
            raise
    else:
        print("⚠️  No observations to insert")
        logger.warning("No observations to insert")
    
    # Step 5: Model configuration is now loaded directly from CSV, no need to save to database
    print("\n" + "=" * 80)
    print("STEP 5: Model Configuration")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 5: Model configuration (loaded from CSV)")
    logger.info("-" * 80)
    print("   ℹ️  Model configuration is loaded directly from CSV, no database storage needed")
    
    # Step 6: Update status (matching initialization.py)
    print("\n" + "=" * 80)
    print("STEP 6: Updating Status")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 6: Updating status")
    logger.info("-" * 80)
    
    if vintage_id:
        print("   📅 Updating vintage status to 'completed'...")
        try:
            update_vintage_status(
                vintage_id=vintage_id,
                status='completed',
                client=client
            )
            print("   ✅ Vintage status updated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to update vintage status: {e}")
            print(f"   ⚠️  Failed to update vintage status (continuing)")
    
    if vintage_id:
        print("   📋 Finalizing vintage...")
        try:
            finalize_ingestion_job(
                vintage_id=vintage_id,
                status='completed',
                successful_series=stats['successful'],
                failed_series=stats['failed'],
                total_series=stats['total'],
                client=client
            )
            print("   ✅ Vintage finalized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to finalize vintage: {e}")
            print(f"   ⚠️  Failed to finalize vintage (continuing)")
    
    # Step 7: Verify data in database
    print("\n" + "=" * 80)
    print("STEP 7: Verifying Data in Database")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 4: Verifying data in database")
    logger.info("-" * 80)
    
    print(f"\n🔍 Verifying {len(processed_series)} processed series...")
    for series_id in processed_series:
        try:
            print(f"   Checking {series_id}...")
            result = client.table('observations').select('*').eq('series_id', series_id).limit(5).execute()
            count = len(result.data) if result.data else 0
            print(f"   ✅ {series_id}: {count} observations found")
            if result.data:
                print(f"      Sample: {result.data[0]}")
        except Exception as e:
            logger.error(f"❌ Error verifying {series_id}: {e}")
            print(f"   ❌ Error verifying {series_id}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ TEST SUMMARY")
    print("=" * 80)
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test Summary")
    logger.info("=" * 80)
    
    # Add additional stats for test
    stats['observations_inserted'] = stats.get('observations_inserted', 0)
    
    print_statistics_summary(stats, vintage_id)
    
    # Additional test-specific stats
    print(f"\n📊 By Frequency:")
    for freq, count in stats['by_frequency'].items():
        if count > 0:
            print(f"   {freq.upper()}: {count}")
    
    print(f"\n📊 By Source:")
    for source, count in stats['by_source'].items():
        if count > 0:
            print(f"   {source}: {count}")
    
    print(f"\n📅 Vintage date: {vintage_date}")
    logger.info(f"Vintage date: {vintage_date}")
    
    if stats['successful'] == stats['total']:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {stats['failed']} series failed, {stats['successful']} succeeded")
    
    print("=" * 80)
    
except ImportError as e:
    print("\n❌ Import error occurred!")
    print(f"   Error: {e}")
    logger.error(f"❌ Import error: {e}")
    logger.error("   Make sure you're in the virtual environment:")
    logger.error("   source .venv/bin/activate")
    print("\n💡 Tip: Activate virtual environment with:")
    print("   source .venv/bin/activate")
except Exception as e:
    print("\n❌ Unexpected error occurred!")
    print(f"   Error: {e}")
    logger.error(f"❌ Error: {e}", exc_info=True)
