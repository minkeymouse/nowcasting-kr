#!/usr/bin/env python3
"""
Test script for API data stream initialization.
Tests data flow with a small sample (2-3 series) to verify everything works.
"""
import sys
import logging
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv
import os

# Add project root to path (script is at project root)
project_root = Path(__file__).resolve().parent
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
        get_source_id, upsert_series, insert_observations_from_dataframe,
        save_model_config, create_vintage, update_vintage_status, get_series,
        create_ingestion_job, update_ingestion_job
    )
    from database.settings import BOKAPIConfig, KOSISAPIConfig
    from services.api.bok_client import BOKAPIClient
    from services.api.kosis_client import KOSISAPIClient
    from initialization import fetch_series_data  # Reuse to avoid duplication
    
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
    
    test_series = []
    if spec_csv_path.exists():
        print(f"📋 Loading series from: {spec_csv_path}")
        logger.info(f"Loading series from CSV: {spec_csv_path}")
        df_spec = pd.read_csv(spec_csv_path)
        
        # Use ALL series from the CSV for comprehensive testing
        # This ensures the test validates the complete initialization process
        test_series_ids = df_spec['series_id'].tolist()
        
        logger.info(f"Loaded {len(test_series_ids)} series from CSV for testing")
        print(f"   ✅ Loaded {len(test_series_ids)} series from CSV (all series)")
        
        # Convert to test_series format
        for series_id in test_series_ids:
            row = df_spec[df_spec['series_id'] == series_id].iloc[0]
            test_series.append({
                'series_id': row['series_id'],
                'series_name': row['series_name'],
                'frequency': row['frequency'],
                'transformation': row['transformation'],
                'category': row['category'],
                'units': row['units'],
                'api_code': row['api_code'],
                'api_source': row['api_source'],
                'blocks': {
                    'Global': int(row.get('Global', 0)),
                    'Consumption': int(row.get('Consumption', 0)),
                    'Investment': int(row.get('Investment', 0)),
                    'External': int(row.get('External', 0))
                }
            })
        
        print(f"   ✅ Loaded {len(test_series)} test series from CSV")
        logger.info(f"Loaded {len(test_series)} test series from CSV")
    else:
        # Fallback to hardcoded test data if CSV not found
        logger.warning(f"CSV spec not found at {spec_csv_path}, using hardcoded test data")
        print(f"⚠️  CSV spec not found, using hardcoded test data")
        test_series = [
            {
                'series_id': 'BOK_200Y109',
                'series_name': '2.1.2.2.3. 국내총생산에 대한 지출(원계열, 명목, 분기 및 연간)',
                'frequency': 'q',
                'transformation': 'pch',
                'category': 'GDP',
                'units': 'PCT',
                'api_code': '200Y109',
                'api_source': 'BOK',
                'blocks': {'Global': 1, 'Consumption': 0, 'Investment': 0, 'External': 0}
            },
            {
                'series_id': 'BOK_200Y105',
                'series_name': '2.1.2.1.3. 경제활동별 GDP 및 GNI(원계열, 명목, 분기 및 연간)',
                'frequency': 'q',
                'transformation': 'pch',
                'category': 'GDP',
                'units': 'PCT',
                'api_code': '200Y105',
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
    
    # Step 1: Get source IDs
    print("\n" + "=" * 80)
    print("STEP 1: Getting Source IDs")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Getting source IDs")
    logger.info("-" * 80)
    
    print("📋 Retrieving BOK source ID...")
    bok_source_id = get_source_id('BOK', client=client)
    logger.info(f"✅ BOK source_id: {bok_source_id}")
    print(f"   ✅ BOK source_id: {bok_source_id}")
    
    print("📋 Retrieving KOSIS source ID...")
    kosis_source_id = get_source_id('KOSIS', client=client)
    logger.info(f"✅ KOSIS source_id: {kosis_source_id}")
    print(f"   ✅ KOSIS source_id: {kosis_source_id}")
    
    # Step 2: Initialize API clients
    print("\n" + "=" * 80)
    print("STEP 2: Initializing API Clients")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 2: Initializing API clients")
    logger.info("-" * 80)
    
    print("🔑 Checking API keys...")
    bok_api_key = os.getenv('BOK_API_KEY')
    kosis_api_key = os.getenv('KOSIS_API_KEY')
    
    bok_client = None
    if bok_api_key:
        print("   ✅ BOK_API_KEY found")
        print("   🔧 Initializing BOK API client...")
        bok_config = BOKAPIConfig(auth_key=bok_api_key)
        bok_client = BOKAPIClient(bok_config)
        logger.info("✅ BOK API client initialized")
        print("   ✅ BOK API client initialized")
    else:
        logger.warning("⚠️  BOK_API_KEY not set - skipping BOK API tests")
        print("   ⚠️  BOK_API_KEY not set - skipping BOK API tests")
    
    kosis_client = None
    if kosis_api_key:
        print("   ✅ KOSIS_API_KEY found")
        print("   🔧 Initializing KOSIS API client...")
        kosis_config = KOSISAPIConfig(api_key=kosis_api_key)
        kosis_client = KOSISAPIClient(kosis_config)
        logger.info("✅ KOSIS API client initialized")
        print("   ✅ KOSIS API client initialized")
    else:
        logger.warning("⚠️  KOSIS_API_KEY not set - skipping KOSIS API tests")
        print("   ⚠️  KOSIS_API_KEY not set - skipping KOSIS API tests")
    
    # Step 3: Test fetching data for one series
    print("\n" + "=" * 80)
    print("STEP 3: Fetching and Processing Data")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Testing data fetch (limited to last 4 periods)")
    logger.info("-" * 80)
    
    vintage_date = date.today()
    vintage_id = None  # Will be set when creating vintage
    job_id = None  # Will be set when creating ingestion job
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
    
    # Create vintage and ingestion job (matching initialization.py)
    print("\n📦 Creating vintage and ingestion job...")
    print("   📅 Creating/getting vintage...")
    try:
        vintage = create_vintage(vintage_date=vintage_date, country='KR', client=client)
        vintage_id = vintage['vintage_id'] if isinstance(vintage, dict) else vintage
    except Exception as e:
        # If vintage already exists, get it
        if 'duplicate key' in str(e).lower() or '23505' in str(e):
            result = client.table('data_vintages').select('vintage_id').eq('vintage_date', vintage_date.isoformat()).eq('country', 'KR').execute()
            if result.data:
                vintage_id = result.data[0]['vintage_id']
                logger.info(f"   ✅ Using existing vintage_id: {vintage_id}")
                print(f"   ✅ Using existing vintage ID: {vintage_id}")
            else:
                raise ValueError(f"Vintage exists but could not retrieve ID")
        else:
            raise
    else:
        logger.info(f"   ✅ Created new vintage_id: {vintage_id}")
        print(f"   ✅ Vintage ID: {vintage_id}")
    
    # Create ingestion job (matching initialization.py signature)
    print("   📋 Creating ingestion job...")
    try:
        job = create_ingestion_job(
            github_run_id='test-run',  # Required parameter
            vintage_date=vintage_date,  # Uses vintage_date, not vintage_id
            github_workflow_run_url=None,
            client=client
        )
        job_id = job['job_id'] if isinstance(job, dict) else job
        logger.info(f"   ✅ Created ingestion job: {job_id}")
        print(f"   ✅ Ingestion job ID: {job_id}")
    except Exception as e:
        logger.warning(f"   ⚠️  Failed to create ingestion job: {e}")
        print(f"   ⚠️  Failed to create ingestion job (continuing anyway)")
        job_id = None
    
    print(f"\n🔄 Processing {len(test_series)} series...")
    print(f"   Testing frequencies: {set(s['frequency'] for s in test_series)}")
    print(f"   Testing sources: {set(s['api_source'] for s in test_series)}")
    logger.info(f"Processing {len(test_series)} series")
    logger.info(f"Frequencies: {set(s['frequency'] for s in test_series)}")
    logger.info(f"Sources: {set(s['api_source'] for s in test_series)}")
    
    for idx, series_cfg in enumerate(test_series, 1):
        if series_cfg['api_source'] == 'BOK' and not bok_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - BOK client not available")
            stats['skipped'] += 1
            continue
        if series_cfg['api_source'] == 'KOSIS' and not kosis_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - KOSIS client not available")
            stats['skipped'] += 1
            continue
        
        print(f"\n[{idx}/{len(test_series)}] Processing: {series_cfg['series_id']}")
        print(f"   Name: {series_cfg['series_name'][:60]}...")
        logger.info(f"\n📥 Fetching data for {series_cfg['series_id']}...")
        
        try:
            # Get appropriate API client
            api_client = None
            if series_cfg['api_source'] == 'BOK' and bok_client:
                api_client = bok_client
            elif series_cfg['api_source'] == 'KOSIS' and kosis_client:
                api_client = kosis_client
            else:
                logger.warning(f"⚠️  No API client available for {series_cfg['api_source']}")
                stats['skipped'] += 1
                continue
            
            # Fetch data using shared function (limited date range for testing)
            print(f"   🌐 Fetching data from {series_cfg['api_source']} API...")
            logger.info(f"  Fetching data for {series_cfg['series_id']}...")
            
            # Set limited date range for testing (2023-2024)
            frequency = series_cfg['frequency'].lower()
            if frequency == 'q':
                start_date, end_date = '2023Q1', '2024Q4'
            elif frequency == 'm':
                start_date, end_date = '202301', '202412'
            elif frequency == 'd':
                start_date, end_date = '20230101', '20241231'
            else:
                start_date, end_date = '2023', '2024'
            
            df_data = fetch_series_data(
                series_id=series_cfg['series_id'],
                api_code=series_cfg['api_code'],
                api_client=api_client,
                source=series_cfg['api_source'],
                frequency=frequency,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_data.empty:
                logger.warning(f"  ⚠ No data fetched for {series_cfg['series_id']}")
                stats['skipped'] += 1
                continue
            
            print(f"   ✅ Fetched {len(df_data)} data points")
            if len(df_data) > 0:
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            
            # Save series metadata (workaround for trigger issue - see note below)
            print(f"   💾 Saving series metadata...")
            from database.models import SeriesModel
            from database.operations import get_series
            
            series_model = SeriesModel(
                series_id=series_cfg['series_id'],
                series_name=series_cfg['series_name'],
                frequency=series_cfg['frequency'],
                transformation=series_cfg['transformation'],
                category=series_cfg['category'],
                units=series_cfg['units'],
                api_source=series_cfg['api_source'],
                api_code=series_cfg['api_code'],
                is_active=True
            )
            
            # Workaround: Skip UPDATE for existing series (trigger issue - needs DB fix)
            existing_series = get_series(series_cfg['series_id'], client=client)
            if not existing_series:
                data = series_model.model_dump(exclude_none=True)
                data.pop('updated_at', None)
                data.pop('created_at', None)
                client.table('series').insert(data).execute()
                print(f"   ✅ Series metadata saved")
            
            # Limit data for testing (use first 12 rows)
            max_test_rows = 12
            df_test = df_data.head(max_test_rows).copy()
            df_test = df_test.drop_duplicates(subset=['series_id', 'date'], keep='first')
            
            # Add vintage_id and job_id
            df_test['vintage_id'] = vintage_id
            if job_id:
                df_test['job_id'] = job_id
            
            # Collect for batch insertion
            all_observations.append(df_test)
            print(f"   ✅ Collected {len(df_test)} observations")
            
            processed_series.append(series_cfg['series_id'])
            stats['successful'] += 1
            stats['by_frequency'][series_cfg['frequency']] = stats['by_frequency'].get(series_cfg['frequency'], 0) + 1
            stats['by_source'][series_cfg['api_source']] = stats['by_source'].get(series_cfg['api_source'], 0) + 1
            print(f"   ✅ Successfully processed {series_cfg['series_id']}")
            
        except Exception as e:
            error_msg = str(e)
            stats['failed'] += 1
            stats['errors'].append({
                'series_id': series_cfg['series_id'],
                'error': error_msg
            })
            logger.error(f"   ❌ Error processing {series_cfg['series_id']}: {e}", exc_info=True)
            print(f"   ❌ Error: {error_msg[:100]}")  # Truncate long errors
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
            inserted_count = insert_observations_from_dataframe(
                df=df_obs,
                vintage_id=vintage_id,
                job_id=job_id,
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
    
    # Step 5: Save model configuration (matching initialization.py)
    print("\n" + "=" * 80)
    print("STEP 5: Saving Model Configuration")
    print("=" * 80)
    logger.info("\n" + "-" * 80)
    logger.info("Step 5: Saving model configuration")
    logger.info("-" * 80)
    
    print("📋 Preparing model configuration...")
    logger.info("Preparing model configuration")
    
    # Extract block assignments from test series
    block_names = ['Global', 'Consumption', 'Investment', 'External']
    block_records = []
    config_series = []
    
    for series_cfg in test_series:
        if series_cfg['series_id'] in processed_series:
            blocks = series_cfg.get('blocks', {})
            config_series.append({
                'series_id': series_cfg['series_id'],
                'series_name': series_cfg['series_name'],
                'frequency': series_cfg['frequency'],
                'transformation': series_cfg['transformation'],
                'units': series_cfg.get('units'),
                'category': series_cfg.get('category'),
                'blocks': blocks
            })
            
            # Create block assignments
            for block_idx, block_name in enumerate(block_names):
                if blocks.get(block_name, 0) == 1:
                    block_records.append({
                        'series_id': series_cfg['series_id'],
                        'block_name': block_name,
                        'block_index': block_idx
                    })
    
    # Create config_json
    config_json = {
        'block_names': block_names,
        'series': config_series
    }
    
    # Save model config
    config_name = 'test-initial-spec'  # Test config name
    print(f"   Config name: {config_name}")
    print(f"   Block names: {', '.join(block_names)}")
    print(f"   Series count: {len(config_series)}")
    print(f"   Block assignments: {len(block_records)}")
    logger.info(f"Config name: {config_name}")
    logger.info(f"Block names: {block_names}")
    logger.info(f"Series count: {len(config_series)}")
    
    try:
        print("   💾 Saving model configuration to database...")
        config_result = save_model_config(
            config_name=config_name,
            config_json=config_json,
            block_names=block_names,
            description=f"Test model configuration from test_initialization.py",
            country='KR',
            client=client
        )
        config_id = config_result['config_id']
        print(f"   ✅ Saved model configuration: {config_name} (ID: {config_id})")
        
        # Save block assignments
        if block_records:
            print(f"   📊 Saving {len(block_records)} block assignments...")
            from database.operations import TABLES
            
            # Delete existing assignments for this config and insert new ones
            client.table(TABLES['model_block_assignments']).delete().eq('config_id', config_id).execute()
            
            # Insert in batches
            batch_size = 100
            batches = (len(block_records) + batch_size - 1) // batch_size
            for i in range(0, len(block_records), batch_size):
                batch_num = i // batch_size + 1
                batch = [
                    {**rec, 'config_id': config_id}
                    for rec in block_records[i:i + batch_size]
                ]
                client.table(TABLES['model_block_assignments']).insert(batch).execute()
                if batches > 1:
                    print(f"      Inserted batch {batch_num}/{batches}...")
            
            print(f"   ✅ Saved {len(block_records)} block assignments")
        else:
            print("   ⚠️  No block assignments to save")
    except Exception as e:
        logger.error(f"❌ Failed to save model configuration: {e}")
        print(f"   ❌ Failed to save model configuration: {e}")
        # Don't raise - this is test, continue
    
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
    
    if job_id:
        print("   📋 Updating ingestion job status...")
        try:
            update_ingestion_job(
                job_id=job_id,
                status='completed',
                successful_series=stats['successful'],
                failed_series=stats['failed'],
                total_series=stats['total'],
                client=client
            )
            print("   ✅ Ingestion job status updated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to update job status: {e}")
            print(f"   ⚠️  Failed to update job status (continuing)")
    
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
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total series: {stats['total']}")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   ⏭️  Skipped: {stats['skipped']}")
    print(f"   📈 Observations inserted: {stats['observations_inserted']}")
    
    print(f"\n📊 By Frequency:")
    for freq, count in stats['by_frequency'].items():
        if count > 0:
            print(f"   {freq.upper()}: {count}")
    
    print(f"\n📊 By Source:")
    for source, count in stats['by_source'].items():
        if count > 0:
            print(f"   {source}: {count}")
    
    print(f"\n📅 Vintage date: {vintage_date}")
    print(f"📅 Vintage ID: {vintage_id}")
    if job_id:
        print(f"📋 Job ID: {job_id}")
    
    if stats['errors']:
        print(f"\n❌ Errors encountered ({len(stats['errors'])}):")
        for err in stats['errors'][:5]:  # Show first 5 errors
            print(f"   - {err['series_id']}: {err['error'][:80]}")
        if len(stats['errors']) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more errors")
    
    logger.info(f"Processed series: {stats['successful']}/{stats['total']}")
    logger.info(f"Observations inserted: {stats['observations_inserted']}")
    logger.info(f"Vintage date: {vintage_date}")
    logger.info(f"Vintage ID: {vintage_id}")
    
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
