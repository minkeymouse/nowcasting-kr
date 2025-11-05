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
        save_model_config, create_vintage, update_vintage_status
    )
    from database.settings import BOKAPIConfig, KOSISAPIConfig
    from services.api.bok_client import BOKAPIClient
    from services.api.kosis_client import KOSISAPIClient
    
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
    
    # Test data: Use only 2-3 series from the spec
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
    print(f"📅 Vintage date: {vintage_date}")
    processed_series = []
    
    print(f"\n🔄 Processing {len(test_series)} series...")
    for idx, series_cfg in enumerate(test_series, 1):
        if series_cfg['api_source'] == 'BOK' and not bok_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - BOK client not available")
            continue
        if series_cfg['api_source'] == 'KOSIS' and not kosis_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - KOSIS client not available")
            continue
        
        print(f"\n[{idx}/{len(test_series)}] Processing: {series_cfg['series_id']}")
        print(f"   Name: {series_cfg['series_name'][:60]}...")
        logger.info(f"\n📥 Fetching data for {series_cfg['series_id']}...")
        
        try:
            # Fetch data (limited to last 4 periods for testing)
            print(f"   🌐 Connecting to {series_cfg['api_source']} API...")
            if series_cfg['api_source'] == 'BOK':
                stat_code = series_cfg['api_code']
                print(f"   📊 Fetching data (stat_code: {stat_code}, period: 2023Q1-2024Q4)...")
                # Fetch only last 4 quarters - check available methods
                if hasattr(bok_client, 'fetch_statistic_data'):
                    data = bok_client.fetch_statistic_data(
                        stat_code=stat_code,
                        item_code1='*',
                        start_date='2023Q1',  # Recent data only
                        end_date='2024Q4',
                        cycle='Q'
                    )
                elif hasattr(bok_client, 'fetch_data'):
                    data = bok_client.fetch_data(
                        stat_code=stat_code,
                        item_code='*',
                        start_date='2023Q1',
                        end_date='2024Q4'
                    )
                else:
                    logger.warning(f"   ⚠️  Unknown BOK client method - skipping")
                    continue
            else:
                logger.warning(f"⚠️  KOSIS fetch not implemented in test - skipping")
                continue
            
            if not data or len(data) == 0:
                logger.warning(f"⚠️  No data returned for {series_cfg['series_id']}")
                print(f"   ⚠️  No data returned - skipping")
                continue
            
            print(f"   ✅ Fetched {len(data)} data points")
            logger.info(f"   ✅ Fetched {len(data)} data points")
            if data:
                sample = data[0] if isinstance(data[0], dict) else 'N/A'
                logger.info(f"   Sample: {sample}")
                print(f"   📋 Sample data: {sample}")
            
            # Step 4: Upsert series metadata
            print(f"   💾 Saving series metadata to database...")
            logger.info(f"   📝 Upserting series metadata...")
            series_metadata = {
                'series_id': series_cfg['series_id'],
                'series_name': series_cfg['series_name'],
                'source_id': bok_source_id if series_cfg['api_source'] == 'BOK' else kosis_source_id,
                'frequency': series_cfg['frequency'],
                'transformation': series_cfg['transformation'],
                'category': series_cfg['category'],
                'units': series_cfg['units'],
                'is_active': True,
            }
            upsert_series(series_metadata, client=client)
            logger.info(f"   ✅ Series metadata upserted")
            print(f"   ✅ Series metadata saved")
            
            # Step 5: Convert and insert observations
            print(f"   📊 Preparing observations for insertion...")
            logger.info(f"   📊 Inserting observations...")
            import pandas as pd
            
            # Convert API data to DataFrame
            obs_data = []
            for item in data[:4]:  # Only use first 4 for testing
                time_val = item.get('time', item.get('TIME', ''))
                value = item.get('data_value', item.get('DATA_VALUE', 0))
                if time_val and value:
                    obs_data.append({
                        'series_id': series_cfg['series_id'],
                        'observation_date': pd.to_datetime(time_val),
                        'value': float(value),
                    })
            
            if obs_data:
                obs_df = pd.DataFrame(obs_data)
                obs_df['observation_date'] = pd.to_datetime(obs_df['observation_date'])
                
                # Get or create vintage
                print(f"   📅 Creating/getting vintage...")
                vintage = create_vintage(vintage_date, 'KR', client=client)
                vintage_id = vintage['vintage_id'] if isinstance(vintage, dict) else vintage
                logger.info(f"   ✅ Using vintage_id: {vintage_id}")
                print(f"   ✅ Vintage ID: {vintage_id}")
                
                # Insert observations
                print(f"   💾 Inserting {len(obs_df)} observations into database...")
                insert_observations_from_dataframe(
                    obs_df,
                    vintage_id=vintage_id,
                    client=client
                )
                logger.info(f"   ✅ Inserted {len(obs_df)} observations")
                print(f"   ✅ Inserted {len(obs_df)} observations")
            
            processed_series.append(series_cfg['series_id'])
            print(f"   ✅ Successfully processed {series_cfg['series_id']}")
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {series_cfg['series_id']}: {e}", exc_info=True)
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Step 6: Verify data in database
    print("\n" + "=" * 80)
    print("STEP 4: Verifying Data in Database")
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
            logger.info(f"✅ {series_id}: {count} observations found in database")
            print(f"   ✅ {series_id}: {count} observations found")
            if result.data:
                logger.info(f"   Sample: {result.data[0]}")
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
    print(f"📊 Processed series: {len(processed_series)}/{len(test_series)}")
    print(f"📅 Vintage date: {vintage_date}")
    logger.info(f"Processed series: {len(processed_series)}/{len(test_series)}")
    logger.info(f"Vintage date: {vintage_date}")
    if len(processed_series) == len(test_series):
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {len(test_series) - len(processed_series)} series failed")
    logger.info("\n✅ Data flow test completed successfully!")
    logger.info("=" * 80)
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
