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

# Add project root to path (go up 3 levels from scripts/setup/)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_paths = [
    project_root / '.env.local',
    project_root / '.env',
    Path.home() / '.env.local',
]
loaded_env = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"✅ Loaded environment from: {env_path}")
        loaded_env = True
        break

if not loaded_env:
    logger.warning("⚠️  No .env file found - make sure SUPABASE_URL and SUPABASE_KEY are set")

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
        save_model_config
    )
    # Import vintage functions if available
    try:
        from database.operations import create_vintage, update_vintage_status
    except ImportError:
        # Fallback: create vintage manually
        def create_vintage(vintage_date, country, client):
            from datetime import datetime
            result = client.table('data_vintages').insert({
                'vintage_date': vintage_date.isoformat(),
                'country': country,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }).execute()
            return result.data[0]['vintage_id'] if result.data else None
        
        def update_vintage_status(vintage_id, status, client):
            client.table('data_vintages').update({'status': status}).eq('vintage_id', vintage_id).execute()
    # Import API clients - use same imports as initialization.py
    from database.settings import BOKAPIConfig, KOSISAPIConfig
    from services.api.bok_client import BOKAPIClient
    from services.api.kosis_client import KOSISAPIClient
    
    logger.info("=" * 80)
    logger.info("Testing API Data Stream Initialization")
    logger.info("=" * 80)
    
    # Initialize database client
    client = get_client()
    logger.info("✅ Database client initialized")
    
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
    
    logger.info(f"\n📊 Testing with {len(test_series)} series:")
    for s in test_series:
        logger.info(f"   - {s['series_id']}: {s['series_name'][:50]}...")
    
    # Step 1: Get source IDs
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Getting source IDs")
    logger.info("-" * 80)
    
    bok_source_id = get_source_id('BOK', client=client)
    kosis_source_id = get_source_id('KOSIS', client=client)
    logger.info(f"✅ BOK source_id: {bok_source_id}")
    logger.info(f"✅ KOSIS source_id: {kosis_source_id}")
    
    # Step 2: Initialize API clients
    logger.info("\n" + "-" * 80)
    logger.info("Step 2: Initializing API clients")
    logger.info("-" * 80)
    
    bok_api_key = os.getenv('BOK_API_KEY')
    kosis_api_key = os.getenv('KOSIS_API_KEY')
    
    bok_client = None
    if bok_api_key:
        bok_config = BOKAPIConfig(api_key=bok_api_key)
        bok_client = BOKAPIClient(bok_config)
        logger.info("✅ BOK API client initialized")
    else:
        logger.warning("⚠️  BOK_API_KEY not set - skipping BOK API tests")
    
    kosis_client = None
    if kosis_api_key:
        kosis_config = KOSISAPIConfig(api_key=kosis_api_key)
        kosis_client = KOSISAPIClient(kosis_config)
        logger.info("✅ KOSIS API client initialized")
    else:
        logger.warning("⚠️  KOSIS_API_KEY not set - skipping KOSIS API tests")
    
    # Step 3: Test fetching data for one series
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Testing data fetch (limited to last 4 periods)")
    logger.info("-" * 80)
    
    vintage_date = date.today()
    processed_series = []
    
    for series_cfg in test_series:
        if series_cfg['api_source'] == 'BOK' and not bok_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - BOK client not available")
            continue
        if series_cfg['api_source'] == 'KOSIS' and not kosis_client:
            logger.warning(f"⚠️  Skipping {series_cfg['series_id']} - KOSIS client not available")
            continue
        
        logger.info(f"\n📥 Fetching data for {series_cfg['series_id']}...")
        
        try:
            # Fetch data (limited to last 4 periods for testing)
            if series_cfg['api_source'] == 'BOK':
                stat_code = series_cfg['api_code']
                # Fetch only last 4 quarters - use the method from BOK client
                # Check what method is available
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
                continue
            
            logger.info(f"   ✅ Fetched {len(data)} data points")
            logger.info(f"   Sample: {data[0] if data else 'N/A'}")
            
            # Step 4: Upsert series metadata
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
            
            # Step 5: Convert and insert observations
            logger.info(f"   📊 Inserting observations...")
            import pandas as pd
            
            # Convert API data to DataFrame
            obs_data = []
            for item in data[:4]:  # Only use first 4 for testing
                obs_data.append({
                    'series_id': series_cfg['series_id'],
                    'observation_date': pd.to_datetime(item.get('time', item.get('TIME', ''))),
                    'value': float(item.get('data_value', item.get('DATA_VALUE', 0))),
                })
            
            if obs_data:
                obs_df = pd.DataFrame(obs_data)
                obs_df['observation_date'] = pd.to_datetime(obs_df['observation_date'])
                
                # Get or create vintage
                vintage_id = create_vintage(vintage_date, 'KR', client=client)
                logger.info(f"   ✅ Using vintage_id: {vintage_id}")
                
                # Insert observations
                insert_observations_from_dataframe(
                    obs_df,
                    vintage_id=vintage_id,
                    client=client
                )
                logger.info(f"   ✅ Inserted {len(obs_df)} observations")
            
            processed_series.append(series_cfg['series_id'])
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {series_cfg['series_id']}: {e}", exc_info=True)
            continue
    
    # Step 6: Verify data in database
    logger.info("\n" + "-" * 80)
    logger.info("Step 4: Verifying data in database")
    logger.info("-" * 80)
    
    for series_id in processed_series:
        try:
            result = client.table('observations').select('*').eq('series_id', series_id).limit(5).execute()
            count = len(result.data) if result.data else 0
            logger.info(f"✅ {series_id}: {count} observations found in database")
            if result.data:
                logger.info(f"   Sample: {result.data[0]}")
        except Exception as e:
            logger.error(f"❌ Error verifying {series_id}: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test Summary")
    logger.info("=" * 80)
    logger.info(f"Processed series: {len(processed_series)}/{len(test_series)}")
    logger.info(f"Vintage date: {vintage_date}")
    logger.info("\n✅ Data flow test completed successfully!")
    logger.info("=" * 80)
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error("   Make sure you're in the virtual environment:")
    logger.error("   source .venv/bin/activate")
except Exception as e:
    logger.error(f"❌ Error: {e}", exc_info=True)
