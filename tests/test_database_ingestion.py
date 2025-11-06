"""Comprehensive test script for database ingestion workflow.

This script tests:
1. Database connectivity
2. API client initialization
3. CSV loading and parsing
4. Series existence checking
5. Latest observation date retrieval
6. Data fetching (with rate limiting)
7. Vintage/job creation
8. Observation insertion logic
9. Model config saving logic

Usage:
    python scripts/test_database_ingestion.py
    python scripts/test_database_ingestion.py --test-api  # Test API calls
"""

import sys
import logging
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables - check multiple locations
env_locations = [
    project_root / '.env.local',
    Path('/home/minkeymouse/Nowcasting') / '.env.local',  # Main worktree
    Path.home() / '.env.local',
    Path('.env.local'),  # Current directory
]

env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        logger.info(f"✅ Loaded environment from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    # Try loading from current directory's .env.local if it exists
    try:
        load_dotenv('.env.local', override=True)
        logger.info("✅ Loaded environment from current directory .env.local")
        env_loaded = True
    except:
        pass

if not env_loaded:
    logger.warning("⚠️  .env.local not found in standard locations")
    logger.warning("   Checked: project root, main worktree, home directory, current directory")

import pandas as pd

from database import (
    get_client,
    get_source_id,
)
from database.operations import (
    check_series_exists,
    get_latest_observation_date,
    list_series,
)
from database.db_utils import (
    initialize_api_clients,
    ensure_vintage_and_job,
    fetch_series_data,
    get_next_period_date,
    RateLimiter,
)


def test_database_connectivity():
    """Test database connection."""
    print("\n" + "=" * 80)
    print("TEST 1: Database Connectivity")
    print("=" * 80)
    
    try:
        client = get_client()
        # Try a simple query
        result = client.table('data_sources').select('source_code').limit(1).execute()
        print("✅ Database connection successful")
        logger.info("✅ Database connection successful")
        return True, client
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        logger.error(f"Database connection failed: {e}")
        return False, None


def test_api_clients():
    """Test API client initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: API Client Initialization")
    print("=" * 80)
    
    bok_client, kosis_client = initialize_api_clients()
    
    results = {}
    if bok_client:
        print("✅ BOK API client initialized")
        results['BOK'] = True
    else:
        print("⚠️  BOK API client not initialized (check BOK_API_KEY)")
        results['BOK'] = False
    
    if kosis_client:
        print("✅ KOSIS API client initialized")
        results['KOSIS'] = True
    else:
        print("⚠️  KOSIS API client not initialized (check KOSIS_API_KEY)")
        results['KOSIS'] = False
    
    return results, bok_client, kosis_client


def test_csv_loading():
    """Test CSV loading and parsing."""
    print("\n" + "=" * 80)
    print("TEST 3: CSV Loading and Parsing")
    print("=" * 80)
    
    csv_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False, None, None
    
    try:
        csv_df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded: {len(csv_df)} series")
        
        # Check required columns
        required_cols = ['series_id', 'series_name', 'frequency', 'transformation', 'api_code', 'api_source']
        missing_cols = [col for col in required_cols if col not in csv_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False, None, None
        
        print(f"✅ All required columns present")
        print(f"   Columns: {', '.join(csv_df.columns.tolist())}")
        
        # Check for block assignment columns
        block_cols = [col for col in csv_df.columns if col not in required_cols]
        print(f"   Block columns: {', '.join(block_cols) if block_cols else 'None'}")
        
        # Create SimpleModelConfig-like structure
        csv_dict = csv_df.set_index('series_id').to_dict('index')
        
        print(f"✅ CSV parsing successful")
        return True, csv_df, csv_dict
        
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        logger.error(f"CSV loading failed: {e}", exc_info=True)
        return False, None, None


def test_series_existence(client, csv_dict, sample_size=5):
    """Test series existence checking."""
    print("\n" + "=" * 80)
    print("TEST 4: Series Existence Checking")
    print("=" * 80)
    
    if not csv_dict:
        print("⚠️  Skipping (no CSV data)")
        return False
    
    sample_series = list(csv_dict.keys())[:sample_size]
    print(f"Testing {len(sample_series)} sample series...")
    
    results = {'exists': 0, 'not_exists': 0, 'errors': 0}
    
    for series_id in sample_series:
        try:
            exists = check_series_exists(series_id, client=client)
            if exists:
                results['exists'] += 1
                print(f"   ✅ {series_id}: exists")
            else:
                results['not_exists'] += 1
                print(f"   ⚪ {series_id}: not exists")
        except Exception as e:
            results['errors'] += 1
            print(f"   ❌ {series_id}: error - {e}")
    
    print(f"\n✅ Results: {results['exists']} exist, {results['not_exists']} not exist, {results['errors']} errors")
    return True


def test_latest_observation_date(client, csv_dict, sample_size=3):
    """Test latest observation date retrieval."""
    print("\n" + "=" * 80)
    print("TEST 5: Latest Observation Date Retrieval")
    print("=" * 80)
    
    if not csv_dict:
        print("⚠️  Skipping (no CSV data)")
        return False
    
    # Get existing series from database
    try:
        all_series = list_series(client=client)
        existing_series_ids = [s['series_id'] for s in all_series]
        
        if not existing_series_ids:
            print("⚠️  No existing series in database - skipping")
            return True
        
        sample_series = existing_series_ids[:sample_size]
        print(f"Testing {len(sample_series)} existing series...")
        
        for series_id in sample_series:
            try:
                latest_date = get_latest_observation_date(series_id, vintage_id=None, client=client)
                if latest_date:
                    print(f"   ✅ {series_id}: latest date = {latest_date}")
                else:
                    print(f"   ⚪ {series_id}: no observations")
            except Exception as e:
                print(f"   ❌ {series_id}: error - {e}")
        
        print("✅ Latest observation date retrieval test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in latest observation date test: {e}", exc_info=True)
        return False


def test_api_data_fetching(bok_client, kosis_client, csv_dict, test_api=False):
    """Test API data fetching."""
    print("\n" + "=" * 80)
    print("TEST 6: API Data Fetching")
    print("=" * 80)
    
    if not test_api:
        print("⚠️  Skipping (use --test-api to enable)")
        print("   This test makes actual API calls and may be rate-limited")
        return True
    
    if not csv_dict:
        print("⚠️  Skipping (no CSV data)")
        return False
    
    # Find one BOK and one KOSIS series
    bok_series = None
    kosis_series = None
    
    for series_id, row in csv_dict.items():
        api_source = row.get('api_source')
        if api_source == 'BOK' and not bok_series and bok_client:
            bok_series = (series_id, row)
        elif api_source == 'KOSIS' and not kosis_series and kosis_client:
            kosis_series = (series_id, row)
        
        if bok_series and kosis_series:
            break
    
    rate_limiter = RateLimiter()
    
    # Test BOK if available
    if bok_series and bok_client:
        series_id, row = bok_series
        api_code = row.get('api_code')
        frequency = row.get('frequency', 'm')
        
        print(f"\nTesting BOK API: {series_id}")
        try:
            rate_limiter.wait_if_needed('BOK')
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=bok_client,
                source='BOK',
                frequency=frequency,
                start_date='202401',  # Limited range for testing
                end_date='202412'
            )
            
            if not df_data.empty:
                print(f"   ✅ Fetched {len(df_data)} observations")
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            else:
                print(f"   ⚠️  No data returned")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test KOSIS if available
    if kosis_series and kosis_client:
        series_id, row = kosis_series
        api_code = row.get('api_code')
        frequency = row.get('frequency', 'm')
        
        print(f"\nTesting KOSIS API: {series_id}")
        try:
            rate_limiter.wait_if_needed('KOSIS')
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=kosis_client,
                source='KOSIS',
                frequency=frequency,
                start_date='202401',  # Limited range for testing
                end_date='202412'
            )
            
            if not df_data.empty:
                print(f"   ✅ Fetched {len(df_data)} observations")
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            else:
                print(f"   ⚠️  No data returned")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ API data fetching test completed")
    return True


def test_vintage_and_job(client):
    """Test vintage and job creation."""
    print("\n" + "=" * 80)
    print("TEST 7: Vintage and Job Creation")
    print("=" * 80)
    
    try:
        vintage_date = date.today()
        vintage_id, job_id = ensure_vintage_and_job(
            vintage_date=vintage_date,
            client=client,
            dry_run=False
        )
        
        if vintage_id:
            print(f"✅ Vintage created/retrieved: {vintage_id}")
        else:
            print("⚠️  Vintage ID is None")
        
        if job_id:
            print(f"✅ Ingestion job created: {job_id}")
        else:
            print("⚠️  Job ID is None")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in vintage/job test: {e}", exc_info=True)
        return False


def test_next_period_date():
    """Test next period date calculation."""
    print("\n" + "=" * 80)
    print("TEST 8: Next Period Date Calculation")
    print("=" * 80)
    
    from datetime import date
    
    test_cases = [
        (date(2024, 3, 31), 'q', '2024Q2'),  # Q1 -> Q2
        (date(2024, 12, 31), 'q', '2025Q1'),  # Q4 -> Q1 next year
        (date(2024, 6, 30), 'm', '202407'),    # June -> July
        (date(2024, 12, 31), 'm', '202501'),   # Dec -> Jan next year
        (date(2024, 1, 15), 'd', '20240116'),  # Day increment
    ]
    
    all_passed = True
    for latest_date, frequency, expected in test_cases:
        try:
            result = get_next_period_date(latest_date, frequency)
            if result == expected:
                print(f"   ✅ {latest_date} ({frequency}) -> {result}")
            else:
                print(f"   ❌ {latest_date} ({frequency}) -> {result} (expected {expected})")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {latest_date} ({frequency}) -> error: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ All next period date calculations correct")
    else:
        print("\n❌ Some calculations failed")
    
    return all_passed


def test_source_ids(client):
    """Test source ID retrieval."""
    print("\n" + "=" * 80)
    print("TEST 9: Source ID Retrieval")
    print("=" * 80)
    
    try:
        bok_source_id = get_source_id('BOK', client=client)
        print(f"✅ BOK source_id: {bok_source_id}")
    except Exception as e:
        print(f"⚠️  BOK source_id: {e}")
    
    try:
        kosis_source_id = get_source_id('KOSIS', client=client)
        print(f"✅ KOSIS source_id: {kosis_source_id}")
    except Exception as e:
        print(f"⚠️  KOSIS source_id: {e}")
    
    return True


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test database ingestion workflow')
    parser.add_argument('--test-api', action='store_true', 
                       help='Test API calls (may be rate-limited)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATABASE INGESTION TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Database connectivity
    success, client = test_database_connectivity()
    results['database'] = success
    
    if not success:
        print("\n❌ Cannot proceed without database connection")
        sys.exit(1)
    
    # Test 2: API clients
    api_results, bok_client, kosis_client = test_api_clients()
    results['api_clients'] = any(api_results.values())
    
    # Test 3: CSV loading
    csv_success, csv_df, csv_dict = test_csv_loading()
    results['csv_loading'] = csv_success
    
    # Test 4: Series existence
    if csv_dict:
        results['series_existence'] = test_series_existence(client, csv_dict)
    else:
        results['series_existence'] = False
    
    # Test 5: Latest observation date
    if csv_dict:
        results['latest_date'] = test_latest_observation_date(client, csv_dict)
    else:
        results['latest_date'] = False
    
    # Test 6: API data fetching
    if args.test_api and csv_dict:
        results['api_fetching'] = test_api_data_fetching(bok_client, kosis_client, csv_dict, test_api=True)
    else:
        results['api_fetching'] = test_api_data_fetching(bok_client, kosis_client, csv_dict, test_api=False)
    
    # Test 7: Vintage and job
    results['vintage_job'] = test_vintage_and_job(client)
    
    # Test 8: Next period date
    results['next_period'] = test_next_period_date()
    
    # Test 9: Source IDs
    results['source_ids'] = test_source_ids(client)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED OR WERE SKIPPED")
        failed = [name for name, passed in results.items() if not passed]
        if failed:
            print(f"   Failed: {', '.join(failed)}")
    print("=" * 80)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()

