"""Consolidated initialization script for DFM database setup.

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
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
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
from scripts.db_utils import (
    RateLimiter,
    initialize_api_clients,
    ensure_vintage_and_job,
    finalize_ingestion_job,
)
# Import load_config_from_csv inside main to avoid Hydra initialization issues


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
        # For KOSIS: item_code1=objL1 (필수), item_code2=itmId (필수)
        # Different tables require different parameters:
        # - DT_1J22003 (CPI): objL1='ALL', itmId='T'
        # - DT_1JH20201 (Production Index): objL1='0 1', itmId='T1'
        # - DT_1DA7002S (Employment Rate): objL1='ALL', itmId='T90'
        # For BOK: these parameters are ignored
        if source == 'KOSIS':
            # Determine parameters based on table ID
            if 'DT_1JH20201' in api_code:
                # 전산업생산지수: objL1='0 1', itmId='T1'
                item_code1_val = '0 1'
                item_code2_val = 'T1'
            elif 'DT_1DA7002S' in api_code:
                # 고용율: objL1='ALL', itmId='T90'
                item_code1_val = 'ALL'
                item_code2_val = 'T90'
            else:
                # Default (CPI and others): objL1='ALL', itmId='T'
                item_code1_val = 'ALL'
                item_code2_val = 'T'
        else:
            item_code1_val = '?'
            item_code2_val = '?'
        
        data_result = api_client.fetch_statistic_data(
            stat_code=api_code,
            frequency=api_freq,
            start_date=start_date,
            end_date=end_date,
            item_code1=item_code1_val,
            item_code2=item_code2_val,
            item_code3=None,  # Don't pass '?' for optional params
            item_code4=None
        )
        
        # KOSIS returns list directly, BOK returns dict with 'StatisticSearch'
        if source == 'KOSIS':
            # KOSIS response is a list directly (wrapped in {'data': [...]} by client)
            if isinstance(data_result, dict) and 'data' in data_result:
                rows = data_result['data']
            elif isinstance(data_result, list):
                rows = data_result
            else:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()
        else:
            # BOK response format - check for rate limiting errors
            if 'RESULT' in data_result:
                result = data_result['RESULT']
                if isinstance(result, dict) and 'CODE' in result:
                    error_code = result.get('CODE', '')
                    error_msg = result.get('MESSAGE', '')
                    if 'ERROR' in error_code or '602' in error_code:
                        logger.error(f"BOK API rate limit error for {series_id}: {error_msg}")
                        print(f"   ⚠️  BOK API rate limited: {error_msg[:50]}...")
                        raise Exception(f"BOK API rate limit: {error_msg}")
                    else:
                        logger.warning(f"BOK API error for {series_id}: {error_code} - {error_msg}")
                        print(f"   ⚠️  BOK API error: {error_code}")
                        return pd.DataFrame()
            
            if 'StatisticSearch' not in data_result:
                logger.warning(f"No data returned for {series_id}")
                print(f"   ⚠️  No StatisticSearch in response for {series_id}")
                return pd.DataFrame()
            rows = data_result['StatisticSearch'].get('row', [])
        if not rows:
            logger.warning(f"Empty data for {series_id}")
            print(f"   ⚠️  Empty data for {series_id}")
            return pd.DataFrame()
        
        # Parse into DataFrame
        data_list = []
        for row in rows:
            # KOSIS uses PRD_DE and DT, BOK uses TIME and DATA_VALUE
            if source == 'KOSIS':
                time_str = row.get('PRD_DE', '')
                value = row.get('DT')
                # Filter by C1 value based on table:
                # - DT_1J22003 (CPI): C1='T10' (전국/whole country)
                # - DT_1JH20201 (Production Index): C1='0' (전산업생산지수)
                # - DT_1DA7002S (Employment Rate): Filter to overall rate (typically C1='T90' or similar)
                c1 = row.get('C1', '')
                c1_nm = row.get('C1_NM', '')
                
                if 'DT_1J22003' in series_id:
                    # CPI: Use C1='T10' (전국/whole country)
                    if c1 and c1 != 'T10':
                        continue
                elif 'DT_1JH20201' in series_id:
                    # Production Index: Use C1='0' (전산업생산지수)
                    if c1 and c1 != '0':
                        continue
                elif 'DT_1DA7002S' in series_id:
                    # Employment Rate: Filter to overall rate
                    # Based on URL, itmId='T90' suggests overall rate
                    # Filter to records where C1 might indicate overall/total
                    # For now, accept all but could be refined if needed
                    pass  # Accept all for Employment Rate (can filter later if needed)
            else:
                time_str = row.get('TIME', '')
                value = row.get('DATA_VALUE')
            
            if value is None or value == '':
                continue
            
            # Parse date based on frequency
            try:
                if api_freq == 'Q':
                    # Format: 2023Q1 or YYYYMM (for monthly data that's quarterly)
                    if 'Q' in time_str:
                        year, q = time_str.split('Q')
                        month = int(q) * 3
                        date_obj = datetime(int(year), month, 1).date()
                    else:
                        # Try YYYYMM format
                        date_obj = datetime.strptime(time_str[:6], '%Y%m').date()
                elif api_freq == 'M':
                    # Format: YYYYMM (e.g., 202301)
                    date_obj = datetime.strptime(time_str[:6], '%Y%m').date()
                elif api_freq == 'D':
                    # Format: YYYYMMDD (e.g., 20230101)
                    date_obj = datetime.strptime(time_str[:8], '%Y%m%d').date()
                else:
                    # Annual or other
                    date_obj = datetime(int(time_str[:4]), 12, 31).date()
                
                data_list.append({
                    'date': date_obj,
                    'value': float(value),
                    'series_id': series_id
                })
            except Exception as e:
                logger.warning(f"Failed to parse date {time_str} for {series_id}: {e}")
                print(f"   ⚠️  Failed to parse date {time_str}: {e}")
                continue
        
        if not data_list:
            logger.warning(f"No valid data points after parsing for {series_id}")
            print(f"   ⚠️  No valid data points after parsing")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        logger.info(f"Parsed {len(df)} valid observations for {series_id}")
        print(f"   ✅ Parsed {len(df)} valid observations")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {e}")
        print(f"   ❌ Error fetching data: {str(e)[:100]}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def main() -> None:
    """Main initialization workflow."""
    parser = argparse.ArgumentParser(
        description='Initialize DFM database with data from CSV specification'
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
    print("🚀 DFM DATABASE INITIALIZATION")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("DFM Database Initialization")
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
        # Load CSV directly to avoid Hydra initialization issues
        csv_df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded CSV: {len(csv_df)} series")
        logger.info(f"Loaded CSV: {len(csv_df)} series")
        
        # Create a simple config-like object from CSV
        class SimpleSeriesConfig:
            def __init__(self, row):
                self.series_id = row['series_id']
                self.series_name = row['series_name']
                self.frequency = row['frequency']
                self.transformation = row['transformation']
                self.category = row.get('category')
                self.units = row.get('units')
                self.api_code = row.get('api_code')
                self.api_source = row.get('api_source')
                # Block assignments
                self.blocks = {
                    'Global': int(row.get('Global', 0)),
                    'Consumption': int(row.get('Consumption', 0)),
                    'Investment': int(row.get('Investment', 0)),
                    'External': int(row.get('External', 0))
                }
        
        class SimpleModelConfig:
            def __init__(self, df):
                self.series = [SimpleSeriesConfig(row) for _, row in df.iterrows()]
                self.config_name = csv_path.stem.replace('_', '-')
                self.country = 'KR'
                # Detect block names from columns
                block_cols = [col for col in df.columns if col not in 
                             ['series_id', 'series_name', 'frequency', 'transformation', 
                              'category', 'units', 'api_code', 'api_source']]
                self.block_names = block_cols if block_cols else ['Global', 'Consumption', 'Investment', 'External']
        
        model_cfg = SimpleModelConfig(csv_df)
        print(f"   ✅ Parsed model config: {len(model_cfg.series)} series")
        print(f"   📊 Block names: {', '.join(model_cfg.block_names)}")
        logger.info(f"Parsed model config: {len(model_cfg.series)} series")
        logger.info(f"Block names: {model_cfg.block_names}")
    except Exception as e:
        print(f"❌ Failed to load model config: {e}")
        logger.error(f"Failed to load model config from CSV: {e}")
        import traceback
        traceback.print_exc()
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
    logger.info()
    
    # Create vintage and ingestion job
    print("\n📦 Creating vintage and ingestion job...")
    client = get_client()
    vintage_id, job_id = ensure_vintage_and_job(
        vintage_date=vintage_date,
        client=client,
        dry_run=dry_run
    )
    if vintage_id:
        print(f"   ✅ Vintage: {vintage_id}")
    if job_id:
        print(f"   ✅ Ingestion job: {job_id}")
    if not dry_run and not vintage_id:
        vintage_id = None
        job_id = None
        print("   🧪 Dry run: Skipping vintage/job creation")
        logger.info("Dry run: Skipping vintage/job creation")
    
    print()
    logger.info()
    
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
    logger.info()
    
    # Get source IDs
    print("\n📋 Getting source IDs from database...")
    try:
        if not dry_run:
            print("   🔍 Retrieving BOK source ID...")
            bok_source_id = get_source_id('BOK', client=client)
            print(f"   ✅ BOK source_id: {bok_source_id}")
            print("   🔍 Retrieving KOSIS source ID...")
            kosis_source_id = get_source_id('KOSIS', client=client)
            print(f"   ✅ KOSIS source_id: {kosis_source_id}")
        else:
            bok_source_id = None
            kosis_source_id = None
            print("   🧪 Dry run: Skipping source ID retrieval")
    except ValueError as e:
        print(f"❌ Source not found: {e}")
        print("   💡 Run initial setup to create data sources first")
        logger.error(f"Source not found: {e}")
        logger.error("Run initial setup to create data sources first")
        sys.exit(1)
    
    # Process each series
    print("\n" + "=" * 80)
    print(f"🔄 PROCESSING {len(model_cfg.series)} SERIES")
    print("=" * 80)
    logger.info(f"Processing {len(model_cfg.series)} series...")
    print()
    logger.info()
    
    # Rate limiting
    rate_limiter = RateLimiter(bok_delay=0.6, kosis_delay=0.5)
    
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
        
        print(f"\n[{i}/{len(model_cfg.series)}] {series_id}")
        print(f"   Name: {series_name[:70]}...")
        print(f"   Frequency: {frequency}, Transformation: {transformation}")
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
            
            # Get appropriate API client with rate limiting
            api_client = None
            source_id = None
            if api_source == 'BOK' and bok_client:
                api_client = bok_client
                source_id = bok_source_id
                rate_limiter.wait_if_needed(api_source)
            elif api_source == 'KOSIS' and kosis_client:
                api_client = kosis_client
                source_id = kosis_source_id
                rate_limiter.wait_if_needed(api_source)
            else:
                logger.warning(f"  ⚠ No API client available for {api_source}")
                stats['skipped'] += 1
                continue
            
            # Fetch data
            print(f"   🌐 Fetching data from {api_source} API (code: {api_code})...")
            logger.info(f"  Fetching data from {api_source} API (code: {api_code})...")
            df_data = fetch_series_data(
                series_id=series_id,
                api_code=api_code,
                api_client=api_client,
                source=api_source,
                frequency=frequency
            )
            
            if df_data.empty:
                print(f"   ⚠️  No data fetched - skipping")
                logger.warning(f"  ⚠ No data fetched for {series_id}")
                stats['skipped'] += 1
                continue
            
            print(f"   ✅ Fetched {len(df_data)} data points")
            if len(df_data) > 0:
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            
            # Create/update series metadata
            if not dry_run:
                print(f"   💾 Saving series metadata...")
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
                print(f"   ✅ Series metadata saved")
                logger.info(f"  ✓ Updated series metadata")
            else:
                print(f"   🧪 Dry run: Skipping metadata save")
            
            # Add to observations list
            df_data['vintage_id'] = vintage_id
            df_data['job_id'] = job_id
            all_observations.append(df_data)
            
            print(f"   ✅ Series {series_id} processed successfully")
            stats['successful'] += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            logger.error(f"  ❌ Error processing {series_id}: {e}")
            stats['failed'] += 1
            stats['errors'].append(f"{series_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
        logger.info()
    
    # Insert all observations
    print("\n" + "=" * 80)
    print("💾 INSERTING OBSERVATIONS INTO DATABASE")
    print("=" * 80)
    if all_observations and not dry_run:
        print(f"📊 Preparing {len(all_observations)} series for batch insertion...")
        logger.info("Inserting observations into database...")
        df_obs = pd.concat(all_observations, ignore_index=True)
        print(f"   Total observations: {len(df_obs)}")
        print("   💾 Inserting into database...")
        
        result = insert_observations_from_dataframe(
            df=df_obs,
            vintage_id=vintage_id,
            job_id=job_id,
            client=client
        )
        
        print(f"   ✅ Successfully inserted {len(df_obs)} observations")
        logger.info(f"✓ Inserted {len(df_obs)} observations")
    elif all_observations and dry_run:
        df_obs = pd.concat(all_observations, ignore_index=True)
        print(f"🧪 Dry run: Would insert {len(df_obs)} observations")
    else:
        print("⚠️  No observations to insert")
    
    # Save model configuration
    print("\n" + "=" * 80)
    print("💾 SAVING MODEL CONFIGURATION")
    print("=" * 80)
    if not dry_run:
        print("📋 Preparing model configuration...")
        logger.info()
        logger.info("Saving model configuration to database...")
        
        # Use CSV filename as config name (without extension)
        config_name = csv_path.stem.replace('_', '-')  # e.g., 001-initial-spec
        print(f"   Config name: {config_name}")
        
        # Extract block assignments from ModelConfig
        block_names = model_cfg.block_names
        print(f"   Block names: {', '.join(block_names)}")
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
        print("   💾 Saving model configuration to database...")
        config_result = save_model_config(
            config_name=config_name,
            config_json=config_json,
            block_names=block_names,
            description=f"DFM model configuration loaded from CSV: {csv_path.name}",
            country='KR',
            client=client
        )
        
        config_id = config_result['config_id']
        print(f"   ✅ Saved model configuration: {config_name} (ID: {config_id})")
        logger.info(f"✓ Saved model configuration: {config_name} (ID: {config_id})")
        
        # Save block assignments
        if block_records:
            print(f"   📊 Saving {len(block_records)} block assignments...")
            from database.operations import TABLES
            
            # Delete existing assignments and insert new ones
            client.table(TABLES['model_block_assignments']).delete().eq('config_id', config_id).execute()
            
            # Insert in batches
            batch_size = 100
            batches = (len(block_records) + batch_size - 1) // batch_size
            for i in range(0, len(block_records), batch_size):
                batch_num = i // batch_size + 1
                print(f"      Inserting batch {batch_num}/{batches}...")
                batch = [
                    {**rec, 'config_id': config_id}
                    for rec in block_records[i:i + batch_size]
                ]
                client.table(TABLES['model_block_assignments']).insert(batch).execute()
            
            print(f"   ✅ Saved {len(block_records)} block assignments")
            logger.info(f"✓ Saved {len(block_records)} block assignments")
        else:
            print("   ⚠️  No block assignments to save")
    else:
        print("🧪 Dry run: Skipping model configuration save")
    
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
        
        print("   📋 Updating ingestion job status...")
        finalize_ingestion_job(
            job_id=job_id,
            status='completed',
            successful_series=stats['successful'],
            failed_series=stats['failed'],
            total_series=stats['total'],
            client=client
        )
        print("   ✅ Ingestion job status updated")
    else:
        print("🧪 Dry run: Skipping status updates")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ INITIALIZATION SUMMARY")
    print("=" * 80)
    logger.info()
    logger.info("=" * 80)
    logger.info("Initialization Summary")
    logger.info("=" * 80)
    if not dry_run:
        print(f"📦 Vintage ID: {vintage_id}")
        print(f"📋 Job ID: {job_id}")
        logger.info(f"Vintage ID: {vintage_id}")
        logger.info(f"Job ID: {job_id}")
    print(f"\n📊 Statistics:")
    print(f"   Total series: {stats['total']}")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   ⏭️  Skipped: {stats['skipped']}")
    logger.info(f"Total series: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors encountered ({len(stats['errors'])}):")
        logger.warning("Errors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"   - {error}")
            logger.warning(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more errors")
    
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    if stats['failed'] == 0 and stats['successful'] == stats['total']:
        print("\n🎉 All series processed successfully!")
    elif stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} series failed to process")
    
    print("=" * 80)
    logger.info("=" * 80)
    
    if stats['failed'] > 0 and not args.dry_run:
        print("\n❌ Exiting with error code due to failed series")
        sys.exit(1)
    else:
        print("\n✅ Initialization completed successfully!")


if __name__ == '__main__':
    main()

