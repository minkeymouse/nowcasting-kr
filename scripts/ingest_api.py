"""Main entry point for data ingestion (GitHub Actions).

This script reads from src/spec/001_initial_spec.csv and updates the database:
- For new series: fetches full history and inserts series metadata + observations
- For existing series: fetches incremental data (from latest observation date) and inserts observations only

Usage:
    python scripts/ingest_api.py
"""

import sys
import logging
import os
from pathlib import Path
from datetime import date
from typing import Dict, Any, Optional, Set

# Optional dotenv import (for local development only)
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    load_dotenv = None

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (script is in scripts/ directory)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (local development only)
# In GitHub Actions, environment variables come from secrets
if HAS_DOTENV and not os.getenv('GITHUB_ACTIONS'):
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
elif os.getenv('GITHUB_ACTIONS'):
    logger.info("Running in GitHub Actions - using environment variables from secrets")
    env_loaded = True  # In GitHub Actions, we use secrets, so consider it "loaded"
elif not HAS_DOTENV:
    logger.info("dotenv not available - using environment variables from system")
    env_loaded = True  # If no dotenv, we rely on system env vars

if not env_loaded and HAS_DOTENV and not os.getenv('GITHUB_ACTIONS'):
    logger.warning("⚠️  .env.local not found in standard locations")
    logger.warning("   Checked: project root, main worktree, home directory, current directory")
    logger.warning("   Will use environment variables from system")

import pandas as pd

from database import (
    get_client,
    upsert_series,
    get_series,
    insert_observations_from_dataframe,
    save_model_config,
    update_vintage_status,
)
from database.operations import (
    get_latest_observation_date,
    check_series_exists,
)
from database.models import SeriesModel
from database.operations import TABLES
from database.db_utils import (
    RateLimiter,
    initialize_api_clients,
    fetch_series_data,
    get_next_period_date,
)
from database import ensure_vintage_and_job, finalize_ingestion_job, delete_old_vintages


def main() -> None:
    """Main ingestion workflow."""
    
    print("=" * 80)
    print("🔄 MACROECONOMIC FORECASTING DATABASE INGESTION")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("Macroeconomic Forecasting Database Ingestion")
    logger.info("=" * 80)
    
    # Load CSV specification from database storage bucket (primary source)
    # Fallback to local file only if database is unavailable
    csv_df = None
    spec_source = None
    spec_filename = None
    
    # Priority 1: Load from database storage bucket
    try:
        # Import only what we need, handle missing dependencies gracefully
        try:
            from adapters.adapter_database import download_spec_csv_from_storage, get_latest_spec_csv_filename
        except ImportError as import_err:
            # If adapter import fails, try importing database functions directly
            logger.warning(f"Could not import adapter functions: {import_err}")
            raise ImportError("Database adapter not available") from import_err
        
        try:
            from scripts.utils import get_db_client
        except ImportError:
            # Fallback: try importing database client directly
            try:
                from database import get_client as get_db_client
            except ImportError:
                raise ImportError("Database client not available")
        
        logger.info("Loading spec CSV from database storage bucket...")
        client = get_db_client()
        spec_filename = get_latest_spec_csv_filename("spec", client)
        
        if spec_filename:
            logger.info(f"Found spec file in storage: {spec_filename}")
            csv_content = download_spec_csv_from_storage(spec_filename, "spec", client)
            
            if csv_content:
                import io
                csv_df = pd.read_csv(io.BytesIO(csv_content))
                spec_source = "database_storage"
                print(f"\n📄 CSV file: {spec_filename} (from database storage)")
                logger.info(f"✅ Loaded CSV specification from database storage: {spec_filename} ({len(csv_df)} series)")
            else:
                logger.warning(f"Spec file {spec_filename} found in storage but download returned empty content")
        else:
            logger.warning("No spec CSV files found in database storage bucket 'spec'")
            
    except ImportError as e:
        logger.warning(f"Database adapter not available: {e}")
        logger.info("Will try local file fallback...")
    except Exception as e:
        logger.warning(f"Failed to load spec from database storage: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Priority 2: Fallback to local file (for local development or if DB unavailable)
    if csv_df is None or csv_df.empty:
        csv_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
        if csv_path.exists():
            try:
                csv_df = pd.read_csv(csv_path)
                spec_source = "local_file"
                print(f"\n📄 CSV file: {csv_path} (from local file)")
                logger.info(f"✅ Loaded CSV specification from local file: {len(csv_df)} series")
                logger.warning("⚠️  Using local spec file - database storage should be the primary source")
            except Exception as e:
                logger.error(f"Failed to load local spec file: {e}")
                csv_df = None
    
    # Validate loaded CSV
    if csv_df is None or csv_df.empty:
        error_msg = (
            f"\n❌ ERROR: Could not load CSV specification file\n"
            f"   Tried sources:\n"
            f"   1. Database storage bucket 'spec' (primary)\n"
            f"   2. Local file: {project_root / 'src' / 'spec' / '001_initial_spec.csv'}\n\n"
            f"   Solutions:\n"
            f"   - Upload spec CSV to database storage bucket 'spec'\n"
            f"   - Or ensure local spec file exists at: src/spec/001_initial_spec.csv\n"
        )
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    
    # Validate CSV structure
    required_columns = ['api_source', 'data_code', 'item_id', 'series_name']
    missing_columns = [col for col in required_columns if col not in csv_df.columns]
    if missing_columns:
        error_msg = (
            f"\n❌ ERROR: CSV specification missing required columns: {missing_columns}\n"
            f"   Required columns: {required_columns}\n"
            f"   Found columns: {list(csv_df.columns)}\n"
        )
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    
    if len(csv_df) == 0:
        error_msg = "\n❌ ERROR: CSV specification file is empty\n"
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    
    # Generate series_id from data_code, item_id, and api_source
    # Note: CSV 'id' column is just an index, not the actual series_id
    # The actual series_id is generated using generate_series_id() for consistency
    from database.operations import generate_series_id
    csv_df['series_id'] = csv_df.apply(
        lambda row: generate_series_id(
            row.get('api_source', ''),
            row.get('data_code', ''),
            row.get('item_id', '')
        ),
        axis=1
    )
    logger.info("Generated series_id from data_code, item_id, and api_source")
    
    print(f"   ✅ Loaded CSV: {len(csv_df)} series")
    logger.info(f"Loaded CSV: {len(csv_df)} series")
    
    # Create a simple config-like object from CSV
    class SimpleSeriesConfig:
        def __init__(self, row):
            # series_id should already be generated from CSV loading
            self.series_id = row.get('series_id')
            if not self.series_id:
                raise ValueError(f"Row missing 'series_id' column: {row.to_dict()}")
            # Ensure it's a string
            self.series_id = str(self.series_id)
            self.series_name = row['series_name']
            self.frequency = row['frequency']
            self.transformation = row['transformation']
            self.category = row.get('category')
            self.units = row.get('units')
            self.api_code = row.get('api_code')
            self.api_source = row.get('api_source')
            # Block assignments - handle both 'Block_X' and 'X' formats
            self.blocks = {}
            # Try Block_Global, Block_Consumption, etc. first
            for block_name in ['Global', 'Consumption', 'Investment', 'External']:
                block_value = row.get(f'Block_{block_name}', row.get(block_name, 0))
                self.blocks[block_name] = int(block_value) if block_value not in [None, ''] else 0
    
    class SimpleModelConfig:
        def __init__(self, df):
            self.series = [SimpleSeriesConfig(row) for _, row in df.iterrows()]
            self.config_name = csv_path.stem.replace('_', '-')
            self.country = 'KR'
            # Detect block names from columns (only Block_X columns)
            block_cols = [col.replace('Block_', '') for col in df.columns if col.startswith('Block_')]
            # Fallback to default if no Block_X columns found
            if not block_cols:
                block_cols = ['Global', 'Consumption', 'Investment', 'External']
            self.block_names = block_cols
    
    model_cfg = SimpleModelConfig(csv_df)
    # Ensure series_id column exists for indexing
    if 'series_id' not in csv_df.columns and 'id' in csv_df.columns:
        csv_df['series_id'] = csv_df['id'].astype(str)
    csv_dict = csv_df.set_index('series_id').to_dict('index')
    
    print(f"   ✅ Parsed model config: {len(model_cfg.series)} series")
    print(f"   📊 Block names: {', '.join(model_cfg.block_names)}")
    logger.info(f"Parsed model config: {len(model_cfg.series)} series")
    logger.info(f"Block names: {model_cfg.block_names}")
    
    # Initialize database client
    print("\n📡 Connecting to database...")
    client = get_client()
    logger.info("✅ Database client initialized")
    print("✅ Database connection established")
    
    # Initialize API clients
    print("\n🔧 Initializing API clients...")
    bok_client, kosis_client = initialize_api_clients()
    
    if not bok_client and not kosis_client:
        logger.error("No API clients available. Check API keys.")
        print("❌ Error: No API clients available")
        sys.exit(1)
    
    # Get source IDs
    # data_sources removed, source_id no longer needed
    bok_source_id = 'BOK' if bok_client else None
    kosis_source_id = 'KOSIS' if kosis_client else None
    
    print(f"   ✅ BOK source_id: {bok_source_id}")
    print(f"   ✅ KOSIS source_id: {kosis_source_id}")
    
    # Create vintage (job tracking is now integrated into data_vintages)
    vintage_date = date.today()
    print(f"\n📦 Creating vintage for {vintage_date}...")
    vintage_id = ensure_vintage_and_job(
        vintage_date=vintage_date,
        client=client,
        dry_run=False
    )
    
    if vintage_id:
        print(f"   ✅ Vintage: {vintage_id}")
    else:
        logger.error("Failed to create or retrieve vintage")
        print("❌ Error: Failed to create vintage")
        sys.exit(1)
    
    # job_id is no longer separate - tracking is in data_vintages table
    job_id = None
    
    # Rate limiting
    rate_limiter = RateLimiter(bok_delay=0.6, kosis_delay=0.5)
    
    all_observations = []
    successfully_processed_series = set()  # Track series that were successfully saved
    stats = {
        'total': len(model_cfg.series),
        'new_series': 0,
        'existing_series': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    print("\n" + "=" * 80)
    print(f"🔄 PROCESSING {len(model_cfg.series)} SERIES")
    print("=" * 80)
    logger.info(f"Processing {len(model_cfg.series)} series...")
    
    # Process each series
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
            # Check if series exists
            series_exists = check_series_exists(series_id, client)
            
            # Get api_code and api_source from CSV
            csv_row = csv_dict.get(series_id)
            
            if not csv_row:
                logger.warning(f"  ⚠ Series {series_id} not found in CSV")
                stats['skipped'] += 1
                continue
            
            # CSV has data_code and item_id, not api_code
            # api_code is derived from data_code and item_id
            data_code = csv_row.get('data_code')
            item_id = csv_row.get('item_id')
            api_source = csv_row.get('api_source')
            
            if not data_code or not item_id or not api_source:
                logger.warning(f"  ⚠ Missing data_code, item_id, or api_source for {series_id}")
                logger.warning(f"     data_code={data_code}, item_id={item_id}, api_source={api_source}")
                stats['skipped'] += 1
                continue
            
            # For API calls, we need api_code which is typically data_code
            # But some APIs use item_id, so we'll pass both
            api_code = data_code  # Use data_code as api_code
            
            # Get appropriate API client
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
            
            # Determine start_date based on whether series exists
            start_date = None
            end_date = None
            
            if series_exists:
                # Incremental update: fetch from latest observation date forward
                latest_date = get_latest_observation_date(series_id, vintage_id=None, client=client)
                if latest_date:
                    start_date = get_next_period_date(latest_date, frequency)
                    print(f"   📅 Latest observation in DB: {latest_date}, fetching new data from {start_date}...")
                    logger.info(f"  Latest observation: {latest_date}, fetching from {start_date}")
                else:
                    # No observations yet - fetch full history
                    start_date = None
                    print(f"   📅 No observations found - fetching full history")
                    logger.info(f"  No observations found - fetching full history")
            else:
                # New series - fetch full history
                stats['new_series'] += 1
                print(f"   ✨ New series - fetching full history")
                logger.info(f"  New series - fetching full history")
            
            # Fetch data
            print(f"   🌐 Fetching data from {api_source} API (code: {api_code})...")
            logger.info(f"  Fetching data from {api_source} API (code: {api_code})...")
            
            try:
                df_data = fetch_series_data(
                    series_id=series_id,
                    api_code=api_code,
                    api_client=api_client,
                    source=api_source,
                    frequency=frequency,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as api_error:
                # Handle API-specific errors (wrong API code, invalid parameters, etc.)
                error_msg = str(api_error)
                if 'rate limit' in error_msg.lower() or '602' in error_msg:
                    logger.error(f"  ❌ {series_id}: API rate limit error - {error_msg}")
                    print(f"   ❌ API rate limit error")
                    print(f"      → Please wait and retry later")
                    stats['failed'] += 1
                    stats['errors'].append(f"{series_id}: Rate limit error - {error_msg[:100]}")
                elif 'invalid' in error_msg.lower() or 'not found' in error_msg.lower() or 'error' in error_msg.lower():
                    logger.error(f"  ❌ {series_id}: Invalid API code or parameters - {error_msg}")
                    print(f"   ❌ Invalid API code or parameters")
                    print(f"      → Check spec file: api_source={api_source}, data_code={data_code}, item_id={item_id}")
                    print(f"      → Error: {error_msg[:200]}")
                    stats['failed'] += 1
                    stats['errors'].append(f"{series_id}: Invalid API code - {error_msg[:100]}")
                else:
                    logger.error(f"  ❌ {series_id}: API error - {error_msg}")
                    print(f"   ❌ API error: {error_msg[:200]}")
                    stats['failed'] += 1
                    stats['errors'].append(f"{series_id}: {error_msg[:100]}")
                continue
            
            if df_data is None or df_data.empty:
                # Determine reason for no data
                if series_exists:
                    reason = "No new data available (already up-to-date or future data not yet published)"
                    logger.info(f"  ℹ️  {series_id}: {reason}")
                    print(f"   ℹ️  No new data available")
                    print(f"      → Series is already up-to-date or future data is not yet published by API")
                else:
                    reason = "No data available from API"
                    logger.warning(f"  ⚠ {series_id}: {reason}")
                    print(f"   ⚠️  No data available from API")
                    print(f"      → API returned no data for this series")
                    print(f"      → This may indicate wrong API code in spec file")
                stats['skipped'] += 1
                continue
            
            print(f"   ✅ Fetched {len(df_data)} data points")
            if len(df_data) > 0:
                print(f"      Date range: {df_data['date'].min()} to {df_data['date'].max()}")
            logger.info(f"  ✓ Fetched {len(df_data)} data points")
            
            # Insert/update series metadata (only for new series)
            if not series_exists:
                print(f"   💾 Saving series metadata...")
                
                # Get is_kpi from CSV if available
                is_kpi = csv_row.get('is_kpi', False)
                if isinstance(is_kpi, str):
                    is_kpi = is_kpi.lower() in ('true', '1', 'yes', 'y')
                elif isinstance(is_kpi, (int, float)):
                    is_kpi = bool(is_kpi)
                
                series_model = SeriesModel(
                    series_id=series_id,
                    series_name=series_name,
                    frequency=frequency,
                    transformation=transformation,
                    units=units,
                    category=category,
                    api_source=api_source,
                    data_code=data_code,
                    item_id=item_id,
                    is_active=True,
                    is_kpi=is_kpi
                )
                
                # Workaround for trigger issue - use insert for new, skip for existing
                existing_series = get_series(series_id, client=client)
                if not existing_series:
                    # New series - use direct insert to avoid trigger issue
                    data = series_model.model_dump(exclude_none=True)
                    data.pop('updated_at', None)
                    data.pop('created_at', None)
                    client.table('series').insert(data).execute()
                    print(f"   ✅ Series metadata saved (new series, is_kpi={is_kpi})")
                    logger.info(f"  ✓ Inserted series metadata for new series (is_kpi={is_kpi})")
            
            # Add to observations list
            df_data['vintage_id'] = vintage_id
            # job_id is no longer needed - tracking is in data_vintages table
            all_observations.append(df_data)
            
            successfully_processed_series.add(series_id)  # Track successful series
            if series_exists:
                stats['existing_series'] += 1
                print(f"   ✅ Series {series_id} updated successfully (incremental update)")
            else:
                stats['new_series'] += 1
                print(f"   ✅ Series {series_id} processed successfully (new series)")
            stats['successful'] += 1
            logger.info(f"  ✓ Series {series_id} processed successfully")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            logger.error(f"  ❌ Error processing {series_id}: {e}", exc_info=True)
            stats['failed'] += 1
            stats['errors'].append(f"{series_id}: {str(e)[:100]}")
        
        print()
    
    # Insert all observations
    print("\n" + "=" * 80)
    print("💾 INSERTING OBSERVATIONS INTO DATABASE")
    print("=" * 80)
    if all_observations:
        print(f"📊 Preparing {len(all_observations)} series for batch insertion...")
        logger.info("Inserting observations into database...")
        df_obs = pd.concat(all_observations, ignore_index=True)
        
        # Deduplicate observations
        df_obs = df_obs.drop_duplicates(subset=['series_id', 'vintage_id', 'date'], keep='first')
        print(f"   Total observations: {len(df_obs)} (after deduplication)")
        print("   💾 Inserting into database...")
        
        result = insert_observations_from_dataframe(
            df=df_obs,
            vintage_id=vintage_id,
            client=client
        )
        
        print(f"   ✅ Successfully inserted {len(df_obs)} observations")
        logger.info(f"✓ Inserted {len(df_obs)} observations")
    else:
        print("⚠️  No observations to insert")
    
    # Save model configuration
    print("\n" + "=" * 80)
    print("💾 SAVING MODEL CONFIGURATION")
    print("=" * 80)
    print("📋 Preparing model configuration...")
    logger.info("Saving model configuration to database...")
    
    # Use CSV filename as config name
    config_name = model_cfg.config_name
    print(f"   Config name: {config_name}")
    
    # Extract block assignments from ModelConfig
    block_names = model_cfg.block_names
    print(f"   Block names: {', '.join(block_names)}")
    block_records = []
    
    # Create series_order mapping (CSV row order)
    for series_order, series_cfg in enumerate(model_cfg.series):
        # Only create block assignments for series that were successfully saved
        if series_cfg.series_id not in successfully_processed_series:
            logger.debug(f"Skipping block assignment for {series_cfg.series_id} (not in database)")
            continue
        
        blocks = getattr(series_cfg, 'blocks', None)
        if blocks:
            for block_name in block_names:
                # blocks is a dict with block_name as key
                if blocks.get(block_name, 0) == 1:
                    block_records.append({
                        'config_name': config_name,
                        'series_id': series_cfg.series_id,
                        'block_name': block_name,
                        'series_order': series_order
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
            if s.series_id in successfully_processed_series
        ]
    }
    
    # Save model config
    print("   💾 Saving model configuration to database...")
    config_result = save_model_config(
        config_name=config_name,
        config_json=config_json,
        block_names=block_names,
        description=f"Macroeconomic forecasting model configuration from {csv_path.name}",
        country='KR',
        client=client
    )
    
    if config_result:
        config_id = config_result.get('config_id')  # May not exist if using blocks table
        if config_id:
            print(f"   ✅ Saved model configuration: {config_name} (ID: {config_id})")
            logger.info(f"✓ Saved model configuration: {config_name} (ID: {config_id})")
        else:
            print(f"   ✅ Saved model configuration: {config_name}")
            logger.info(f"✓ Saved model configuration: {config_name}")
    else:
        print(f"   ⚠️  Model config table not available, using blocks table only")
        logger.info(f"⚠ Model config table not available, using blocks table only")
    
    # Save block assignments to blocks table
    # Delete existing blocks for this config_name first, then insert new ones
    if block_records:
        print(f"   📊 Saving {len(block_records)} block assignments to blocks table...")
        logger.info(f"Saving {len(block_records)} block assignments for config {config_name}")
        
        # Delete existing blocks for this config_name and insert new ones
        client.table('blocks').delete().eq('config_name', config_name).execute()
        
        # Insert using batch_insert helper
        from database.helpers import batch_insert
        total_inserted = batch_insert(
            client=client,
            table_name='blocks',
            records=block_records,
            batch_size=100
        )
        
        print(f"   ✅ Saved {total_inserted} block assignments")
        logger.info(f"✓ Saved {total_inserted} block assignments")
    else:
        print("   ⚠️  No block assignments to save (no series were successfully processed)")
        logger.warning("No block assignments to save - no series were successfully processed")
    
    # Update status with full statistics
    print("\n" + "=" * 80)
    print("📝 UPDATING STATUS")
    print("=" * 80)
    print("   📅 Finalizing ingestion job with statistics...")
    
    # Get GitHub run ID if available
    github_run_id = os.getenv('GITHUB_RUN_ID')
    github_workflow_run_url = os.getenv('GITHUB_SERVER_URL') and os.getenv('GITHUB_REPOSITORY') and os.getenv('GITHUB_RUN_ID') and \
        f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
    
    # Prepare error message if there are failures
    error_message = None
    if stats['errors']:
        error_message = f"{len(stats['errors'])} series failed: " + "; ".join(stats['errors'][:3])
        if len(stats['errors']) > 3:
            error_message += f" ... and {len(stats['errors']) - 3} more"
    
    # Finalize ingestion job with full statistics
    finalize_ingestion_job(
        vintage_id=vintage_id,
        status='completed' if stats['failed'] == 0 else 'partial',
        total_series=stats['total'],
        successful_series=stats['successful'],
        failed_series=stats['failed'],
        error_message=error_message,
        client=client
    )
    
    # Update GitHub workflow URL if available
    if github_workflow_run_url:
        try:
            client.table('data_vintages').update({
                'github_workflow_run_url': github_workflow_run_url
            }).eq('vintage_id', vintage_id).execute()
        except Exception as e:
            logger.debug(f"Could not update github_workflow_run_url: {e}")
    
    print("   ✅ Vintage status updated with full statistics")
    print(f"      Total: {stats['total']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"   Total series: {stats['total']}")
    print(f"   ✨ New series: {stats['new_series']}")
    print(f"   🔄 Existing series: {stats['existing_series']}")
    print(f"   ✅ Successful: {stats['successful']} (new data fetched and inserted)")
    print(f"   ❌ Failed: {stats['failed']} (errors during processing)")
    print(f"   ⏭️  Skipped: {stats['skipped']} (no new data - already up-to-date or future data not yet published)")
    
    # Calculate success rate
    if stats['total'] > 0:
        success_rate = (stats['successful'] / stats['total']) * 100
        print(f"\n   📈 Success rate: {success_rate:.1f}%")
        if success_rate < 100:
            print(f"   ℹ️  Note: Skipped series are normal for incremental updates.")
            print(f"      They indicate series are already up-to-date or future data is not yet published.")
    
    logger.info(f"Total series: {stats['total']}, Successful: {stats['successful']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    logger.info(f"New series: {stats['new_series']}")
    logger.info(f"Existing series: {stats['existing_series']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:
            print(f"   - {error}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more errors")
    
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    if stats['failed'] == 0 and stats['successful'] == stats['total']:
        print("\n🎉 All series processed successfully!")
    elif stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} series failed to process")
    
    print("=" * 80)
    
    # Cleanup old vintages (6 months or older)
    print("\n🧹 Cleaning up old vintages (6 months or older)...")
    try:
        cleanup_result = delete_old_vintages(months=6, dry_run=False, client=client)
        if cleanup_result['deleted_count'] > 0:
            print(f"   ✅ Deleted {cleanup_result['deleted_count']} old vintages (cutoff: {cleanup_result['cutoff_date']})")
            logger.info(f"Cleaned up {cleanup_result['deleted_count']} old vintages")
        else:
            print(f"   ℹ️  No old vintages to delete (cutoff: {cleanup_result['cutoff_date']})")
    except Exception as e:
        logger.warning(f"Failed to cleanup old vintages: {e}", exc_info=True)
        print(f"   ⚠️  Cleanup failed: {e}")
    
    print("=" * 80)
    
    # Exit with error code if failures
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
