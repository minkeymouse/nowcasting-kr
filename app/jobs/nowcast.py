"""Job 3: Generate nowcasts/forecasts and update database.

This job:
- Queries database for latest data vintage
- Downloads latest model weights from Supabase storage bucket
- Loads model configuration (from DB storage CSV or Hydra YAML)
- Generates nowcasts (current period) and forecasts (future periods)
- Saves nowcasts and forecasts to database with news decomposition

Usage:
    python -m app.jobs.nowcast --config-name=test series=test_series
"""

import sys
import logging
import pickle
import os
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np

# Add project root to path (script is in app/jobs/ directory)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Optional dotenv import (for local development only)
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    load_dotenv = None

from dfm_python import load_data, dfm, update_nowcast
from app.adapters.adapter_database import load_data_from_db, save_nowcast_to_db
from app.utils import (
    load_model_config_with_hydra_fallback,
    get_db_client,
    get_latest_vintage_with_fallback,
    extract_hydra_config_dicts
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local file (local development only)
# In GitHub Actions, environment variables come from secrets
env_loaded = False

if HAS_DOTENV and not os.getenv('GITHUB_ACTIONS'):
    # Use .env.local for local development
    env_locations = [
        project_root / '.env.local',
        Path('.env.local'),  # Current directory
    ]
    
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"✅ Loaded environment from: {env_path}")
            env_loaded = True
            break

if not env_loaded and os.getenv('GITHUB_ACTIONS'):
    logger.info("Running in GitHub Actions - using environment variables from secrets")
    env_loaded = True  # In GitHub Actions, we use secrets, so consider it "loaded"
elif not env_loaded and not HAS_DOTENV:
    logger.info("dotenv not available - using environment variables from system")
    env_loaded = True  # If no dotenv, we rely on system env vars


@hydra.main(version_base=None, config_path="../../app/config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run nowcasting with Hydra configuration (MATLAB-compatible).
    
    This script performs nowcasting (current period forecast comparison):
    1. Loads latest model spec from database (or CSV/YAML fallback)
    2. Loads data from database or files
    3. Estimates DFM model (if not already trained)
    4. Compares old vs new vintage forecasts for current period
    5. Decomposes changes into news components (MATLAB news_dfm functionality)
    
    This matches MATLAB functionality - no forward nowcasting, only nowcasting.
    
    Usage:
        python nowcast_dfm.py                                    # Use defaults
        python nowcast_dfm.py data.vintage_old=2016-12-16        # Specify old vintage
        python nowcast_dfm.py data.vintage_new=2016-12-23        # Specify new vintage
        python nowcast_dfm.py series=GDPC1 period=2016q4         # Specific series and period
    """
    # GitHub Actions context
    github_run_id = os.getenv('GITHUB_RUN_ID')
    if github_run_id:
        logger.info(f"GitHub Actions Run ID: {github_run_id}")
    
    try:
        # Load model configuration with priority: CSV from DB storage → Hydra YAML
        use_db_for_config = True
        if hasattr(cfg, 'model'):
            model_dicts = extract_hydra_config_dicts(cfg, sections=['model'])
            model_dict = model_dicts.get('model', {})
            use_db_for_config = model_dict.get('use_db', True) if model_dict else True
        
        model_cfg = load_model_config_with_hydra_fallback(
            cfg,
            script_path=Path(__file__),
            use_db=use_db_for_config
        )
        
        # Extract config dicts
        config_dicts = extract_hydra_config_dicts(cfg, sections=['data', 'dfm'])
        data_cfg_dict = config_dicts['data']
        dfm_cfg_dict = config_dicts['dfm']
        
        # Extract settings
        use_database = data_cfg_dict.get('use_database', True)
        vintage_old = cfg.get('vintage_old') or data_cfg_dict.get('vintage_old')
        vintage_new = cfg.get('vintage_new') or data_cfg_dict.get('vintage_new') or data_cfg_dict.get('vintage')
        threshold = dfm_cfg_dict.get('threshold', 1e-5)
        max_iter = dfm_cfg_dict.get('max_iter', 5000)
        
        # Initialize database client if needed
        db_client = None
        if use_database:
            try:
                db_client = get_db_client()
            except Exception:
                pass  # Will handle errors in specific operations
        
        # If vintages not specified, use latest vintage for both (for testing)
        if use_database and (not vintage_old or not vintage_new):
            try:
                if db_client is None:
                    db_client = get_db_client()
                result = get_latest_vintage_with_fallback(client=db_client)
                if result:
                    latest_vintage_id, vintage_info = result
                    latest_vintage_date = vintage_info['vintage_date']
                    # Normalize vintage_date to string if it's a date object
                    if hasattr(latest_vintage_date, 'isoformat'):
                        latest_vintage_date = latest_vintage_date.isoformat()
                    elif not isinstance(latest_vintage_date, str):
                        latest_vintage_date = str(latest_vintage_date)
                    if not vintage_new:
                        vintage_new = latest_vintage_date
                    if not vintage_old:
                        # Use same vintage for both (for testing - in production should use different vintages)
                        vintage_old = latest_vintage_date
                    logger.info(f"Using latest vintage: {latest_vintage_date} (ID: {latest_vintage_id})")
                else:
                    logger.warning("No vintage found in database. Run ingest_api.py first.")
            except Exception as e:
                logger.warning(f"Could not get latest vintage: {e}")
        
        if not vintage_old or not vintage_new:
            raise ValueError(
                "Both vintage_old and vintage_new must be specified for nowcasting. "
                "Nowcasting compares forecasts between two vintages for the same period."
            )
        
        # Handle series parameter - can be a string (series ID) or DictConfig (series config)
        series_raw = cfg.get('series', data_cfg_dict.get('target_series', 'GDPC1'))
        # If it's a DictConfig (from Hydra config group), extract the actual series ID
        if hasattr(series_raw, '__class__') and 'DictConfig' in str(type(series_raw)):
            # Series config is a dict of series definitions, get the first series ID
            series_dict = OmegaConf.to_container(series_raw, resolve=True)
            if isinstance(series_dict, dict) and 'series' in series_dict:
                # Extract first series ID from the series dict
                series_dict_inner = series_dict.get('series', {})
                if isinstance(series_dict_inner, dict) and series_dict_inner:
                    series = list(series_dict_inner.keys())[0]
                else:
                    series = 'GDPC1'  # Default fallback
            else:
                series = 'GDPC1'  # Default fallback
        else:
            series = str(series_raw) if series_raw else 'GDPC1'
        
        period_raw = cfg.get('period', data_cfg_dict.get('target_period', '2016q4'))
        period = str(period_raw) if period_raw else '2016q4'
        
        # Determine series frequency from config to convert period format
        # update_nowcast expects period format matching the series' original frequency
        # Quarterly series needs YYYYqQ format, Monthly series needs YYYYmMM format
        series_frequency = 'm'  # Default to monthly
        if hasattr(model_cfg, 'SeriesID') and hasattr(model_cfg, 'Frequency'):
            try:
                series_idx = model_cfg.SeriesID.index(series) if series in model_cfg.SeriesID else -1
                if series_idx >= 0 and series_idx < len(model_cfg.Frequency):
                    series_frequency = model_cfg.Frequency[series_idx].lower()
                    logger.info(f"Series {series} has frequency: {series_frequency}")
            except (ValueError, AttributeError, IndexError):
                pass
        
        # Convert period to match series frequency format
        # update_nowcast requires period format to match the series' native frequency
        import re
        if isinstance(period_raw, str):
            period_lower = period_raw.lower()
            # Check if period is in quarterly format (YYYYqQ)
            q_match = re.match(r'(\d{4})q([1-4])', period_lower)
            # Check if period is in monthly format (YYYYmMM)
            m_match = re.match(r'(\d{4})m(\d{1,2})', period_lower)
            
            if q_match:
                # Period is in quarterly format
                if series_frequency == 'q':
                    # Series is quarterly, keep quarterly format
                    period = period_raw  # Keep as YYYYqQ
                    logger.info(f"Keeping quarterly period format: {period}")
                elif series_frequency == 'm':
                    # Series is monthly, convert to monthly format
                    year = q_match.group(1)
                    quarter = int(q_match.group(2))
                    month = quarter * 3  # Q1=3, Q2=6, Q3=9, Q4=12
                    period = f"{year}m{month:02d}"
                    logger.info(f"Converted quarterly period to monthly: {period_raw} -> {period}")
                else:
                    # Unknown frequency, keep as is
                    period = period_raw
            elif m_match:
                # Period is in monthly format
                if series_frequency == 'q':
                    # Series is quarterly, convert to quarterly format
                    year = m_match.group(1)
                    month = int(m_match.group(2))
                    quarter = (month - 1) // 3 + 1  # 1-3->Q1, 4-6->Q2, 7-9->Q3, 10-12->Q4
                    period = f"{year}q{quarter}"
                    logger.info(f"Converted monthly period to quarterly: {period_raw} -> {period}")
                elif series_frequency == 'm':
                    # Series is monthly, keep monthly format
                    period = period_raw  # Keep as YYYYmMM
                    logger.info(f"Keeping monthly period format: {period}")
                else:
                    # Unknown frequency, keep as is
                    period = period_raw
            else:
                # Period format not recognized, keep as is
                logger.warning(f"Period format not recognized: {period_raw}, keeping as is")
                period = period_raw
        
        config_id = data_cfg_dict.get('config_id')
        strict_mode = data_cfg_dict.get('strict_mode', False)
        forecast_periods = cfg.get('forecast_periods', data_cfg_dict.get('forecast_periods', 2))
        
        logger.info("=" * 80)
        logger.info(f"Nowcasting - Series: {series}, Period: {period}")
        logger.info(f"Vintage (old): {vintage_old}, Vintage (new): {vintage_new}")
        logger.info("=" * 80)
        
        # Load or estimate DFM model
        base_dir = Path(__file__).parent.parent
        res_file = base_dir / 'ResDFM.pkl'
        Res = None
        
        # Use config_id as model_id, or generate a simple hash-based ID
        if config_id:
            model_id = int(config_id) if isinstance(config_id, (int, str)) and str(config_id).isdigit() else hash(str(config_id)) % 2147483647
        else:
            # Generate model_id from config hash
            model_id = hash(str(model_cfg.SeriesID)) % 2147483647
        
        # Try loading from Supabase storage first (primary source for latest weights)
        model_weights_from_storage = None
        model_weights_found = False
        if use_database and vintage_new:
            try:
                from app.adapters.adapter_database import download_model_weights_from_storage
                
                # Try config-specific file first
                model_filename = None
                if config_id:
                    model_filename = f"dfm_config_{config_id}_{vintage_new}.pkl"
                    model_weights_from_storage = download_model_weights_from_storage(
                        filename=model_filename,
                        bucket_name="model-weights",
                        client=db_client if db_client else get_db_client()
                    )
                
                # Fallback to vintage-only file
                if model_weights_from_storage is None:
                    model_filename = f"dfm_{vintage_new}.pkl"
                    model_weights_from_storage = download_model_weights_from_storage(
                        filename=model_filename,
                        bucket_name="model-weights",
                        client=db_client if db_client else get_db_client()
                    )
                
                if model_weights_from_storage:
                    logger.info(f"✅ Loaded model weights from Supabase storage: {model_filename}")
                    model_weights_found = True
                    # Note: We still need full Res for nowcasting, so we'll estimate if ResDFM.pkl not available
                else:
                    logger.warning(f"⚠️  No model weights found in storage for vintage {vintage_new}")
                    logger.info("Will check local files or re-estimate model")
            except ImportError:
                logger.debug("Supabase storage not available, trying local files...")
            except Exception as e:
                logger.warning(f"Failed to load from Supabase storage: {e}. Trying local files...")
        
        # Try loading full results from ResDFM.pkl (preferred for nowcasting)
        if not Res:
            try:
                logger.info(f"Loading full DFM results from {res_file}")
                with open(res_file, 'rb') as f:
                    data = pickle.load(f)
                    saved_config = data.get('Config', data.get('Spec'))
                    if saved_config and 'Res' in data:
                        # Check config consistency
                        if hasattr(saved_config, 'SeriesID') and saved_config.SeriesID != model_cfg.SeriesID:
                            logger.warning('Configuration mismatch. Re-estimating...')
                            raise FileNotFoundError
                        Res = data.get('Res', data)
                    else:
                        Res = data.get('Res', data)
                logger.info("DFM results loaded successfully from ResDFM.pkl")
            except FileNotFoundError:
                Res = None
            except Exception as e:
                logger.warning(f"Failed to load from ResDFM.pkl: {e}")
                Res = None
        
        # Re-estimate if not found
        if Res is None:
            if not model_weights_found and use_database:
                logger.warning(
                    "⚠️  No model weights found in storage and ResDFM.pkl not available. "
                    "Will re-estimate model, but this may take time. "
                    "Consider running train_dfm.py first to generate model weights."
                )
            logger.info('Estimating DFM model...')
            
            if use_database:
                X, Time, Z = load_data_from_db(
                    vintage_date=vintage_new,
                    config=model_cfg,
                    config_id=config_id,
                    strict_mode=strict_mode
                )
                # X and Z are already numpy arrays, no conversion needed
            else:
                data_file = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_new}.csv'
                if not data_file.exists():
                    raise FileNotFoundError(
                        f"CSV file not found: {data_file}\n"
                        f"Use database mode (data.use_database=true) or provide CSV file"
                    )
                X, Time, Z = load_data(data_file, model_cfg)
            
            Res = dfm(X, model_cfg, threshold=threshold, max_iter=max_iter)
            
            # Save model weights to Supabase storage (if re-estimated)
            # Note: Cleanup is only done in train job, not here
            if use_database and vintage_new:
                try:
                    from app.adapters.adapter_database import upload_model_weights_to_storage
                    
                    model_filename = f"dfm_{vintage_new}.pkl"
                    if config_id:
                        model_filename = f"dfm_config_{config_id}_{vintage_new}.pkl"
                    
                    model_weights = {
                        'C': Res.C, 'R': Res.R, 'A': Res.A, 'Q': Res.Q,
                        'Z_0': Res.Z_0, 'V_0': Res.V_0, 'Mx': Res.Mx, 'Wx': Res.Wx,
                        'threshold': threshold, 'vintage': vintage_new, 'config_id': config_id,
                        'convergence_iter': getattr(Res, 'convergence_iter', None),
                        'log_likelihood': getattr(Res, 'loglik', None),
                    }
                    
                    storage_url = upload_model_weights_to_storage(
                        model_weights=model_weights,
                        filename=model_filename,
                        bucket_name="model-weights",
                        client=db_client if db_client else get_db_client()
                    )
                    logger.info(f"Model weights uploaded to Supabase storage: {storage_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload model weights to storage: {e}")
            
            # Save factors, factor_values, and factor_loadings to database for frontend visualization
            if use_database:
                try:
                    from app.adapters.adapter_database import save_factors_to_db
            from app.database import get_latest_vintage_id
                    
                    # Get vintage_id for factor_values
                    vintage_id_new = None
                    if isinstance(vintage_new, int):
                        vintage_id_new = vintage_new
                    else:
                        if db_client is None:
                            db_client = get_db_client()
                        vintage_id_new = get_latest_vintage_id(vintage_date=vintage_new, client=db_client)
                    
                    if vintage_id_new:
                        save_factors_to_db(
                            Res=Res,
                            model_id=model_id,
                            config=model_cfg,
                            vintage_id=vintage_id_new,
                            Time=Time,
                            client=db_client if db_client else get_db_client()
                        )
                        logger.info(f"Saved factors, factor_values, and factor_loadings to database for model_id={model_id}")
                    else:
                        logger.warning(f"Could not resolve vintage_id for {vintage_new}. Skipping factor save.")
                except ImportError:
                    logger.warning("Database module not available. Cannot save factors to database.")
                except Exception as e:
                    logger.warning(f"Failed to save factors to database: {e}", exc_info=True)
            
            # Save full results to ResDFM.pkl (legacy support)
            with open(res_file, 'wb') as f:
                pickle.dump({'Res': Res, 'Config': model_cfg, 'model_id': model_id}, f)
            logger.info(f"Full DFM results saved to {res_file}")
        
        # Load datasets for vintage comparison (nowcasting)
        logger.info("Loading vintages for nowcast comparison...")
        
        if use_database:
            # Load old vintage
            X_old, Time_old, _ = load_data_from_db(
                vintage_date=vintage_old,
                config=model_cfg,
                config_id=config_id,
                strict_mode=strict_mode
            )
            # Load new vintage
            X_new, Time, _ = load_data_from_db(
                vintage_date=vintage_new,
                config=model_cfg,
                config_id=config_id,
                strict_mode=strict_mode
            )
            logger.info("Vintages loaded successfully from database")
        else:
            # Load from files
            datafile_old = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_old}.csv'
            datafile_new = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_new}.csv'
            
            if not datafile_old.exists() or not datafile_new.exists():
                raise FileNotFoundError(
                    f"CSV files not found: {datafile_old} or {datafile_new}\n"
                    f"Use database mode (data.use_database=true) or provide CSV files"
                )
            
            logger.info(f"Loading data from {datafile_old} and {datafile_new}")
            X_old, Time_old, _ = load_data(datafile_old, model_cfg)
            X_new, Time, _ = load_data(datafile_new, model_cfg)
        
        # Update nowcast (news decomposition) - MATLAB functionality
        # series is already converted to string above
        logger.info(f"Running nowcast update for {series}, period {period}")
        logger.info("This compares old vs new vintage forecasts for the current period (nowcasting)")
        
        # Create save callback if database saving is enabled
        # Note: model_id is not stored in DB (models are pkl files only)
        # For forecasts table, model_id can be None or a reference ID
        save_callback = None
        if use_database:
            save_callback = lambda **kwargs: save_nowcast_to_db(**kwargs)
        
        update_nowcast(
            X_old, X_new, Time, model_cfg, Res, series, period,
            vintage_old, vintage_new,
            model_id=None,  # Models are pkl files, not in DB
            save_callback=save_callback
        )
        
        # Generate forward forecasts if forecast_periods > 0
        if forecast_periods > 0:
            logger.info("=" * 80)
            logger.info(f"Generating forward forecasts for {forecast_periods} periods ahead")
            logger.info("=" * 80)
            
            # Generate forecasts
            from app.utils import generate_forecasts
            
            generate_forecasts(
                X_new, Time, model_cfg, Res, series, forecast_periods,
                str(vintage_new), model_id=None, use_database=use_database
        )
        
        logger.info("=" * 80)
        logger.info("Nowcasting completed successfully")
        if forecast_periods > 0:
            logger.info(f"Forward forecasts generated for {forecast_periods} periods")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in nowcasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
