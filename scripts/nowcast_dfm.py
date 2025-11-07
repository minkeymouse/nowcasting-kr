"""Nowcasting script for DFM models (MATLAB-compatible).

This script performs nowcasting (vintage comparison) using trained DFM models.
It matches MATLAB functionality: compares old vs new vintage forecasts for the current period,
and decomposes changes into news components.

This is NOT forward nowcasting - it only nowcasts the current period (t=now).
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dfm_python import load_data, dfm, update_nowcast
from adapters.adapter_database import load_data_from_db, save_nowcast_to_db
from scripts.utils import (
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


@hydra.main(version_base=None, config_path="../config", config_name="default")
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
        
        # If vintages not specified, use latest vintage for both (for testing)
        if use_database and (not vintage_old or not vintage_new):
            try:
                if db_client is None:
                    db_client = get_db_client()
                result = get_latest_vintage_with_fallback(client=db_client)
                if result:
                    latest_vintage_id, vintage_info = result
                    latest_vintage_date = vintage_info['vintage_date']
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
        
        series = cfg.get('series', data_cfg_dict.get('target_series', 'GDPC1'))
        period = cfg.get('period', data_cfg_dict.get('target_period', '2016q4'))
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
                from adapters.adapter_database import download_model_weights_from_storage
                
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
            
            # Save model weights to Supabase storage
            if use_database and vintage_new:
                try:
                    from adapters.adapter_database import upload_model_weights_to_storage
                    
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
                    from adapters.adapter_database import save_factors_to_db
                    from database import get_latest_vintage_id
                    
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
            from scripts.utils import generate_forecasts
            
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
