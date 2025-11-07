"""Nowcasting script for DFM models (MATLAB-compatible).

This script performs nowcasting (vintage comparison) using trained DFM models.
It matches MATLAB functionality: compares old vs new vintage forecasts for the current period,
and decomposes changes into news components.

This is NOT forward nowcasting - it only nowcasts the current period (t=now).
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting import load_data, dfm, update_nowcast
from adapters.adapter_database import load_data_from_db, save_nowcast_to_db
from scripts.utils import load_model_config_from_hydra, get_db_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
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
        # Load model configuration - try DB first, then CSV/YAML
        # Researchers update spec in DB or src/spec/001_initial_spec.csv
        use_db_for_config = cfg.get('model', {}).get('use_db', True)
        model_cfg = load_model_config_from_hydra(
            cfg.model,
            use_db=use_db_for_config,
            script_path=Path(__file__)
        )
        
        # Load data and DFM configs from Hydra
        data_cfg_dict = OmegaConf.to_container(cfg.data, resolve=True)
        dfm_cfg_dict = OmegaConf.to_container(cfg.dfm, resolve=True)
        
        # Extract settings
        use_database = data_cfg_dict.get('use_database', True)
        vintage_old = cfg.get('vintage_old') or data_cfg_dict.get('vintage_old')
        vintage_new = cfg.get('vintage_new') or data_cfg_dict.get('vintage_new') or data_cfg_dict.get('vintage')
        threshold = dfm_cfg_dict.get('threshold', 1e-5)
        max_iter = dfm_cfg_dict.get('max_iter', 5000)
        
        # If vintages not specified, use latest vintage for both (for testing)
        if use_database and (not vintage_old or not vintage_new):
            try:
                from database import get_latest_vintage_id, get_vintage
                client = get_db_client()
                latest_vintage_id = get_latest_vintage_id(client=client)
                if latest_vintage_id:
                    vintage_info = get_vintage(vintage_id=latest_vintage_id, client=client)
                    if vintage_info:
                        latest_vintage_date = vintage_info['vintage_date']
                        if not vintage_new:
                            vintage_new = latest_vintage_date
                        if not vintage_old:
                            # Use same vintage for both (for testing - in production should use different vintages)
                            vintage_old = latest_vintage_date
                        logger.info(f"Using latest vintage: {latest_vintage_date} (ID: {latest_vintage_id})")
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
        
        logger.info("=" * 80)
        logger.info(f"Nowcasting - Series: {series}, Period: {period}")
        logger.info(f"Vintage (old): {vintage_old}, Vintage (new): {vintage_new}")
        logger.info("=" * 80)
        
        # Load or estimate DFM model
        base_dir = Path(__file__).parent.parent
        model_dir = base_dir / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Fallback to ResDFM.pkl if model file not found
        res_file = base_dir / 'ResDFM.pkl'
        Res = None
        model_id = None
        
        # Try to load model weights from model/ directory
        model_file = None
        if vintage_new:
            # Try config-specific file first
            if config_id:
                model_file = model_dir / f"dfm_config_{config_id}_{vintage_new}.pkl"
                if not model_file.exists():
                    model_file = None
            # Fallback to vintage-only file
            if model_file is None or not model_file.exists():
                model_file = model_dir / f"dfm_{vintage_new}.pkl"
        
        # Try loading from model/ directory first
        if model_file and model_file.exists():
            try:
                logger.info(f"Loading model weights from {model_file}")
                with open(model_file, 'rb') as f:
                    model_weights = pickle.load(f)
                
                # Reconstruct DFMResult from weights (need to estimate full model)
                # For nowcasting, we need full Res object, so we'll still need to estimate
                # But we can skip if we have ResDFM.pkl with full results
                logger.info("Model weights loaded, but full DFM estimation needed for nowcasting")
                logger.info("Checking for full DFM results in ResDFM.pkl...")
            except Exception as e:
                logger.warning(f"Failed to load model weights from {model_file}: {e}")
                model_file = None
        
        # Try loading from model/ directory first
        if model_file and model_file.exists():
            try:
                logger.info(f"Loading model weights from {model_file}")
                with open(model_file, 'rb') as f:
                    model_weights = pickle.load(f)
                
                # Reconstruct DFMResult from weights (need to estimate full model)
                # For nowcasting, we need full Res object, so we'll still need to estimate
                # But we can skip if we have ResDFM.pkl with full results
                logger.info("Model weights loaded, but full DFM estimation needed for nowcasting")
                logger.info("Checking for full DFM results in ResDFM.pkl...")
            except Exception as e:
                logger.warning(f"Failed to load model weights from {model_file}: {e}")
                model_file = None
        
        # Try loading full results from ResDFM.pkl
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
            logger.info('Estimating DFM model...')
            
            if use_database:
                data_df, Time, Z_df, _ = load_data_from_db(
                    vintage_date=vintage_new,
                    config=model_cfg,
                    config_id=config_id,
                    strict_mode=strict_mode
                )
                # Convert DataFrame to numpy array for DFM
                X = data_df.values
                Z = Z_df.values if Z_df is not None else None
            else:
                data_file = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_new}.csv'
                if not data_file.exists():
                    raise FileNotFoundError(
                        f"CSV file not found: {data_file}\n"
                        f"Use database mode (data.use_database=true) or provide CSV file"
                    )
                X, Time, Z = load_data(data_file, model_cfg)
            
            Res = dfm(X, model_cfg, threshold=threshold, max_iter=max_iter)
            
            # Save model weights to model/ directory (not to database yet)
            if vintage_new:
                model_filename = f"dfm_{vintage_new}.pkl"
                if config_id:
                    model_filename = f"dfm_config_{config_id}_{vintage_new}.pkl"
                model_file = model_dir / model_filename
                
                model_weights = {
                    'C': Res.C,
                    'R': Res.R,
                    'A': Res.A,
                    'Q': Res.Q,
                    'Z_0': Res.Z_0,
                    'V_0': Res.V_0,
                    'Mx': Res.Mx,
                    'Wx': Res.Wx,
                    'threshold': threshold,
                    'vintage': vintage_new,
                    'config_id': config_id,
                    'convergence_iter': getattr(Res, 'convergence_iter', None),
                    'log_likelihood': getattr(Res, 'loglik', None),
                }
                
                with open(model_file, 'wb') as f:
                    pickle.dump(model_weights, f)
                logger.info(f"Model weights saved to {model_file}")
            
            # Save full results to ResDFM.pkl (legacy support)
            with open(res_file, 'wb') as f:
                pickle.dump({'Res': Res, 'Config': model_cfg, 'model_id': model_id}, f)
            logger.info(f"Full DFM results saved to {res_file}")
        
        # Load datasets for vintage comparison (nowcasting)
        logger.info("Loading vintages for nowcast comparison...")
        
        if use_database:
            # Load old vintage
            data_df_old, Time_old, _, _ = load_data_from_db(
                vintage_date=vintage_old,
                config=model_cfg,
                config_id=config_id,
                strict_mode=strict_mode
            )
            X_old = data_df_old.values
            # Load new vintage
            data_df_new, Time, _, _ = load_data_from_db(
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
        
        logger.info("=" * 80)
        logger.info("Nowcasting completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in nowcasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
