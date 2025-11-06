"""Nowcasting script for DFM models (MATLAB-compatible).

This script performs nowcasting (vintage comparison) using trained DFM models.
It matches MATLAB functionality: compares old vs new vintage forecasts for the current period,
and decomposes changes into news components.

This is NOT forward forecasting - it only nowcasts the current period (t=now).
"""

import sys
import os
import logging
from pathlib import Path
from datetime import date, datetime
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting import load_data, dfm, update_nowcast, load_config, ModelConfig
from adapters.database import load_data_from_db, save_nowcast_to_db
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> ModelConfig:
    """Load model configuration from database (latest) or CSV/YAML file.
    
    Application-specific config loader for Hydra workflows.
    Priority: Database → CSV → YAML
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
    use_db : bool, default=True
        Whether to try loading from database first
    script_path : Path, optional
        Path to the calling script (for relative path resolution)
        If None, uses current working directory
    
    Returns
    -------
    ModelConfig
        Loaded model configuration
    
    Raises
    ------
    FileNotFoundError
        If config file is specified but not found
    """
    # Try database first (if enabled)
    if use_db:
        try:
            from database import get_client, load_model_config
            
            client = get_client()
            config_name = cfg_model.get('config_name', '001-initial-spec')
            
            db_config = load_model_config(config_name, client=client)
            if db_config and 'config_json' in db_config:
                config_dict = db_config['config_json']
                if 'block_names' not in config_dict and 'block_names' in db_config:
                    config_dict['block_names'] = db_config['block_names']
                return ModelConfig.from_dict(config_dict)
        except (ImportError, Exception):
            pass  # Fall back to file
    
    # Load from CSV or YAML file
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        config_file = Path(model_config_path)
        
        # Resolve relative paths
        if not config_file.is_absolute():
            # Start from script location or current directory
            if script_path:
                base_dir = script_path.parent.parent
            else:
                base_dir = Path.cwd()
            
            # Try multiple possible locations
            search_paths = [
                base_dir / model_config_path,  # Relative to project root
                Path.cwd() / model_config_path,  # Relative to current dir
            ]
            
            # Also try parent directories
            for parent in [base_dir] + list(base_dir.parents):
                search_paths.append(parent / model_config_path)
            
            # Find first existing path
            for candidate in search_paths:
                if candidate.exists():
                    config_file = candidate
                    break
            else:
                # If not found, use default location relative to script
                if script_path:
                    config_file = script_path.parent.parent / model_config_path
                else:
                    config_file = Path(model_config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}\n"
                f"Researchers should update: src/spec/001_initial_spec.csv"
            )
        
        # Load config from file
        model_config = load_config(config_file)
        
        # If loading from CSV and use_db is enabled, save blocks to database
        if use_db and config_file.suffix.lower() == '.csv':
            try:
                from adapters.database import save_blocks_to_db
                
                # Derive config_name from CSV filename
                # Example: '001_initial_spec.csv' → '001-initial-spec'
                config_name = config_file.stem.replace('_', '-')
                
                # Save blocks to database
                save_blocks_to_db(model_config, config_name)
            except (ImportError, Exception) as e:
                # Log warning but don't fail - block saving is optional
                logger.warning(
                    f"Could not save blocks to database for {config_file.name}: {e}. "
                    f"Continuing without saving blocks."
                )
        
        return model_config
    else:
        # Fallback to YAML config (convert DictConfig to dict, then to ModelConfig)
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """Run nowcasting with Hydra configuration (MATLAB-compatible).
    
    This script performs nowcasting (current period forecast comparison):
    1. Loads latest model spec from database (or CSV/YAML fallback)
    2. Loads data from database or files
    3. Estimates DFM model (if not already trained)
    4. Compares old vs new vintage forecasts for current period
    5. Decomposes changes into news components (MATLAB news_dfm functionality)
    
    This matches MATLAB functionality - no forward forecasting, only nowcasting.
    
    Usage:
        python forecast_dfm.py                                    # Use defaults
        python forecast_dfm.py data.vintage_old=2016-12-16        # Specify old vintage
        python forecast_dfm.py data.vintage_new=2016-12-23        # Specify new vintage
        python forecast_dfm.py series=GDPC1 period=2016q4         # Specific series and period
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
        
        if not vintage_old or not vintage_new:
            raise ValueError(
                "Both vintage_old and vintage_new must be specified for nowcasting. "
                "Nowcasting compares forecasts between two vintages for the same period."
            )
        
        series = cfg.get('series', data_cfg_dict.get('target_series', 'GDPC1'))
        period = cfg.get('period', data_cfg_dict.get('target_period', '2016q4'))
        threshold = dfm_cfg_dict.get('threshold', 1e-5)
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
                X, Time, Z = load_data_from_db(
                    vintage_date=vintage_new,
                    config=model_cfg,
                    config_id=config_id,
                    strict_mode=strict_mode
                )
            else:
                data_file = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_new}.csv'
                if not data_file.exists():
                    raise FileNotFoundError(
                        f"CSV file not found: {data_file}\n"
                        f"Use database mode (data.use_database=true) or provide CSV file"
                    )
                X, Time, Z = load_data(data_file, model_cfg)
            
            Res = dfm(X, model_cfg, threshold=threshold)
            
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
        
        logger.info("=" * 80)
        logger.info("Nowcasting completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in nowcasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
