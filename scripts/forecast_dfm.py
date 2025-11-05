"""Forecast generation script for DFM models.

This script generates forecasts using trained DFM models and saves them to the database.
It consolidates both nowcasting (vintage comparison) and forecasting functionality.
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

from src.nowcasting import (
    load_config, load_data, dfm, update_nowcast, load_data_from_db,
    load_model_config_from_hydra
)
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """Run forecasting with Hydra configuration.
    
    This script:
    1. Loads latest model spec from database (or CSV/YAML fallback)
    2. Loads data from database or files
    3. Estimates DFM model (if not already trained)
    4. Generates forecasts
    5. Saves forecasts to database
    
    Usage:
        python forecast_dfm.py                                    # Use defaults
        python forecast_dfm.py data.vintage_old=2016-12-16        # Specify vintage
        python forecast_dfm.py series=GDPC1 period=2016q4         # Specific series
        python forecast_dfm.py forecast.use_db=false              # Don't save to DB
    """
    # GitHub Actions context
    github_run_id = os.getenv('GITHUB_RUN_ID')
    if github_run_id:
        logger.info(f"GitHub Actions Run ID: {github_run_id}")
    
    try:
        # Load model configuration - try DB first, then CSV/YAML
        # Researchers update spec in DB or migrations/001_initial_spec.csv
        use_db_for_config = cfg.get('model', {}).get('use_db', True)
        model_cfg = load_model_config_from_hydra(cfg.model, use_db=use_db_for_config)
        
        # Load data and DFM configs from Hydra
        data_cfg_dict = OmegaConf.to_container(cfg.data, resolve=True)
        dfm_cfg_dict = OmegaConf.to_container(cfg.dfm, resolve=True)
        
        # Extract settings
        use_database = data_cfg_dict.get('use_database', True)
        vintage_old = cfg.get('vintage_old') or data_cfg_dict.get('vintage')
        vintage_new = cfg.get('vintage_new') or data_cfg_dict.get('vintage')
        series = cfg.get('series', data_cfg_dict.get('target_series', 'GDPC1'))
        period = cfg.get('period', data_cfg_dict.get('target_period', '2016q4'))
        threshold = dfm_cfg_dict.get('threshold', 1e-5)
        config_id = data_cfg_dict.get('config_id')
        strict_mode = data_cfg_dict.get('strict_mode', False)
        
        logger.info("=" * 80)
        logger.info(f"Forecasting - Series: {series}, Period: {period}")
        logger.info(f"Vintage (old): {vintage_old}, Vintage (new): {vintage_new}")
        logger.info("=" * 80)
        
        # Load or estimate DFM model
        base_dir = Path(__file__).parent.parent
        res_file = base_dir / 'ResDFM.pkl'
        model_id = None
        
        try:
            logger.info(f"Loading DFM results from {res_file}")
            with open(res_file, 'rb') as f:
                data = pickle.load(f)
                saved_config = data.get('Config', data.get('Spec'))
                if saved_config and 'Res' in data:
                    # Check config consistency
                    if hasattr(saved_config, 'SeriesID') and saved_config.SeriesID != model_cfg.SeriesID:
                        logger.warning('Configuration mismatch. Re-estimating...')
                        raise FileNotFoundError
                    Res = data.get('Res', data)
                    # Try to get model_id from saved data
                    model_id = data.get('model_id')
                else:
                    Res = data.get('Res', data)
            logger.info("DFM results loaded successfully")
        except FileNotFoundError:
            # Re-estimate if file not found or config mismatch
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
            
            # Save model weights to database if available
            if use_database and config_id:
                try:
                    from database import get_client, save_model_weights, get_latest_vintage_id
                    
                    client = get_client()
                    vintage_id = get_latest_vintage_id(vintage_date=vintage_new, client=client)
                    
                    if vintage_id:
                        # Serialize DFMResult to dict
                        params = {
                            'C': Res.C,
                            'R': Res.R,
                            'A': Res.A,
                            'Q': Res.Q,
                            'Z_0': Res.Z_0,
                            'V_0': Res.V_0,
                            'Mx': Res.Mx,
                            'Wx': Res.Wx,
                        }
                        
                        model_record = save_model_weights(
                            config_id=config_id,
                            vintage_id=vintage_id,
                            parameters=params,
                            threshold=threshold,
                            convergence_iter=Res.convergence_iter if hasattr(Res, 'convergence_iter') else None,
                            log_likelihood=Res.loglik if hasattr(Res, 'loglik') else None,
                            client=client
                        )
                        model_id = model_record.get('model_id') if model_record else None
                        logger.info(f"Model weights saved to database (model_id={model_id})")
                except Exception as e:
                    logger.warning(f"Failed to save model weights to database: {e}")
            
            # Save to file
            with open(res_file, 'wb') as f:
                pickle.dump({'Res': Res, 'Config': model_cfg, 'model_id': model_id}, f)
            logger.info(f"DFM model estimated and saved to {res_file}")
        
        # Load datasets for vintage comparison (nowcasting)
        if vintage_old and vintage_old != vintage_new:
            logger.info("Loading vintages for nowcast update...")
            
            if use_database:
                X_old, Time_old, _ = load_data_from_db(
                    vintage_date=vintage_old,
                    config=model_cfg,
                    config_id=config_id,
                    strict_mode=strict_mode
                )
                X_new, Time, _ = load_data_from_db(
                    vintage_date=vintage_new,
                    config=model_cfg,
                    config_id=config_id,
                    strict_mode=strict_mode
                )
            else:
                datafile_old = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_old}.csv'
                datafile_new = base_dir / 'data' / data_cfg_dict.get('country', 'KR') / f'{vintage_new}.csv'
                
                if not datafile_old.exists() or not datafile_new.exists():
                    raise FileNotFoundError("CSV data files not found")
                
                X_old, Time_old, _ = load_data(datafile_old, model_cfg)
                X_new, Time, _ = load_data(datafile_new, model_cfg)
            
            # Update nowcast (news decomposition)
            logger.info(f"Running nowcast update for {series}, period {period}")
            update_nowcast(X_old, X_new, Time, model_cfg, Res, series, period,
                          vintage_old, vintage_new)
        
        # Generate forecasts
        logger.info("Generating forecasts...")
        
        # Extract forecasts from DFM result
        # Forecast logic: use smoothed factors to predict future values
        if hasattr(Res, 'Zsmooth') and Res.Zsmooth is not None:
            # Use smoothed factors for forecasting
            factors = Res.Zsmooth  # (n_factors, T)
            n_factors = factors.shape[0]
            T = factors.shape[1]
            
            # Forecast horizon (can be configured)
            forecast_horizon = cfg.get('forecast', {}).get('horizon', 4)
            
            # Get loading matrix C
            C = Res.C  # (N, n_factors * p)
            
            # For simplicity, forecast using last factor values
            # More sophisticated forecasting would use AR dynamics
            last_factors = factors[:, -1:]  # (n_factors, 1)
            
            # Forecast values (simplified - uses last factor values)
            # This is a placeholder - actual forecasting should use AR dynamics
            forecast_values = {}
            
            # Get series index for target series
            if series in model_cfg.SeriesID:
                series_idx = model_cfg.SeriesID.index(series)
                
                # Extract relevant loadings for this series
                # Simplified: use first factor loading
                loading = C[series_idx, 0] if C.shape[1] > 0 else 0
                
                # Forecast (simplified - should use proper AR forecasting)
                base_value = loading * last_factors[0, 0] if last_factors.shape[0] > 0 else 0
                
                # Store forecast for target date
                forecast_date = pd.to_datetime(period).to_pydatetime().date() if isinstance(period, str) else period
                
                forecast_values[series] = {
                    'value': float(base_value),
                    'date': forecast_date
                }
                
                logger.info(f"Forecast for {series}: {base_value:.4f} on {forecast_date}")
                
                # Save forecast to database
                if use_database and model_id:
                    try:
                        from database import get_client, save_forecast
                        
                        client = get_client()
                        
                        save_forecast(
                            model_id=model_id,
                            series_id=series,
                            forecast_date=forecast_date,
                            forecast_value=base_value,
                            lower_bound=None,  # Can be calculated from confidence intervals
                            upper_bound=None,
                            confidence_level=0.95,
                            client=client
                        )
                        logger.info(f"Forecast saved to database for {series}")
                    except Exception as e:
                        logger.warning(f"Failed to save forecast to database: {e}")
            
            else:
                logger.warning(f"Series {series} not found in model configuration")
        else:
            logger.warning("No smoothed factors available for forecasting")
        
        logger.info("=" * 80)
        logger.info("Forecasting completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in forecasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
