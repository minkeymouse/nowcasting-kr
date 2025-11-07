"""Shared utilities for DFM scripts.

This module consolidates common functionality used across train_dfm.py,
nowcast_dfm.py, and other scripts to reduce code duplication.
"""

import logging
import io
from pathlib import Path
from typing import Optional, Any, Union
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd

from dfm_python import load_config, DFMConfig
# Backward compatibility alias
ModelConfig = DFMConfig

logger = logging.getLogger(__name__)


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> ModelConfig:
    """Load model configuration with priority: CSV from DB storage → Hydra YAML.
    
    Application-specific config loader for Hydra workflows.
    Priority: CSV from DB storage bucket → Local CSV → Hydra YAML
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
    use_db : bool, default=True
        Whether to try loading from database storage first
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
    # Priority 1: Try to load CSV from database storage bucket
    if use_db:
        try:
            from adapters.adapter_database import (
                download_spec_csv_from_storage,
                get_latest_spec_csv_filename
            )
            
            # Get DB client
            db_client = get_db_client()
            
            # Determine which spec file to load
            spec_filename = None
            
            # If config_path is specified, try that first
            model_config_path = cfg_model.get('config_path')
            if model_config_path and Path(model_config_path).suffix.lower() == '.csv':
                spec_filename = Path(model_config_path).name
                logger.info(f"Trying specified spec file: {spec_filename}")
            else:
                # Get the latest spec file (highest number)
                spec_filename = get_latest_spec_csv_filename(
                    bucket_name="spec",
                    client=db_client
                )
                if spec_filename:
                    logger.info(f"Using latest spec file: {spec_filename}")
            
            # Download and load the spec file
            if spec_filename:
                csv_content = download_spec_csv_from_storage(
                    filename=spec_filename,
                    bucket_name="spec",
                    client=db_client
                )
                
                if csv_content:
                    loaded_filename = spec_filename
                    logger.info(f"✅ Loaded spec CSV from database storage: {spec_filename}")
                else:
                    logger.debug(f"Spec file not found in storage: {spec_filename}")
                    csv_content = None
            else:
                logger.debug("No spec CSV files found in storage bucket")
                csv_content = None
            
            if csv_content:
                # Load config from CSV bytes
                csv_file_like = io.BytesIO(csv_content)
                model_config = load_config(csv_file_like)
                
                # Save blocks to database
                try:
                    from adapters.adapter_database import save_blocks_to_db
                    config_name = Path(loaded_filename).stem.replace('_', '-')
                    save_blocks_to_db(model_config, config_name)
                    logger.info(f"Saved blocks to database for {config_name}")
                except Exception as e:
                    logger.warning(f"Could not save blocks to database: {e}")
                
                return model_config
        except (ImportError, Exception) as e:
            logger.debug(f"Could not load CSV from database storage: {e}. Falling back to Hydra YAML...")
            pass  # Fall back to Hydra YAML
    
    # Priority 2: Try local CSV file if config_path specified
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        config_file = Path(model_config_path)
        
        # Resolve relative paths
        if not config_file.is_absolute():
            if script_path:
                base_dir = script_path.parent.parent
            else:
                base_dir = Path.cwd()
            
            # Try multiple possible locations
            search_paths = [
                base_dir / model_config_path,
                Path.cwd() / model_config_path,
            ]
            
            for parent in [base_dir] + list(base_dir.parents):
                search_paths.append(parent / model_config_path)
            
            for candidate in search_paths:
                if candidate.exists():
                    config_file = candidate
                    break
            else:
                if script_path:
                    config_file = script_path.parent.parent / model_config_path
                else:
                    config_file = Path(model_config_path)
        
        if config_file.exists():
            logger.info(f"Loading config from local CSV file: {config_file}")
            model_config = load_config(config_file)
            
            # Save blocks to database if enabled
            if use_db and config_file.suffix.lower() == '.csv':
                try:
                    from adapters.adapter_database import save_blocks_to_db
                    config_name = config_file.stem.replace('_', '-')
                    save_blocks_to_db(model_config, config_name)
                except Exception as e:
                    logger.warning(f"Could not save blocks to database: {e}")
            
            return model_config
    
    # Priority 3: Fallback to Hydra YAML config
    logger.info("Falling back to Hydra YAML config structure")
    try:
        # Try to construct from Hydra YAML structure
        # This requires cfg.series and cfg.model to be available
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))
    except Exception:
        # If that fails, try to get from parent config (if called from script)
        # This is a last resort fallback
        raise ValueError(
            "Could not load config from database storage, local CSV, or Hydra YAML. "
            "Please ensure spec CSV is uploaded to database storage bucket 'spec' "
            "or provide a valid config_path."
        )


def get_db_client():
    """Get database client with consistent error handling.
    
    Returns
    -------
    Client
        Database client instance
    
    Raises
    ------
    ImportError
        If database module is not available
    Exception
        If client initialization fails
    """
    try:
        from adapters.adapter_database import _get_db_client
        return _get_db_client()
    except ImportError:
        # Fallback to database.get_client
        from database import get_client
        return get_client()


def generate_forecasts(
    X: np.ndarray,
    Time: pd.DatetimeIndex,
    config: DFMConfig,
    Res: Any,
    series: str,
    forecast_periods: int,
    vintage_new: Union[str, int],
    model_id: Optional[int] = None,
    use_database: bool = True
) -> None:
    """Generate forward forecasts for specified number of periods ahead.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N) from latest vintage
    Time : pd.DatetimeIndex
        Time index for data
    config : DFMConfig
        DFM model configuration
    Res : DFMResult
        DFM estimation results
    series : str
        Target series ID to forecast
    forecast_periods : int
        Number of periods ahead to forecast (default: 2)
    vintage_new : str
        Vintage date used for forecasting
    model_id : int, optional
        Model ID for saving (default: None)
    use_database : bool
        Whether to save forecasts to database (default: True)
    
    Notes
    -----
    Forecasts are generated using the DFM state-space model:
    - Factors evolve as: Z_{t+h} = A^h @ Z_T
    - Forecasts are: y_{t+h} = C @ Z_{t+h}
    - Multiple rows are saved to database (one per forecast period)
    - Each row has run_type='forecast', vintage_id_new only, vintage_id_old=NULL
    """
    from dfm_python.kalman import run_kf
    
    logger.info(f"Generating {forecast_periods} period(s) ahead forecasts for {series}")
    
    # Find series index
    try:
        i_series = config.SeriesID.index(series)
    except ValueError:
        logger.error(f"Series {series} not found in configuration")
        return
    
    series_name = config.SeriesName[i_series]
    freq = config.Frequency[i_series]
    
    # Get the last smoothed factor estimate (from Kalman smoother)
    # Run Kalman filter/smoother on the latest data to get Z_T
    T, N = X.shape
    
    # Standardize data using Res parameters
    x = (X - Res.Mx) / Res.Wx
    
    # Run Kalman filter and smoother to get final factor estimate
    # Note: y needs to be n x T (transpose)
    y = x.T  # n x T
    
    # Get smoothed factors
    zsmooth, _, _, _ = run_kf(
        Y=y,
        A=Res.A,
        C=Res.C,
        Q=Res.Q,
        R=Res.R,
        Z_0=Res.Z_0,
        V_0=Res.V_0
    )
    
    # zsmooth is (m x (T+1)) from run_kf, get the last factor vector
    # Note: zsmooth[:, t+1] = Z_t|T, so last column is Z_T|T
    if zsmooth.shape[0] < zsmooth.shape[1]:
        # If shape is (m x T), get last column
        Z_T = zsmooth[:, -1]  # (m,)
    else:
        # If shape is (T x m), get last row
        Z_T = zsmooth[-1, :]  # (m,)
    
    # Generate forecasts for each period ahead
    forecasts = []
    last_date = Time.iloc[-1] if hasattr(Time, 'iloc') else Time[-1]
    
    for h in range(1, forecast_periods + 1):
        # Project factor forward: Z_{T+h} = A^h @ Z_T
        A_power = np.linalg.matrix_power(Res.A, h)
        Z_T_plus_h = A_power @ Z_T
        
        # Forecast: y_{T+h} = C @ Z_{T+h}
        # Get forecast for the target series
        y_forecast = Res.C[i_series, :] @ Z_T_plus_h
        
        # Unstandardize: X_forecast = y_forecast * Wx + Mx
        X_forecast = y_forecast * Res.Wx[i_series] + Res.Mx[i_series]
        
        # Calculate forecast date based on frequency
        if freq.lower() == 'q':
            # Quarterly: add h quarters
            forecast_date = last_date + pd.DateOffset(months=3 * h)
        elif freq.lower() == 'm':
            # Monthly: add h months
            forecast_date = last_date + pd.DateOffset(months=h)
        elif freq.lower() == 'd':
            # Daily: add h days
            forecast_date = last_date + pd.DateOffset(days=h)
        else:
            # Default: assume monthly
            forecast_date = last_date + pd.DateOffset(months=h)
        
        forecasts.append({
            'period': h,
            'forecast_date': forecast_date,
            'forecast_value': X_forecast
        })
        
        logger.info(
            f"  Forecast {h} period(s) ahead: {forecast_date.date()} = {X_forecast:.4f}"
        )
    
    # Save forecasts to database
    if use_database:
        try:
            from adapters.adapter_database import save_forecast_to_db
            from database import get_latest_vintage_id
            
            db_client = get_db_client()
            
            # Resolve vintage_id_new
            vintage_id_new = None
            if isinstance(vintage_new, int):
                vintage_id_new = vintage_new
            else:
                try:
                    vintage_id_new = get_latest_vintage_id(
                        vintage_date=vintage_new,
                        client=db_client
                    )
                except Exception as e:
                    logger.warning(f"Could not resolve vintage_id_new: {e}")
            
            # Save each forecast as a separate row
            for fcst in forecasts:
                save_forecast_to_db(
                    model_id=model_id if model_id is not None else 0,
                    series_id=series,
                    forecast_date=fcst['forecast_date'],
                    forecast_value=fcst['forecast_value'],
                    vintage_id_new=vintage_id_new,
                    client=db_client
                )
            
            logger.info(f"✅ Saved {len(forecasts)} forecast(s) to database")
            
        except ImportError:
            logger.warning("Database module not available. Cannot save forecasts to database.")
        except Exception as e:
            logger.warning(f"Failed to save forecasts to database: {e}")



