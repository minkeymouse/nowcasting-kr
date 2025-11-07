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

logger = logging.getLogger(__name__)


def summarize(X: np.ndarray, Time, config: DFMConfig, vintage: Optional[str] = None) -> None:
    """Print summary statistics for data matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    Time : pd.DatetimeIndex or similar
        Time index
    config : DFMConfig
        Model configuration
    vintage : str, optional
        Vintage identifier for display
    """
    T, N = X.shape
    
    print("\n" + "=" * 80)
    print("Table 2: Data Summary")
    print("=" * 80)
    print(f"N = {N:4d} data series")
    print(f"T = {T:4d} observations from {Time[0]} to {Time[-1]}")
    
    if vintage:
        print(f"Vintage: {vintage}")
    
    print(f"{'Data Series':<40} | {'Observations':<20} {'Units':<15} {'Frequency':<12} {'Mean':<10} {'Std. Dev.':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    for i in range(N):
        series_id = config.SeriesID[i] if hasattr(config, 'SeriesID') and i < len(config.SeriesID) else f"Series_{i}"
        series_name = config.SeriesName[i] if hasattr(config, 'SeriesName') and i < len(config.SeriesName) else series_id
        freq = config.Frequency[i] if hasattr(config, 'Frequency') and i < len(config.Frequency) else "N/A"
        units = getattr(config, 'Units', [None] * N)[i] if hasattr(config, 'Units') and i < len(getattr(config, 'Units', [])) else "N/A"
        
        x_series = X[:, i]
        non_nan = ~np.isnan(x_series)
        n_obs = np.sum(non_nan)
        
        if n_obs > 0:
            x_valid = x_series[non_nan]
            mean_val = np.mean(x_valid)
            std_val = np.std(x_valid)
            min_val = np.min(x_valid)
            max_val = np.max(x_valid)
            
            # Format time range
            time_range = f"{Time[non_nan][0]} - {Time[non_nan][-1]}" if n_obs > 0 else "N/A"
        else:
            mean_val = std_val = min_val = max_val = np.nan
            time_range = "N/A"
        
        # Truncate long names
        display_name = (series_name[:37] + "...") if len(series_name) > 40 else series_name
        print(f"{display_name:<40} | {n_obs:>10} obs    {str(units):<15} {freq:<12} {mean_val:>10.1f} {std_val:>10.1f} {min_val:>10.1f} {max_val:>10.1f}")
        if series_id != series_name:
            print(f"[{series_id}]".rjust(42) + " | " + time_range)
    
    print("=" * 80)


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> DFMConfig:
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
    DFMConfig
        Loaded model configuration
    
    Raises
    ------
    FileNotFoundError
        If config file is specified but not found
    """
    # Priority 1: Try to load CSV from database storage bucket
    if use_db:
        try:
            from app.adapters.adapter_database import (
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
                
                # CSV provides series info but NOT model-level config (factors_per_block, etc.)
                # We need to merge model config from Hydra (cfg_model) if available
                # Extract factors_per_block from cfg_model.blocks if available
                if cfg_model and hasattr(cfg_model, 'get'):
                    model_dict = OmegaConf.to_container(cfg_model, resolve=True) if hasattr(cfg_model, '__class__') else dict(cfg_model)
                    blocks_dict = model_dict.get('blocks', {})
                    
                    # Extract factors_per_block from blocks dict
                    # Format: {'Global': {'factors': 3}, 'Investment': {'factors': 2}}
                    if blocks_dict and isinstance(blocks_dict, dict):
                        factors_per_block = []
                        block_names_from_blocks = []
                        for block_name, block_cfg in blocks_dict.items():
                            if isinstance(block_cfg, dict):
                                factors = block_cfg.get('factors', 1)
                            else:
                                factors = 1
                            factors_per_block.append(factors)
                            block_names_from_blocks.append(block_name)
                        
                        if factors_per_block:
                            # Update model_config with factors_per_block from Hydra
                            model_config.factors_per_block = factors_per_block
                            logger.info(f"Merged factors_per_block from Hydra: {factors_per_block}")
                
                # Save blocks to database
                try:
                    from app.adapters.adapter_database import save_blocks_to_db
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
            
            # CSV provides series info but NOT model-level config (factors_per_block, etc.)
            # Merge model config from Hydra (cfg_model) if available
            if cfg_model and hasattr(cfg_model, 'get'):
                model_dict = OmegaConf.to_container(cfg_model, resolve=True) if hasattr(cfg_model, '__class__') else dict(cfg_model)
                blocks_dict = model_dict.get('blocks', {})
                
                # Extract factors_per_block from blocks dict
                if blocks_dict and isinstance(blocks_dict, dict):
                    factors_per_block = []
                    for block_name, block_cfg in blocks_dict.items():
                        if isinstance(block_cfg, dict):
                            factors = block_cfg.get('factors', 1)
                        else:
                            factors = 1
                        factors_per_block.append(factors)
                    
                    if factors_per_block:
                        model_config.factors_per_block = factors_per_block
                        logger.info(f"Merged factors_per_block from Hydra: {factors_per_block}")
            
            # Save blocks to database if enabled
            if use_db and config_file.suffix.lower() == '.csv':
                try:
                    from app.adapters.adapter_database import save_blocks_to_db
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
        return DFMConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))
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
        from app.adapters.adapter_database import _get_db_client
        return _get_db_client()
    except ImportError:
        # Fallback to database.get_client
        from app.database import get_client
        return get_client()


def get_latest_vintage_with_fallback(
    client: Optional[Any] = None,
    allow_partial: bool = True
) -> Optional[tuple[int, dict]]:
    """Get latest vintage ID with fallback from 'completed' to 'partial'.
    
    Parameters
    ----------
    client : Client, optional
        Database client. If None, will get from get_db_client()
    allow_partial : bool, default=True
        If True, fallback to 'partial' status if no 'completed' vintage found
    
    Returns
    -------
    tuple[int, dict] or None
        (vintage_id, vintage_info) if found, None otherwise
    """
    from app.database import get_latest_vintage_id, get_vintage
    
    if client is None:
        client = get_db_client()
    
    # Try 'completed' first
    vintage_id = get_latest_vintage_id(status='completed', client=client)
    if not vintage_id and allow_partial:
        vintage_id = get_latest_vintage_id(status='partial', client=client)
        if vintage_id:
            logger.warning(f"Using 'partial' vintage {vintage_id} (no 'completed' vintage found)")
    
    if vintage_id:
        vintage_info = get_vintage(vintage_id=vintage_id, client=client)
        if vintage_info:
            return vintage_id, vintage_info
    
    return None


def load_model_config_with_hydra_fallback(
    cfg: DictConfig,
    script_path: Optional[Path] = None,
    use_db: bool = True
) -> DFMConfig:
    """Load model config with CSV → Hydra YAML fallback.
    
    Consolidated config loading logic used by train_dfm.py and nowcast_dfm.py.
    
    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration
    script_path : Path, optional
        Path to calling script for relative path resolution
    use_db : bool, default=True
        Whether to try loading from database storage first
    
    Returns
    -------
    DFMConfig
        Loaded model configuration
    
    Raises
    ------
    ValueError
        If config cannot be loaded from any source
    """
    # Try loading from CSV (DB storage or local)
    try:
        use_db_for_config = use_db
        if hasattr(cfg, 'model'):
            model_dict = OmegaConf.to_container(cfg.model, resolve=True)
            use_db_for_config = model_dict.get('use_db', use_db)
        
        model_cfg = load_model_config_from_hydra(
            cfg.model,
            use_db=use_db_for_config,
            script_path=script_path
        )
    except (ValueError, FileNotFoundError):
        # Fallback to Hydra YAML structure
        logger.info("CSV config not found, trying Hydra YAML structure...")
        from dfm_python.config import DFMConfig
        
        series_dict = OmegaConf.to_container(cfg.series, resolve=True) if hasattr(cfg, 'series') else {}
        model_dict = OmegaConf.to_container(cfg.model, resolve=True) if hasattr(cfg, 'model') else {}
        
        combined_dict = {
            'series': series_dict.get('series', {}),
            'blocks': model_dict.get('blocks', {}),
            'block_names': model_dict.get('block_names', None),
            'factors_per_block': model_dict.get('factors_per_block', None)
        }
        
        model_cfg = DFMConfig.from_dict(combined_dict)
        logger.info("✅ Loaded config from Hydra YAML structure")
    
    # Merge factors_per_block from Hydra if available
    if hasattr(cfg, 'model'):
        merge_factors_per_block_from_hydra(model_cfg, cfg.model)
    
    return model_cfg


def extract_hydra_config_dicts(
    cfg: DictConfig,
    sections: list[str] = ['data', 'dfm']
) -> dict[str, dict]:
    """Extract and convert Hydra config sections to dictionaries.
    
    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration
    sections : list[str]
        List of config sections to extract (default: ['data', 'dfm'])
    
    Returns
    -------
    dict[str, dict]
        Dictionary mapping section names to their config dicts
    """
    result = {}
    for section in sections:
        if hasattr(cfg, section):
            result[section] = OmegaConf.to_container(getattr(cfg, section), resolve=True)
        else:
            result[section] = {}
    return result


def merge_factors_per_block_from_hydra(
    model_cfg: DFMConfig,
    cfg_model: DictConfig
) -> None:
    """Merge factors_per_block from Hydra config into model config.
    
    CSV provides series info but NOT model-level config (factors_per_block),
    so we need to merge it from Hydra YAML config.
    
    Parameters
    ----------
    model_cfg : ModelConfig
        Model configuration to update
    cfg_model : DictConfig
        Hydra model configuration dict containing blocks with factors
    """
    if not hasattr(cfg_model, '__class__'):
        return
    
    model_dict = OmegaConf.to_container(cfg_model, resolve=True)
    blocks_dict = model_dict.get('blocks', {})
    
    if blocks_dict and isinstance(blocks_dict, dict):
        # Extract factors_per_block from blocks dict: {'Global': {'factors': 3}, ...}
        factors_per_block = []
        for block_name, block_cfg in blocks_dict.items():
            if isinstance(block_cfg, dict):
                factors = block_cfg.get('factors', 1)
            else:
                factors = 1
            factors_per_block.append(factors)
        
        if factors_per_block:
            model_cfg.factors_per_block = factors_per_block
            logger.info(f"Merged factors_per_block from Hydra model.blocks: {factors_per_block}")


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
            from app.adapters.adapter_database import save_forecast_to_db
            from app.database import get_latest_vintage_id
            
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



