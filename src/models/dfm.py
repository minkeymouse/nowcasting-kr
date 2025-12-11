"""DFM model training and forecasting."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.utils import ValidationError
from src.train.preprocess import (
    prepare_multivariate_data,
    set_dataframe_frequency
)
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_dfm(
    data: pd.DataFrame,
    target_series: str,
    config: Dict[str, Any],
    cfg: DictConfig,
    checkpoint_dir: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Train DFM model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (max_iter, threshold, clock, etc.)
    cfg : DictConfig
        Full experiment config
    checkpoint_dir : Path
        Directory to save checkpoint
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    from dfm_python.config import DFMConfig
    from dfm_python import DFM, DFMDataModule, DFMTrainer
    from src.train.train_dfm_python import (
        _create_preprocessing_pipeline,
        _build_dfm_config,
        _create_time_index_from_data
    )
    
    # Get model parameters
    max_iter = config.get('max_iter', 5000)
    threshold = config.get('threshold', 1e-5)
    mixed_freq = config.get('mixed_freq', False)
    clock = config.get('clock', 'w')
    
    # Prepare data
    config_dict = config.copy()
    config_dict['clock'] = clock
    y_train, available_series = prepare_multivariate_data(
        data, config_dict, cfg, target_series, model_type='dfm'
    )
    y_train = set_dataframe_frequency(y_train)
    
    logger.info(f"Training DFM: {y_train.shape[1]} series, {len(y_train)} observations")
    logger.info(f"Max iterations: {max_iter}, Threshold: {threshold}, Clock: {clock}")
    
    # Convert to DFMConfig
    from src.utils import get_project_root
    config_path = str(get_project_root() / "config")
    
    dfm_config = _build_dfm_config(
        cfg=cfg,
        available_series=available_series,
        model_config=config_dict,
        config_path=config_path,
        clock=clock,
        mixed_freq=mixed_freq
    )
    
    # Create time index
    time_index = _create_time_index_from_data(y_train)
    
    # Create model
    model = DFM(
        config=dfm_config,
        mixed_freq=mixed_freq,
        max_iter=max_iter,
        threshold=threshold
    )
    
    # Create preprocessing pipeline
    preprocessing_pipeline = _create_preprocessing_pipeline(model)
    
    # Create DataModule
    data_module = DFMDataModule(
        config=dfm_config,
        pipeline=preprocessing_pipeline,
        data=y_train.values,
        time_index=time_index
    )
    data_module.setup()
    
    # Train
    trainer = DFMTrainer(max_epochs=max_iter)
    trainer.fit(model, data_module)
    
    logger.info("DFM training completed")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'dfm',
        'target_series': target_series,
        'max_iter': max_iter,
        'threshold': threshold,
        'clock': clock,
        'mixed_freq': mixed_freq,
        'series': list(y_train.columns),
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index)
    }
    save_model_checkpoint(model, checkpoint_path, metadata)
    
    return model, metadata


def forecast_dfm(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    y_recent: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Generate DFM forecasts.
    
    This function uses the update().predict() pattern which is the strength of DFM.
    If y_recent is provided, the model state is updated with latest data before forecasting.
    
    Parameters
    ----------
    model : Any
        Trained DFM model
    horizon : int
        Forecast horizon (in model's clock frequency: weeks if clock='w', months if clock='m')
    last_date : Optional[pd.Timestamp]
        Last date in training data
    y_recent : Optional[pd.DataFrame]
        Recent data for state update. Should be original data (before transformations).
        update() will handle transformations and standardization internally.
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly if clock='w', monthly if clock='m')
    """
    # Get clock frequency from model config
    clock = getattr(model.config, 'clock', 'w') if hasattr(model, 'config') else 'w'
    
    # Update model state with recent data if provided
    # Updated to use original data (not standardized) - update() now handles preprocessing internally
    if y_recent is not None:
        try:
            # Pass original data to update() - it will handle transformations and standardization internally
            if isinstance(y_recent, pd.DataFrame):
                # Use DataFrame directly (update() accepts DataFrame)
                model.update(y_recent, history=None)
                logger.info(f"Updated DFM state with {len(y_recent)} periods (original data) before forecasting")
            else:
                # Convert to DataFrame if numpy array
                y_recent_df = pd.DataFrame(y_recent)
                model.update(y_recent_df, history=None)
                logger.info(f"Updated DFM state with {len(y_recent_df)} periods (original data) before forecasting")
        except Exception as e:
            logger.warning(f"Failed to update DFM with recent data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Generate forecasts
    result = model.predict(horizon=horizon, return_series=True, return_factors=True)
    
    # Handle return value - could be tuple or single array
    if isinstance(result, tuple):
        X_forecast, Z_forecast = result
    else:
        X_forecast = result
        Z_forecast = None
    
    # Convert to DataFrame
    if isinstance(X_forecast, np.ndarray):
        # Ensure X_forecast is 2D (horizon x n_series)
        if X_forecast.ndim == 1:
            # Reshape to (horizon, 1) if single series
            X_forecast = X_forecast.reshape(-1, 1)
        elif X_forecast.ndim == 0:
            # Scalar case - reshape to (1, 1)
            X_forecast = np.array([[X_forecast.item()]])
        
        # Get series names from model config if available
        try:
            series_names = [s.series_id for s in model.config.series]
        except:
            n_series = X_forecast.shape[1] if X_forecast.ndim > 1 else 1
            series_names = [f"series_{i}" for i in range(n_series)]
        
        # Ensure series_names matches X_forecast shape
        n_series = X_forecast.shape[1] if X_forecast.ndim > 1 else 1
        forecast_df = pd.DataFrame(
            X_forecast,
            columns=series_names[:n_series]
        )
    else:
        forecast_df = pd.DataFrame(X_forecast)
    
    # Create index based on clock frequency
    # Always create DatetimeIndex for proper aggregation
    if last_date is not None:
        if clock == 'w':
            # Weekly frequency
            forecast_df.index = pd.date_range(
                start=last_date + pd.DateOffset(weeks=1),
                periods=horizon,
                freq='W'
            )
        else:
            # Monthly frequency (default)
            forecast_df.index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
    else:
        # If last_date is None, create weekly index from current date (for weekly models)
        # This ensures DatetimeIndex exists for aggregation
        if clock == 'w':
            from datetime import datetime
            start_date = datetime.now()
            forecast_df.index = pd.date_range(
                start=start_date,
                periods=horizon,
                freq='W'
            )
        else:
            # For monthly, use RangeIndex if no date available
            forecast_df.index = pd.RangeIndex(start=0, stop=horizon)
    
    return forecast_df

