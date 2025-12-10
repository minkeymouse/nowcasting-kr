"""DFM model training and forecasting."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils import ValidationError
from src.preprocessing import (
    prepare_multivariate_data,
    set_dataframe_frequency,
    create_transformer_from_config
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
    from src.models.models_forecasters import _create_preprocessing_pipeline
    from src.train import _convert_experiment_to_dfm_config_fallback, _create_time_index_from_data
    
    # Get model parameters
    max_iter = config.get('max_iter', 5000)
    threshold = config.get('threshold', 1e-5)
    mixed_freq = config.get('mixed_freq', False)
    clock = config.get('clock', 'w')
    
    # Prepare data
    config_dict = config.copy()
    config_dict['clock'] = clock
    y_train, available_series_list = prepare_multivariate_data(
        data, config_dict, cfg, target_series, model_type='dfm'
    )
    y_train = set_dataframe_frequency(y_train)
    
    logger.info(f"Training DFM: {y_train.shape[1]} series, {len(y_train)} observations")
    logger.info(f"Max iterations: {max_iter}, Threshold: {threshold}, Clock: {clock}")
    
    # Convert to DFMConfig
    project_root = Path(__file__).parent.parent.parent
    config_path = str(project_root / "config")
    
    dfm_config = _convert_experiment_to_dfm_config_fallback(
        cfg=cfg,
        available_series_list=available_series_list,
        model_cfg_dict=config_dict,
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
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate DFM forecasts.
    
    Parameters
    ----------
    model : Any
        Trained DFM model
    horizon : int
        Forecast horizon
    last_date : Optional[pd.Timestamp]
        Last date in training data
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex
    """
    # Generate forecasts
    X_forecast, Z_forecast = model.predict(horizon=horizon, return_series=True, return_factors=True)
    
    # Convert to DataFrame
    if isinstance(X_forecast, np.ndarray):
        # Get series names from model config if available
        try:
            series_names = [s.series_id for s in model.config.series]
        except:
            series_names = [f"series_{i}" for i in range(X_forecast.shape[1])]
        
        forecast_df = pd.DataFrame(
            X_forecast,
            columns=series_names[:X_forecast.shape[1]]
        )
    else:
        forecast_df = pd.DataFrame(X_forecast)
    
    # Create index
    if last_date is not None:
        forecast_df.index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )
    
    return forecast_df

