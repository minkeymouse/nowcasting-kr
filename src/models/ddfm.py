"""DDFM model training and forecasting."""

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


def train_ddfm(
    data: pd.DataFrame,
    target_series: str,
    config: Dict[str, Any],
    cfg: DictConfig,
    checkpoint_dir: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Train DDFM model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (epochs, encoder_layers, etc.)
    cfg : DictConfig
        Full experiment config
    checkpoint_dir : Path
        Directory to save checkpoint
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    from dfm_python import DDFM, DFMDataModule, DDFMTrainer
    from src.train.train_dfm_python import (
        _create_preprocessing_pipeline,
        _build_dfm_config,
        _create_time_index_from_data
    )
    # Get model parameters
    epochs = config.get('epochs', 100)
    encoder_layers = config.get('encoder_layers', [16, 4])
    num_factors = config.get('num_factors', 1)
    learning_rate = config.get('learning_rate', 0.005)
    batch_size = config.get('batch_size', 100)
    activation = config.get('activation', 'relu')
    loss_function = config.get('loss_function', 'mse')
    weight_decay = config.get('weight_decay', 0.0)
    grad_clip_val = config.get('grad_clip_val', 1.0)
    factor_order = config.get('factor_order', 1)
    mult_epoch_pretrain = config.get('mult_epoch_pretrain', 1)
    clock = config.get('clock', 'w')  # Default to weekly, but should come from Hydra config
    mixed_freq = config.get('mixed_freq', False)
    
    # Target-specific adjustments (simplified - can be moved to config)
    if target_series == 'KOEQUIPTE':
        if encoder_layers == [16, 4] or encoder_layers is None:
            encoder_layers = [64, 32, 16]
        if activation == 'relu' and 'activation' not in config:
            activation = 'tanh'
        if epochs == 100:
            epochs = 150
        if weight_decay == 0.0:
            weight_decay = 1e-4
        if batch_size >= 100:
            batch_size = 64
        if mult_epoch_pretrain == 1:
            mult_epoch_pretrain = 2
    
    # Prepare data
    config_dict = config.copy()
    # Add clock to config_dict so prepare_multivariate_data can use it
    config_dict['clock'] = clock
    y_train, available_series = prepare_multivariate_data(
        data, config_dict, cfg, target_series, model_type='ddfm'
    )
    y_train = set_dataframe_frequency(y_train)
    
    logger.info(f"Training DDFM: {y_train.shape[1]} series, {len(y_train)} observations")
    logger.info(f"Epochs: {epochs}, Encoder: {encoder_layers}, Factors: {num_factors}")
    
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
    model = DDFM(
        config=dfm_config,
        encoder_layers=encoder_layers,
        num_factors=num_factors,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        activation=activation,
        loss_function=loss_function,
        weight_decay=weight_decay,
        grad_clip_val=grad_clip_val,
        factor_order=factor_order,
        mult_epoch_pretrain=mult_epoch_pretrain
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
    # Note: DDFM training happens entirely in on_train_start() via MCMC.
    # PyTorch Lightning should not run additional epochs, so set max_epochs=1.
    # The epochs parameter is used internally by MCMC (epochs_per_iter), not by Lightning.
    trainer = DDFMTrainer(max_epochs=1)
    logger.info(f"DDFM trainer created with max_epochs={trainer.max_epochs}")
    trainer.fit(model, data_module)
    
    logger.info("DDFM training completed")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'ddfm',
        'target_series': target_series,
        'epochs': epochs,
        'encoder_layers': encoder_layers,
        'num_factors': num_factors,
        'series': list(y_train.columns),
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index)
    }
    save_model_checkpoint(model, checkpoint_path, metadata)
    
    return model, metadata


def forecast_ddfm(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    y_recent: Optional[pd.DataFrame] = None,
    target_series: Optional[str] = None,
    original_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Generate DDFM forecasts.
    
    This function uses the update().predict() pattern which is the strength of DDFM.
    If y_recent is provided, the model state is updated with latest data before forecasting.
    
    Parameters
    ----------
    model : Any
        Trained DDFM model
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
    # IMPORTANT: Apply transformations first (matching training data preparation)
    # Training: Raw data → prepare_multivariate_data → transformations → preprocessing pipeline (standardization)
    # Prediction: Raw data → transformations → update() → preprocessing pipeline (standardization)
    if y_recent is not None:
        try:
            # Convert to DataFrame if needed
            if isinstance(y_recent, pd.DataFrame):
                y_recent_df = y_recent
            elif isinstance(y_recent, np.ndarray):
                y_recent_df = pd.DataFrame(y_recent)
            else:
                y_recent_df = pd.DataFrame(y_recent)
            
            # Apply transformations first (matching prepare_multivariate_data)
            # This ensures the same data preparation as training
            from src.train.preprocess import apply_transformations
            from src.utils import get_config_path
            
            config_path = str(get_config_path())
            # Get series IDs from model config
            try:
                series_ids = [s.series_id for s in model.config.series] if hasattr(model, 'config') else list(y_recent_df.columns)
            except:
                series_ids = list(y_recent_df.columns)
            
            # Apply transformations to match training data preparation
            y_recent_transformed = apply_transformations(y_recent_df, config_path=config_path, series_ids=series_ids)
            
            # Now update() will apply preprocessing pipeline (standardization only)
            # This matches the training flow: transformed data → standardization
            model.update(y_recent_transformed, history=None)
            logger.info(f"Updated DDFM state with {len(y_recent_transformed)} periods (transformations applied, matching training)")
            logger.info("  Preprocessing pipeline (standardization) applied internally using model.preprocess")
        except Exception as e:
            logger.warning(f"Failed to update DDFM with recent data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Generate forecasts
    # model.predict() returns forecasts in the same scale as training data
    # Since transformations were applied at src/ level (prepare_multivariate_data),
    # we need to manually inverse transform the forecasts (matching previous implementation)
    X_forecast, Z_forecast = model.predict(horizon=horizon, return_series=True, return_factors=True)
    
    # Convert to DataFrame
    if isinstance(X_forecast, np.ndarray):
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
    
    # Manually inverse transform forecasts (matching previous implementation)
    # Transformations were applied at src/ level (prepare_multivariate_data),
    # so we need to inverse transform here to get original scale
    # This matches the previous implementation where transformations were applied at src/ level
    # and manually inverse transformed after prediction
    if y_recent is not None and len(y_recent) > 0:
        from src.utils import get_config_path
        import yaml
        from pathlib import Path
        
        config_path = str(get_config_path())
        series_dir = Path(config_path) / "series"
        
        # Inverse transform all series that have transformations
        for col in forecast_df.columns:
            if col not in y_recent.columns:
                continue
            
            # Get transformation type for this series
            series_config_file = series_dir / f"{col}.yaml"
            trans = 'lin'  # default
            freq = 'm'  # default
            
            if series_config_file.exists():
                try:
                    with open(series_config_file, 'r') as f:
                        series_config = yaml.safe_load(f) or {}
                    trans = str(series_config.get('transformation', 'lin')).lower()
                    freq = str(series_config.get('frequency', 'm')).lower()
                except Exception:
                    pass
            
            # Get last value from y_recent (raw data, before transformations)
            last_value_original = y_recent[col].iloc[-1] if len(y_recent) > 0 else None
            
            if pd.isna(last_value_original):
                continue
            
            # Inverse transform based on transformation type
            if trans in ('chg', 'ch1'):
                # For differencing: cumsum with initial value
                # Forecast is in differenced space, need to cumsum to get back to original
                forecast_df[col] = forecast_df[col].cumsum() + last_value_original
                logger.debug(f"Applied inverse differencing to {col} forecast (last value: {last_value_original:.4f})")
            elif trans == 'log':
                # Inverse log: exp
                # Forecast is in log space, need to exp to get back to original
                forecast_df[col] = np.exp(forecast_df[col])
                logger.debug(f"Applied inverse log transform to {col} forecast")
            elif trans in ('pch', 'pc1'):
                # Inverse percent change: need to reconstruct from base value
                # Forecast is in percent change space, need to reconstruct from last value
                # pct_change = (new - old) / old * 100
                # new = old * (pct_change / 100 + 1)
                forecast_df[col] = (forecast_df[col] / 100 + 1).cumprod() * last_value_original
                logger.debug(f"Applied inverse percent change to {col} forecast (last value: {last_value_original:.4f})")
        
        logger.info("Applied manual inverse transformations to forecasts (matching previous implementation)")
    
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

