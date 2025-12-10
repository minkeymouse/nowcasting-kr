"""DDFM model training and forecasting."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.utils import ValidationError
from src.preprocessing import (
    prepare_multivariate_data,
    set_dataframe_frequency,
    create_transformer_from_config
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
    from src.models.models_forecasters import _create_preprocessing_pipeline
    from src.train import _convert_experiment_to_dfm_config_fallback, _create_time_index_from_data
    from src.utils import extract_experiment_params
    
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
    exp_params = extract_experiment_params(cfg)
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
    y_train, available_series_list = prepare_multivariate_data(
        data, config_dict, cfg, target_series, model_type='ddfm'
    )
    y_train = set_dataframe_frequency(y_train)
    
    logger.info(f"Training DDFM: {y_train.shape[1]} series, {len(y_train)} observations")
    logger.info(f"Epochs: {epochs}, Encoder: {encoder_layers}, Factors: {num_factors}")
    
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
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate DDFM forecasts.
    
    Parameters
    ----------
    model : Any
        Trained DDFM model
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

