"""Forecasting module - unified interface for generating forecasts.

This module provides a simple interface to load trained models and generate forecasts.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from omegaconf import DictConfig
from src.utils import ValidationError
from src.models import (
    train_arima, forecast_arima,
    train_var, forecast_var,
    train_dfm, forecast_dfm,
    train_ddfm, forecast_ddfm,
    load_model_checkpoint
)

logger = logging.getLogger(__name__)


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: Optional[str] = None
) -> pd.DataFrame:
    """Generate forecasts from a trained model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon (number of periods ahead)
    model_type : Optional[str]
        Model type (arima, var, dfm, ddfm). If None, inferred from checkpoint.
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex
    """
    # Load checkpoint
    model, metadata = load_model_checkpoint(checkpoint_path)
    
    # Get model type
    if model_type is None:
        model_type = metadata.get('model_type')
        if model_type is None:
            raise ValidationError("Cannot determine model type from checkpoint")
    
    model_type = model_type.lower()
    
    # Get last date from metadata
    training_index = metadata.get('training_data_index', [])
    last_date = None
    if training_index:
        try:
            last_date = pd.to_datetime(training_index[-1])
        except:
            pass
    
    # Generate forecasts based on model type
    if model_type == 'arima':
        return forecast_arima(model, horizon, last_date)
    elif model_type == 'var':
        return forecast_var(model, horizon, last_date)
    elif model_type == 'dfm':
        return forecast_dfm(model, horizon, last_date)
    elif model_type == 'ddfm':
        return forecast_ddfm(model, horizon, last_date)
    else:
        raise ValidationError(f"Unknown model type: {model_type}")


def forecast_multiple_horizons(
    checkpoint_path: Path,
    horizons: List[int],
    model_type: Optional[str] = None
) -> Dict[int, pd.DataFrame]:
    """Generate forecasts for multiple horizons.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizons : List[int]
        List of forecast horizons
    model_type : Optional[str]
        Model type. If None, inferred from checkpoint.
        
    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping horizon to forecast DataFrame
    """
    results = {}
    for horizon in horizons:
        results[horizon] = forecast(checkpoint_path, horizon, model_type)
    return results

