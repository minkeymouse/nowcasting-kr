"""Forecasting module for sktime-based models (ARIMA, VAR, TFT, LSTM, Chronos).

This module provides forecasting functionality for models that use sktime
forecaster interface or are implemented in src.models modules.
"""

import logging
from pathlib import Path
from typing import Optional, Any, List, Dict
import pandas as pd

from src.utils import ValidationError
from src.models import (
    load_model_checkpoint,
    forecast_arima,
    forecast_var,
    forecast_tft,
    forecast_lstm,
    forecast_chronos
)

logger = logging.getLogger(__name__)


def forecast_sktime(
    model: Any,
    horizon: int,
    model_type: str,
    last_date: Optional[pd.Timestamp] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Generate forecasts from a trained sktime-based model.
    
    All models now forecast in weekly frequency, then aggregate to monthly.
    
    Parameters
    ----------
    model : Any
        Trained model object (already loaded from checkpoint)
    horizon : int
        Forecast horizon in weeks (88 weeks = 2024 JAN to 2025 OCT)
    model_type : str
        Model type ('arima', 'var', 'tft', 'lstm', 'chronos')
    last_date : Optional[pd.Timestamp]
        Last date from training data (for creating forecast index)
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly frequency, will be aggregated to monthly in evaluation)
    """
    # Generate weekly forecasts based on model type
    # Horizon is already in weeks (88 weeks)
    if model_type == 'arima':
        return forecast_arima(model, horizon, last_date)
    elif model_type == 'var':
        return forecast_var(model, horizon, last_date)
    elif model_type == 'tft':
        return forecast_tft(model, horizon, last_date, metadata)
    elif model_type == 'lstm':
        return forecast_lstm(model, horizon, last_date, metadata)
    elif model_type == 'chronos':
        return forecast_chronos(model, horizon, last_date, metadata)
    else:
        raise ValidationError(f"Unknown sktime model type: {model_type}")


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: Optional[str] = None
) -> pd.DataFrame:
    """Generate forecasts from a trained sktime-based model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon in weeks (88 weeks = 2024 JAN to 2025 OCT)
    model_type : Optional[str]
        Model type (arima, var, tft, lstm, chronos). If None, inferred from checkpoint.
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly frequency)
    """
    # Load checkpoint
    model, metadata = load_model_checkpoint(checkpoint_path)
    
    # Get model type
    if model_type is None:
        model_type = metadata.get('model_type')
        if model_type is None:
            raise ValidationError("Cannot determine model type from checkpoint")
    
    model_type = model_type.lower()
    
    # Validate model type
    if model_type not in ['arima', 'var', 'tft', 'lstm', 'chronos']:
        raise ValidationError(f"Model type '{model_type}' is not a sktime-based model")
    
    # Get last date from metadata
    training_index = metadata.get('training_data_index', [])
    last_date = None
    if training_index:
        try:
            last_date = pd.to_datetime(training_index[-1])
        except:
            pass
    
    # Generate weekly forecasts (horizon converted to weeks internally)
    return forecast_sktime(model, horizon, model_type, last_date, metadata)


def forecast_multiple_horizons(
    checkpoint_path: Path,
    horizons: List[int],
    model_type: Optional[str] = None
) -> Dict[int, pd.DataFrame]:
    """Generate forecasts for multiple horizons from a sktime-based model checkpoint.
    
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
