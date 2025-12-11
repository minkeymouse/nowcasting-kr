"""Forecasting module for dfm-python-based models (DFM, DDFM).

This module provides forecasting functionality for Dynamic Factor Model (DFM)
and Deep Dynamic Factor Model (DDFM) using dfm-python package.
"""

import logging
from pathlib import Path
from typing import Optional, Any, List, Dict
import pandas as pd

from src.utils import ValidationError
from src.models import (
    load_model_checkpoint,
    forecast_dfm,
    forecast_ddfm
)

logger = logging.getLogger(__name__)


def forecast_dfm_python(
    model: Any,
    horizon: int,
    model_type: str,
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate forecasts from a trained dfm-python model.
    
    Parameters
    ----------
    model : Any
        Trained model object (already loaded from checkpoint)
    horizon : int
        Forecast horizon (number of periods ahead)
    model_type : str
        Model type ('dfm' or 'ddfm')
    last_date : Optional[pd.Timestamp]
        Last date from training data (for creating forecast index)
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex
    """
    # Generate forecasts based on model type
    if model_type == 'dfm':
        return forecast_dfm(model, horizon, last_date)
    elif model_type == 'ddfm':
        return forecast_ddfm(model, horizon, last_date)
    else:
        raise ValidationError(f"Unknown dfm-python model type: {model_type}")


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: Optional[str] = None
) -> pd.DataFrame:
    """Generate forecasts from a trained dfm-python model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon in months (will be converted to weeks for weekly models)
    model_type : Optional[str]
        Model type (dfm, ddfm). If None, inferred from checkpoint.
        
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
    if model_type not in ['dfm', 'ddfm']:
        raise ValidationError(f"Model type '{model_type}' is not a dfm-python model")
    
    # Get clock frequency from model config
    clock = getattr(model.config, 'clock', 'w') if hasattr(model, 'config') else 'w'
    
    # Convert horizon from months to weeks for weekly models
    weeks_per_month = 4
    if clock == 'w':
        horizon_weeks = horizon * weeks_per_month
    else:
        horizon_weeks = horizon  # Keep as months for monthly models
    
    # Get last date from metadata
    training_index = metadata.get('training_data_index', [])
    last_date = None
    if training_index:
        try:
            last_date = pd.to_datetime(training_index[-1])
        except:
            pass
    
    # Generate forecasts (horizon is in model's clock frequency)
    # forecast_dfm/forecast_ddfm already return weekly forecasts with DatetimeIndex when clock='w'
    forecast_result = forecast_dfm_python(model, horizon_weeks, model_type, last_date)
    
    # Aggregate weekly forecasts to monthly if needed
    # Note: forecast_dfm/forecast_ddfm already create weekly DatetimeIndex when clock='w'
    if clock == 'w' and isinstance(forecast_result.index, pd.DatetimeIndex):
        from src.utils import aggregate_weekly_to_monthly
        # Verify it's weekly before aggregating
        inferred_freq = pd.infer_freq(forecast_result.index)
        if inferred_freq and ('W' in inferred_freq or inferred_freq.startswith('W')):
            forecast_result = aggregate_weekly_to_monthly(forecast_result, weeks_per_month=weeks_per_month)
    
    return forecast_result


def forecast_multiple_horizons(
    checkpoint_path: Path,
    horizons: List[int],
    model_type: Optional[str] = None
) -> Dict[int, pd.DataFrame]:
    """Generate forecasts for multiple horizons from a dfm-python model checkpoint.
    
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
