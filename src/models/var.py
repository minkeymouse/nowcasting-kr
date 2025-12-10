"""VAR model training and forecasting."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from sktime.forecasting.var import VAR as SktimeVAR
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import ValidationError
from src.preprocessing import (
    prepare_multivariate_data,
    set_dataframe_frequency,
    impute_missing_values,
    apply_scaling
)
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_var(
    data: pd.DataFrame,
    target_series: str,
    config: Dict[str, Any],
    cfg: Any,
    checkpoint_dir: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Train VAR model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (lag_order, trend, auto_lag, etc.)
    cfg : Any
        Full experiment config (for series list)
    checkpoint_dir : Path
        Directory to save checkpoint
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    # Preprocess multivariate data (transformation applied in prepare_multivariate_data)
    y_train, available_series_list = prepare_multivariate_data(
        data, None, cfg, target_series, model_type='var'
    )
    y_train = set_dataframe_frequency(y_train)
    y_train = impute_missing_values(y_train, model_type='var')
    
    if len(y_train) == 0:
        raise ValidationError("VAR: No valid data after imputation.")
    
    if y_train.shape[1] < 2:
        raise ValidationError(f"VAR requires at least 2 series. Found {y_train.shape[1]} series.")
    
    if y_train.isnull().sum().sum() > 0:
        nan_count = int(y_train.isnull().sum().sum())
        raise ValidationError(f"VAR cannot handle missing data. Found {nan_count} NaN values.")
    
    # Get scaler type
    scaler_type = config.get('scaler', 'robust')
    
    # Create scaler transformer for pipeline (will be fitted during pipeline.fit())
    scaler_transformer = None
    scaler = None
    if scaler_type and scaler_type != 'null':
        scaler_type_lower = scaler_type.lower()
        if scaler_type_lower == 'robust':
            sklearn_scaler = RobustScaler()
        elif scaler_type_lower == 'standard':
            sklearn_scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler")
            sklearn_scaler = RobustScaler()
        
        # Wrap sklearn scaler in TabularToSeriesAdaptor for sktime compatibility
        scaler_transformer = TabularToSeriesAdaptor(sklearn_scaler)
        scaler = sklearn_scaler
        logger.info(f"Will apply {scaler_type} scaling in ForecastingPipeline")
    else:
        logger.info("No scaling applied (scaler='null')")
    
    logger.info(f"Training VAR on {y_train.shape[1]} series ({len(y_train)} observations)")
    
    # Get model parameters
    lag_order = config.get('lag_order')
    auto_lag = config.get('auto_lag', {})
    trend = config.get('trend', 'c') or 'c'
    
    # Create base forecaster
    if lag_order is None and auto_lag and auto_lag.get('enabled', False):
        maxlags = auto_lag.get('maxlags', 12) or 12
        ic = auto_lag.get('ic', 'aic') or 'aic'
        base_forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend), ic=str(ic))
    else:
        maxlags = lag_order if lag_order is not None else 1
        base_forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend))
    
    # Create pipeline with scaling (for automatic inverse transform)
    if scaler_transformer is not None:
        forecaster = ForecastingPipeline([
            ('scaler', scaler_transformer),
            ('forecaster', base_forecaster)
        ])
    else:
        forecaster = base_forecaster
    
    # Train (scaler will be fitted automatically during pipeline.fit())
    forecaster.fit(y_train)
    logger.info(f"VAR training completed. Max lags: {maxlags}, Trend: {trend}")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'var',
        'target_series': target_series,
        'lag_order': maxlags,
        'trend': trend,
        'scaler': scaler,
        'scaler_type': scaler_type if scaler else None,
        'series': list(y_train.columns),
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index)
    }
    save_model_checkpoint(forecaster, checkpoint_path, metadata)
    
    return forecaster, metadata


def forecast_var(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate VAR forecasts.
    
    Parameters
    ----------
    model : Any
        Trained VAR forecaster
    horizon : int
        Forecast horizon
    last_date : Optional[pd.Timestamp]
        Last date in training data (for index generation)
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex
    """
    # Generate forecasts
    fh = np.arange(1, horizon + 1)
    forecast = model.predict(fh=fh)
    
    # Create index if needed
    if last_date is not None and isinstance(forecast, pd.DataFrame):
        forecast.index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )
    
    return forecast

