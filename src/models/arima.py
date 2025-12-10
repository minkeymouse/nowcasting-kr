"""ARIMA model training and forecasting."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from sktime.forecasting.arima import ARIMA as SktimeARIMA, AutoARIMA
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import ValidationError
from src.preprocessing import prepare_univariate_data, set_dataframe_frequency, apply_scaling
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_arima(
    data: pd.DataFrame,
    target_series: str,
    config: Dict[str, Any],
    checkpoint_dir: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Train ARIMA model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (order, auto_arima, etc.)
    checkpoint_dir : Path
        Directory to save checkpoint
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    # Preprocess univariate data (transformation applied in prepare_univariate_data)
    y_train = prepare_univariate_data(data, target_series)
    if len(y_train) == 0:
        raise ValidationError(f"No valid monthly data for target series '{target_series}' after resampling")
    
    y_train = set_dataframe_frequency(y_train.to_frame()).iloc[:, 0]
    
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
    
    logger.info(f"Training ARIMA on {target_series} ({len(y_train)} observations)")
    
    # Get model parameters
    order = config.get('order', [1, 1, 1])
    auto_arima = config.get('auto_arima', {})
    
    # Create base forecaster
    if auto_arima and auto_arima.get('enabled', False):
        base_forecaster = AutoARIMA(
            max_p=auto_arima.get('max_p', 5),
            max_d=auto_arima.get('max_d', 2),
            max_q=auto_arima.get('max_q', 5),
            information_criterion=auto_arima.get('information_criterion', 'aic')
        )
    else:
        base_forecaster = SktimeARIMA(order=tuple(order) if isinstance(order, list) else order)
    
    # Create pipeline with imputation and scaling
    pipeline_steps = [
        ('imputer_ffill', Imputer(method="ffill")),
        ('imputer_bfill', Imputer(method="bfill")),
        ('imputer_forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))),
    ]
    
    # Add scaler to pipeline if specified (before forecaster for automatic inverse transform)
    if scaler_transformer is not None:
        pipeline_steps.append(('scaler', scaler_transformer))
    
    pipeline_steps.append(('forecaster', base_forecaster))
    
    forecaster = ForecastingPipeline(pipeline_steps)
    
    # Train (scaler will be fitted automatically during pipeline.fit())
    forecaster.fit(y_train)
    logger.info(f"ARIMA training completed. Order: {order}")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'arima',
        'target_series': target_series,
        'order': order,
        'auto_arima': auto_arima,
        'scaler': scaler,
        'scaler_type': scaler_type if scaler else None,
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index)
    }
    save_model_checkpoint(forecaster, checkpoint_path, metadata)
    
    return forecaster, metadata


def forecast_arima(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate ARIMA forecasts.
    
    Parameters
    ----------
    model : Any
        Trained ARIMA forecaster
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
    if last_date is not None:
        if isinstance(forecast, pd.Series):
            forecast.index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
        elif isinstance(forecast, pd.DataFrame):
            forecast.index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
    
    return forecast if isinstance(forecast, pd.DataFrame) else forecast.to_frame()

