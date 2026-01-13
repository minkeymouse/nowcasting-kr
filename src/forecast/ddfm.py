"""Forecasting functions for DDFM (Deep Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import joblib
import pandas as pd
import numpy as np

from src.utils import (
    get_target_series_from_dataset,
    filter_and_prepare_test_data,
    extract_forecast_values
)

logger = logging.getLogger(__name__)


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "ddfm",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_model: bool = True
) -> None:
    """Load trained DDFM model and generate forecasts.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to the saved model checkpoint (model.pkl)
    horizon : int
        Forecast horizon (number of periods ahead)
    model_type : str, default "ddfm"
        Model type: 'ddfm'
    recursive : bool, default False
        If True, run recursive forecasting experiment (weekly updates)
    test_data : pd.DataFrame, optional
        Test data for recursive experiments (required if recursive=True)
    update_model : bool, default True
        Whether to update model with new data (always True for DDFM)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    logger.info(f"Loading DDFM model from: {checkpoint_path}")
    
    # Load dataset from the same directory (required for DDFM)
    dataset_path = checkpoint_path.parent / "dataset.pkl"
    
    if recursive:
        _run_recursive_forecast(
            checkpoint_path, dataset_path, test_data,
            horizon=horizon, start_date=None, end_date=None,
            target_series=None, data_loader=None
        )
    else:
        _forecast_ddfm(checkpoint_path, dataset_path, horizon)


def _forecast_ddfm(checkpoint_path: Path, dataset_path: Path, horizon: int) -> None:
    """Forecast using DDFM model."""
    try:
        from dfm_python import DDFM, DDFMDataset
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load dataset first (required for DDFM.load())
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            f"DDFM requires dataset for loading."
        )
    
    logger.info("Loading DDFM dataset...")
    dataset = joblib.load(dataset_path)
    
    # Load model (requires dataset)
    logger.info("Loading DDFM model...")
    model = DDFM.load(checkpoint_path, dataset=dataset)
    
    # Generate forecasts
    logger.info(f"Generating forecasts with horizon={horizon}...")
    
    try:
        predictions = model.predict(horizon=horizon)
        
        # Log and save forecasts
        _log_and_save_forecast(predictions, checkpoint_path)
            
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        raise


def _log_and_save_forecast(predictions: Any, checkpoint_path: Path) -> None:
    """Helper function to log and save forecast results.
    
    Parameters
    ----------
    predictions : Any
        Prediction result (tuple or array)
    checkpoint_path : Path
        Path to checkpoint (parent dir used for output)
    """
    # Extract forecast arrays
    if isinstance(predictions, tuple):
        X_forecast, Z_forecast = predictions
        logger.info(f"Forecast completed:")
        logger.info(f"  Series forecast shape: {X_forecast.shape}")
        logger.info(f"  Factor forecast shape: {Z_forecast.shape}")
        if X_forecast.shape[1] > 0:
            logger.info(f"  First forecast value: {X_forecast[0, 0]:.6f}")
        else:
            logger.info("  No target series")
    else:
        X_forecast = predictions
        logger.info(f"Forecast completed:")
        logger.info(f"  Forecast shape: {X_forecast.shape}")
        if X_forecast.shape[1] > 0 if X_forecast.ndim > 1 else True:
            logger.info(f"  First forecast value: {X_forecast[0, 0]:.6f}" if X_forecast.ndim > 1 else f"  Forecast value: {X_forecast[0]:.6f}")
        else:
            logger.info("  No target series")
    
    # Save forecasts to outputs directory
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(predictions, tuple):
        X_forecast, Z_forecast = predictions
        np.save(output_dir / "series_forecast.npy", X_forecast)
        np.save(output_dir / "factor_forecast.npy", Z_forecast)
    else:
        np.save(output_dir / "series_forecast.npy", X_forecast)
    
    logger.info(f"Forecasts saved to: {output_dir}")


def _run_recursive_forecast(
    checkpoint_path: Path,
    dataset_path: Path,
    test_data: pd.DataFrame,
    horizon: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run recursive forecasting for DDFM model.
    
    For factor models, recursive forecasting means:
    1. Update factors with new observations (via model.update() which uses neural network forward pass)
    2. Predict 1 step ahead from updated factors
    3. Repeat for each time step
    
    Unlike attention-based models, factor models maintain latent factors that capture
    underlying dynamics. The update() method refreshes these factors when new data arrives.
    
    Note: DDFM update() requires a DDFMDataset, so we need to create datasets for new data.
    """
    try:
        from dfm_python import DDFM, DDFMDataset
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info("Loading DDFM dataset and model for recursive forecasting...")
    dataset = joblib.load(dataset_path)
    model = DDFM.load(checkpoint_path, dataset=dataset)
    
    # Get target series using helper
    available_targets = get_target_series_from_dataset(dataset, test_data, target_series)
    
    # Filter and prepare test data using helper
    test_data_filtered, start_ts, end_ts = filter_and_prepare_test_data(
        test_data, start_date, end_date, available_targets
    )
    
    # Use actual weekly dates from test data (handles any day-of-week)
    weekly_dates = test_data_filtered.index[(test_data_filtered.index >= start_ts) & 
                                             (test_data_filtered.index <= end_ts)]
    weekly_dates = weekly_dates.sort_values()
    
    if len(weekly_dates) < 2:
        raise ValueError(f"Not enough weekly dates in test data. Found {len(weekly_dates)} dates.")
    
    logger.info(f"Running recursive DDFM forecast from {start_ts.date()} to {end_ts.date()}")
    logger.info(f"Factor model: updating factors at each step, then predicting 1 step ahead")
    logger.info(f"Total weeks: {len(weekly_dates) - 1}")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Initialize with data up to first forecast date
    train_data_initial = test_data[test_data.index < weekly_dates[0]].copy()
    
    # Update factors with initial data (if any)
    if len(train_data_initial) > 0:
        # Create dataset for initial data (DDFM requires DDFMDataset for update)
        # Use the original dataset's configuration but with new data
        initial_dataset = DDFMDataset(
            data=train_data_initial,
            time_idx='index',
            target_series=available_targets,
            target_scaler=dataset.target_scaler if hasattr(dataset, 'target_scaler') else None
        )
        logger.info(f"Updating factors with initial data ({len(train_data_initial)} rows)...")
        model.update(initial_dataset)
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        logger.info(f"Week {i+1}/{len(weekly_dates)-1}: Updating factors, then forecasting {next_date.date()}")
        
        # Get new data since last update
        new_data = test_data[
            (test_data.index >= cutoff_date) & 
            (test_data.index < next_date)
        ].copy()
        
        if len(new_data) > 0:
            # Create dataset for new data (DDFM requires DDFMDataset for update)
            new_dataset = DDFMDataset(
                data=new_data,
                time_idx='index',
                target_series=available_targets,
                target_scaler=dataset.target_scaler if hasattr(dataset, 'target_scaler') else None
            )
            
            # Update factors with new observations (neural network forward pass updates factors)
            # This is the key difference from attention models: we update latent factors
            model.update(new_dataset)
            logger.debug(f"Factors updated with {len(new_data)} new observations")
        
        # Predict 1 step ahead from updated factors
        # Factor models are structural: factors evolve via AR dynamics, forecast is deterministic
        forecast_result = model.predict(horizon=1)  # 1 step ahead from current factor state
        forecast_values = extract_forecast_values(
            forecast_result, len(available_targets), horizon_idx=0
        )
        
        predictions.append(forecast_values)
        forecast_dates.append(next_date)
        
        # Get actual values for comparison
        if next_date in test_data_filtered.index:
            actual_values = test_data_filtered.loc[next_date, available_targets].values
            actuals.append(actual_values)
        else:
            logger.warning(f"No actual data for {next_date.date()}, using NaN")
            actuals.append(np.full(len(available_targets), np.nan))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    logger.info(f"Recursive forecasting completed: {len(predictions)} forecasts generated")
    logger.info(f"Factor model approach: factors updated at each step, structural AR dynamics used for forecasting")
    
    return predictions, actuals, forecast_dates


def _run_multi_horizon_forecast(
    checkpoint_path: Path,
    dataset_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame],
    target_series: Optional[List[str]],
    data_loader: Optional[Any]
) -> Dict[int, np.ndarray]:
    """Run multi-horizon forecasting for DDFM.
    
    For structural factor models, multi-horizon forecasting works differently:
    1. Update factors once with data up to start_date
    2. Predict once with max(horizons) - the model structure naturally handles all horizons
    3. Extract different time steps from the single forecast (structural AR dynamics)
    
    Unlike attention-based models, factor models don't need separate training for
    different horizons. The structural state-space model defines the forecast dynamics.
    """
    try:
        from dfm_python import DDFM, DDFMDataset
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info("Loading DDFM model for multi-horizon forecasting...")
    dataset = joblib.load(dataset_path)
    model = DDFM.load(checkpoint_path, dataset=dataset)
    
    start_date_ts = pd.Timestamp(start_date)
    max_horizon = max(horizons) if horizons else 1
    
    logger.info(f"Multi-horizon forecasting from {start_date_ts.date()}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Factor model: updating factors once, then extracting all horizons from structural forecast")
    
    forecasts = {}
    
    # Update factors once with all data up to start_date
    # This is the key: factor models update their latent state, then the structure
    # naturally defines forecasts at all horizons via AR dynamics
    if test_data is not None:
        train_data = test_data[test_data.index < start_date_ts].copy()
        if len(train_data) > 0:
            # Get target series for dataset creation
            available_targets = get_target_series_from_dataset(dataset, test_data, target_series)
            
            # Create dataset for update (DDFM requires DDFMDataset)
            update_dataset = DDFMDataset(
                data=train_data,
                time_idx='index',
                target_series=available_targets,
                target_scaler=dataset.target_scaler if hasattr(dataset, 'target_scaler') else None
            )
            
            # Update factors with data up to start_date
            # This updates the latent factor state via neural network forward pass
            model.update(update_dataset)
            logger.info(f"Factors updated with {len(train_data)} observations up to {start_date_ts.date()}")
    
    # Determine number of series
    available_targets = get_target_series_from_dataset(dataset, test_data, target_series)
    n_series = len(available_targets)
    
    # For structural factor models: predict once with max horizon, then extract time steps
    # The model structure (AR dynamics of factors) naturally defines all horizons
    logger.info(f"Predicting up to {max_horizon} weeks ahead (structural forecast)...")
    forecast_result = model.predict(horizon=max_horizon)
    
    # Extract the forecast array
    if isinstance(forecast_result, tuple):
        X_forecast, _ = forecast_result
    else:
        X_forecast = forecast_result
    
    # Extract forecasts for each requested horizon from the single structural forecast
    # Factor models have deterministic AR dynamics, so extracting time steps is valid
    for horizon in horizons:
        logger.debug(f"Extracting forecast for horizon: {horizon} weeks")
        
        # Extract forecast at the specified horizon (0-indexed, so horizon-1)
        # For factor models, horizon represents "steps ahead", so index is horizon-1
        horizon_idx = horizon - 1 if horizon > 0 else 0
        if horizon_idx < X_forecast.shape[0]:
            forecast_values = X_forecast[horizon_idx, :]
        else:
            # If horizon exceeds forecast length, use last available
            logger.warning(f"Horizon {horizon} exceeds forecast length {X_forecast.shape[0]}, using last step")
            forecast_values = X_forecast[-1, :]
        
        forecasts[horizon] = forecast_values
        logger.debug(f"Horizon {horizon}w: extracted from structural forecast (time step {horizon_idx})")
    
    logger.info(f"Multi-horizon forecasting completed: extracted {len(forecasts)} horizons from structural forecast")
    
    return forecasts


def run_recursive_forecast(
    checkpoint_path: Path,
    dataset_path: Path,
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str = "ddfm",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run recursive forecasting experiment with weekly updates."""
    return _run_recursive_forecast(
        checkpoint_path, dataset_path, test_data,
        horizon=1, start_date=start_date, end_date=end_date,
        target_series=target_series, data_loader=data_loader
    )


def run_multi_horizon_forecast(
    checkpoint_path: Path,
    dataset_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "ddfm",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Dict[int, np.ndarray]:
    """Run multi-horizon forecasting from fixed start point."""
    return _run_multi_horizon_forecast(
        checkpoint_path, dataset_path, horizons, start_date,
        test_data, target_series, data_loader
    )
