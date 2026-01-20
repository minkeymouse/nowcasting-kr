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


def _get_update_data_source(
    test_data: Optional[pd.DataFrame],
    data_loader: Optional[Any],
    dataset: Any
) -> Optional[pd.DataFrame]:
    """Get processed data for DDFM updates (aligned with training space)."""
    if data_loader is None or not hasattr(data_loader, 'processed') or data_loader.processed is None:
        return test_data
    processed = data_loader.processed.copy()
    date_cols = ['date', 'date_w', 'year', 'month', 'day']
    processed = processed.drop(columns=[c for c in date_cols if c in processed.columns], errors='ignore')
    if test_data is not None and isinstance(test_data.index, pd.DatetimeIndex):
        processed = processed.loc[processed.index.intersection(test_data.index)]
    train_cols = list(dataset.data.columns)
    processed = processed[[c for c in train_cols if c in processed.columns]].copy()
    return processed


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "ddfm",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_model: bool = True
) -> None:
    """Load trained DDFM model and generate forecasts."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    dataset_path = checkpoint_path.parent / "dataset.pkl"
    
    if recursive:
        if test_data is None:
            raise ValueError("test_data is required when recursive=True")
        _run_recursive_forecast(
            checkpoint_path, dataset_path, test_data,
            horizon=horizon, start_date=None, end_date=None,
            target_series=None, data_loader=None
        )
    else:
        _forecast_ddfm(checkpoint_path, dataset_path, horizon)


def _forecast_ddfm(checkpoint_path: Path, dataset_path: Path, horizon: int) -> None:
    """Forecast using DDFM model."""
    from dfm_python import DDFM
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    dataset = joblib.load(dataset_path)
    model = DDFM.load(checkpoint_path, dataset=dataset)
    predictions = model.predict(horizon=horizon)
    _log_and_save_forecast(predictions, checkpoint_path)


def _log_and_save_forecast(predictions: Any, checkpoint_path: Path) -> None:
    """Save forecast results."""
    # Extract forecast array
    X_forecast = predictions[0] if isinstance(predictions, tuple) else predictions
    logger.info(f"Forecast shape: {X_forecast.shape}")
    
    # Save forecasts
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "series_forecast.npy", X_forecast)
    
    if isinstance(predictions, tuple):
        np.save(output_dir / "factor_forecast.npy", predictions[1])


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
    update_data_source = _get_update_data_source(test_data, data_loader, dataset)
    
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
    
    logger.info(f"Running recursive DDFM forecast from {start_ts.date()} to {end_ts.date()} ({len(weekly_dates) - 1} weeks)")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Helper to create dataset with consistent config
    def _create_dataset(data: pd.DataFrame) -> DDFMDataset:
        train_cols = list(dataset.data.columns)
        data_aligned = data[[c for c in train_cols if c in data.columns]].copy()
        return DDFMDataset(
            data=data_aligned,
            time_idx='index',
            target_series=available_targets,
            target_scaler=getattr(dataset, 'target_scaler', None),
            feature_scaler=getattr(dataset, 'feature_scaler', None)
        )
    
    # Initialize with data up to first forecast date
    train_data_initial = update_data_source[update_data_source.index < weekly_dates[0]].copy()
    last_update_date = weekly_dates[0] if len(train_data_initial) == 0 else train_data_initial.index[-1]
    
    if len(train_data_initial) > 0:
        model.update(_create_dataset(train_data_initial))
        last_update_date = train_data_initial.index[-1]
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        # Get new data since last update (including cutoff_date)
        new_data = update_data_source[
            (update_data_source.index > last_update_date) &
            (update_data_source.index <= cutoff_date)
        ].copy()
        
        if len(new_data) > 0:
            model.update(_create_dataset(new_data))
            last_update_date = cutoff_date
        
        forecast_result = model.predict(horizon=1)
        forecast_values = extract_forecast_values(forecast_result, len(available_targets), horizon_idx=0)
        
        predictions.append(forecast_values)
        forecast_dates.append(next_date)
        
        if next_date in test_data_filtered.index:
            actuals.append(test_data_filtered.loc[next_date, available_targets].values)
        else:
            actuals.append(np.full(len(available_targets), np.nan))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    logger.info(f"Recursive forecasting completed: {len(predictions)} forecasts generated")
    
    return predictions, actuals, forecast_dates


def _run_multi_horizon_forecast(
    checkpoint_path: Path,
    dataset_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame],
    target_series: Optional[List[str]],
    data_loader: Optional[Any],
    return_weekly_forecasts: bool = False
) -> Dict[int, Any]:
    """Run multi-horizon forecasting for DDFM."""
    from dfm_python import DDFM, DDFMDataset
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    dataset = joblib.load(dataset_path)
    model = DDFM.load(checkpoint_path, dataset=dataset)
    start_date_ts = pd.Timestamp(start_date)
    max_horizon = max(horizons) if horizons else 1
    update_data_source = _get_update_data_source(test_data, data_loader, dataset)
    available_targets = get_target_series_from_dataset(dataset, test_data, target_series)
    
    # Update factors with data up to start_date
    if update_data_source is not None:
        train_data = update_data_source[update_data_source.index < start_date_ts].copy()
        if len(train_data) > 0:
            train_cols = list(dataset.data.columns)
            train_data = train_data[[c for c in train_cols if c in train_data.columns]].copy()
            update_dataset = DDFMDataset(
                data=train_data,
                time_idx='index',
                target_series=available_targets,
                target_scaler=getattr(dataset, 'target_scaler', None),
                feature_scaler=getattr(dataset, 'feature_scaler', None)
            )
            model.update(update_dataset)
    
    # Predict once with max horizon, then extract time steps
    forecast_result = model.predict(horizon=max_horizon)
    X_forecast = forecast_result[0] if isinstance(forecast_result, tuple) else forecast_result

    # Align forecast columns to available_targets if the forecast is for all series
    X_forecast = np.asarray(X_forecast)
    if hasattr(dataset, "data") and hasattr(dataset.data, "columns"):
        all_cols = list(dataset.data.columns)
        if X_forecast.ndim == 2 and X_forecast.shape[1] == len(all_cols):
            idxs = [all_cols.index(t) for t in available_targets if t in all_cols]
            if idxs:
                X_forecast = X_forecast[:, idxs]
    
    # Extract forecasts for each horizon
    forecasts = {}
    for horizon in horizons:
        horizon_idx = max(0, horizon - 1)
        if return_weekly_forecasts:
            # Return weekly forecasts inside the month containing horizon_date
            horizon_date = start_date_ts + pd.Timedelta(weeks=horizon)
            month_start = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)

            month_weekly_dates: List[pd.Timestamp] = []
            month_forecast_values: List[np.ndarray] = []

            first_week_in_month = month_start
            while first_week_in_month < start_date_ts:
                first_week_in_month += pd.Timedelta(weeks=1)

            current_date = first_week_in_month
            while current_date <= month_end:
                weeks_from_start = (current_date - start_date_ts).days // 7
                if 0 <= weeks_from_start < X_forecast.shape[0]:
                    month_forecast_values.append(X_forecast[weeks_from_start, :])
                    month_weekly_dates.append(current_date)
                current_date += pd.Timedelta(weeks=1)

            if month_forecast_values:
                forecasts[horizon] = {
                    "weekly_forecasts": np.array(month_forecast_values),
                    "dates": pd.DatetimeIndex(month_weekly_dates),
                    "_available_targets": available_targets,
                    "_primary_target": available_targets[0] if available_targets else None,
                }
            else:
                forecasts[horizon] = X_forecast[min(horizon_idx, X_forecast.shape[0] - 1), :]
        else:
            forecasts[horizon] = X_forecast[min(horizon_idx, X_forecast.shape[0] - 1), :]
    
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
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Dict[int, Any]:
    """Run multi-horizon forecasting from fixed start point."""
    return _run_multi_horizon_forecast(
        checkpoint_path, dataset_path, horizons, start_date,
        test_data, target_series, data_loader, return_weekly_forecasts=return_weekly_forecasts
    )
