"""Forecasting functions for NeuralForecast models (PatchTST, TFT, iTransformer, TimeMixer)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from src.utils import (
    get_target_series_from_model,
    filter_and_prepare_test_data,
    load_model_checkpoint,
    interpolate_missing_values,
    convert_to_neuralforecast_format,
    extract_neuralforecast_forecasts,
)

logger = logging.getLogger(__name__)


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str,
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_params: bool = False
) -> None:
    """Load trained model and generate forecasts.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon
    model_type : str
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    recursive : bool, default False
        Whether to use recursive forecasting
    test_data : pd.DataFrame, optional
        Test data for recursive forecasting
    update_params : bool, default False
        Whether to update model parameters (retrain) during recursive forecasting
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    logger.info(f"Loading {model_type.upper()} model from: {checkpoint_path}")
    
    if recursive:
        run_recursive_forecast_neuralforecast(
            checkpoint_path, test_data, horizon, model_type, update_params
        )
    else:
        _forecast_neuralforecast_models(checkpoint_path, horizon, model_type)


def run_recursive_forecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    update_params: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Run recursive forecasting experiment with weekly updates.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint
    test_data : pd.DataFrame
        Test data with datetime index
    start_date : str
        Start date for experiment (YYYY-MM-DD)
    end_date : str
        End date for experiment (YYYY-MM-DD)
    model_type : str
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    update_params : bool, default False
        If False, only move cutoff (no retraining). If True, full retraining.
    
    Returns
    -------
    tuple
        (predictions, actuals, dates, target_series)
    """
    return run_recursive_forecast_neuralforecast(
        checkpoint_path, test_data, horizon=1,
        model_type=model_type, update_params=update_params,
        start_date=start_date, end_date=end_date,
        target_series=target_series, data_loader=data_loader
    )


def run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "patchtst",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting from fixed start point.
    
    Uses ONE model (trained with max horizon) to predict all horizons.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint (should be trained with prediction_length >= max(horizons))
    horizons : list
        List of forecast horizons (weeks), e.g. [4, 8, 12, ..., 40]
    start_date : str
        Fixed start date for all forecasts (YYYY-MM-DD)
    test_data : pd.DataFrame, optional
        Test data for extracting actuals
    model_type : str, default "patchtst"
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    return_weekly_forecasts : bool, default False
        If True, return weekly forecasts within each horizon month
    
    Returns
    -------
    tuple
        (horizon_forecasts dict, target_series list)
    """
    return _run_multi_horizon_forecast_neuralforecast(
        checkpoint_path, horizons, start_date, test_data,
        model_type=model_type, target_series=target_series,
        data_loader=data_loader, return_weekly_forecasts=return_weekly_forecasts
    )


def run_recursive_forecast_neuralforecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    horizon: int,
    model_type: str,
    update_params: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Run recursive forecasting for NeuralForecast models.
    
    NeuralForecast uses long format (unique_id, ds, y) and requires rebuilding
    the dataset for each prediction step.
    
    When update_params=True, training data from data_loader is included to provide
    sufficient historical context for retraining, especially important for models
    like iTransformer that use normalization statistics.
    """
    # Load model using helper
    logger.info(f"Loading {model_type.upper()} model for recursive forecasting...")
    nf_model = load_model_checkpoint(checkpoint_path)
    
    # Get available target series first (filter out series not in test_data)
    available_targets = get_target_series_from_model(nf_model, test_data, target_series)
    
    # Convert dates to timestamps
    start_ts = pd.Timestamp(start_date) if start_date else test_data.index.min()
    end_ts = pd.Timestamp(end_date) if end_date else test_data.index.max()
    
    # Filter test data to experiment date range
    test_data_for_forecasts, _, _ = filter_and_prepare_test_data(
        test_data, start_date, end_date, available_targets
    )
    
    # Use full test_data with only target series for model updates
    full_data_for_updates = test_data[available_targets].copy()
    
    # Use actual weekly dates from filtered test data
    weekly_dates = test_data_for_forecasts.index[(test_data_for_forecasts.index >= start_ts) & 
                                                   (test_data_for_forecasts.index <= end_ts)]
    weekly_dates = weekly_dates.sort_values()
    
    if len(weekly_dates) < 2:
        raise ValueError(f"Not enough weekly dates. Found {len(weekly_dates)} dates.")
    
    logger.info(f"Running recursive {model_type.upper()} forecast from {start_ts.date()} to {end_ts.date()} ({len(weekly_dates)} weeks)")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        # Get data up to cutoff from FULL test_data
        train_data_up_to_cutoff = full_data_for_updates[full_data_for_updates.index <= cutoff_date].copy()
        
        # Impute missing values
        train_data_up_to_cutoff = interpolate_missing_values(train_data_up_to_cutoff, data_loader)
        
        # Convert to NeuralForecast long format
        nf_df = convert_to_neuralforecast_format(train_data_up_to_cutoff, available_targets)
        
        # Predict
        forecast_df = nf_model.predict(df=nf_df)
        
        # Extract 1-step-ahead forecasts
        # NeuralForecast models handle scaling/transformation internally, so predictions
        # are already in the correct scale (no inverse transformation needed)
        forecast_values = extract_neuralforecast_forecasts(forecast_df, available_targets, horizon_idx=0)
        predictions.append(forecast_values)
        
        forecast_dates.append(next_date)
        
        # Get actual values
        if next_date in test_data_for_forecasts.index:
            actual_values = test_data_for_forecasts.loc[next_date, available_targets].values
            actuals.append(actual_values)
        else:
            logger.warning(f"No actual data for {next_date.date()}")
            actuals.append(np.full(len(available_targets), np.nan))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    return predictions, actuals, forecast_dates, available_targets


def _forecast_neuralforecast_models(checkpoint_path: Path, horizon: int, model_type: str) -> None:
    """Forecast using NeuralForecast models (PatchTST, TFT, iTransformer, TimeMixer).
    
    Note: NeuralForecast models require data (df) for prediction. This simple mode
    logs that the model is ready for forecasting. Use experiment modes for actual forecasts.
    """
    from src.utils import load_model_checkpoint
    
    logger.info(f"Loading {model_type.upper()} model...")
    nf_model = load_model_checkpoint(checkpoint_path)
    
    logger.info(f"Model loaded successfully. Model is ready for forecasting with horizon={horizon}.")
    logger.info("Note: NeuralForecast models require data (df) for prediction. "
                "Use experiment modes (short_term/long_term) for actual forecasts with data.")
    
    # Save a placeholder to indicate forecast mode was run
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple status file
    status_file = output_dir / "forecast_status.txt"
    with open(status_file, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Horizon: {horizon}\n")
        f.write(f"Status: Model loaded and ready for forecasting\n")
        f.write(f"Note: Use experiment modes for actual forecasts with data\n")
    
    logger.info(f"Forecast status saved to: {output_dir}")


def _run_multi_horizon_forecast_neuralforecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "patchtst",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting from fixed start point for NeuralForecast models."""
    from src.utils import (
        get_target_series_from_model,
        load_model_checkpoint,
        interpolate_missing_values,
        convert_to_neuralforecast_format,
        extract_neuralforecast_forecasts,
    )
    
    # Load model
    logger.info(f"Loading {model_type.upper()} model for multi-horizon forecasting from {start_date}")
    model = load_model_checkpoint(checkpoint_path)
    
    start_date_ts = pd.Timestamp(start_date)
    forecasts = {}
    
    # Get actual target series
    available_targets = None
    if test_data is not None:
        available_targets = get_target_series_from_model(model, test_data, target_series)
    
    # Rebuild dataset for each horizon and predict
    for horizon in horizons:
        if test_data is not None and available_targets:
            train_data = test_data[test_data.index < start_date_ts][available_targets].copy()
            train_data = interpolate_missing_values(train_data, data_loader)
            
            # Convert to NeuralForecast long format and predict
            nf_df = convert_to_neuralforecast_format(train_data, available_targets)
            forecast_df = model.predict(df=nf_df)
            
            if return_weekly_forecasts:
                # Get all weeks in the month containing horizon_date
                horizon_date = start_date_ts + pd.Timedelta(weeks=horizon)
                month_start = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1)
                month_end = month_start + pd.offsets.MonthEnd(0)
                
                month_weekly_dates = []
                month_forecast_values = []
                
                # Find the first week in the month
                first_week_in_month = month_start
                while first_week_in_month < start_date_ts:
                    first_week_in_month += pd.Timedelta(weeks=1)
                
                # Get all weeks from first_week_in_month to month_end
                current_date = first_week_in_month
                while current_date <= month_end:
                    weeks_from_start = (current_date - start_date_ts).days // 7
                    if 0 <= weeks_from_start < max(horizons):
                        # NeuralForecast models handle scaling internally, no inverse transform needed
                        forecast_val = extract_neuralforecast_forecasts(
                            forecast_df, available_targets, horizon_idx=weeks_from_start
                        )
                        month_forecast_values.append(forecast_val)
                        month_weekly_dates.append(current_date)
                    current_date += pd.Timedelta(weeks=1)
                    if weeks_from_start >= max(horizons):
                        break
                
                if month_forecast_values:
                    forecasts[horizon] = {
                        'weekly_forecasts': np.array(month_forecast_values),
                        'dates': pd.DatetimeIndex(month_weekly_dates)
                    }
                else:
                    # Fallback: single forecast
                    # NeuralForecast models handle scaling internally, no inverse transform needed
                    forecast_values = extract_neuralforecast_forecasts(forecast_df, available_targets, horizon_idx=horizon - 1)
                    forecasts[horizon] = forecast_values
            else:
                # Extract forecast (no inverse transform needed - models handle scaling internally)
                forecast_values = extract_neuralforecast_forecasts(forecast_df, available_targets, horizon_idx=horizon - 1)
                forecasts[horizon] = forecast_values
        else:
            logger.warning(f"No test_data available for horizon {horizon}w")
            forecasts[horizon] = np.full(len(available_targets) if available_targets else 1, np.nan)
    
    # Return forecasts and actual target series used
    actual_targets = available_targets if available_targets else target_series
    return forecasts, actual_targets if actual_targets else []
