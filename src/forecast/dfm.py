"""Forecasting functions for DFM (Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import joblib
import pandas as pd
import numpy as np

from src.utils import (
    get_target_series_from_dataset,
    filter_and_prepare_test_data,
    preprocess_data_for_model,
    extract_forecast_values
)

logger = logging.getLogger(__name__)


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "dfm",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_model: bool = True
) -> None:
    """Load trained DFM model and generate forecasts.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to the saved model checkpoint (model.pkl)
    horizon : int
        Forecast horizon (number of periods ahead)
    model_type : str, default "dfm"
        Model type: 'dfm'
    recursive : bool, default False
        If True, run recursive forecasting experiment (weekly updates)
    test_data : pd.DataFrame, optional
        Test data for recursive experiments (required if recursive=True)
    update_model : bool, default True
        Whether to update model with new data (always True for DFM)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    logger.info(f"Loading DFM model from: {checkpoint_path}")
    
    # Load dataset from the same directory (needed for target_series info)
    dataset_path = checkpoint_path.parent / "dataset.pkl"
    
    if recursive:
        _run_recursive_forecast(
            checkpoint_path, dataset_path, test_data, 
            horizon=horizon, start_date=None, end_date=None,
            target_series=None, data_loader=None
        )
    else:
        _forecast_dfm(checkpoint_path, dataset_path, horizon)


def _forecast_dfm(checkpoint_path: Path, dataset_path: Path, horizon: int) -> None:
    """Forecast using DFM model."""
    try:
        from dfm_python import DFM
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load model
    logger.info("Loading DFM model...")
    model = DFM.load(checkpoint_path)
    
    # Load dataset if available (for target_series info)
    dataset = None
    if dataset_path.exists():
        try:
            dataset = joblib.load(dataset_path)
            logger.info(f"Dataset loaded from: {dataset_path}")
            
            # Try to attach dataset to model if needed
            # Some models may need this for predict() to access target_series
            if hasattr(model, '_dataset') or hasattr(model, 'dataset'):
                if hasattr(model, '_dataset'):
                    model._dataset = dataset
                elif hasattr(model, 'dataset'):
                    model.dataset = dataset
                logger.info("Dataset attached to model")
        except Exception as e:
            logger.warning(f"Could not load dataset: {e}. Model may have target_series stored internally.")

    # Generate forecasts
    logger.info(f"Generating forecasts with horizon={horizon}...")
    
    try:
        # Generate predictions
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
    """Run recursive forecasting for DFM model.
    
    For factor models, recursive forecasting means:
    1. Update factors with new observations (via model.update() which uses Kalman filtering)
    2. Predict 1 step ahead from updated factors
    3. Repeat for each time step
    
    Unlike attention-based models, factor models maintain latent factors that capture
    underlying dynamics. The update() method refreshes these factors when new data arrives.
    """
    try:
        from dfm_python import DFM
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load model and dataset
    logger.info("Loading DFM model for recursive forecasting...")
    model = DFM.load(checkpoint_path)
    dataset = joblib.load(dataset_path) if dataset_path.exists() else None
    
    # region agent log
    try:
        import json, time
        # Check if target_scaler was loaded
        has_target_scaler = False
        target_scaler_source = None
        if hasattr(model, '_result') and model._result is not None:
            has_target_scaler = (model._result.target_scaler is not None)
            target_scaler_source = "result"
        elif hasattr(model, 'target_scaler'):
            has_target_scaler = (model.target_scaler is not None)
            target_scaler_source = "model"
        
        with open("/data/nowcasting-kr/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "dfm-forecast",
                "hypothesisId": "H3_target_scaler_loaded",
                "location": "src/forecast/dfm.py:model_loaded",
                "message": "Checking if target_scaler was loaded from checkpoint",
                "data": {
                    "has_target_scaler": has_target_scaler,
                    "target_scaler_source": target_scaler_source,
                    "model_has_result": hasattr(model, '_result') and model._result is not None,
                },
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion
    
    # Get target series from config first (this is what the model was trained with)
    config_target_series = None
    if hasattr(model, '_config') and model._config is not None:
        config = model._config
        if hasattr(config, 'target_series') and config.target_series:
            config_target_series = config.target_series if isinstance(config.target_series, list) else [config.target_series]
            logger.info(f"Found target_series in model config: {config_target_series}")
    
    # Get target series using helper (use full test_data before filtering)
    available_targets = get_target_series_from_dataset(dataset, test_data, target_series)
    
    # Prefer config target_series if available, otherwise use available_targets
    if config_target_series:
        # Verify config target_series are in test_data
        config_targets_in_data = [t for t in config_target_series if t in test_data.columns]
        if config_targets_in_data:
            available_targets = config_targets_in_data
            logger.info(f"Using target_series from model config: {available_targets}")
        else:
            logger.warning(f"Config target_series {config_target_series} not in test_data, using {available_targets}")
    
    # Set target_series on dataset (needed for predict() to resolve indices correctly)
    if dataset is not None:
        dataset.target_series = available_targets
        logger.info(f"Set target_series on dataset: {available_targets}")
    
    # Attach dataset to model if available (needed for predict() to find target_series)
    if dataset is not None and hasattr(model, '_dataset'):
        model._dataset = dataset

    # Convert date strings to timestamps
    start_ts = pd.Timestamp(start_date) if start_date else test_data.index.min()
    end_ts = pd.Timestamp(end_date) if end_date else test_data.index.max()
    
    # Filter to experiment date range for forecasting
    test_data_filtered = test_data[
        (test_data.index >= start_ts) & 
        (test_data.index <= end_ts)
    ].copy()
    
    # Use actual weekly dates from test data (handles any day-of-week)
    weekly_dates = test_data_filtered.index.sort_values()
    
    if len(weekly_dates) < 2:
        raise ValueError(f"Not enough weekly dates in test data. Found {len(weekly_dates)} dates.")
    
    logger.info(f"Running recursive forecast from {start_ts.date()} to {end_ts.date()}")
    logger.info(f"Factor model: updating factors at each step, then predicting 1 step ahead")
    logger.info(f"Total weeks: {len(weekly_dates) - 1} (forecasting {len(weekly_dates) - 1} weeks)")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Initialize with data up to first forecast date (use full test_data for factor updates)
    train_data_initial = test_data[test_data.index < weekly_dates[0]].copy()
    
    # Preprocess initial training data (use all series for factor model, not just targets)
    # Note: DFM can handle NaNs in mixed-frequency data, preserve them for tent kernel
    train_data_standardized = preprocess_data_for_model(
        train_data_initial, data_loader, columns=None, impute_missing=False  # DFM handles NaNs
    )
    
    # Track last update date for incremental updates
    last_update_date = weekly_dates[0] if len(train_data_standardized) == 0 else pd.Timestamp(train_data_initial.index[-1])
    
    # Update factors with initial training data
    # This updates the latent factors via Kalman filtering, keeping model parameters fixed
    if len(train_data_standardized) > 0:
        logger.info(f"Updating factors with initial data ({len(train_data_standardized)} rows)...")
        model.update(train_data_standardized)
        last_update_date = pd.Timestamp(train_data_initial.index[-1])
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        logger.info(f"Week {i+1}/{len(weekly_dates)-1}: Updating factors, then forecasting {next_date.date()}")
        
        # IMPORTANT: Synchronize with NeuralForecast models - get NEW data since last update,
        # INCLUDING cutoff_date. This ensures we use the latest available data (cutoff_date)
        # to update factors before predicting next_date, matching attention-based models.
        # DFM's update() extends factors incrementally, so we only pass new data.
        new_data = test_data[
            (test_data.index > last_update_date) & 
            (test_data.index <= cutoff_date)
        ].copy()
        
        if len(new_data) > 0:
            # Preprocess new data (use all series for factor model)
            # Note: DFM can handle NaNs in mixed-frequency data, preserve them
            new_data_standardized = preprocess_data_for_model(
                new_data, data_loader, columns=None, impute_missing=False  # DFM handles NaNs
            )

            # Update factors with new observations (Kalman filtering updates factor state)
            # This includes cutoff_date data, synchronized with attention-based models
            model.update(new_data_standardized)
            logger.debug(f"Factors updated with {len(new_data_standardized)} new observations (including {cutoff_date.date()})")
            last_update_date = cutoff_date
        
        # Predict 1 step ahead from updated factors
        # Factor models are structural: factors evolve via AR dynamics, forecast is deterministic
        forecast_result = model.predict(horizon=1)  # 1 step ahead from current factor state
        forecast_values = extract_forecast_values(
            forecast_result, len(available_targets), horizon_idx=0
        )
        
        # region agent log
        try:
            import json, time
            # Check target_scaler status and get its parameters
            has_target_scaler = False
            scaler_mean = None
            scaler_scale = None
            if hasattr(model, '_result') and model._result is not None:
                if model._result.target_scaler is not None:
                    has_target_scaler = True
                    scaler_mean = model._result.target_scaler.mean_.tolist() if hasattr(model._result.target_scaler, 'mean_') else None
                    scaler_scale = model._result.target_scaler.scale_.tolist() if hasattr(model._result.target_scaler, 'scale_') else None
            elif hasattr(model, 'target_scaler') and model.target_scaler is not None:
                has_target_scaler = True
                scaler_mean = model.target_scaler.mean_.tolist() if hasattr(model.target_scaler, 'mean_') else None
                scaler_scale = model.target_scaler.scale_.tolist() if hasattr(model.target_scaler, 'scale_') else None
            
            with open("/data/nowcasting-kr/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "dfm-forecast",
                    "hypothesisId": "H4_predict_uses_scaler",
                    "location": "src/forecast/dfm.py:after_predict",
                    "message": "Prediction values after model.predict()",
                    "data": {
                        "has_target_scaler": has_target_scaler,
                        "scaler_mean": scaler_mean,
                        "scaler_scale": scaler_scale,
                        "forecast_values_shape": forecast_values.shape if hasattr(forecast_values, 'shape') else None,
                        "forecast_values_mean": float(np.mean(forecast_values)) if isinstance(forecast_values, np.ndarray) else None,
                        "forecast_values_std": float(np.std(forecast_values)) if isinstance(forecast_values, np.ndarray) else None,
                        "forecast_values_min": float(np.min(forecast_values)) if isinstance(forecast_values, np.ndarray) else None,
                        "forecast_values_max": float(np.max(forecast_values)) if isinstance(forecast_values, np.ndarray) else None,
                        "week": i+1,
                    },
                    "timestamp": int(time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # endregion
        
        predictions.append(forecast_values)
        forecast_dates.append(next_date)
        
        # Get actual values for comparison
        if next_date in test_data_filtered.index:
            actual_values = test_data_filtered.loc[next_date, available_targets].values
            actuals.append(actual_values)
        else:
            logger.warning(f"No actual data for {next_date.date()}, using NaN")
            actuals.append(np.full(len(available_targets), np.nan))
        
    
    # Convert to numpy arrays
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
    """Run multi-horizon forecasting for DFM.
    
    For structural factor models, multi-horizon forecasting works differently:
    1. Update factors once with data up to start_date
    2. Predict once with max(horizons) - the model structure naturally handles all horizons
    3. Extract different time steps from the single forecast (structural AR dynamics)
    
    Unlike attention-based models, factor models don't need separate training for
    different horizons. The structural state-space model defines the forecast dynamics.
    """
    try:
        from dfm_python import DFM
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load model
    logger.info("Loading DFM model for multi-horizon forecasting...")
    model = DFM.load(checkpoint_path)
    dataset = joblib.load(dataset_path) if dataset_path.exists() else None
    
    # Ensure target_series is set on dataset if provided
    if dataset is not None:
        if target_series is None:
            if hasattr(dataset, 'target_series'):
                target_series = dataset.target_series
        else:
            # Set target_series on dataset if it's missing or empty
            if not hasattr(dataset, 'target_series') or dataset.target_series is None or len(dataset.target_series) == 0:
                dataset.target_series = target_series
                logger.info(f"Set target_series on dataset: {target_series}")
        
        # Attach dataset to model for predict() to access target_series
        if hasattr(model, '_dataset'):
            model._dataset = dataset
        elif hasattr(model, 'dataset'):
            model.dataset = dataset
    
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
        # Use all series for factor model (factors capture common dynamics)
        if len(train_data) > 0:
            train_data_standardized = preprocess_data_for_model(
                train_data, data_loader, columns=None, impute_missing=False  # DFM handles NaNs
            )
            # Update factors with data up to start_date
            # This updates the latent factor state via Kalman filtering
            model.update(train_data_standardized)
            logger.info(f"Factors updated with {len(train_data_standardized)} observations up to {start_date_ts.date()}")
    
    # Determine number of series for forecast extraction
    if dataset is not None and hasattr(dataset, 'target_series'):
        target_list = dataset.target_series
        n_series = len(target_list) if isinstance(target_list, list) else 1
    elif test_data is not None:
        train_data_check = test_data[test_data.index < start_date_ts].copy()
        if len(train_data_check) > 0:
            if target_series:
                available_targets = [t for t in target_series if t in train_data_check.columns]
                n_series = len(available_targets) if available_targets else len(train_data_check.columns)
            else:
                n_series = len(train_data_check.columns)
        else:
            n_series = 1
    else:
        n_series = 1
    
    # For structural factor models: predict once with max horizon, then extract time steps
    # The model structure (AR dynamics of factors) naturally defines all horizons
    logger.info(f"Predicting up to {max_horizon} weeks ahead (structural forecast)...")
    forecast_result = model.predict(horizon=max_horizon)
    
    # Extract forecasts for each requested horizon from the single structural forecast
    # Factor models have deterministic AR dynamics, so extracting time steps is valid
    for horizon in horizons:
        logger.debug(f"Extracting forecast for horizon: {horizon} weeks")
        
        # Extract forecast at the specified horizon (0-indexed, so horizon-1)
        # For factor models, horizon represents "steps ahead", so index is horizon-1
        horizon_idx = horizon - 1 if horizon > 0 else 0
        forecast_values = extract_forecast_values(
            forecast_result, n_series, horizon_idx=horizon_idx
        )
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
    model_type: str = "dfm",
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
    model_type: str = "dfm",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Dict[int, np.ndarray]:
    """Run multi-horizon forecasting from fixed start point."""
    return _run_multi_horizon_forecast(
        checkpoint_path, dataset_path, horizons, start_date,
        test_data, target_series, data_loader
    )
