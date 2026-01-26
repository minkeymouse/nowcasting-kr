"""Forecasting functions for DDFM (Deep Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
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
    update_model: bool = True,
    data_loader: Optional[Any] = None,
    window_size: int = 52
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
    data_loader : Any, optional
        Data loader to get latest processed data for context. If provided,
        uses latest data to initialize factor state before forecasting.
        Latest data must match training data format exactly (same columns, same order).
    window_size : int, default 52
        Number of weeks of latest data to use for forecast initialization.
        Only the most recent window_size weeks will be used (rolling window).
        Data must be in the same format and column order as training data.
    
    Raises
    ------
    ValueError
        If training column order cannot be determined, or if latest data format/columns
        don't match training data exactly (column count, column order, or data types).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if recursive:
        if test_data is None:
            raise ValueError("test_data is required when recursive=True")
        _run_recursive_forecast(
            checkpoint_path, test_data,
            horizon=horizon, start_date=None, end_date=None,
            covariates=None, data_loader=data_loader
        )
    else:
        _forecast_ddfm(checkpoint_path, horizon, data_loader=data_loader, window_size=window_size)


def _get_ddfm_training_column_order(
    model: Any,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[List[str]]:
    """Get training column order from DDFM model.
    
    Parameters
    ----------
    model : Any
        Trained DDFM model
    logger_instance : logging.Logger, optional
        Logger instance for warnings
    
    Returns
    -------
    List[str] or None
        Training column order if it can be determined
    """
    if logger_instance is None:
        logger_instance = logger
    
    training_column_order = None
    
    # Try to get from scaler feature_names_in_ (preferred)
    if hasattr(model, 'scaler') and model.scaler is not None:
        if hasattr(model.scaler, 'feature_names_in_') and model.scaler.feature_names_in_ is not None:
            training_column_order = list(model.scaler.feature_names_in_)
            logger_instance.debug(f"DDFM training column order from scaler.feature_names_in_: {len(training_column_order)} columns")
    
    # Fallback: try to get from model's dataset
    if training_column_order is None and hasattr(model, '_dataset') and model._dataset is not None:
        dataset = model._dataset
        if hasattr(dataset, 'data') and dataset.data is not None:
            training_column_order = list(dataset.data.columns)
            logger_instance.debug(f"DDFM training column order from dataset.data.columns: {len(training_column_order)} columns")
    
    # Fallback: try to get from checkpoint metadata
    if training_column_order is None and hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
        dataset_metadata = model._checkpoint_metadata.get('dataset_metadata')
        if dataset_metadata and dataset_metadata.get('_processed_columns'):
            training_column_order = [c for c in dataset_metadata['_processed_columns'] 
                                    if c not in {'date', 'date_w', 'year', 'month', 'day'}]
            logger_instance.debug(f"DDFM training column order from checkpoint metadata: {len(training_column_order)} columns")
    
    if training_column_order is None:
        logger_instance.warning("Could not determine DDFM training column order. Column order validation will be skipped.")
    
    return training_column_order


def _validate_ddfm_data_format(
    data: pd.DataFrame,
    training_column_order: List[str],
    model: Any,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """Validate that data format matches training data exactly.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    training_column_order : List[str]
        Expected column order from training
    model : Any
        DDFM model for additional validation
    logger_instance : logging.Logger, optional
        Logger instance for errors
    
    Raises
    ------
    ValueError
        If data format doesn't match training data exactly
    """
    if logger_instance is None:
        logger_instance = logger
    
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    training_numeric_cols = [c for c in training_column_order if c not in date_cols]
    data_cols = [c for c in data.columns if c not in date_cols]
    
    # Validate all required columns are present
    missing_cols = [c for c in training_numeric_cols if c not in data_cols]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in latest data: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}.\n"
            f"Latest data must contain all {len(training_numeric_cols)} columns from training data.\n"
            f"Found {len(data_cols)} columns, expected {len(training_numeric_cols)}.\n"
            f"Expected columns (first 10): {training_numeric_cols[:10]}"
        )
    
    # Validate column count matches
    if len(data_cols) != len(training_numeric_cols):
        raise ValueError(
            f"Column count mismatch: latest data has {len(data_cols)} columns, "
            f"but training data has {len(training_numeric_cols)} columns.\n"
            f"All training columns must be present in the same order."
        )
    
    # Validate column order matches exactly
    if data_cols != training_numeric_cols:
        # Find first mismatch
        first_mismatch = next((i for i, (expected, actual) in enumerate(zip(training_numeric_cols, data_cols)) if expected != actual), None)
        if first_mismatch is not None:
            raise ValueError(
                f"Column order mismatch at index {first_mismatch}: "
                f"expected '{training_numeric_cols[first_mismatch]}', got '{data_cols[first_mismatch]}'.\n"
                f"Latest data columns must be in the exact same order as training data.\n"
                f"Expected order (first 5): {training_numeric_cols[:5]}, "
                f"Got order (first 5): {data_cols[:5]}"
            )
    
    # Validate all columns are numeric
    non_numeric_cols = [col for col in training_numeric_cols if col in data.columns and not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(
            f"Non-numeric columns found: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}.\n"
            f"All columns must be numeric to match training data format."
        )
    
    # Validate against model's dataset if available
    if hasattr(model, '_dataset') and model._dataset is not None:
        train_dataset = model._dataset
        if hasattr(train_dataset, 'data') and train_dataset.data is not None:
            train_cols = list(train_dataset.data.columns)
            if train_cols != training_column_order:
                logger_instance.warning(
                    f"Model dataset columns ({len(train_cols)}) don't match training column order ({len(training_column_order)}). "
                    f"Using training column order for validation."
                )


def _forecast_ddfm(checkpoint_path: Path, horizon: int, data_loader: Optional[Any] = None, window_size: int = 52) -> None:
    """Forecast using DDFM model.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon
    data_loader : Any, optional
        Data loader to get latest processed data for context. If provided,
        uses latest data to initialize factor state before forecasting.
        Latest data must match training data format exactly (same columns, same order).
    window_size : int, default 52
        Number of weeks of latest data to use for forecast initialization.
        Only the most recent window_size weeks will be used (rolling window).
        Data must be in the same format and column order as training data.
    
    Raises
    ------
    ValueError
        If training column order cannot be determined, or if latest data format/columns
        don't match training data exactly (column count, column order, or data types).
    """
    from dfm_python import DDFM, DDFMDataset
    
    # Dataset metadata is now stored in model.pkl, so dataset.pkl is no longer needed
    logger.info("Loading DDFM model...")
    model = DDFM.load(checkpoint_path, dataset=None)  # None = create from metadata
    
    # Get training column order for data validation (REQUIRED)
    training_column_order = _get_ddfm_training_column_order(model, logger)
    if training_column_order is None:
        raise ValueError(
            "Cannot determine training column order from DDFM model. "
            "This is required to ensure data format matches training exactly. "
            "Model may be corrupted or was saved without column order metadata."
        )
    
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    training_numeric_cols = [c for c in training_column_order if c not in date_cols]
    
    # Prepare latest data if data_loader is provided
    latest_data_dataset = None
    if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
        processed_data = data_loader.processed.copy()
        
        # Validate data format matches training exactly (REQUIRED - must match exactly)
        _validate_ddfm_data_format(processed_data, training_column_order, model, logger)
        
        # Use a rolling window of recent data (not all data)
        if len(processed_data) < window_size:
            logger.warning(
                f"Available data ({len(processed_data)} weeks) is less than requested window size ({window_size} weeks). "
                f"Using all available data."
            )
            actual_window_size = len(processed_data)
        else:
            actual_window_size = window_size
        
        update_data_df = processed_data.iloc[-actual_window_size:].copy()
        
        # Ensure columns are in training order (validation already confirmed all columns exist)
        update_data_df = update_data_df[training_column_order]
        
        # Get dataset from model to create compatible DDFMDataset
        dataset = model._dataset
        if dataset is None:
            raise ValueError(
                "Cannot create DDFMDataset: model._dataset is None. "
                "Cannot determine dataset configuration for latest data."
            )
        
        # Create DDFMDataset with same configuration as training
        covariates = getattr(dataset, 'covariates', None) or []
        available_covariates = [c for c in covariates if c in update_data_df.columns]
        
        latest_data_dataset = DDFMDataset(
            data=update_data_df,
            time_idx='index',
            covariates=available_covariates if available_covariates else None,
            scaler=dataset.scaler,
            feature_scaler=getattr(dataset, 'feature_scaler', None)
        )
        
        logger.info(f"Using latest {actual_window_size} weeks of data for forecast initialization")
        logger.info(f"Latest data date range: {update_data_df.index.min()} to {update_data_df.index.max()}")
        logger.info(f"Latest data shape: {update_data_df.shape} (matches training format: {actual_window_size} time steps × {len(training_numeric_cols)} features)")
    else:
        if data_loader is None:
            logger.info("No data_loader provided, using training state for forecast")
        else:
            raise ValueError(
                "Data loader provided but has no 'processed' attribute. "
                "Latest data must be in the same processed format as training data."
            )
    
    # Generate forecasts
    logger.info(f"Generating forecasts with horizon={horizon}...")
    
    try:
        # Generate predictions (with latest data if available, update=True by default)
        if latest_data_dataset is not None:
            predictions = model.predict(horizon=horizon, data=latest_data_dataset, update=True)
        else:
            predictions = model.predict(horizon=horizon)
        
        # Log and save forecasts
        _log_and_save_forecast(predictions, checkpoint_path)
            
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        raise


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
    test_data: pd.DataFrame,
    horizon: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run recursive forecasting for DDFM model.
    
    For factor models, recursive forecasting means:
    1. Update factors with new observations and predict (via model.predict(update=True, data=...) which uses neural network forward pass)
    2. Repeat for each time step
    
    Unlike attention-based models, factor models maintain latent factors that capture
    underlying dynamics. The update() method refreshes these factors when new data arrives.
    
    Note: DDFM update() requires a DDFMDataset, so we need to create datasets for new data.
    """
    try:
        from dfm_python import DDFM, DDFMDataset
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Dataset metadata is now stored in model.pkl, so dataset.pkl is no longer needed
    # DDFM.load() can create a minimal dataset from checkpoint metadata
    logger.info("Loading DDFM model for recursive forecasting...")
    model = DDFM.load(checkpoint_path, dataset=None)  # None = create from metadata
    
    # Get dataset from model (created from metadata) for update_data_source
    dataset = model._dataset
    
    # Ensure model has scaler (should be set from checkpoint, but verify)
    if model.scaler is None and dataset is not None and hasattr(dataset, 'scaler'):
        model.scaler = dataset.scaler
        logger.info("Set model.scaler from dataset")
    elif model.scaler is None:
        logger.warning("Model scaler is None - forecasts may not be properly scaled")
    
    update_data_source = _get_update_data_source(test_data, data_loader, dataset)
    
    
    # Compute available_targets from covariates (or use all series if no covariates)
    if covariates:
        available_covariates = [c for c in covariates if c in test_data.columns]
        all_series = list(test_data.columns)
        available_targets = [s for s in all_series if s not in available_covariates]
        logger.info(f"Using {len(available_targets)} target series (excluded {len(available_covariates)} covariates)")
    else:
        # No covariates: use helper to get targets (or all series)
        available_targets = get_target_series_from_dataset(dataset, test_data, None)
        if available_targets is None or len(available_targets) == 0:
            available_targets = list(test_data.columns)
        logger.info(f"No covariates specified, using {len(available_targets)} target series")
    
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
    
    # Helper to create dataset with consistent config and validation
    def _create_dataset(data: pd.DataFrame) -> DDFMDataset:
        train_cols = list(dataset.data.columns)
        
        # Validate all training columns are present
        missing_cols = [c for c in train_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in update data: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}.\n"
                f"Update data must contain all {len(train_cols)} columns from training data.\n"
                f"Found {len([c for c in train_cols if c in data.columns])} columns, expected {len(train_cols)}."
            )
        
        # Align columns to training order (CRITICAL for correct scaling)
        data_aligned = data[train_cols].copy()
        
        # Validate column order matches exactly
        if list(data_aligned.columns) != train_cols:
            first_mismatch = next((i for i, (expected, actual) in enumerate(zip(train_cols, data_aligned.columns)) if expected != actual), None)
            if first_mismatch is not None:
                raise ValueError(
                    f"Column order mismatch at index {first_mismatch}: "
                    f"expected '{train_cols[first_mismatch]}', got '{data_aligned.columns[first_mismatch]}'.\n"
                    f"Update data columns must be in the exact same order as training data.\n"
                    f"Expected order (first 5): {train_cols[:5]}, "
                    f"Got order (first 5): {list(data_aligned.columns[:5])}"
                )
        
        if covariates:
            available_covariates = [c for c in covariates if c in data_aligned.columns]
            return DDFMDataset(
                data=data_aligned,
                time_idx='index',
                covariates=available_covariates,
                scaler=dataset.scaler,
                feature_scaler=getattr(dataset, 'feature_scaler', None)
            )
        else:
            return DDFMDataset(
                data=data_aligned,
                time_idx='index',
                scaler=dataset.scaler,
                feature_scaler=getattr(dataset, 'feature_scaler', None)
            )
    
    # Rolling window size for factor updates (similar to attention models' context_length)
    # Factor models need sufficient history to properly estimate factors
    # Use 48 weeks as default (similar to attention models using 96 weeks)
    window_size_weeks = 48
    
    # Save original training factors for resetting each week (rolling window approach)
    # DDFM's update() appends factors, so we need to reset to avoid accumulation
    original_factors = model.factors.copy() if hasattr(model, 'factors') and model.factors is not None else None
    logger.debug(f"Saved original training factors for rolling window approach (factors shape: {original_factors.shape if original_factors is not None else None})")
    
    # Loop over weekly cutoffs (rolling window pattern, similar to attention models)
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        logger.info(f"Week {i+1}/{len(weekly_dates)-1}: Using rolling window up to {cutoff_date.date()}, then forecasting {next_date.date()}")
        
        # Reset model factors to original training state (fresh state each week)
        if original_factors is not None:
            model.factors = original_factors.copy()
            logger.debug(f"Reset model factors to training state for rolling window")
        
        # Get rolling window of data up to cutoff_date (past N weeks)
        window_data = update_data_source[update_data_source.index <= cutoff_date].copy()
        
        # If we have enough data, use only the last window_size_weeks
        if len(window_data) > window_size_weeks:
            window_data = window_data.iloc[-window_size_weeks:].copy()
            logger.debug(f"Using rolling window of {len(window_data)} weeks (last {window_size_weeks} weeks up to {cutoff_date.date()})")
        else:
            logger.debug(f"Using all available data ({len(window_data)} weeks) up to {cutoff_date.date()}")
        
        if len(window_data) > 0:
            # Update model state and predict in one call
            update_dataset = _create_dataset(window_data)
            forecast_result = model.predict(horizon=1, data=update_dataset, update=True)
            logger.debug(f"Updated factors with rolling window of {len(window_data)} observations and generated forecast")
        else:
            logger.warning(f"No data available for window up to {cutoff_date.date()}, using training state")
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
    logger.info(f"Factor model approach: rolling window updates ({window_size_weeks} weeks) for each forecast (provides sufficient context for factor estimation)")
    
    return predictions, actuals, forecast_dates


def _run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame],
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Dict[int, Any]:
    """Run multi-horizon forecasting for DDFM."""
    from dfm_python import DDFM, DDFMDataset
    
    # Dataset metadata is now stored in model.pkl, so dataset.pkl is no longer needed
    # DDFM.load() can create a minimal dataset from checkpoint metadata
    model = DDFM.load(checkpoint_path, dataset=None)  # None = create from metadata
    
    # Get dataset from model (created from metadata) for update_data_source
    dataset = model._dataset
    start_date_ts = pd.Timestamp(start_date)
    max_horizon = max(horizons) if horizons else 1
    update_data_source = _get_update_data_source(test_data, data_loader, dataset)
    
    # Compute available_targets from covariates (or use all series if no covariates)
    if covariates and test_data is not None:
        available_covariates = [c for c in covariates if c in test_data.columns]
        all_series = list(test_data.columns)
        available_targets = [s for s in all_series if s not in available_covariates]
        logger.info(f"Using {len(available_targets)} target series (excluded {len(available_covariates)} covariates)")
    else:
        # No covariates: use helper to get targets (or all series)
        available_targets = get_target_series_from_dataset(dataset, test_data, None)
        if available_targets is None or len(available_targets) == 0 and test_data is not None:
            available_targets = list(test_data.columns)
        logger.info(f"No covariates specified, using {len(available_targets) if available_targets else 'all'} target series")
    
    # Update factors with data up to start_date
    update_dataset = None
    if update_data_source is not None:
        train_data = update_data_source[update_data_source.index < start_date_ts].copy()
        if len(train_data) > 0:
            train_cols = list(dataset.data.columns)
            
            # Validate all training columns are present
            missing_cols = [c for c in train_cols if c not in train_data.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in update data: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}.\n"
                    f"Update data must contain all {len(train_cols)} columns from training data.\n"
                    f"Found {len([c for c in train_cols if c in train_data.columns])} columns, expected {len(train_cols)}."
                )
            
            # Align columns to training order (CRITICAL for correct scaling)
            train_data = train_data[train_cols].copy()
            
            # Validate column order matches exactly
            if list(train_data.columns) != train_cols:
                first_mismatch = next((i for i, (expected, actual) in enumerate(zip(train_cols, train_data.columns)) if expected != actual), None)
                if first_mismatch is not None:
                    raise ValueError(
                        f"Column order mismatch at index {first_mismatch}: "
                        f"expected '{train_cols[first_mismatch]}', got '{train_data.columns[first_mismatch]}'.\n"
                        f"Update data columns must be in the exact same order as training data.\n"
                        f"Expected order (first 5): {train_cols[:5]}, "
                        f"Got order (first 5): {list(train_data.columns[:5])}"
                    )
            
            if covariates:
                available_covariates = [c for c in covariates if c in train_data.columns]
                update_dataset = DDFMDataset(
                    data=train_data,
                    time_idx='index',
                    covariates=available_covariates,
                    scaler=dataset.scaler,
                    feature_scaler=getattr(dataset, 'feature_scaler', None)
                )
            else:
                update_dataset = DDFMDataset(
                    data=train_data,
                    time_idx='index',
                    scaler=dataset.scaler,
                    feature_scaler=getattr(dataset, 'feature_scaler', None)
                )
            logger.info(f"Will update factors with {len(train_data)} observations up to {start_date_ts.date()}")
    
    # Predict once with max horizon, then extract time steps
    # Update model state if data was provided
    if update_dataset is not None:
        forecast_result = model.predict(horizon=max_horizon, data=update_dataset, update=True)
    else:
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
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str = "ddfm",
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run recursive forecasting experiment with weekly updates."""
    return _run_recursive_forecast(
        checkpoint_path, test_data,
        horizon=1, start_date=start_date, end_date=end_date,
        covariates=covariates, data_loader=data_loader
    )


def run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "ddfm",
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Dict[int, Any]:
    """Run multi-horizon forecasting from fixed start point."""
    return _run_multi_horizon_forecast(
        checkpoint_path, horizons, start_date,
        test_data, covariates=covariates, data_loader=data_loader,
        return_weekly_forecasts=return_weekly_forecasts
    )
