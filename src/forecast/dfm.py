"""Forecasting functions for DFM (Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.utils import (
    get_target_series_from_dataset,
    filter_and_prepare_test_data,
    preprocess_data_for_model,
    extract_forecast_values,
    inverse_transform_predictions
)
from src.metric import (
    compute_smse,
    compute_smae,
    compute_test_data_std,
    validate_std_for_metrics,
    save_experiment_results
)
# ColumnOrderManager removed - now using scaler.feature_names_in_ directly

logger = logging.getLogger(__name__)

# #region agent log
DEBUG_LOG_PATH = Path("/data/nowcasting-kr/.cursor/debug.log")
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    """Write debug log entry."""
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            entry = {
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": location,
                "message": message,
                "data": data,
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id
            }
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion


def _get_training_column_order(
    model: Any,
    test_data: Optional[pd.DataFrame] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[List[str]]:
    """Get training column order from model.
    
    Parameters
    ----------
    model : Any
        Trained DFM model
    test_data : pd.DataFrame, optional
        Test data for fallback inference
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
    
    # Try to get from scaler feature_names_in_ (preferred - set by our implementation)
    if hasattr(model, 'scaler') and model.scaler is not None:
        if hasattr(model.scaler, 'feature_names_in_') and model.scaler.feature_names_in_ is not None:
            training_column_order = list(model.scaler.feature_names_in_)
            logger_instance.debug(f"Training column order from scaler.feature_names_in_: {len(training_column_order)} columns")
    
    # Fallback: try to get from model checkpoint metadata
    if training_column_order is None and hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
        dataset_metadata = model._checkpoint_metadata.get('dataset_metadata')
        if dataset_metadata and dataset_metadata.get('_processed_columns'):
            training_column_order = [c for c in dataset_metadata['_processed_columns'] 
                                    if c not in {'date', 'date_w', 'year', 'month', 'day'}]
            logger_instance.debug(f"Training column order from checkpoint metadata: {len(training_column_order)} columns")
    
    # Last resort: try to infer from model's C matrix (observation dimension)
    if training_column_order is None and test_data is not None and hasattr(model, 'C') and model.C is not None:
        n_obs = model.C.shape[0]  # Observation dimension = number of series
        logger_instance.debug(f"Inferred {n_obs} series from model.C.shape[0]")
        # Use test_data columns but limit to n_obs to match training
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        numeric_series = [c for c in test_data.columns if c not in date_cols and pd.api.types.is_numeric_dtype(test_data[c])]
        if len(numeric_series) >= n_obs:
            # Use first n_obs series (assumes consistent ordering)
            training_column_order = numeric_series[:n_obs]
            logger_instance.warning(
                f"Inferred training column order from model.C: {len(training_column_order)} series "
                f"(limited from {len(numeric_series)} test_data series). "
                f"This may be incorrect if test_data column order differs from training!"
            )
    
    if training_column_order is None:
        logger_instance.warning("Could not determine training column order. Column order validation will be skipped.")
    
    return training_column_order


def _align_dataframe_columns(
    df: pd.DataFrame,
    training_column_order: List[str],
    logger_instance: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Align DataFrame columns to match training column order.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to align
    training_column_order : List[str]
        Expected column order from training
    logger_instance : logging.Logger, optional
        Logger instance for warnings
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns aligned to training order
    """
    if logger_instance is None:
        logger_instance = logger
    
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    data_cols = [c for c in df.columns if c not in date_cols]
    
    # Check if columns match (strict validation)
    missing_cols = [c for c in training_column_order if c not in data_cols and c not in date_cols]
    extra_cols = [c for c in data_cols if c not in training_column_order]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns in data: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}.\n"
            f"Data must contain all {len(training_column_order)} columns from training data.\n"
            f"Found {len(data_cols)} columns, expected {len([c for c in training_column_order if c not in date_cols])} numeric columns.\n"
            f"Expected columns (first 10): {[c for c in training_column_order if c not in date_cols][:10]}"
        )
    
    if extra_cols:
        logger_instance.debug(f"Extra columns in data (will be ignored): {extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}")
    
    # Reorder columns to match training order
    aligned_cols = [c for c in training_column_order if c in data_cols]
    
    # Create aligned DataFrame
    result_df = df.copy()
    date_cols_present = [c for c in date_cols if c in df.columns]
    
    # Reorder: date cols first, then data cols in training order
    reordered_cols = date_cols_present + aligned_cols
    result_df = result_df[reordered_cols]
    
    return result_df


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "dfm",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_model: bool = True,
    data_loader: Optional[Any] = None,
    window_size: int = 52
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
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    logger.info(f"Loading DFM model from: {checkpoint_path}")
    
    # Dataset metadata is now stored in model.pkl, so dataset.pkl is no longer needed
    if recursive:
        _run_recursive_forecast(
            checkpoint_path, test_data, 
            horizon=horizon, start_date=None, end_date=None,
            covariates=None, data_loader=data_loader
        )
    else:
        _forecast_dfm(
            checkpoint_path, 
            horizon, 
            data_loader=data_loader, 
            window_size=window_size,
            test_data=test_data
        )


def _forecast_dfm(
    checkpoint_path: Path, 
    horizon: int, 
    data_loader: Optional[Any] = None, 
    window_size: int = 52,
    test_data: Optional[pd.DataFrame] = None
) -> None:
    """Forecast using DFM model.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon
    data_loader : Any, optional
        Data loader to get latest processed data for context. If provided,
        uses latest data to initialize factor state before forecasting.
    window_size : int, default 52
        Number of weeks of latest data to use for forecast initialization.
        Must match the format and column order of training data exactly.
    
    Raises
    ------
    ValueError
        If training column order cannot be determined, or if data format/columns
        don't match training data exactly.
    """
    try:
        from dfm_python import DFM
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load model (dataset metadata is now stored in model.pkl)
    logger.info("Loading DFM model...")
    model = DFM.load(checkpoint_path)

    # Get training column order for data alignment (REQUIRED)
    training_column_order = _get_training_column_order(model, None, logger)
    if training_column_order is None:
        raise ValueError(
            "Cannot determine training column order from model. "
            "This is required to ensure data format matches training exactly. "
            "Model may be corrupted or was saved without column order metadata."
        )
    
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    training_numeric_cols = [c for c in training_column_order if c not in date_cols]
    
    # Validate model has expected number of features
    expected_num_features = len(training_numeric_cols)
    if hasattr(model, 'scaler') and model.scaler is not None:
        if hasattr(model.scaler, 'n_features_in_'):
            scaler_features = model.scaler.n_features_in_
            if scaler_features != expected_num_features:
                raise ValueError(
                    f"Training column order mismatch: scaler expects {scaler_features} features, "
                    f"but training column order has {expected_num_features} numeric columns. "
                    f"This indicates a data format mismatch."
                )
    
    # Prepare latest data if data_loader is provided
    latest_data = None
    if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
        processed_data = data_loader.processed.copy()
        
        # Align columns to training order (REQUIRED - must match exactly)
        processed_data_aligned = _align_dataframe_columns(processed_data, training_column_order, logger)
        
        # Validate all required columns are present
        data_cols = [c for c in training_numeric_cols if c in processed_data_aligned.columns]
        missing_cols = [c for c in training_numeric_cols if c not in processed_data_aligned.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns in latest data: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}. "
                f"Latest data must contain all {len(training_numeric_cols)} columns from training data. "
                f"Found {len(data_cols)} columns, expected {len(training_numeric_cols)}."
            )
        
        if len(data_cols) != len(training_numeric_cols):
            raise ValueError(
                f"Column count mismatch: latest data has {len(data_cols)} columns, "
                f"but training data has {len(training_numeric_cols)} columns. "
                f"All training columns must be present in the same order."
            )
        
        # Validate column order matches exactly
        if data_cols != training_numeric_cols:
            # Find first mismatch
            first_mismatch = next((i for i, (expected, actual) in enumerate(zip(training_numeric_cols, data_cols)) if expected != actual), None)
            if first_mismatch is not None:
                raise ValueError(
                    f"Column order mismatch at index {first_mismatch}: "
                    f"expected '{training_numeric_cols[first_mismatch]}', got '{data_cols[first_mismatch]}'. "
                    f"Latest data columns must be in the exact same order as training data. "
                    f"Expected order (first 5): {training_numeric_cols[:5]}, "
                    f"Got order (first 5): {data_cols[:5]}"
                )
        
        # Use a rolling window of recent data (not all data)
        if len(processed_data_aligned) < window_size:
            logger.warning(
                f"Available data ({len(processed_data_aligned)} weeks) is less than requested window size ({window_size} weeks). "
                f"Using all available data."
            )
            actual_window_size = len(processed_data_aligned)
        else:
            actual_window_size = window_size
        
        update_data_df = processed_data_aligned.iloc[-actual_window_size:].copy()
        
        # Validate data format: must be numeric
        for col in data_cols:
            if not pd.api.types.is_numeric_dtype(update_data_df[col]):
                raise ValueError(
                    f"Column '{col}' is not numeric (dtype: {update_data_df[col].dtype}). "
                    f"All columns must be numeric to match training data format."
                )
        
        # Extract numeric values only (exclude date columns) - this creates numpy array
        latest_data = update_data_df[data_cols].values
        
        # Validate array shape matches expected format
        if latest_data.ndim != 2:
            raise ValueError(
                f"Latest data must be 2D array (time_steps x features), got shape {latest_data.shape}"
            )
        
        if latest_data.shape[1] != expected_num_features:
            raise ValueError(
                f"Latest data feature count mismatch: got {latest_data.shape[1]} features, "
                f"expected {expected_num_features} (from training data)."
            )
        
        logger.info(f"Using latest {actual_window_size} weeks of data for forecast initialization")
        logger.info(f"Latest data date range: {update_data_df.index.min()} to {update_data_df.index.max()}")
        logger.info(f"Latest data shape: {latest_data.shape} (matches training format: {actual_window_size} time steps × {expected_num_features} features)")
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
        if latest_data is not None:
            predictions = model.predict(horizon=horizon, data=latest_data, update=True)
        else:
            predictions = model.predict(horizon=horizon)

        # Log and save forecasts
        _log_and_save_forecast(predictions, checkpoint_path)
        
        # Compute metrics if test_data is available
        if test_data is not None:
            _compute_and_save_metrics(
                predictions=predictions,
                model=model,
                test_data=test_data,
                data_loader=data_loader,
                horizon=horizon,
                checkpoint_path=checkpoint_path,
                training_column_order=training_column_order
            )
            
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


def _compute_and_save_metrics(
    predictions: Any,
    model: Any,
    test_data: pd.DataFrame,
    data_loader: Optional[Any],
    horizon: int,
    checkpoint_path: Path,
    training_column_order: Optional[List[str]]
) -> None:
    """Compute and save metrics for DFM forecasts.
    
    Parameters
    ----------
    predictions : Any
        Model predictions (tuple or array)
    model : Any
        Trained DFM model
    test_data : pd.DataFrame
        Test data with actual values
    data_loader : Any, optional
        Data loader for inverse transformations
    horizon : int
        Forecast horizon
    checkpoint_path : Path
        Path to checkpoint (for output directory)
    training_column_order : List[str], optional
        Training column order for extracting target series
    """
    # #region agent log
    import json
    from datetime import datetime
    DEBUG_LOG_PATH = Path("/data/nowcasting-kr/.cursor/debug.log")
    def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                entry = {
                    "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "location": location,
                    "message": message,
                    "data": data,
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": hypothesis_id
                }
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
    # #endregion
    
    # #region agent log
    _debug_log(
        "dfm.py:456-entry",
        "Entering _compute_and_save_metrics",
        {
            "horizon": horizon,
            "test_data_shape": list(test_data.shape) if test_data is not None else None,
            "test_data_columns": list(test_data.columns[:5]) if test_data is not None and len(test_data.columns) > 0 else None,
            "has_data_loader": data_loader is not None,
            "training_column_order_len": len(training_column_order) if training_column_order else None
        },
        "H1,H2,H3,H4,H5"
    )
    # #endregion
    
    # Extract forecast values
    if isinstance(predictions, tuple):
        X_forecast, Z_forecast = predictions
    else:
        X_forecast = predictions
    
    # #region agent log
    _debug_log(
        "dfm.py:488-after-extract",
        "Extracted X_forecast from predictions",
        {
            "X_forecast_shape": list(X_forecast.shape) if hasattr(X_forecast, 'shape') else None,
            "X_forecast_dtype": str(X_forecast.dtype) if hasattr(X_forecast, 'dtype') else None,
            "X_forecast_min": float(X_forecast.min()) if hasattr(X_forecast, 'min') else None,
            "X_forecast_max": float(X_forecast.max()) if hasattr(X_forecast, 'max') else None,
            "X_forecast_mean": float(X_forecast.mean()) if hasattr(X_forecast, 'mean') else None,
            "X_forecast_sample": X_forecast[0, :5].tolist() if X_forecast.ndim == 2 and X_forecast.shape[0] > 0 and X_forecast.shape[1] >= 5 else None
        },
        "H1,H4"
    )
    # #endregion
    
    # Get target series from model
    target_series = None
    
    # Try to get from model config
    if hasattr(model, '_config') and model._config is not None:
        config = model._config
        target_series = getattr(config, 'target_series', None)
        if target_series:
            if isinstance(target_series, str):
                target_series = [target_series]
            logger.info(f"Using target series from model config: {target_series}")
    
    # Fallback: try to get from training column order (exclude covariates)
    if target_series is None and training_column_order is not None:
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
        
        # Check for covariates in config
        covariates = None
        if hasattr(model, '_config') and model._config is not None:
            covariates = getattr(model._config, 'covariates', None)
            if covariates and isinstance(covariates, str):
                covariates = [covariates]
        
        if covariates:
            target_series = [c for c in training_numeric_cols if c not in covariates]
            logger.info(f"Using target series from training columns (excluded {len(covariates)} covariates): {len(target_series)} series")
        else:
            target_series = training_numeric_cols
            logger.info(f"Using all training series as targets: {len(target_series)} series")
    
    # Final fallback: use test_data columns
    if target_series is None:
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        target_series = [c for c in test_data.columns if c not in date_cols and pd.api.types.is_numeric_dtype(test_data[c])]
        logger.warning(f"Could not determine target series from model, using test_data columns: {len(target_series)} series")
    
    if not target_series:
        logger.warning("No target series found, skipping metrics computation")
        return
    
    # Filter to series that exist in test_data
    available_targets = [t for t in target_series if t in test_data.columns]
    if len(available_targets) != len(target_series):
        missing = set(target_series) - set(available_targets)
        logger.warning(f"Some target series not found in test_data: {missing}. Using {len(available_targets)}/{len(target_series)} series.")
    
    if not available_targets:
        logger.warning("No available target series in test_data, skipping metrics computation")
        return
    
    # Extract predictions for target series only
    # X_forecast shape: (horizon, n_series) where n_series matches training data
    # Need to map to target_series indices
    if training_column_order is not None:
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
        target_indices = [training_numeric_cols.index(t) for t in available_targets if t in training_numeric_cols]
    else:
        # Fallback: assume predictions are in same order as available_targets
        target_indices = list(range(len(available_targets)))
    
    if not target_indices:
        logger.warning("Could not map target series to prediction indices, skipping metrics computation")
        return
    
    # Extract predictions for target series
    if X_forecast.ndim == 2:
        # Shape: (horizon, n_series)
        predictions_array = X_forecast[:, target_indices]  # (horizon, n_targets)
    else:
        # Shape: (horizon * n_series,) - need to reshape
        n_series = len(training_numeric_cols) if training_column_order else X_forecast.shape[0] // horizon
        X_forecast_reshaped = X_forecast.reshape(horizon, n_series)
        predictions_array = X_forecast_reshaped[:, target_indices]  # (horizon, n_targets)
    
    # #region agent log
    _debug_log(
        "dfm.py:559-after-extract-targets",
        "Extracted predictions for target series",
        {
            "predictions_array_shape": list(predictions_array.shape),
            "target_indices": target_indices[:10] if len(target_indices) > 10 else target_indices,
            "available_targets": available_targets[:5] if len(available_targets) > 5 else available_targets,
            "predictions_array_sample": predictions_array[0, :5].tolist() if predictions_array.shape[0] > 0 and predictions_array.shape[1] >= 5 else None,
            "predictions_array_min": float(predictions_array.min()),
            "predictions_array_max": float(predictions_array.max()),
            "predictions_array_mean": float(predictions_array.mean())
        },
        "H3,H4"
    )
    # #endregion
    
    # CRITICAL: DFM returns predictions in TRANSFORMED space (chg/logdiff)
    # DFM's scaler is fitted on transformed data, so inverse transform returns chg values
    # We need to apply accumulation to convert chg → levels
    if data_loader is not None and available_targets is not None and len(available_targets) > 0:
        try:
            predictions_original = inverse_transform_predictions(
                predictions=predictions_array,
                available_targets=available_targets,
                data_loader=data_loader,
                reverse_transformations=True,
                test_data=test_data
            )
            # #region agent log
            _debug_log(
                "dfm.py:636-after-inverse-transform",
                "After inverse transformation (chg → levels via accumulation)",
                {
                    "predictions_original_shape": list(predictions_original.shape),
                    "predictions_original_sample": predictions_original[0, :5].tolist() if predictions_original.ndim == 2 and predictions_original.shape[0] > 0 and predictions_original.shape[1] >= 5 else None,
                    "predictions_original_min": float(predictions_original.min()),
                    "predictions_original_max": float(predictions_original.max()),
                    "predictions_original_mean": float(predictions_original.mean())
                },
                "H1,H4"
            )
            # #endregion
        except Exception as e:
            logger.warning(f"Failed to inverse transform predictions: {e}. Using predictions as-is.")
            predictions_original = predictions_array
            # #region agent log
            _debug_log(
                "dfm.py:636-inverse-transform-failed",
                "Inverse transformation failed",
                {"error": str(e)},
                "H4"
            )
            # #endregion
    else:
        logger.warning("No data_loader provided, using predictions as-is (may be in transformed scale)")
        predictions_original = predictions_array
    
    # Get actuals from test_data
    # For horizon > 1, we need to get actuals for each horizon step
    # For simplicity, use the first horizon step's actuals
    forecast_date = test_data.index[-1] if len(test_data) > 0 else None
    
    # #region agent log
    _debug_log(
        "dfm.py:586-before-actuals-extraction",
        "Before extracting actuals",
        {
            "test_data_index_range": [str(test_data.index[0]), str(test_data.index[-1])] if len(test_data) > 0 else None,
            "test_data_len": len(test_data),
            "forecast_date": str(forecast_date) if forecast_date is not None else None,
            "horizon": horizon,
            "available_targets": available_targets[:5] if len(available_targets) > 5 else available_targets
        },
        "H2"
    )
    # #endregion
    
    # Get actuals for target series
    actuals_array = test_data[available_targets].iloc[-horizon:].values  # (horizon, n_targets)
    
    # #region agent log
    _debug_log(
        "dfm.py:592-after-actuals-extraction",
        "After extracting actuals",
        {
            "actuals_array_shape": list(actuals_array.shape) if actuals_array.ndim > 0 else None,
            "actuals_array_sample": actuals_array[0, :5].tolist() if actuals_array.ndim == 2 and actuals_array.shape[0] > 0 and actuals_array.shape[1] >= 5 else (actuals_array[:5].tolist() if actuals_array.ndim == 1 else None),
            "actuals_array_min": float(actuals_array.min()) if actuals_array.size > 0 else None,
            "actuals_array_max": float(actuals_array.max()) if actuals_array.size > 0 else None,
            "actuals_array_mean": float(actuals_array.mean()) if actuals_array.size > 0 else None,
            "extracted_dates": [str(test_data.index[i]) for i in range(max(0, len(test_data) - horizon), len(test_data))] if len(test_data) > 0 else None
        },
        "H2,H5"
    )
    # #endregion
    
    # If horizon is 1, reshape to (1, n_targets)
    if actuals_array.ndim == 1:
        actuals_array = actuals_array.reshape(1, -1)
    
    # Ensure predictions and actuals have same shape
    if predictions_original.shape != actuals_array.shape:
        # Take first horizon step if shapes don't match
        if predictions_original.shape[0] > 0 and actuals_array.shape[0] > 0:
            predictions_array_1step = predictions_original[0:1, :]  # (1, n_targets)
            actuals_array_1step = actuals_array[0:1, :]  # (1, n_targets)
            predictions_original = predictions_array_1step
            actuals_array = actuals_array_1step
        else:
            logger.warning(f"Shape mismatch: predictions {predictions_original.shape} vs actuals {actuals_array.shape}, skipping metrics")
            return
    
    # Compute test_data_std for normalization
    monthly_series = None
    if data_loader is not None and hasattr(data_loader, 'monthly_series'):
        monthly_series = data_loader.monthly_series
    
    test_data_std = compute_test_data_std(
        test_data=test_data,
        target_series=available_targets,
        monthly_series=monthly_series
    )
    
    # Compute metrics
    std_for_metrics = validate_std_for_metrics(test_data_std, len(available_targets))
    
    if std_for_metrics is not None and not np.isnan(std_for_metrics).all():
        # #region agent log
        _debug_log(
            "dfm.py:621-before-metrics",
            "Before computing metrics",
            {
                "predictions_original_shape": list(predictions_original.shape),
                "actuals_array_shape": list(actuals_array.shape),
                "std_for_metrics": std_for_metrics[:5].tolist() if len(std_for_metrics) >= 5 else std_for_metrics.tolist(),
                "predictions_sample": predictions_original[0, :5].tolist() if predictions_original.shape[0] > 0 and predictions_original.shape[1] >= 5 else None,
                "actuals_sample": actuals_array[0, :5].tolist() if actuals_array.shape[0] > 0 and actuals_array.shape[1] >= 5 else None,
                "diff_sample": (predictions_original[0, :5] - actuals_array[0, :5]).tolist() if predictions_original.shape[0] > 0 and actuals_array.shape[0] > 0 and predictions_original.shape[1] >= 5 and actuals_array.shape[1] >= 5 else None
            },
            "H1,H2,H3,H4,H5"
        )
        # #endregion
        
        smse = compute_smse(actuals_array, predictions_original, test_data_std=std_for_metrics)
        smae = compute_smae(actuals_array, predictions_original, test_data_std=std_for_metrics)
        
        # #region agent log
        _debug_log(
            "dfm.py:625-after-metrics",
            "After computing metrics",
            {
                "smse": float(smse) if not np.isnan(smse) else None,
                "smae": float(smae) if not np.isnan(smae) else None,
                "is_smse_nan": bool(np.isnan(smse)),
                "is_smae_nan": bool(np.isnan(smae))
            },
            "H1,H2,H3,H4,H5"
        )
        # #endregion
        
        metrics = {
            "smse": float(smse) if not np.isnan(smse) else None,
            "smae": float(smae) if not np.isnan(smae) else None
        }
        
        # Log metrics
        smse_str = f"{smse:.6f}" if not np.isnan(smse) else "NaN"
        smae_str = f"{smae:.6f}" if not np.isnan(smae) else "NaN"
        logger.info(f"DFM Forecast Metrics (horizon={horizon}): sMSE={smse_str}, sMAE={smae_str}")
        
        # Save metrics
        output_dir = checkpoint_path.parent / "forecasts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dates for predictions
        if forecast_date is not None:
            dates = pd.DatetimeIndex([forecast_date] * predictions_original.shape[0])
        else:
            dates = None
        
        save_experiment_results(
            output_dir=output_dir,
            predictions=predictions_original,
            actuals=actuals_array,
            dates=dates,
            target_series=available_targets,
            metrics=metrics
        )
        
        logger.info(f"Metrics saved to: {output_dir / 'metrics.json'}")
    else:
        logger.warning("Could not compute test_data_std, skipping metrics computation")


def _run_recursive_forecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    horizon: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run recursive forecasting for DFM model.
    
    For factor models, recursive forecasting means:
    1. Update factors with new observations and predict (via model.predict(update=True, data=...) which uses Kalman filtering)
    2. Repeat for each time step
    
    Unlike attention-based models, factor models maintain latent factors that capture
    underlying dynamics. The update() method refreshes these factors when new data arrives.
    """
    try:
        from dfm_python import DFM
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    # Load model (dataset metadata is now stored in model checkpoint)
    logger.info("Loading DFM model for recursive forecasting...")
    model = DFM.load(checkpoint_path)
    
    # Get dataset metadata from model checkpoint
    dataset_metadata = None
    if hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
        dataset_metadata = model._checkpoint_metadata.get('dataset_metadata')
    
    
    # Get covariates from config
    config_covariates = None
    if hasattr(model, '_config') and model._config is not None:
        config = model._config
        config_covariates = getattr(config, 'covariates', None)
        if config_covariates:
            logger.info(f"Found covariates in model config: {config_covariates}")
    
    # Get training column order from model (critical for proper standardization)
    # MUST be done before computing available_targets
    training_column_order = _get_training_column_order(model, test_data, logger)
    
    # #region agent log
    _debug_log(
        "dfm.py:291",
        "Training column order retrieved",
        {
            "training_column_order": training_column_order[:10] if training_column_order else None,
            "training_column_order_len": len(training_column_order) if training_column_order else 0,
            "is_none": training_column_order is None
        },
        "H1"
    )
    # #endregion
    
    # Use config covariates if available, otherwise use provided covariates
    if config_covariates:
        covariates = config_covariates if isinstance(config_covariates, list) else [config_covariates]
    
    # Compute available_targets from training column order (CRITICAL: must match training data)
    # CRITICAL: DFM is a factor model that predicts ALL series, not just target_series
    # target_series in config is for evaluation/metrics only, not for limiting predictions
    # CRITICAL: Use training column order to ensure series count matches training data
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    
    # Get training series from model (must match training data exactly)
    training_series = None
    if training_column_order is not None:
        # Filter to only numeric columns (exclude date columns from training order)
        training_series = [c for c in training_column_order if c not in date_cols]
        logger.debug(f"Using training column order: {len(training_series)} series")
    elif hasattr(model, 'scaler') and model.scaler is not None and hasattr(model.scaler, 'feature_names_in_'):
        training_series = [c for c in model.scaler.feature_names_in_ if c not in date_cols]
        logger.debug(f"Using scaler feature names: {len(training_series)} series")
    
    # Fallback: use test_data numeric columns (may cause mismatch if test_data has different columns)
    if training_series is None:
        numeric_series = [c for c in test_data.columns if c not in date_cols and pd.api.types.is_numeric_dtype(test_data[c])]
        training_series = numeric_series
        logger.warning(f"Could not get training column order, using test_data columns: {len(training_series)} series. This may cause errors if count doesn't match training data.")
    
    # Filter to only series that exist in test_data
    available_training_series = [s for s in training_series if s in test_data.columns]
    if len(available_training_series) != len(training_series):
        missing = set(training_series) - set(available_training_series)
        logger.warning(f"Some training series not found in test_data: {missing}. Using {len(available_training_series)}/{len(training_series)} series.")
    
    if covariates:
        available_covariates = [c for c in covariates if c in available_training_series]
        available_targets = [s for s in available_training_series if s not in available_covariates]
        logger.info(f"Using {len(available_targets)} target series (excluded {len(available_covariates)} covariates from {len(available_training_series)} training series)")
    else:
        # No covariates: all training series are targets (default for factor models)
        available_targets = available_training_series
        logger.info(f"No covariates specified, using all {len(available_targets)} training series as targets (factor model predicts all series)")
    
    # Use dataset metadata from checkpoint
    if dataset_metadata:
        # Use metadata from model checkpoint
        if covariates is None and dataset_metadata.get('covariates'):
            covariates = dataset_metadata['covariates']
        # Try to get training column order from dataset metadata
        if training_column_order is None and dataset_metadata.get('_processed_columns'):
            training_column_order = list(dataset_metadata['_processed_columns'])
            logger.debug(f"Training column order from dataset metadata: {len(training_column_order)} columns")
    
    # If still None, try to get from model's scaler (second attempt, in case first attempt failed)
    if training_column_order is None and hasattr(model, 'scaler') and model.scaler is not None:
        if hasattr(model.scaler, 'feature_names_in_'):
            training_column_order = list(model.scaler.feature_names_in_)
            logger.debug(f"Training column order from scaler (second attempt): {len(training_column_order)} columns")

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
    # PATTERN: Pass transformed (chg, log, etc.) but NOT standardized data to model.update()
    #          model.update() will standardize internally using model.scaler
    # CRITICAL: Column order must match training_column_order for correct standardization
    if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
        # Use processed data (transformed but not standardized)
        # Align indices: use intersection of available dates
        common_dates = train_data_initial.index.intersection(data_loader.processed.index)
        if len(common_dates) > 0:
            processed_df = data_loader.processed.loc[common_dates]
            # CRITICAL: Align columns to match training order exactly
            # This ensures scaler applies correct mean/scale to each column
            if training_column_order is not None:
                # Align columns to training order
                processed_df_aligned = _align_dataframe_columns(processed_df, training_column_order, logger)
                # Extract values in correct order
                date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                data_cols = [c for c in training_column_order if c in processed_df_aligned.columns]
                train_data_processed = processed_df_aligned[data_cols].values
            else:
                # Fallback: extract numeric columns (order may not match training - RISKY!)
                logger.warning("Training column order not available. Using fallback column extraction (may cause scaling errors!)")
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                train_data_processed = processed_df[numeric_cols].values
            
            # Validate column count matches training data
            if hasattr(model, 'data_processed') and model.data_processed is not None:
                expected_cols = model.data_processed.shape[1]
                actual_cols = train_data_processed.shape[1] if train_data_processed.ndim > 1 else 1
                if actual_cols != expected_cols:
                    raise ValueError(
                        f"Column count mismatch: update data has {actual_cols} columns, "
                        f"but training data has {expected_cols} columns. "
                        f"Training columns: {len(training_column_order) if training_column_order else 'unknown'}, "
                        f"Update columns: {len(numeric_cols)}"
                    )
        else:
            # Fallback: use raw data if no overlap
            if training_column_order is not None:
                date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                training_numeric_cols = [c for c in training_column_order if c not in date_cols and c in train_data_initial.columns]
                train_data_processed = train_data_initial[training_numeric_cols].values
            else:
                numeric_cols = train_data_initial.select_dtypes(include=[np.number]).columns.tolist()
                train_data_processed = train_data_initial[numeric_cols].values
    else:
        # Fallback: use preprocess_data_for_model but skip standardization
        # Select numeric columns only, filtered to training columns if available
        if training_column_order is not None:
            date_cols = {'date', 'date_w', 'year', 'month', 'day'}
            training_numeric_cols = [c for c in training_column_order if c not in date_cols and c in train_data_initial.columns]
            train_data_processed = train_data_initial[training_numeric_cols].values
        else:
            numeric_cols = train_data_initial.select_dtypes(include=[np.number]).columns.tolist()
            train_data_processed = train_data_initial[numeric_cols].values
        # Impute if needed (DFM can handle NaNs but imputation helps)
        if train_data_processed.dtype == float and np.isnan(train_data_processed).any():
            train_data_processed = pd.DataFrame(train_data_processed).ffill().bfill().fillna(0).values
    
    # CRITICAL: Update model._config.frequency to match training column order before any predict() calls
    # This ensures target_indices in model.predict() resolve correctly (matches C matrix size)
    if training_column_order is not None and hasattr(model, '_config') and model._config is not None:
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
        config_series_ids = model._config.get_series_ids() if hasattr(model._config, 'get_series_ids') else None
        if config_series_ids is None or len(config_series_ids) != len(training_numeric_cols):
            if hasattr(model._config, 'frequency'):
                clock_freq = getattr(model._config, 'clock', 'w')
                model._config.frequency = {col: clock_freq for col in training_numeric_cols}
                logger.info(f"Updated model config frequency dict to match {len(training_numeric_cols)} training series")
    
    # Get update data source (processed data aligned with training space)
    # Similar to DDFM: use processed data from data_loader if available
    update_data_source = None
    if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
        processed = data_loader.processed.copy()
        date_cols = ['date', 'date_w', 'year', 'month', 'day']
        processed = processed.drop(columns=[c for c in date_cols if c in processed.columns], errors='ignore')
        if isinstance(test_data.index, pd.DatetimeIndex):
            processed = processed.loc[processed.index.intersection(test_data.index)]
        if training_column_order is not None:
            date_cols = {'date', 'date_w', 'year', 'month', 'day'}
            training_numeric_cols = [c for c in training_column_order if c not in date_cols]
            processed = processed[[c for c in training_numeric_cols if c in processed.columns]].copy()
        update_data_source = processed
    else:
        # Fallback: use test_data directly (will need preprocessing in loop)
        update_data_source = test_data
    
    # Rolling window size for factor updates (similar to attention models' context_length)
    # Factor models need sufficient history to properly estimate factors
    # Use 48 weeks as default (similar to attention models using 96 weeks)
    window_size_weeks = 48
    
    # Save original training state for resetting each week (rolling window approach)
    result = model._ensure_result()
    original_Z = result.Z.copy() if result.Z is not None else None
    original_x_sm = result.x_sm.copy() if result.x_sm is not None else None
    original_data_processed = model.data_processed.copy() if model.data_processed is not None else None
    logger.debug(f"Saved original training state for rolling window approach (Z shape: {original_Z.shape if original_Z is not None else None})")
    
    # Loop over weekly cutoffs (rolling window pattern, similar to attention models)
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        logger.info(f"Week {i+1}/{len(weekly_dates)-1}: Using rolling window up to {cutoff_date.date()}, then forecasting {next_date.date()}")
        
        # Reset model state to original training state (fresh state each week)
        result = model._ensure_result()
        if original_Z is not None:
            result.Z = original_Z.copy()  # Reset to training factors
        if original_x_sm is not None:
            result.x_sm = original_x_sm.copy()  # Reset smoothed data
        if original_data_processed is not None:
            model.data_processed = original_data_processed.copy()  # Reset processed data
        logger.debug(f"Reset model state to training state for rolling window")
        
        # Get rolling window of data up to cutoff_date (past N weeks)
        window_data = update_data_source[update_data_source.index <= cutoff_date].copy()
        
        # If we have enough data, use only the last window_size_weeks
        if len(window_data) > window_size_weeks:
            window_data = window_data.iloc[-window_size_weeks:].copy()
            logger.debug(f"Using rolling window of {len(window_data)} weeks (last {window_size_weeks} weeks up to {cutoff_date.date()})")
        else:
            logger.debug(f"Using all available data ({len(window_data)} weeks) up to {cutoff_date.date()}")
        
        if len(window_data) > 0:
            # Preprocess window data
            if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
                # Already processed, just need to extract in training order
                if training_column_order is not None:
                    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                    training_numeric_cols = [c for c in training_column_order if c not in date_cols]
                    window_data_processed = window_data[training_numeric_cols].values
                    
                    # #region agent log
                    _debug_log(
                        f"dfm.py:513-{i}",
                        "Window data extracted with training column order",
                        {
                            "week": i+1,
                            "training_numeric_cols_count": len(training_numeric_cols),
                            "window_data_shape": window_data_processed.shape,
                            "first_5_cols": training_numeric_cols[:5],
                            "window_data_mean": window_data_processed.mean(axis=0)[:5].tolist() if window_data_processed.ndim > 1 else [float(window_data_processed.mean())],
                            "window_data_std": window_data_processed.std(axis=0)[:5].tolist() if window_data_processed.ndim > 1 else [float(window_data_processed.std())],
                            "target_series": available_targets[:3] if 'available_targets' in locals() else None
                        },
                        "H1"
                    )
                    # #endregion
                else:
                    window_data_processed = window_data.values
            else:
                # Need to preprocess (extract numeric columns in training order)
                if training_column_order is not None:
                    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                    training_numeric_cols = [c for c in training_column_order if c not in date_cols and c in window_data.columns]
                    window_data_processed = window_data[training_numeric_cols].values
                else:
                    numeric_cols = window_data.select_dtypes(include=[np.number]).columns.tolist()
                    window_data_processed = window_data[numeric_cols].values
                # Impute if needed
                if window_data_processed.dtype == float and np.isnan(window_data_processed).any():
                    window_data_processed = pd.DataFrame(window_data_processed).ffill().bfill().fillna(0).values
            
            # Validate column count
            if hasattr(model, 'data_processed') and model.data_processed is not None:
                expected_cols = model.data_processed.shape[1]
                actual_cols = window_data_processed.shape[1] if window_data_processed.ndim > 1 else 1
                if actual_cols != expected_cols:
                    raise ValueError(
                        f"Column count mismatch: window data has {actual_cols} columns, "
                        f"but training data has {expected_cols} columns."
                    )
            
            # Update model with rolling window (provides sufficient context for factor estimation)
            # CRITICAL: Pass raw transformed data (not standardized) - model.predict(update=True) standardizes internally
            # Column order must match training_column_order for correct standardization
            
            # #region agent log
            if hasattr(model, 'scaler') and model.scaler is not None:
                scaler_mean = model.scaler.mean_[:5].tolist() if hasattr(model.scaler, 'mean_') else None
                scaler_scale = model.scaler.scale_[:5].tolist() if hasattr(model.scaler, 'scale_') else None
                _debug_log(
                    f"dfm.py:542-before-update-{i}",
                    "Before model.predict(update=True) - scaler statistics",
                    {
                        "week": i+1,
                        "scaler_mean_first5": scaler_mean,
                        "scaler_scale_first5": scaler_scale,
                        "window_data_mean_first5": window_data_processed.mean(axis=0)[:5].tolist() if window_data_processed.ndim > 1 else [float(window_data_processed.mean())],
                        "window_data_std_first5": window_data_processed.std(axis=0)[:5].tolist() if window_data_processed.ndim > 1 else [float(window_data_processed.std())]
                    },
                    "H1"
                )
            # #endregion
            
            # Update model state and predict in one call
            # #region agent log
            _debug_log(
                f"dfm.py:1158-before-predict-{i}",
                "Before model.predict(update=True)",
                {
                    "week": i+1,
                    "window_data_shape": window_data_processed.shape,
                    "window_data_mean": float(window_data_processed.mean()),
                    "window_data_std": float(window_data_processed.std()),
                    "window_data_min": float(window_data_processed.min()),
                    "window_data_max": float(window_data_processed.max()),
                    "update": True
                },
                "H1,H2"
            )
            # #endregion
            forecast_result = model.predict(horizon=1, data=window_data_processed, update=True)
            logger.debug(f"Updated factors with rolling window of {len(window_data_processed)} observations and generated forecast")
            
            # #region agent log
            if isinstance(forecast_result, tuple):
                X_forecast_log, _ = forecast_result
            else:
                X_forecast_log = forecast_result
            X_forecast_log = np.asarray(X_forecast_log)
            
            # Find KOIPALL.G index if available
            koipall_idx = None
            if training_column_order and 'KOIPALL.G' in training_column_order:
                training_numeric = [c for c in training_column_order if c not in {'date', 'date_w', 'year', 'month', 'day'}]
                koipall_idx = training_numeric.index('KOIPALL.G') if 'KOIPALL.G' in training_numeric else None
            
            _debug_log(
                f"dfm.py:1158-after-predict-{i}",
                "After model.predict(update=True)",
                {
                    "week": i+1,
                    "X_forecast_shape": list(X_forecast_log.shape),
                    "X_forecast_mean": float(X_forecast_log.mean()),
                    "X_forecast_std": float(X_forecast_log.std()),
                    "koipall_idx": koipall_idx,
                    "koipall_value": float(X_forecast_log[0, koipall_idx]) if koipall_idx is not None and X_forecast_log.ndim > 1 and X_forecast_log.shape[1] > koipall_idx else None,
                    "X_forecast_first5": X_forecast_log[0, :5].tolist() if X_forecast_log.ndim > 1 and X_forecast_log.shape[1] >= 5 else X_forecast_log.flatten()[:5].tolist()
                },
                "H1,H2"
            )
            # #endregion
        else:
            logger.warning(f"No data available for window up to {cutoff_date.date()}, using training state")
            # Predict 1 step ahead from training state
            forecast_result = model.predict(horizon=1)  # 1 step ahead from training state
        
        # #region agent log
        # Log what predict() actually returns
        if isinstance(forecast_result, tuple):
            X_forecast, _ = forecast_result
        else:
            X_forecast = forecast_result
        X_forecast = np.asarray(X_forecast)
        
        # Log to regular logger for first week
        if i == 0:
            logger.info(f"DEBUG Week {i+1}: model.predict() returned shape {X_forecast.shape}")
            if X_forecast.ndim > 1 and X_forecast.shape[1] > 12:
                logger.info(f"DEBUG Week {i+1}: Value at index 12 in raw prediction: {X_forecast[0, 12]}")
                logger.info(f"DEBUG Week {i+1}: First 5 values in raw prediction: {X_forecast[0, :5].tolist()}")
            elif X_forecast.size > 12:
                logger.info(f"DEBUG Week {i+1}: Value at index 12 in raw prediction: {X_forecast[12]}")
                logger.info(f"DEBUG Week {i+1}: First 5 values in raw prediction: {X_forecast[:5].tolist()}")
        
        _debug_log(
            f"dfm.py:548-predict-result-{i}",
            "model.predict() result",
            {
                "week": i+1,
                "forecast_result_type": type(forecast_result).__name__,
                "X_forecast_shape": list(X_forecast.shape) if hasattr(X_forecast, 'shape') else None,
                "X_forecast_first5": X_forecast.flatten()[:5].tolist() if X_forecast.size > 0 else [],
                "X_forecast_at_12": float(X_forecast[0, 12]) if X_forecast.ndim > 1 and X_forecast.shape[1] > 12 else (float(X_forecast[12]) if X_forecast.size > 12 else None),
                "len_available_targets": len(available_targets),
                "available_targets_first5": available_targets[:5]
            },
            "H2"
        )
        # #endregion
        
        # CRITICAL: Model may return predictions only for target_series from config, not all available_targets
        # Extract the actual number of predictions returned, then map to available_targets order
        if isinstance(forecast_result, tuple):
            X_forecast_actual, _ = forecast_result
        else:
            X_forecast_actual = forecast_result
        X_forecast_actual = np.asarray(X_forecast_actual)
        
        # Get actual number of series in predictions
        if X_forecast_actual.ndim > 1:
            n_series_actual = X_forecast_actual.shape[1]
        else:
            n_series_actual = 1 if X_forecast_actual.size > 0 else 0
        
        # Extract predictions using actual number of series
        forecast_values = extract_forecast_values(
            forecast_result, n_series_actual, horizon_idx=0
        )
        
        # #region agent log
        # Log what we extracted and check KOEQUIPTE specifically
        if 'KOEQUIPTE' in available_targets and i == 0:  # Only log first week to avoid spam
            koequipte_idx_in_available = available_targets.index('KOEQUIPTE')
            # Check if predictions are in training order (KOEQUIPTE should be at index 12)
            if training_column_order and 'KOEQUIPTE' in training_column_order:
                training_numeric = [c for c in training_column_order if c not in {'date', 'date_w', 'year', 'month', 'day'}]
                koequipte_idx_in_training = training_numeric.index('KOEQUIPTE')
                
                # Log to both debug log and regular logger
                logger.info(f"DEBUG Week {i+1}: KOEQUIPTE at training_idx={koequipte_idx_in_training}, available_idx={koequipte_idx_in_available}")
                logger.info(f"DEBUG Week {i+1}: Value at training_idx: {forecast_values[koequipte_idx_in_training] if koequipte_idx_in_training < len(forecast_values) else 'N/A'}")
                logger.info(f"DEBUG Week {i+1}: Value at available_idx: {forecast_values[koequipte_idx_in_available] if koequipte_idx_in_available < len(forecast_values) else 'N/A'}")
                logger.info(f"DEBUG Week {i+1}: Value at index 12: {forecast_values[12] if len(forecast_values) > 12 else 'N/A'}")
                logger.info(f"DEBUG Week {i+1}: First 5 values: {forecast_values[:5].tolist()}")
                logger.info(f"DEBUG Week {i+1}: Training order first 5: {training_numeric[:5]}")
                logger.info(f"DEBUG Week {i+1}: Available targets first 5: {available_targets[:5]}")
                
                _debug_log(
                    f"dfm.py:625-after-extract-{i}",
                    "After extract_forecast_values - checking KOEQUIPTE",
                    {
                        "week": i+1,
                        "forecast_values_shape": list(forecast_values.shape) if hasattr(forecast_values, 'shape') else None,
                        "n_series_actual": n_series_actual,
                        "len_available_targets": len(available_targets),
                        "koequipte_idx_in_training": koequipte_idx_in_training,
                        "koequipte_idx_in_available": koequipte_idx_in_available,
                        "value_at_training_idx": float(forecast_values[koequipte_idx_in_training]) if koequipte_idx_in_training < len(forecast_values) else None,
                        "value_at_available_idx": float(forecast_values[koequipte_idx_in_available]) if koequipte_idx_in_available < len(forecast_values) else None,
                        "forecast_values_first5": forecast_values[:5].tolist() if len(forecast_values) >= 5 else forecast_values.tolist(),
                        "forecast_values_at_12": float(forecast_values[12]) if len(forecast_values) > 12 else None,
                        "training_order_first5": training_numeric[:5],
                        "available_targets_first5": available_targets[:5]
                    },
                    "H5"
                )
        # #endregion
        
        # If model returned fewer predictions than available_targets, we need to map them correctly
        # The model returns predictions in the order of target_series from config
        # We need to map these to available_targets order
        if n_series_actual < len(available_targets):
            # Model returned predictions only for target_series from config
            # Map these to available_targets order
            # Get target_series from model config
            target_series_from_model = None
            if hasattr(model, '_config') and model._config is not None:
                config = model._config
                target_series_from_model = getattr(config, 'target_series', None)
                if target_series_from_model and not isinstance(target_series_from_model, list):
                    target_series_from_model = [target_series_from_model]
            
            if target_series_from_model:
                # Create mapping: forecast_values[i] corresponds to target_series_from_model[i]
                # Map to available_targets order
                forecast_values_mapped = np.full(len(available_targets), np.nan)
                for idx, ts in enumerate(target_series_from_model):
                    if ts in available_targets:
                        available_idx = available_targets.index(ts)
                        if idx < len(forecast_values):
                            forecast_values_mapped[available_idx] = forecast_values[idx]
                forecast_values = forecast_values_mapped
                logger.debug(f"Mapped {n_series_actual} predictions from model to {len(available_targets)} available_targets")
            else:
                # Fallback: assume predictions are in same order as first n_series_actual of available_targets
                logger.warning(f"Model returned {n_series_actual} predictions but {len(available_targets)} available_targets. Cannot map without target_series from config.")
                # Pad with NaN or repeat last value
                if len(forecast_values) < len(available_targets):
                    forecast_values = np.pad(forecast_values, (0, len(available_targets) - len(forecast_values)), mode='constant', constant_values=np.nan)
        
        # #region agent log
        _debug_log(
            f"dfm.py:548-after-predict-{i}",
            "After model.predict() - forecast values",
            {
                "week": i+1,
                "forecast_values": forecast_values.tolist() if isinstance(forecast_values, np.ndarray) else [float(forecast_values)],
                "forecast_values_mean": float(np.mean(forecast_values)) if isinstance(forecast_values, np.ndarray) else float(forecast_values),
                "forecast_values_std": float(np.std(forecast_values)) if isinstance(forecast_values, np.ndarray) else 0.0,
                "available_targets": available_targets[:3],
                "koequipte_idx_in_available": available_targets.index('KOEQUIPTE') if 'KOEQUIPTE' in available_targets else None,
                "koequipte_value": float(forecast_values[available_targets.index('KOEQUIPTE')]) if 'KOEQUIPTE' in available_targets and isinstance(forecast_values, np.ndarray) and len(forecast_values) > available_targets.index('KOEQUIPTE') else None
            },
            "H2"
        )
        # #endregion
        
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
    logger.info(f"Factor model approach: rolling window updates ({window_size_weeks} weeks) for each forecast (provides sufficient context for factor estimation)")
    
    # Compute and save metrics
    if len(predictions) > 0 and len(actuals) > 0:
        actuals_np = np.asarray(actuals, dtype=float)
        if not np.isnan(actuals_np).all():
            try:
                # #region agent log
                _debug_log(
                    "dfm.py:1350-before-recursive-metrics",
                "Before computing recursive forecast metrics",
                {
                    "predictions_shape": list(predictions.shape),
                    "actuals_shape": list(actuals.shape),
                    "predictions_sample": predictions[0, :5].tolist() if predictions.shape[0] > 0 and predictions.shape[1] >= 5 else None,
                    "actuals_sample": actuals[0, :5].tolist() if actuals.shape[0] > 0 and actuals.shape[1] >= 5 else None,
                    "predictions_min": float(predictions.min()),
                    "predictions_max": float(predictions.max()),
                    "predictions_mean": float(predictions.mean()),
                    "actuals_min": float(actuals.min()),
                    "actuals_max": float(actuals.max()),
                    "actuals_mean": float(actuals.mean()),
                    "diff_sample": (predictions[0, :5] - actuals[0, :5]).tolist() if predictions.shape[0] > 0 and actuals.shape[0] > 0 and predictions.shape[1] >= 5 and actuals.shape[1] >= 5 else None,
                    "available_targets": available_targets[:5] if len(available_targets) > 5 else available_targets
                },
                "H1,H2,H3,H4,H5"
            )
                # #endregion
                
                # Get monthly series info for test_data_std computation
                monthly_series = None
                if data_loader is not None and hasattr(data_loader, 'monthly_series'):
                    monthly_series = data_loader.monthly_series
                
                # Compute test_data_std for normalization
                test_data_std = compute_test_data_std(
                    test_data=test_data,
                    target_series=available_targets,
                    monthly_series=monthly_series
                )
                
                # #region agent log
                _debug_log(
                    "dfm.py:1370-after-test-data-std",
                    "After computing test_data_std",
                    {
                        "test_data_std": test_data_std[:5].tolist() if test_data_std is not None and len(test_data_std) >= 5 else (test_data_std.tolist() if test_data_std is not None else None),
                        "monthly_series": list(monthly_series) if monthly_series else None
                    },
                    "H1,H5"
                )
                # #endregion
                
                # Compute metrics
                std_for_metrics = validate_std_for_metrics(test_data_std, len(available_targets))
                
                if std_for_metrics is not None and not np.isnan(std_for_metrics).all():
                    # #region agent log
                    _debug_log(
                        "dfm.py:1378-before-compute-metrics",
                        "Before compute_smse/smae",
                        {
                            "std_for_metrics": std_for_metrics[:5].tolist() if len(std_for_metrics) >= 5 else std_for_metrics.tolist(),
                            "predictions_shape": list(predictions.shape),
                            "actuals_shape": list(actuals.shape)
                        },
                        "H1,H2,H3,H4,H5"
                    )
                    # #endregion
                    
                    smse = compute_smse(actuals, predictions, test_data_std=std_for_metrics)
                    smae = compute_smae(actuals, predictions, test_data_std=std_for_metrics)
                    
                    # #region agent log
                    _debug_log(
                        "dfm.py:1385-after-compute-metrics",
                        "After compute_smse/smae",
                        {
                            "smse": float(smse) if not np.isnan(smse) else None,
                            "smae": float(smae) if not np.isnan(smae) else None,
                            "is_smse_nan": bool(np.isnan(smse)),
                            "is_smae_nan": bool(np.isnan(smae))
                        },
                        "H1,H2,H3,H4,H5"
                    )
                    # #endregion
                    
                    metrics = {
                        "smse": float(smse) if not np.isnan(smse) else None,
                        "smae": float(smae) if not np.isnan(smae) else None
                    }
                    
                    # Log metrics
                    smse_str = f"{smse:.6f}" if not np.isnan(smse) else "NaN"
                    smae_str = f"{smae:.6f}" if not np.isnan(smae) else "NaN"
                    logger.info(f"DFM Recursive Forecast Metrics ({len(predictions)} periods): sMSE={smse_str}, sMAE={smae_str}")
                    
                    # Save metrics
                    output_dir = checkpoint_path.parent / "forecasts"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    save_experiment_results(
                        output_dir=output_dir,
                        predictions=predictions,
                        actuals=actuals,
                        dates=forecast_dates,
                        target_series=available_targets,
                        metrics=metrics
                    )
                    
                    logger.info(f"Metrics saved to: {output_dir / 'metrics.json'}")
                else:
                    logger.warning("Could not compute test_data_std, skipping metrics computation")
            except Exception as e:
                logger.warning(f"Failed to compute metrics: {e}", exc_info=True)
    else:
        logger.warning("No valid predictions or actuals available, skipping metrics computation")
    
    return predictions, actuals, forecast_dates


def _run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame],
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
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
    
    # Load model (dataset metadata is now stored in model checkpoint)
    logger.info("Loading DFM model for multi-horizon forecasting...")
    model = DFM.load(checkpoint_path)
    
    # Get dataset metadata from model checkpoint
    dataset_metadata = None
    if hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
        dataset_metadata = model._checkpoint_metadata.get('dataset_metadata')
    
    # Get covariates from config
    if hasattr(model, '_config') and model._config is not None:
        config = model._config
        config_covariates = getattr(config, 'covariates', None)
        if config_covariates and covariates is None:
            covariates = config_covariates if isinstance(config_covariates, list) else [config_covariates]
            logger.info(f"Using covariates from model config: {covariates}")
    
    # Use dataset metadata from checkpoint if available
    if dataset_metadata:
        if covariates is None and dataset_metadata.get('covariates'):
            covariates = dataset_metadata['covariates']
    
    # Get training column order (critical for proper standardization)
    training_column_order = _get_training_column_order(model, test_data, logger)
    
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
            # PATTERN: Pass transformed (chg, log, etc.) but NOT standardized data to model.predict(update=True, data=...)
            #          model.predict() will standardize internally using model.scaler
            # CRITICAL: Column order must match training_column_order for correct standardization
            if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
                # Get processed data with proper column alignment
                common_dates = train_data.index.intersection(data_loader.processed.index)
                if len(common_dates) > 0:
                    processed_df = data_loader.processed.loc[common_dates]
                    # CRITICAL: Align columns to match training order exactly
                    # This ensures scaler applies correct mean/scale to each column
                    if training_column_order is not None:
                        # Align columns to training order
                        processed_df_aligned = _align_dataframe_columns(processed_df, training_column_order, logger)
                        # Extract values in correct order
                        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                        data_cols = [c for c in training_column_order if c in processed_df_aligned.columns]
                        train_data_processed = processed_df_aligned[data_cols].values
                    else:
                        # Fallback: extract numeric columns (order may not match training - RISKY!)
                        logger.warning("Training column order not available. Using fallback column extraction (may cause scaling errors!)")
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                        train_data_processed = processed_df[numeric_cols].values
                    
                    # Validate column count matches training data
                    if hasattr(model, 'data_processed') and model.data_processed is not None:
                        expected_cols = model.data_processed.shape[1]
                        actual_cols = train_data_processed.shape[1] if train_data_processed.ndim > 1 else 1
                        if actual_cols != expected_cols:
                            raise ValueError(
                                f"Column count mismatch: update data has {actual_cols} columns, "
                                f"but training data has {expected_cols} columns. "
                                f"Training columns: {len(training_column_order) if training_column_order else 'unknown'}"
                            )
                else:
                    # Fallback: use raw data filtered to training columns
                    if training_column_order is not None:
                        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
                        # Filter train_data to only training columns
                        train_data_filtered = train_data[[c for c in training_numeric_cols if c in train_data.columns]].copy()
                        train_data_processed = train_data_filtered.values
                        logger.warning("Using fallback column filtering (may cause scaling errors if order differs!)")
                    else:
                        # Fallback: extract numeric columns (order may not match training - RISKY!)
                        logger.warning("No column order available. Using fallback extraction (may cause scaling errors!)")
                        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
                        train_data_processed = train_data[numeric_cols].values
            else:
                # Fallback: use raw data filtered to training columns
                if training_column_order is not None:
                    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
                    training_numeric_cols = [c for c in training_column_order if c not in date_cols]
                    # Filter train_data to only training columns
                    train_data_filtered = train_data[[c for c in training_numeric_cols if c in train_data.columns]].copy()
                    train_data_processed = train_data_filtered.values
                    logger.warning("Using fallback column filtering (may cause scaling errors if order differs!)")
                else:
                    # Fallback: extract numeric columns (order may not match training - RISKY!)
                    logger.warning("No column order available. Using fallback extraction (may cause scaling errors!)")
                    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
                    train_data_processed = train_data[numeric_cols].values
            # Update factors with data up to start_date
            # This updates the latent factor state via Kalman filtering
            # CRITICAL: Pass raw transformed data (not standardized) - model.predict(update=True) standardizes internally
            # Column order must match training_column_order for correct standardization
            if training_column_order is None:
                logger.warning(
                    f"Could not determine training column order for multi-horizon update. "
                    f"Data columns may not match training order, causing incorrect standardization."
                )
            # Update will be done internally in predict() when update=True
            logger.info(f"Will update factors with {len(train_data_processed)} observations up to {start_date_ts.date()}")
        else:
            train_data_processed = None
    
    # Determine number of series for forecast extraction (must match training data)
    # For DFM, use dataset_metadata or training_column_order (dataset object not available)
    if dataset_metadata and dataset_metadata.get('target_series'):
        target_list = dataset_metadata['target_series']
        n_series = len(target_list) if isinstance(target_list, list) and len(target_list) > 0 else 1
    elif training_column_order is not None:
        # Use training column order to determine number of series (CRITICAL: must match training)
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
        if covariates:
            available_covariates = [c for c in covariates if c in training_numeric_cols]
            n_series = len(training_numeric_cols) - len(available_covariates)
        else:
            n_series = len(training_numeric_cols)
        logger.debug(f"Using training column order: {n_series} series (from {len(training_numeric_cols)} training series)")
    elif hasattr(model, 'data_processed') and model.data_processed is not None:
        # Use model's training data shape
        n_series = model.data_processed.shape[1]
        logger.debug(f"Using model training data shape: {n_series} series")
    elif test_data is not None:
        # Fallback: use test_data (may cause mismatch)
        train_data_check = test_data[test_data.index < start_date_ts].copy()
        if len(train_data_check) > 0:
            date_cols = {'date', 'date_w', 'year', 'month', 'day'}
            numeric_cols = [c for c in train_data_check.columns if c not in date_cols and pd.api.types.is_numeric_dtype(train_data_check[c])]
            if covariates:
                available_covariates = [c for c in covariates if c in numeric_cols]
                n_series = len(numeric_cols) - len(available_covariates)
            else:
                n_series = len(numeric_cols)
        else:
            n_series = 1
        logger.warning(f"Using test_data columns for n_series: {n_series}. This may cause mismatch if test_data has different columns than training data.")
    else:
        n_series = 1
    
    # For structural factor models: predict once with max horizon, then extract time steps
    # The model structure (AR dynamics of factors) naturally defines all horizons
    # CRITICAL: Ensure model config has correct series_ids matching training data
    # If config has wrong series_ids (e.g., 40 instead of 38), update it to match training column order
    if training_column_order is not None and hasattr(model, '_config') and model._config is not None:
        date_cols = {'date', 'date_w', 'year', 'month', 'day'}
        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
        config_series_ids = model._config.get_series_ids() if hasattr(model._config, 'get_series_ids') else None
        if config_series_ids is None or len(config_series_ids) != len(training_numeric_cols):
            # Update config frequency dict to match training column order
            if hasattr(model._config, 'frequency'):
                # Create frequency dict from training columns (use clock frequency for all)
                clock_freq = getattr(model._config, 'clock', 'w')  # Default to weekly
                model._config.frequency = {col: clock_freq for col in training_numeric_cols}
                logger.debug(f"Updated model config frequency dict to match {len(training_numeric_cols)} training series")
    
    logger.info(f"Predicting up to {max_horizon} weeks ahead (structural forecast)...")
    # Predict with update if data was provided
    if train_data_processed is not None:
        forecast_result = model.predict(horizon=max_horizon, data=train_data_processed, update=True)
    else:
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
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str = "dfm",
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
    model_type: str = "dfm",
    covariates: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    dataset_path: Optional[Path] = None  # Deprecated: dataset metadata now stored in checkpoint
) -> Dict[int, np.ndarray]:
    """Run multi-horizon forecasting from fixed start point.
    
    Note: dataset_path parameter is deprecated. Dataset metadata is now stored in model checkpoint.
    """
    return _run_multi_horizon_forecast(
        checkpoint_path, horizons, start_date,
        test_data, covariates=covariates, data_loader=data_loader
    )
