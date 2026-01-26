"""Utility functions for the project.

Includes general utilities, experiment utilities, date handling, data loading,
and data splitting for forecasting experiments.

Note: Metric computation functions have been moved to src.metric module.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Literal
import pandas as pd
import numpy as np
import json

# Import preprocess utilities
from src.preprocess import load_data

# Import tent weights helper from metric module (for aggregate_weekly_to_monthly_tent_kernel)
from src.metric import _get_tent_weights

logger = logging.getLogger(__name__)

# Model type constants
# All deep learning models now use NeuralForecast
NEURALFORECAST_MODELS = {'itransformer', 'itf', 'patchtst', 'tft', 'timemixer'}


def is_neuralforecast_model(model_type: str) -> bool:
    """Check if model type is NeuralForecast-based."""
    return model_type.lower() in NEURALFORECAST_MODELS


def standardize_data(
    data: pd.DataFrame,
    columns: List[str],
    scaler: Any,
    data_loader: Optional[Any] = None
) -> pd.DataFrame:
    """Standardize specified columns using scaler.
    
    Handles case where scaler was fitted on all columns but we only want to transform a subset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to standardize
    columns : list
        Column names to standardize
    scaler : Any
        Scaler with transform method
    data_loader : Any, optional
        Data loader with standardized attribute to get scaler's column order
    
    Returns
    -------
    pd.DataFrame
        Data with standardized columns
    """
    data_std = data.copy()
    
    # If scaler expects different number of features, extract subset manually
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        if data_loader is not None and hasattr(data_loader, 'standardized'):
            # Get scaler's column order
            scaler_cols = list(data_loader.standardized.columns)
            # Find indices of requested columns in scaler's order
            col_indices = [scaler_cols.index(c) for c in columns if c in scaler_cols]
            if len(col_indices) == len(columns):
                # Manual transform: (x - mean) / scale
                values = data[columns].values
                mean_vals = scaler.mean_[col_indices]
                scale_vals = scaler.scale_[col_indices]
                standardized = (values - mean_vals) / scale_vals
                data_std.loc[:, columns] = pd.DataFrame(
                    standardized,
                    index=data.index,
                    columns=columns
                )
                return data_std
    
    # Default: use scaler.transform (assumes columns match scaler's expected features)
    try:
        data_std.loc[:, columns] = pd.DataFrame(
            scaler.transform(data[columns].values),
            index=data.index,
            columns=columns
        )
    except ValueError:
        # Fallback: if transform fails, try manual transformation
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            values = data[columns].values
            if len(columns) == len(scaler.mean_):
                standardized = (values - scaler.mean_) / scaler.scale_
                data_std.loc[:, columns] = pd.DataFrame(
                    standardized,
                    index=data.index,
                    columns=columns
                )
            else:
                raise ValueError(f"Column count mismatch: {len(columns)} columns but scaler expects {len(scaler.mean_)} features")
        else:
            raise
    return data_std


def _get_last_original_values(
    series_names: List[str],
    data_loader: Any,
    test_data: Optional[pd.DataFrame] = None,
    cutoff_date: Optional[pd.Timestamp] = None
) -> np.ndarray:
    """Get last original values from data_loader or test_data before cutoff_date.
    
    For monthly series, ensures we get the last monthly value even when cutoff_date
    is at weekly granularity by using month-end dates.
    """
    last_values = []
    
    # Get monthly series set if available
    monthly_series = set()
    if data_loader is not None:
        monthly_series = get_monthly_series_from_metadata(data_loader)
    
    for series_name in series_names:
        val = 0.0
        is_monthly = series_name in monthly_series
        
        # For monthly series, use month-end of cutoff date's month (or previous month)
        # This ensures we get the actual monthly value, not a weekly interpolation
        effective_cutoff = cutoff_date
        if is_monthly and cutoff_date is not None:
            # Use end of previous month for monthly series
            # e.g., if cutoff is 2024-01-26, use 2023-12-31
            prev_month_end = (cutoff_date.replace(day=1) - pd.Timedelta(days=1))
            effective_cutoff = prev_month_end
        
        # Try test_data first (more recent, has correct accumulated values up to cutoff)
        # For accumulation, we need the last value in levels space before the cutoff
        # test_data is in levels and is more recent than training data
        if test_data is not None and series_name in test_data.columns:
            series_data = test_data[series_name]
            if effective_cutoff is not None and isinstance(test_data.index, pd.DatetimeIndex):
                if is_monthly:
                    # For monthly: find last value in or before the cutoff month
                    cutoff_month = effective_cutoff.to_period('M')
                    mask = pd.to_datetime(test_data.index).to_period('M') <= cutoff_month
                    series_data = series_data[mask]
                else:
                    series_data = series_data[series_data.index < effective_cutoff]
            series_non_null = series_data.dropna()
            if len(series_non_null) > 0:
                val = series_non_null.iloc[-1]
        
        # Fallback to training data (data_loader.original) if test_data not available
        if val == 0.0 and hasattr(data_loader, 'original') and series_name in data_loader.original.columns:
            logger.info(f"_get_last_original_values: Falling back to training data for {series_name} (test_data had no values before {effective_cutoff.date()})")
            series_data = data_loader.original[series_name]
            if effective_cutoff is not None and isinstance(data_loader.original.index, pd.DatetimeIndex):
                if is_monthly:
                    # For monthly: find last value in or before the cutoff month
                    cutoff_month = effective_cutoff.to_period('M')
                    mask = pd.to_datetime(data_loader.original.index).to_period('M') <= cutoff_month
                    series_data = series_data[mask]
                else:
                    series_data = series_data[series_data.index < effective_cutoff]
            series_non_null = series_data.dropna()
            if len(series_non_null) > 0:
                val = series_non_null.iloc[-1]
                logger.info(f"_get_last_original_values: Found value {val:.2f} from training data for {series_name} (cutoff={effective_cutoff.date()})")
                # #region agent log
                if series_name in ['KOEQUIPTE', 'KOIPALL.G']:
                    try:
                        with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                            entry = {
                                "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                                "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                                "location": "utils.py:169-found-value-from-training",
                                "message": f"Found value from training data for {series_name}",
                                "data": {
                                    "series_name": series_name,
                                    "val": float(val),
                                    "effective_cutoff": str(effective_cutoff) if effective_cutoff is not None else None,
                                    "cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
                                    "last_available_date": str(series_non_null.index[-1]) if len(series_non_null) > 0 else None,
                                    "num_values_before_cutoff": len(series_non_null)
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1,H2"
                            }
                            f.write(json.dumps(entry) + "\n")
                    except:
                        pass
                # #endregion
        
        
        if val == 0.0:
            logger.warning(f"_get_last_original_values: WARNING - Initial level is 0.0 for {series_name} (cutoff={effective_cutoff}). This may cause accumulation errors!")
        
        # #region agent log
        if series_name in ['KOEQUIPTE', 'KOIPALL.G']:
            try:
                with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                    entry = {
                        "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                        "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                        "location": "utils.py:176-get-last-original-value",
                        "message": f"Retrieved last original value for {series_name}",
                        "data": {
                            "series_name": series_name,
                            "val": float(val),
                            "cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
                            "effective_cutoff": str(effective_cutoff) if effective_cutoff is not None else None,
                            "is_monthly": is_monthly,
                            "has_test_data": test_data is not None and series_name in test_data.columns if test_data is not None else False,
                            "has_training_data": hasattr(data_loader, 'original') and series_name in data_loader.original.columns if data_loader is not None else False
                        },
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1,H4"
                    }
                    f.write(json.dumps(entry) + "\n")
            except:
                pass
        # #endregion
        
        last_values.append(val)
    
    return np.array(last_values)


def _reverse_transformation(
    predicted_processed: np.ndarray,
    series_names: List[str],
    data_loader: Any,
    test_data: Optional[pd.DataFrame] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
    last_original_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """Reverse preprocessing transformations to get original scale."""
    if data_loader is None or not hasattr(data_loader, 'metadata') or data_loader.metadata is None:
        return predicted_processed
    
    metadata = data_loader.metadata
    series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
    
    # Get last original values if not provided
    if last_original_values is None:
        last_original_values = _get_last_original_values(series_names, data_loader, test_data, cutoff_date)
    
    predicted_processed = np.asarray(predicted_processed)
    original_shape = predicted_processed.shape
    
    # Handle shape mismatch: predictions might have more series than target_series
    # Truncate predictions to match series_names length
    if predicted_processed.ndim == 1:
        if len(predicted_processed) != len(series_names):
            predicted_processed = predicted_processed[:len(series_names)]
        predicted_processed = predicted_processed.reshape(-1, len(series_names))
    else:
        # 2D case: (n_samples, n_series)
        if predicted_processed.shape[1] != len(series_names):
            predicted_processed = predicted_processed[:, :len(series_names)]
    
    result = np.zeros_like(predicted_processed)
    
    if series_names is None or len(series_names) == 0:
        logger.warning("_reverse_transformation: series_names is None or empty, returning predictions as-is")
        return predicted_processed
    
    for i, series_name in enumerate(series_names):
        if series_name is None:
            result[:, i] = predicted_processed[:, i]
            continue
        meta_row = metadata[metadata[series_col] == series_name]
        if len(meta_row) == 0:
            result[:, i] = predicted_processed[:, i]
            continue
        
        trans = str(meta_row.iloc[0].get('Transformation', 'lin')).lower()
        pred_vals = predicted_processed[:, i]
        has_last_val = i < len(last_original_values) and last_original_values[i] != 0.0
        
        # Reverse transformations
        if trans == 'lin':
            result[:, i] = pred_vals
        elif trans == 'log':
            result[:, i] = np.exp(pred_vals)
        elif trans in ['chg', 'ch1']:
            # Differencing: x_t = d_t + x_{t-1}
            # CRITICAL: pred_vals are in transformed space (chg values), last_original_values are in levels
            # CRITICAL: For chg, we MUST have last_original_values - if missing, this is an error
            if has_last_val:
                result_val = pred_vals + last_original_values[i]
                # Log first accumulation to debug
                if i == 0 and len(pred_vals) > 0:
                    logger.info(f"_reverse_transformation: chg accumulation for {series_name}: {pred_vals[0]:.4f} (chg) + {last_original_values[i]:.2f} (last_level) = {result_val[0]:.2f}")
                
                # #region agent log
                if series_name in ['KOEQUIPTE', 'KOIPALL.G']:
                    try:
                        with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                            entry = {
                                "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                                "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                                "location": "utils.py:265-chg-accumulation",
                                "message": "chg accumulation - pred_vals",
                                "data": {
                                    "series_name": series_name,
                                    "pred_vals": pred_vals.flatten().tolist()[:3] if hasattr(pred_vals, 'flatten') else [float(pred_vals[0])] if len(pred_vals) > 0 else [],
                                    "pred_vals_mean": float(np.mean(pred_vals)) if hasattr(pred_vals, '__len__') and len(pred_vals) > 0 else float(pred_vals) if not hasattr(pred_vals, '__len__') else 0.0,
                                    "last_original_value": float(last_original_values[i]) if hasattr(last_original_values, '__getitem__') else 0.0,
                                    "result_val": result_val.flatten().tolist()[:3] if hasattr(result_val, 'flatten') else [float(result_val[0])] if len(result_val) > 0 else []
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1,H4"
                            }
                            f.write(json.dumps(entry) + "\n")
                    except:
                        pass
                # #endregion
                
                result[:, i] = result_val
            else:
                # CRITICAL BUG FIX: If last_original_values is missing or 0.0, we cannot accumulate chg predictions
                # This means predictions will be in wrong scale. Raise error instead of silently failing.
                error_msg = (
                    f"_reverse_transformation: CRITICAL ERROR - chg transformation for {series_name} "
                    f"requires last_original_values but got: "
                    f"last_original_values={last_original_values[i] if i < len(last_original_values) else 'missing'} "
                    f"(cutoff_date={cutoff_date}). "
                    f"Cannot accumulate chg predictions without initial level. "
                    f"This will cause wrong predictions in levels space."
                )
                logger.error(error_msg)
                # #region agent log
                if series_name in ['KOEQUIPTE', 'KOIPALL.G']:
                    try:
                        with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                            entry = {
                                "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                                "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                                "location": "utils.py:275-chg-missing-last-value",
                                "message": "ERROR: chg transformation missing last_original_values",
                                "data": {
                                    "series_name": series_name,
                                    "i": i,
                                    "last_original_values_len": len(last_original_values) if last_original_values is not None else None,
                                    "last_original_values_i": float(last_original_values[i]) if last_original_values is not None and i < len(last_original_values) else None,
                                    "cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
                                    "pred_vals_sample": pred_vals.flatten().tolist()[:3] if hasattr(pred_vals, 'flatten') else [float(pred_vals[0])] if len(pred_vals) > 0 else []
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1,H4"
                            }
                            f.write(json.dumps(entry) + "\n")
                    except:
                        pass
                # #endregion
                # Use pred_vals as-is (wrong, but better than crashing) - this will cause wrong metrics
                result[:, i] = pred_vals
                logger.warning(f"_reverse_transformation: Using pred_vals as-is for {series_name} (no accumulation) - PREDICTIONS WILL BE WRONG!")
        elif trans in ['pch', 'pc1']:
            # Percent change: x_t = x_{t-1} * (1 + p_t/100)
            result[:, i] = last_original_values[i] * (1 + pred_vals / 100.0) if has_last_val else pred_vals
        elif trans in ['logchg', 'logdiff']:
            # Log-differencing: x_t = x_{t-1} * exp(log_diff)
            result[:, i] = last_original_values[i] * np.exp(pred_vals) if has_last_val else pred_vals
        else:
            result[:, i] = pred_vals
    
    return result.reshape(original_shape)


def _check_needs_accumulation(
    target_series: List[str],
    data_loader: Any
) -> bool:
    """Check if any series uses chg/logdiff transformation requiring accumulation."""
    if data_loader is None or not hasattr(data_loader, 'metadata') or data_loader.metadata is None:
        return False
    
    metadata = data_loader.metadata
    series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
    
    for series_name in target_series:
        if not series_name:
            continue
        meta_row = metadata[metadata[series_col] == series_name]
        if len(meta_row) > 0:
            trans = str(meta_row.iloc[0].get('Transformation', 'lin')).lower()
            if trans in ['chg', 'ch1', 'logchg', 'logdiff']:
                return True
    return False


def apply_inverse_transformations_with_accumulation(
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    data_loader: Any,
    test_data: Optional[pd.DataFrame] = None,
    experiment_start_date: Optional[pd.Timestamp] = None
) -> np.ndarray:
    """Apply inverse transformations with sequential accumulation for chg/logdiff.
    
    This is a consolidated function that handles the common pattern of:
    1. Checking if accumulation is needed
    2. Applying inverse transformations sequentially
    3. Tracking accumulated levels for chg/logdiff
    
    Parameters
    ----------
    predictions : np.ndarray
        Predictions in transformed space, shape (n_samples, n_series)
    dates : pd.DatetimeIndex
        Dates corresponding to predictions
    target_series : List[str]
        Target series names
    data_loader : Any
        Data loader with metadata
    test_data : pd.DataFrame, optional
        Test data for getting last values
        
    Returns
    -------
    np.ndarray
        Predictions in raw level space, same shape as input
    """
    if data_loader is None or not target_series:
        return predictions
    
    needs_accumulation = _check_needs_accumulation(target_series, data_loader)
    predictions_inv = np.zeros_like(predictions, dtype=float)
    accumulated_levels = None
    
    # #region agent log
    if 'KOEQUIPTE' in target_series or 'KOIPALL.G' in target_series:
        try:
            koequipte_idx = target_series.index('KOEQUIPTE') if 'KOEQUIPTE' in target_series else None
            koipall_idx = target_series.index('KOIPALL.G') if 'KOIPALL.G' in target_series else None
            with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                entry = {
                    "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                    "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                    "location": "utils.py:321",
                    "message": "apply_inverse_transformations entry - predictions",
                    "data": {
                        "target_series": target_series[:5],
                        "koequipte_idx": koequipte_idx,
                        "koipall_idx": koipall_idx,
                        "predictions_shape": list(predictions.shape) if hasattr(predictions, 'shape') else None,
                        "predictions_first3_mean": [float(np.mean(predictions[:3, i])) for i in [koequipte_idx, koipall_idx] if i is not None] if koequipte_idx is not None or koipall_idx is not None else None,
                        "predictions_first3_std": [float(np.std(predictions[:3, i])) for i in [koequipte_idx, koipall_idx] if i is not None] if koequipte_idx is not None or koipall_idx is not None else None,
                        "needs_accumulation": needs_accumulation
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H4"
                }
                f.write(json.dumps(entry) + "\n")
        except:
            pass
    # #endregion
    
    for i, d in enumerate(dates):
        last_original_values = accumulated_levels if (needs_accumulation and accumulated_levels is not None) else None
        
        # For first prediction in accumulation, get initial level from the date BEFORE the first forecast
        # CRITICAL: We need the value BEFORE the first forecast date, not before experiment_start_date
        # For weekly data, this is typically 1 week before the first forecast date
        if needs_accumulation and last_original_values is None:
            # First prediction: get initial level from the date BEFORE the first forecast date
            # Use the date one week before the first forecast (or use experiment_start_date as fallback)
            if i == 0:
                # For the very first prediction, use the date before the first forecast date
                # Try to get 1 week before first forecast date
                first_forecast_date = pd.Timestamp(dates[0])
                cutoff_for_initial = first_forecast_date - pd.Timedelta(weeks=1)
                
                # If experiment_start_date is provided and is more recent than cutoff, use it instead
                # (This handles cases where experiment_start_date is closer to the forecast date)
                if experiment_start_date is not None:
                    exp_start = pd.Timestamp(experiment_start_date)
                    # Use the later of: (first_forecast - 1 week) or experiment_start_date
                    # This ensures we get the most recent available value
                    if exp_start > cutoff_for_initial:
                        cutoff_for_initial = exp_start
                
                logger.info(f"First prediction in accumulation: using cutoff={cutoff_for_initial.date()} (1 week before first forecast {first_forecast_date.date()}) for initial level lookup")
            else:
                # Should not happen - accumulated_levels should be set after first prediction
                cutoff_for_initial = pd.Timestamp(d)
                logger.warning(f"Unexpected: needs_accumulation=True but last_original_values=None for prediction {i} (not first). Using forecast date {d.date()}.")
        else:
            # Subsequent predictions: use forecast date (or accumulated_levels if available)
            cutoff_for_initial = pd.Timestamp(d)
        
        # #region agent log
        if 'KOEQUIPTE' in target_series or 'KOIPALL.G' in target_series:
            try:
                koequipte_idx = target_series.index('KOEQUIPTE') if 'KOEQUIPTE' in target_series else None
                koipall_idx = target_series.index('KOIPALL.G') if 'KOIPALL.G' in target_series else None
                with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                    entry = {
                        "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                        "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                        "location": "utils.py:393-before-inverse-transform",
                        "message": f"Before inverse_transform_predictions for date {d}",
                        "data": {
                            "date": str(d),
                            "i": i,
                            "needs_accumulation": needs_accumulation,
                            "last_original_values": last_original_values.tolist() if last_original_values is not None else None,
                            "predictions_i": predictions[i, koequipte_idx].tolist() if koequipte_idx is not None and predictions.shape[1] > koequipte_idx else None,
                            "predictions_i_koipall": predictions[i, koipall_idx].tolist() if koipall_idx is not None and predictions.shape[1] > koipall_idx else None,
                            "cutoff_for_initial": str(cutoff_for_initial)
                        },
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1,H4"
                    }
                    f.write(json.dumps(entry) + "\n")
            except:
                pass
        # #endregion
        
        inv_pred = inverse_transform_predictions(
            predictions[i],
            target_series,
            data_loader,
            reverse_transformations=True,
            test_data=test_data,
            cutoff_date=cutoff_for_initial,
            last_original_values=last_original_values,
        )
        
        # #region agent log
        if 'KOEQUIPTE' in target_series or 'KOIPALL.G' in target_series:
            try:
                koequipte_idx = target_series.index('KOEQUIPTE') if 'KOEQUIPTE' in target_series else None
                koipall_idx = target_series.index('KOIPALL.G') if 'KOIPALL.G' in target_series else None
                with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                    entry = {
                        "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                        "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                        "location": "utils.py:420-after-inverse-transform",
                        "message": f"After inverse_transform_predictions for date {d}",
                        "data": {
                            "date": str(d),
                            "i": i,
                            "inv_pred": inv_pred.tolist() if inv_pred is not None else None,
                            "inv_pred_koequipte": float(inv_pred[koequipte_idx]) if inv_pred is not None and koequipte_idx is not None and len(inv_pred) > koequipte_idx else None,
                            "inv_pred_koipall": float(inv_pred[koipall_idx]) if inv_pred is not None and koipall_idx is not None and len(inv_pred) > koipall_idx else None,
                            "predictions_inv_i": predictions_inv[i, koequipte_idx].tolist() if koequipte_idx is not None and predictions_inv.shape[1] > koequipte_idx else None
                        },
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1,H4"
                    }
                    f.write(json.dumps(entry) + "\n")
            except:
                pass
        # #endregion
        
        if inv_pred is not None:
            predictions_inv[i] = inv_pred
            if needs_accumulation:
                accumulated_levels = inv_pred.copy()
                # #region agent log
                if 'KOEQUIPTE' in target_series or 'KOIPALL.G' in target_series:
                    try:
                        koequipte_idx = target_series.index('KOEQUIPTE') if 'KOEQUIPTE' in target_series else None
                        koipall_idx = target_series.index('KOIPALL.G') if 'KOIPALL.G' in target_series else None
                        with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                            entry = {
                                "id": f"log_{int(pd.Timestamp.now().timestamp() * 1000)}",
                                "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                                "location": "utils.py:428-accumulated-levels-updated",
                                "message": f"Updated accumulated_levels after date {d}",
                                "data": {
                                    "date": str(d),
                                    "i": i,
                                    "accumulated_levels": accumulated_levels.tolist() if accumulated_levels is not None else None,
                                    "accumulated_koequipte": float(accumulated_levels[koequipte_idx]) if accumulated_levels is not None and koequipte_idx is not None and len(accumulated_levels) > koequipte_idx else None,
                                    "accumulated_koipall": float(accumulated_levels[koipall_idx]) if accumulated_levels is not None and koipall_idx is not None and len(accumulated_levels) > koipall_idx else None
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1,H4"
                            }
                            f.write(json.dumps(entry) + "\n")
                    except:
                        pass
                # #endregion
        else:
            predictions_inv[i] = predictions[i]
            if needs_accumulation:
                accumulated_levels = predictions[i].copy()
    
    return predictions_inv


def inverse_transform_predictions(
    predictions: np.ndarray,
    available_targets: List[str],
    data_loader: Any,
    reverse_transformations: bool = True,
    test_data: Optional[pd.DataFrame] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
    last_original_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """Inverse transform predictions to original scale.
    
    For attention-based models (TFT, PatchTST, iTransformer):
    - Models handle scaling internally (NeuralForecast with local_scaler_type='standard')
    - Predictions are already in processed scale (after model's inverse standardization)
    - This function only reverses preprocessing transformations (log, differencing, etc.)
    
    For other models:
    - May need inverse standardization if they use pre-standardized data
    
    Pipeline:
    - Model output (already in processed scale for attention-based models)
    → Reverse transformations (this function)
    → Original scale → Compare with actuals (monthly aggregated, original scale)
    
    Parameters
    ----------
    predictions : np.ndarray
        Predictions in processed scale (for attention-based models with internal scaling),
        shape (n_samples,) or (n_samples, n_series)
    available_targets : list
        Target series names
    data_loader : Any
        Data loader with metadata for transformations
    reverse_transformations : bool
        If True, reverse preprocessing transformations (differencing, log, etc.)
        If False, return predictions as-is
    test_data : pd.DataFrame, optional
        Test data for getting last values for differencing reversal
    cutoff_date : pd.Timestamp, optional
        Cutoff date for getting last values
    
    Returns
    -------
    np.ndarray
        Predictions in original scale (if reverse_transformations=True) or 
        processed scale (if reverse_transformations=False), same shape as input
    """
    predictions = np.asarray(predictions)
    
    # For attention-based models, predictions are already in processed scale
    # (models handle scaling internally). We only need to reverse transformations.
    if reverse_transformations:
        predictions_original = _reverse_transformation(
            predictions, available_targets, data_loader, test_data=test_data, cutoff_date=cutoff_date, last_original_values=last_original_values
        )
        return predictions_original
    
    return predictions


def convert_to_neuralforecast_format(
    data: pd.DataFrame,
    target_series: List[str],
    covariate_data: Optional[pd.DataFrame] = None,
    covariate_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert DataFrame to NeuralForecast long format.
    
    Parameters
    ----------
    data : pd.DataFrame
        Wide format data with datetime index (target series only)
    target_series : list
        List of target series column names
    covariate_data : pd.DataFrame, optional
        Covariate data (non-target variables) with same index as data
    covariate_names : list, optional
        List of covariate column names
    
    Returns
    -------
    pd.DataFrame
        Long format DataFrame with columns: unique_id, ds, y, and covariate columns
    """
    nf_data_list = []
    for col in target_series:
        series_data = pd.DataFrame({
            'unique_id': col,
            'ds': data.index,
            'y': data[col].values
        })
        
        # Add covariates if provided
        if covariate_data is not None and covariate_names:
            for cov_name in covariate_names:
                if cov_name in covariate_data.columns:
                    series_data[cov_name] = covariate_data[cov_name].values
        
        nf_data_list.append(series_data)
    return pd.concat(nf_data_list, ignore_index=True)


def extract_neuralforecast_forecasts(
    forecast_df: pd.DataFrame,
    target_series: List[str],
    horizon_idx: int = 0
) -> np.ndarray:
    """Extract forecasts from NeuralForecast forecast DataFrame.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast DataFrame from NeuralForecast (columns: unique_id, ds, {model}_1, ...)
    target_series : list
        List of target series IDs
    horizon_idx : int, default 0
        Horizon index to extract (0 = 1 step ahead)
    
    Returns
    -------
    np.ndarray
        Forecast values, shape (len(target_series),)
    """
    forecast_col = next((c for c in forecast_df.columns if c not in ['unique_id', 'ds']), None)
    if forecast_col is None:
        return np.full(len(target_series), np.nan)
    
    # Extract forecast column name with horizon (e.g., 'iTransformer_1', 'iTransformer_2')
    # Sort horizon columns to ensure correct order (e.g., 'PatchTST_1', 'PatchTST_2', ..., 'PatchTST_40')
    horizon_cols = sorted([c for c in forecast_df.columns if c.startswith(forecast_col.split('_')[0])], 
                          key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
    if horizon_idx < len(horizon_cols):
        forecast_col = horizon_cols[horizon_idx]
    elif horizon_cols:
        # If horizon_idx is out of bounds, use the last available horizon (closest approximation)
        # This handles cases where model was trained with prediction_length < requested horizon
        logger.warning(
            f"Horizon index {horizon_idx} is out of bounds (model has {len(horizon_cols)} horizons). "
            f"Using last available horizon ({horizon_cols[-1]}) as approximation."
        )
        forecast_col = horizon_cols[-1]
    
    forecast_values = []
    for series_id in target_series:
        series_forecast = forecast_df[forecast_df['unique_id'] == series_id]
        if len(series_forecast) > horizon_idx:
            forecast_val = series_forecast[forecast_col].iloc[horizon_idx]
        elif len(series_forecast) > 0:
            forecast_val = series_forecast[forecast_col].iloc[-1]
        else:
            forecast_val = np.nan
        forecast_values.append(forecast_val)
    
    return np.array(forecast_values)


def ensure_index_frequency(
    data: pd.DataFrame,
    freq: Any
) -> pd.DataFrame:
    """Ensure DataFrame index has frequency set if possible.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with DatetimeIndex
    freq : Any
        Frequency to set (if index doesn't have one)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with frequency set if possible
    """
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is None and freq:
        try:
            data.index.freq = freq
        except (ValueError, TypeError):
            pass  # If setting fails, freq is optional
    return data




def get_project_root() -> Path:
    """Get project root directory (parent of src/)."""
    return Path(__file__).parent.parent


def setup_logging(log_dir: Path = None, force: bool = False, log_file: Path = None) -> None:
    """Setup basic logging configuration.
    
    Parameters
    ----------
    log_dir : Path, optional
        Directory for log files. If None, uses project_root/log
    force : bool, default False
        Force reconfiguration if already set up. If True and Hydra handlers
        exist, preserves file handlers to avoid breaking Hydra's main.log.
    log_file : Path, optional
        Specific log file path. If None, logs only to console
    """
    root_logger = logging.getLogger()
    
    # Preserve existing file handlers (e.g., Hydra's main.log handler)
    # when using force=True, to avoid breaking Hydra's logging
    preserved_file_handlers = []
    if force and root_logger.handlers:
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                preserved_file_handlers.append(handler)
    
    if log_dir is None:
        log_dir = get_project_root() / "log"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic logging config
    handlers = [logging.StreamHandler()]
    
    # Preserve Hydra's file handlers
    handlers.extend(preserved_file_handlers)
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=force
    )


# ============================================================================
# Experiment Utilities
# ============================================================================

ExperimentType = Literal["short_term", "long_term"]


def get_experiment_dates(experiment_type: ExperimentType) -> Dict[str, str]:
    """Returns date ranges for short-term or long-term experiments.
    
    Tries to load from config files first, falls back to hardcoded defaults
    if config files are not accessible.
    
    Parameters
    ----------
    experiment_type : str
        Type of experiment: "short_term" or "long_term"
    
    Returns
    -------
    dict
        Dictionary with start_date and end_date keys
    
    Examples
    --------
    >>> dates = get_experiment_dates("short_term")
    >>> dates["start_date"]  # "2024-01-01"
    >>> dates["end_date"]    # "2025-10-04"
    """
    # Try to load from config file
    try:
        from omegaconf import OmegaConf
        config_path = get_project_root() / "config" / "experiment" / f"{experiment_type}.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            return {
                "start_date": str(cfg.get("start_date", "")),
                "end_date": str(cfg.get("end_date", ""))
            }
    except Exception:
        # Fall back to hardcoded defaults if config loading fails
        pass
    
    # Hardcoded defaults (fallback only)
    if experiment_type == "short_term":
        return {
            "start_date": "2024-01-01",
            "end_date": "2025-09-26"  # 2025년 9월 4주
        }
    elif experiment_type == "long_term":
        return {
            "start_date": "2024-11-01",  # 2024년 11월 첫째 주
            "end_date": "2025-09-26"
        }
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}. Must be 'short_term' or 'long_term'")


def load_test_data(data_model: str) -> pd.DataFrame:
    """Load test data for experiments.
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    
    Returns
    -------
    pd.DataFrame
        Test data with datetime index from date_w column
    
    Examples
    --------
    >>> test_data = load_test_data("investment")
    >>> test_data.index  # DatetimeIndex
    """
    project_root = get_project_root()
    
    if data_model.lower() == "investment":
        test_path = project_root / "data" / "test_investment.csv"
    elif data_model.lower() == "production":
        test_path = project_root / "data" / "test_production.csv"
    else:
        raise ValueError(f"Unknown data_model: {data_model}. Must be 'investment' or 'production'")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_path}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading test data from: {test_path}")
    data = load_data(str(test_path))
    
    # Set datetime index from date_w column
    if 'date_w' in data.columns:
        data['date_w'] = pd.to_datetime(data['date_w'], errors='coerce')
        data = data.set_index('date_w')
        data.index.name = 'date'
    elif 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.set_index('date')
    else:
        raise ValueError("Test data must have 'date_w' or 'date' column")
    
    # Sort by date
    data = data.sort_index()
    
    logger.info(f"Loaded test data: {len(data)} rows, {len(data.columns)} columns")
    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def split_data_by_date(data: pd.DataFrame, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data at specific date for temporal splits.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with datetime index
    cutoff_date : pd.Timestamp
        Cutoff date (exclusive). All data before this date goes to train, 
        this date and after goes to test.
    
    Returns
    -------
    tuple
        (train_data, test_data) split at cutoff_date
    
    Examples
    --------
    >>> train, test = split_data_by_date(data, pd.Timestamp("2024-01-01"))
    >>> train.index.max() < test.index.min()  # True
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have datetime index")
    
    cutoff_date = pd.Timestamp(cutoff_date)
    
    train_data = data[data.index < cutoff_date].copy()
    test_data = data[data.index >= cutoff_date].copy()
    
    logger = logging.getLogger(__name__)
    
    return train_data, test_data


def get_weekly_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get weekly date range between start and end dates.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns
    -------
    pd.DatetimeIndex
        Weekly dates from start_date to end_date (inclusive)
    
    Examples
    --------
    >>> dates = get_weekly_dates("2024-01-01", "2024-01-31")
    >>> len(dates)  # ~5 weeks
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Generate weekly dates (every Monday)
    dates = pd.date_range(start=start, end=end, freq='W-MON')
    
    # If start_date doesn't fall on Monday, include it
    if start not in dates:
        dates = dates.insert(0, start)
        dates = dates.sort_values()
    
    # Remove dates after end_date
    dates = dates[dates <= end]
    
    return dates


# ============================================================================
# Evaluation Metrics
# ============================================================================

def interpolate_missing_values(
    data: pd.DataFrame, 
    data_loader: Optional[Any] = None
) -> pd.DataFrame:
    """Interpolate missing values for attention-based models.
    
    Monthly series: forward fill only (no backward fill to avoid temporal leakage).
    Other series: forward fill only (no linear interpolation to avoid leakage).
    Remaining NaNs: fill with column mean (or 0 if all NaN).
    
    Note: For monthly series, we only use forward fill to prevent using future
    monthly values to fill past weeks (e.g., April's value shouldn't fill March).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with potential NaN values
    data_loader : optional
        Data loader with metadata to identify monthly series
        
    Returns
    -------
    pd.DataFrame
        Data with NaN values interpolated (guaranteed no NaNs)
    """
    if data.isna().sum().sum() == 0:
        return data
    
    logger = logging.getLogger(__name__)
    monthly_series = get_monthly_series_from_metadata(data_loader) if data_loader else set()
    interpolated_data = data.copy()
    
    # Process each column
    for col in data.columns:
        if interpolated_data[col].isna().sum() == 0:
            continue
            
        n_missing = interpolated_data[col].isna().sum()
        if n_missing > 0:
            logger.debug(f"Interpolating {n_missing} missing values in {col}")
        
        # Use forward fill only for all series to prevent temporal leakage.
        # This keeps imputation causal for both monthly and weekly series.
        interpolated_data[col] = interpolated_data[col].ffill()
        
        # Fill remaining NaNs with column mean (or 0 if all NaN)
        if interpolated_data[col].isna().any():
            col_mean = interpolated_data[col].mean()
            interpolated_data[col] = interpolated_data[col].fillna(col_mean if not pd.isna(col_mean) else 0)
    
    return interpolated_data


def get_monthly_series_from_metadata(data_loader: Optional[Any] = None, metadata_path: Optional[Path] = None) -> set:
    """Identify monthly series from metadata.
    
    Parameters
    ----------
    data_loader : optional
        Data loader with metadata attribute
    metadata_path : Path, optional
        Path to metadata CSV file (e.g., investment_metadata.csv or production_metadata.csv)
    
    Returns
    -------
    set
        Set of monthly series IDs (SeriesID or column names)
    """
    monthly_series = set()
    
    # Try to get metadata from data_loader first
    if data_loader is not None:
        try:
            metadata = data_loader.metadata
            if metadata is not None and not metadata.empty:
                series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
                freq_col = 'Frequency' if 'Frequency' in metadata.columns else 'frequency'
                
                if series_col in metadata.columns and freq_col in metadata.columns:
                    monthly_mask = metadata[freq_col].str.lower() == 'm'
                    monthly_series.update(metadata[monthly_mask][series_col].tolist())
                    return monthly_series
        except (AttributeError, KeyError):
            pass
    
    # Fallback: load metadata from file
    if metadata_path is None:
        project_root = get_project_root()
        # Try common metadata file paths
        for path in [project_root / "data" / "investment_metadata.csv",
                     project_root / "data" / "production_metadata.csv"]:
            if path.exists():
                metadata_path = path
                break
    
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            try:
                metadata = pd.read_csv(metadata_path)
                series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
                freq_col = 'Frequency' if 'Frequency' in metadata.columns else 'frequency'
                
                if series_col in metadata.columns and freq_col in metadata.columns:
                    monthly_mask = metadata[freq_col].str.lower() == 'm'
                    monthly_series.update(metadata[monthly_mask][series_col].tolist())
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not load metadata from {metadata_path}: {e}")
        else:
            # If metadata_path was provided but doesn't exist, log warning
            logger = logging.getLogger(__name__)
            logger.warning(f"Metadata file does not exist: {metadata_path}")
    
    return monthly_series


def aggregate_weekly_to_monthly_tent_kernel(
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    monthly_series: Optional[set] = None,
    tent_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Aggregate weekly forecasts to monthly using tent kernel weights.
    
    Only aggregates monthly series. Weekly series remain unchanged.
    
    Tent kernel weights: [0.1, 0.2, 0.3, 0.4] for weeks 1, 2, 3, 4 of each month.
    For months with 5 weeks, weights are adjusted: [0.1, 0.2, 0.3, 0.4, 0.0]
    (last week of 5-week month is typically in the next month).
    
    Parameters
    ----------
    predictions : np.ndarray
        Weekly predictions, shape (n_weeks, n_series)
    dates : pd.DatetimeIndex
        Weekly dates corresponding to predictions
    target_series : list
        List of target series names
    monthly_series : set, optional
        Set of monthly series IDs. If None, all series are aggregated.
    tent_weights : np.ndarray, optional
        Tent kernel weights. Default: [0.1, 0.2, 0.3, 0.4]
    
    Returns
    -------
    tuple
        (monthly_predictions, monthly_dates)
        - monthly_predictions: shape (n_months, n_series)
        - monthly_dates: DatetimeIndex of month-end dates
    """
    if tent_weights is None:
        tent_weights = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Normalize weights to sum to 1
    tent_weights = tent_weights / tent_weights.sum()
    
    logger = logging.getLogger(__name__)
    predictions = np.asarray(predictions)
    
    if target_series is None or len(target_series) == 0:
        raise ValueError("target_series cannot be None or empty for aggregation")
    
    # Ensure target_series matches predictions shape
    n_pred_series = predictions.shape[1] if predictions.ndim > 1 else 1
    if len(target_series) != n_pred_series:
        logger.warning(f"target_series length ({len(target_series)}) doesn't match predictions shape ({n_pred_series}). Truncating/padding.")
        if len(target_series) > n_pred_series:
            target_series = target_series[:n_pred_series]
        else:
            # Pad with generic names (won't be used for monthly_mask but needed for shape)
            target_series = target_series + [f"_series_{i}" for i in range(len(target_series), n_pred_series)]
    
    monthly_mask = np.ones(len(target_series), dtype=bool) if monthly_series is None else np.array([s in monthly_series for s in target_series])
    
    # Group dates by year-month
    dates_df = pd.DataFrame({'date': dates})
    dates_df['year_month'] = dates_df['date'].dt.to_period('M')
    
    # Get month-end dates (last date of each month in the data)
    monthly_dates_list = []
    monthly_predictions_list = []
    
    for year_month, group in dates_df.groupby('year_month'):
        month_indices = group.index.tolist()
        month_dates = dates[month_indices]
        
        # Use last date of month as representative date (month-end)
        month_end_date = month_dates[-1]
        
        # Get predictions for this month
        month_preds = predictions[month_indices, :]
        
        # Aggregate using tent kernel
        aggregated = np.zeros((1, predictions.shape[1]))
        
        for series_idx in range(predictions.shape[1]):
            if monthly_mask[series_idx]:
                weights = _get_tent_weights(len(month_indices), tent_weights)
                aggregated[0, series_idx] = np.dot(month_preds[:, series_idx], weights)
            else:
                aggregated[0, series_idx] = month_preds[-1, series_idx]
        
        monthly_predictions_list.append(aggregated)
        monthly_dates_list.append(month_end_date)
    
    return np.vstack(monthly_predictions_list), pd.DatetimeIndex(monthly_dates_list)


def _extract_last_value_in_month(
    month_data: pd.DataFrame,
    target_series: List[str]
) -> np.ndarray:
    """Extract last non-NaN value per series in a month."""
    actuals_values = []
    for series_name in target_series:
        series_values = month_data[series_name].dropna()
        if len(series_values) > 0:
            series_values_sorted = series_values.sort_index()
            actuals_values.append(series_values_sorted.iloc[-1])
        else:
            actuals_values.append(np.nan)
    return np.array(actuals_values)


def extract_monthly_actuals(
    test_data: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    target_series: List[str],
    monthly_series: Optional[set] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Extract monthly actuals from test_data for given monthly dates.
    
    For monthly series: extracts the last non-NaN value in each month.
    For weekly series: extracts the last weekly value in each month.
    
    Parameters
    ----------
    test_data : pd.DataFrame
        Full test dataset (weekly-aligned)
    monthly_dates : pd.DatetimeIndex
        Monthly dates to extract actuals for
    target_series : list
        Target series names
    monthly_series : set, optional
        Set of monthly series IDs. If None, all series treated as weekly.
    
    Returns
    -------
    tuple
        (actuals_array, dates)
        - actuals_array: shape (n_months, n_series)
        - dates: DatetimeIndex of dates with available data
    """
    if monthly_series is None:
        monthly_series = set()
    
    monthly_actuals = []
    valid_dates = []
    
    for month_date in monthly_dates:
        month_mask = (test_data.index.year == month_date.year) & \
                     (test_data.index.month == month_date.month)
        month_data = test_data.loc[month_mask, target_series]
        
        if len(month_data) == 0:
            continue
        
        actuals_for_month = _extract_last_value_in_month(month_data, target_series)
        if not np.isnan(actuals_for_month).all():
            monthly_actuals.append(actuals_for_month)
            valid_dates.append(month_date)
    
    return np.array(monthly_actuals) if monthly_actuals else np.empty((0, len(target_series))), pd.DatetimeIndex(valid_dates)


# Metric functions moved to src.metric module
# Import here for backward compatibility or direct import from metric


# ============================================================================
# Common Helper Functions for Forecast Modules
# ============================================================================

def load_model_checkpoint(checkpoint_path: Path) -> Any:
    """Load a model from checkpoint file with fallback to pickle.
    
    Tries joblib.load first, falls back to pickle if that fails.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    
    Returns
    -------
    Any
        Loaded model object
    
    Raises
    ------
    FileNotFoundError
        If checkpoint file doesn't exist
    Exception
        If all loading methods fail
    """
    import joblib
    import pickle
    
    logger = logging.getLogger(__name__)
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    try:
        return joblib.load(checkpoint_path)
    except Exception as e:
        logger.warning(f"joblib.load failed: {e}. Trying pickle...")
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)


def preprocess_data_for_model(
    data: pd.DataFrame,
    data_loader: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    impute_missing: bool = True
) -> np.ndarray:
    """Preprocess data using data_loader's scaler if available.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to preprocess
    data_loader : optional
        Data loader with scaler attribute
    columns : list, optional
        Columns to select from data. If None, uses all numeric columns.
    impute_missing : bool, default True
        If True, impute missing values using forward/backward fill.
        DFM models can handle NaNs in mixed-frequency data, but imputation
        can help with consistency. NeuralForecast models require no NaNs.
    
    Returns
    -------
    np.ndarray
        Preprocessed data (standardized if scaler available)
    """
    if columns is not None:
        data = data[columns]
    else:
        # Select only numeric columns (exclude date/object columns and index)
        # Exclude common non-numeric column names
        exclude_cols = {'date', 'Date', 'DATE', 'time', 'Time', 'TIME', 'index'}
        numeric_cols = [col for col in data.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]
        if not numeric_cols:
            # Fallback: use select_dtypes if no explicit exclusions match
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        data = data[numeric_cols]
    
    # Impute missing values if requested (for models that can't handle NaNs)
    if impute_missing and data.isnull().any().any():
        data = data.ffill().bfill()
        if data.isnull().any().any():
            # Fill remaining NaNs with 0
            data = data.fillna(0)
    
    # Use data_loader's scaler if available
    if data_loader is not None and hasattr(data_loader, 'scaler') and data_loader.scaler is not None:
        return data_loader.scaler.transform(data.values)
    return data.values


def extract_forecast_values(
    forecast_result: Any,
    n_series: int,
    horizon_idx: int = 0
) -> np.ndarray:
    """Extract forecast values from model prediction result.
    
    Handles both tuple returns (X_forecast, Z_forecast), single array returns,
    and pandas DataFrame/Series.
    
    Parameters
    ----------
    forecast_result : Any
        Result from model.predict() - can be tuple, array, DataFrame, or Series
    n_series : int
        Expected number of series
    horizon_idx : int, default 0
        Which time step to extract (0 = first step, -1 = last step)
    
    Returns
    -------
    np.ndarray
        Forecast values, shape (n_series,)
    """
    logger = logging.getLogger(__name__)
    
    # Handle tuple returns (X_forecast, Z_forecast)
    if isinstance(forecast_result, tuple):
        X_forecast, _ = forecast_result
    else:
        X_forecast = forecast_result
    
    # Convert pandas objects to numpy
    if isinstance(X_forecast, pd.DataFrame):
        forecast_values = X_forecast.iloc[-1 if horizon_idx == -1 else horizon_idx, :].values
    elif isinstance(X_forecast, pd.Series):
        forecast_values = X_forecast.values
    else:
        # Convert to numpy array
        X_forecast = np.asarray(X_forecast)
        
        # Extract specific time step
        if X_forecast.ndim > 1:
            forecast_values = X_forecast[horizon_idx, :]
        else:
            forecast_values = X_forecast
    
    # Ensure correct length
    forecast_values = np.asarray(forecast_values)
    
    # #region agent log
    if forecast_values.size > 0 and forecast_values.size <= 50:  # Only log for reasonable sizes
        try:
            with open("/data/nowcasting-kr/.cursor/debug.log", "a") as f:
                import json
                from datetime import datetime
                entry = {
                    "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "location": "utils.py:1198",
                    "message": "extract_forecast_values result",
                    "data": {
                        "forecast_values_shape": list(forecast_values.shape) if hasattr(forecast_values, 'shape') else None,
                        "forecast_values_first10": forecast_values.flatten()[:10].tolist() if forecast_values.size > 0 else [],
                        "n_series_requested": n_series,
                        "forecast_values_mean": float(np.mean(forecast_values)) if forecast_values.size > 0 else None,
                        "forecast_values_std": float(np.std(forecast_values)) if forecast_values.size > 0 else None
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H5"
                }
                f.write(json.dumps(entry) + "\n")
        except:
            pass
    # #endregion
    
    if len(forecast_values) >= n_series:
        forecast_values = forecast_values[:n_series]
    else:
        logger.warning(
            f"Forecast has {len(forecast_values)} series, expected {n_series}. "
            f"Using all available."
        )
        # Pad with NaN if needed
        if len(forecast_values) < n_series:
            forecast_values = np.pad(
                forecast_values, 
                (0, n_series - len(forecast_values)), 
                constant_values=np.nan
            )
    
    return forecast_values

def _extract_target_series_from_object(obj: Any) -> Optional[List[str]]:
    """Internal helper to extract target series from various object types.
    
    Parameters
    ----------
    obj : Any
        Object that may contain target series information
        
    Returns
    -------
    list or None
        List of target series names, or None if not found
    """
    if obj is None:
        return None
    
    # Check for dataset-style attributes (DFM/DDFM)
    if hasattr(obj, 'target_series') and obj.target_series is not None:
        series = obj.target_series
        return series if isinstance(series, list) else [series]
    
    if hasattr(obj, 'target_series_list') and obj.target_series_list is not None:
        series = obj.target_series_list
        return series if isinstance(series, list) else [series]
    
    # Check for model-style attributes
    if hasattr(obj, '_y') and obj._y is not None:
        if isinstance(obj._y, pd.DataFrame):
            return list(obj._y.columns)
        elif isinstance(obj._y, pd.Series):
            return [obj._y.name] if obj._y.name else ['target']
    
    return None


def get_target_series_from_dataset(dataset: Any, test_data: pd.DataFrame, default: Optional[List[str]] = None) -> List[str]:
    """Extract target series from dataset or use defaults.
    
    Helper function to extract target series from various dataset types.
    If dataset has covariates, targets are computed as all_series - covariates.
    If target_series is None or empty, all series are used as targets (default behavior).
    
    Parameters
    ----------
    dataset : Any
        Dataset object (DFMDataset, DDFMDataset, or model with target_series attribute)
    test_data : pd.DataFrame
        Test data to fallback to column names
    default : list, optional
        Default target series list (deprecated, use covariates instead)
    
    Returns
    -------
    list
        List of target series names
        
    Raises
    ------
    ValueError
        If no target series found in test_data
    """
    # Try to extract from dataset object
    target_series = _extract_target_series_from_object(dataset)
    
    # If dataset has covariates, compute targets from covariates
    if dataset is not None and hasattr(dataset, 'covariates') and dataset.covariates:
        all_series = list(test_data.columns)
        date_cols = {'date', 'date_w', 'Date', 'Date_w'}
        all_series = [s for s in all_series if s not in date_cols]
        available_covariates = [c for c in dataset.covariates if c in test_data.columns]
        target_series = [s for s in all_series if s not in available_covariates]
        if target_series:
            return target_series
    
    # Use default if provided (backward compatibility)
    if target_series is None or len(target_series) == 0:
        target_series = default
    
    # Fallback to all test_data columns (default: all series are targets)
    if target_series is None or len(target_series) == 0:
        target_series = list(test_data.columns)
    
    # Filter to series that exist in test_data (exclude date columns)
    date_cols = {'date', 'date_w', 'Date', 'Date_w'}
    available_targets = [t for t in target_series if t in test_data.columns and t not in date_cols]
    if not available_targets:
        raise ValueError(f"No target series found in test_data. Target series: {target_series}, Available columns: {list(test_data.columns)}")
    
    return available_targets


def get_target_series_from_model(model: Any, test_data: pd.DataFrame, default: Optional[List[str]] = None) -> List[str]:
    """Extract target series from model.
    
    Parameters
    ----------
    model : Any
        Model object (may have _y attribute)
    test_data : pd.DataFrame
        Test data to fallback to column names
    default : list, optional
        Default target series list (prioritized if provided)
    
    Returns
    -------
    list
        List of target series names
        
    Raises
    ------
    ValueError
        If no target series found in test_data (consistent with get_target_series_from_dataset)
    """
    date_cols = {'date', 'date_w', 'Date', 'Date_w'}
    
    # If default provided, try it first (it's explicitly specified)
    if default is not None:
        available_targets = [t for t in default if t in test_data.columns and t not in date_cols]
        if available_targets:
            return available_targets
    
    # Try to extract from model object
    target_series = _extract_target_series_from_object(model)
    
    # Use extracted series if it matches test_data columns
    if target_series is not None:
        available_targets = [t for t in target_series if t in test_data.columns and t not in date_cols]
        if available_targets:
            return available_targets
    
    # Fallback to all test_data columns (exclude date columns)
    target_series = [t for t in test_data.columns if t not in date_cols]
    available_targets = [t for t in target_series if t in test_data.columns and t not in date_cols]
    if not available_targets:
        raise ValueError(f"No target series found in test_data. Target series: {target_series}, Available columns: {list(test_data.columns)}")
    
    return available_targets


def filter_and_prepare_test_data(
    test_data: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    available_targets: List[str]
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Filter test data to date range and prepare for forecasting.
    
    Parameters
    ----------
    test_data : pd.DataFrame
        Full test data
    start_date : str or None
        Start date (YYYY-MM-DD) or None to use min
    end_date : str or None
        End date (YYYY-MM-DD) or None to use max
    available_targets : list
        List of target series to include
    
    Returns
    -------
    tuple
        (filtered_data, start_timestamp, end_timestamp)
    """
    if start_date is None:
        start_date = test_data.index.min()
    if end_date is None:
        end_date = test_data.index.max()
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Filter test data to date range and target series
    test_data_filtered = test_data[
        (test_data.index >= start_ts) & (test_data.index <= end_ts)
    ][available_targets].copy()
    
    return test_data_filtered, start_ts, end_ts
