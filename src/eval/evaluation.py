"""Evaluation helper functions for forecasting experiments.

This module provides standardized metric calculation functions for evaluating
forecasting model performance, including standardized MSE, MAE, and RMSE.
"""

from typing import Union, Dict, Optional, Tuple, Any, List
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from sktime.performance_metrics.forecasting import (
        MeanSquaredError,
        MeanAbsoluteError,
    )
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    MeanSquaredError = None
    MeanAbsoluteError = None


def calculate_standardized_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """Calculate standardized MSE, MAE, and RMSE metrics.
    
    Standardized metrics are normalized by the standard deviation of the
    training data to enable fair comparison across different series and scales.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        Actual/true values (T × N) or (T,)
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values (T × N) or (T,), same shape as y_true
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used to calculate standardization factor (σ).
        If None, uses y_true to calculate σ (not recommended for out-of-sample evaluation)
    target_series : str or int, optional
        If multivariate, specify which series to evaluate (column name or index).
        If None, evaluates all series and returns average.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'sMSE': Standardized Mean Squared Error (MSE / σ²)
        - 'sMAE': Standardized Mean Absolute Error (MAE / σ)
        - 'sRMSE': Standardized Root Mean Squared Error (RMSE / σ)
        - 'MSE': Raw Mean Squared Error
        - 'MAE': Raw Mean Absolute Error
        - 'RMSE': Raw Root Mean Squared Error
        - 'sigma': Standard deviation used for standardization
        
    Notes
    -----
    - Standardization uses training data standard deviation (σ) to normalize metrics
    - For multivariate data, metrics are calculated per series and averaged
    - Missing values (NaN) are automatically excluded from calculations
    - If all values are NaN, returns NaN for all metrics
    """
    # Convert to numpy arrays and track original types
    is_y_true_series = isinstance(y_true, pd.Series)
    is_y_true_dataframe = isinstance(y_true, pd.DataFrame)
    
    if is_y_true_dataframe:
        y_true_arr = y_true.values
        columns = y_true.columns
    elif is_y_true_series:
        y_true_arr = y_true.values.reshape(-1, 1)
        columns = [y_true.name] if y_true.name else [0]
    else:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        columns = list(range(y_true_arr.shape[1]))
    
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    elif isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values.reshape(-1, 1)
    else:
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Ensure same shape
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )
    
    # Select target series if specified
    if target_series is not None:
        if isinstance(target_series, str):
            if is_y_true_dataframe:
                # DataFrame: look up column name
                col_idx = y_true.columns.get_loc(target_series)
            elif is_y_true_series:
                # Series: check if name matches, otherwise use index 0
                if y_true.name == target_series:
                    col_idx = 0
                else:
                    # Name doesn't match, but Series has only one column - use 0
                    col_idx = 0
            else:
                # Not DataFrame or Series - can't use string index
                raise ValueError("target_series must be int if y_true is not DataFrame or Series")
        else:
            col_idx = target_series
        
        y_true_arr = y_true_arr[:, col_idx:col_idx+1]
        y_pred_arr = y_pred_arr[:, col_idx:col_idx+1]
        columns = [columns[col_idx]]
    
    # Calculate standardization factor from training data
    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            train_arr = y_train.values
            if target_series is not None:
                if isinstance(target_series, str):
                    col_idx = y_train.columns.get_loc(target_series)
                else:
                    col_idx = target_series
                train_arr = train_arr[:, col_idx:col_idx+1]
        elif isinstance(y_train, pd.Series):
            train_arr = y_train.values.reshape(-1, 1)
        else:
            train_arr = np.asarray(y_train)
            if train_arr.ndim == 1:
                train_arr = train_arr.reshape(-1, 1)
    else:
        # Use y_true as fallback (not ideal for out-of-sample)
        train_arr = y_true_arr
    
    # Calculate standard deviation per series
    sigma = np.nanstd(train_arr, axis=0, ddof=1)
    # Avoid division by zero
    sigma = np.where(sigma == 0, 1.0, sigma)
    
    # Create mask for valid (non-NaN) values
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    
    # Debug logging for mask calculation
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"calculate_standardized_metrics: y_true_arr shape={y_true_arr.shape}, y_pred_arr shape={y_pred_arr.shape}")
    logger.debug(f"calculate_standardized_metrics: mask sum={np.sum(mask)}, mask shape={mask.shape}, n_valid={int(np.sum(mask))}")
    if np.sum(mask) == 0:
        logger.warning(f"calculate_standardized_metrics: mask is all False! y_true_arr finite={np.sum(np.isfinite(y_true_arr))}, y_pred_arr finite={np.sum(np.isfinite(y_pred_arr))}")
        logger.warning(f"calculate_standardized_metrics: y_true_arr has NaN={np.sum(np.isnan(y_true_arr))}, y_pred_arr has NaN={np.sum(np.isnan(y_pred_arr))}")
        logger.warning(f"calculate_standardized_metrics: y_true_arr has Inf={np.sum(np.isinf(y_true_arr))}, y_pred_arr has Inf={np.sum(np.isinf(y_pred_arr))}")
    
    # Calculate raw metrics per series
    n_series = y_true_arr.shape[1]
    mse_per_series = np.zeros(n_series)
    mae_per_series = np.zeros(n_series)
    rmse_per_series = np.zeros(n_series)
    
    for i in range(n_series):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            y_true_series = y_true_arr[series_mask, i]
            y_pred_series = y_pred_arr[series_mask, i]
            
            # Calculate raw metrics
            mse_per_series[i] = np.mean((y_true_series - y_pred_series) ** 2)
            mae_per_series[i] = np.mean(np.abs(y_true_series - y_pred_series))
            rmse_per_series[i] = np.sqrt(mse_per_series[i])
        else:
            mse_per_series[i] = np.nan
            mae_per_series[i] = np.nan
            rmse_per_series[i] = np.nan
    
    # Average across series
    mse = np.nanmean(mse_per_series)
    mae = np.nanmean(mae_per_series)
    rmse = np.nanmean(rmse_per_series)
    sigma_mean = np.nanmean(sigma)
    
    # Calculate standardized metrics
    sMSE = mse / (sigma_mean ** 2) if sigma_mean > 0 else np.nan
    sMAE = mae / sigma_mean if sigma_mean > 0 else np.nan
    sRMSE = rmse / sigma_mean if sigma_mean > 0 else np.nan
    
    return {
        'sMSE': sMSE,
        'sMAE': sMAE,
        'sRMSE': sRMSE,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'sigma': sigma_mean,
        'n_valid': int(np.sum(mask))
    }


def calculate_metrics_per_horizon(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    horizons: Union[list, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """Calculate standardized metrics for each forecast horizon.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        Actual values (T × N) or (T,)
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted values (T × N) or (T,)
        Predictions from sktime predict(fh=[1,2,...,max_horizon]) are indexed by time,
        where index 0 corresponds to horizon 1 (1 step ahead from training end).
    horizons : list or np.ndarray
        List of forecast horizons (e.g., [1, 7, 28] for 1, 7, 28 days ahead)
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data for standardization
    target_series : str or int, optional
        Target series to evaluate (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary with horizon as key and metrics dict as value
        Example: {1: {'sMSE': 0.5, 'sMAE': 0.3, ...}, 7: {...}, ...}
    """
    horizons = np.asarray(horizons)
    results = {}
    
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        
        # Extract predictions for horizon h
        # When predict(fh=[1,2,...,max_horizon]) is called, predictions are indexed by time:
        # - Position 0 (or index train_end+1) = horizon 1 prediction
        # - Position 1 (or index train_end+2) = horizon 2 prediction
        # - Position h-1 (or index train_end+h) = horizon h prediction
        # Test data y_true starts at the same index (train_end+1), so we can align by position
        # OR by matching indices if both have time-based indices
        
        # Try to extract by matching indices first (if both have time-based indices)
        if isinstance(y_pred, (pd.DataFrame, pd.Series)) and isinstance(y_true, (pd.DataFrame, pd.Series)):
            # Both have indices - try to match by index
            pred_idx = h - 1  # Position in prediction array (0-indexed, so h-1 for horizon h)
            if pred_idx < len(y_pred):
                # Get the index value at position pred_idx
                pred_time_idx = y_pred.index[pred_idx] if hasattr(y_pred.index, '__getitem__') else None
                
                # Try to find matching index in y_true
                if pred_time_idx is not None and pred_time_idx in y_true.index:
                    y_pred_h = y_pred.loc[[pred_time_idx]]
                    y_true_h = y_true.loc[[pred_time_idx]]
                else:
                    # Fall back to position-based extraction
                    y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else (pd.DataFrame() if isinstance(y_pred, pd.DataFrame) else pd.Series())
                    y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else (pd.DataFrame() if isinstance(y_true, pd.DataFrame) else pd.Series())
            else:
                y_pred_h = pd.DataFrame() if isinstance(y_pred, pd.DataFrame) else pd.Series()
                y_true_h = pd.DataFrame() if isinstance(y_true, pd.DataFrame) else pd.Series()
        else:
            # Use position-based extraction for arrays or when indices don't match
            pred_idx = h - 1  # Position for horizon h (0-indexed)
            if isinstance(y_pred, pd.DataFrame):
                y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else pd.DataFrame()
            elif isinstance(y_pred, pd.Series):
                y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else pd.Series()
            else:
                y_pred_h = y_pred[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else np.array([])
            
            if isinstance(y_true, pd.DataFrame):
                y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else pd.DataFrame()
            elif isinstance(y_true, pd.Series):
                y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else pd.Series()
            else:
                y_true_h = y_true[pred_idx:pred_idx+1] if pred_idx < len(y_true) else np.array([])
        
        # Check if we have valid data
        has_pred = len(y_pred_h) > 0 if hasattr(y_pred_h, '__len__') else (y_pred_h.size > 0 if hasattr(y_pred_h, 'size') else False)
        has_true = len(y_true_h) > 0 if hasattr(y_true_h, '__len__') else (y_true_h.size > 0 if hasattr(y_true_h, 'size') else False)
        
        # Calculate metrics for this horizon
        if has_pred and has_true:
            try:
                # Fix: If y_true_h is Series (not DataFrame), target_series must be None or int, not string
                target_series_for_metrics = target_series
                if isinstance(y_true_h, pd.Series) and isinstance(target_series, str):
                    # y_true_h is Series but target_series is string - set to None
                    target_series_for_metrics = None
                elif isinstance(y_true_h, pd.DataFrame) and len(y_true_h.columns) == 1:
                    # Single column DataFrame - can use None or column name
                    if isinstance(target_series, str) and target_series not in y_true_h.columns:
                        # target_series string doesn't match column name - set to None
                        target_series_for_metrics = None
                
                metrics = calculate_standardized_metrics(
                    y_true_h, y_pred_h, y_train=y_train, target_series=target_series_for_metrics
                )
                results[h] = metrics
            except Exception as e:
                # If calculation fails, return NaN metrics
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
        else:
            results[h] = {
                'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'sigma': np.nan, 'n_valid': 0
            }
    
    return results


def evaluate_forecaster(
    forecaster,
    y_train: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.DataFrame, pd.Series],
    horizons: Union[list, np.ndarray],
    target_series: Optional[Union[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """Evaluate a sktime-compatible forecaster on test data.
    
    Parameters
    ----------
    forecaster : sktime BaseForecaster
        Fitted forecaster (must have fit() and predict() methods)
    y_train : pd.DataFrame or pd.Series
        Training data
    y_test : pd.DataFrame or pd.Series
        Test data
    horizons : list or np.ndarray
        Forecast horizons to evaluate
    target_series : str or int, optional
        Target series to evaluate
        
    Returns
    -------
    dict
        Dictionary with horizon as key and metrics dict as value
    """
    # Fit forecaster
    forecaster.fit(y_train)
    
    # Generate predictions for requested horizons only
    horizons_arr = np.asarray(horizons)
    horizons_arr = np.sort(horizons_arr)  # Sort horizons
    
    # Calculate metrics per horizon by matching indices
    results = {}
    for h in horizons_arr:
        h = int(h)
        if h <= 0:
            continue
        
        # Get prediction for horizon h
        # When predict(fh=[h]) is called, it may return:
        # - Single prediction for horizon h (most common)
        # - h predictions for horizons 1 to h (some forecasters)
        # Test data starts at train_end+1, so horizon h corresponds to position h-1 in test data
        try:
            # Predict for this specific horizon
            # Try both fh=[h] (list) and fh=h (int) for compatibility
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                y_pred_h = forecaster.predict(fh=[h])
                logger.info(f"Horizon {h}: predict(fh=[{h}]) succeeded, type={type(y_pred_h)}, shape={getattr(y_pred_h, 'shape', 'N/A')}, length={len(y_pred_h) if hasattr(y_pred_h, '__len__') else 'N/A'}")
            except (TypeError, ValueError) as e:
                # Some forecasters might not accept list, try int
                logger.debug(f"Horizon {h}: predict(fh=[{h}]) failed with {type(e).__name__}: {e}, trying fh={h}")
                try:
                    y_pred_h = forecaster.predict(fh=h)
                    logger.info(f"Horizon {h}: predict(fh={h}) succeeded, type={type(y_pred_h)}, shape={getattr(y_pred_h, 'shape', 'N/A')}, length={len(y_pred_h) if hasattr(y_pred_h, '__len__') else 'N/A'}")
                except Exception as e2:
                    logger.error(f"Horizon {h}: Both predict(fh=[{h}]) and predict(fh={h}) failed. Last error: {type(e2).__name__}: {e2}")
                    raise
            
            # Extract corresponding test data point using position-based matching
            # This is more reliable than index matching since test data is created by splitting
            test_pos = h - 1
            
            # Fix: Check if test data has enough points for this horizon
            if test_pos >= len(y_test):
                logger.warning(f"Horizon {h}: test_pos {test_pos} >= y_test length {len(y_test)}. Skipping horizon {h} - test set too small.")
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
                continue
            
            # Enhanced debug logging
            logger.info(f"Horizon {h}: test_pos={test_pos}, y_test length={len(y_test)}, y_test type={type(y_test)}, y_test shape={getattr(y_test, 'shape', 'N/A')}")
            if hasattr(y_pred_h, 'index'):
                logger.debug(f"Horizon {h}: y_pred_h index type={type(y_pred_h.index)}, y_pred_h index length={len(y_pred_h.index) if hasattr(y_pred_h.index, '__len__') else 'N/A'}")
            if hasattr(y_test, 'index'):
                logger.debug(f"Horizon {h}: y_test index type={type(y_test.index)}, y_test index length={len(y_test.index) if hasattr(y_test.index, '__len__') else 'N/A'}")
            
            # Extract prediction value(s) - handle both Series and DataFrame
            # For predict(fh=[h]), we want the prediction at horizon h
            # If it returns multiple predictions, take the last one (should be for horizon h)
            # If it returns a single prediction, use it directly
            if isinstance(y_pred_h, pd.DataFrame):
                if len(y_pred_h) > 0:
                    # Always take the last row (should be the prediction for horizon h)
                    y_pred_h = y_pred_h.iloc[-1:].copy()
                    # Extract target series if specified
                    if target_series is not None and target_series in y_pred_h.columns:
                        y_pred_h = y_pred_h[[target_series]]
                    elif isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
                        # Single column, use it
                        y_pred_h = y_pred_h.iloc[:, [0]]
                else:
                    logger.warning(f"Horizon {h}: y_pred_h DataFrame is empty.")
                    y_pred_h = pd.DataFrame()
            elif isinstance(y_pred_h, pd.Series):
                if len(y_pred_h) > 0:
                    # Always take the last value (should be the prediction for horizon h)
                    y_pred_h = y_pred_h.iloc[-1:]
                else:
                    logger.warning(f"Horizon {h}: y_pred_h Series is empty.")
                    y_pred_h = pd.Series()
            else:
                # Handle numpy array or other types
                if hasattr(y_pred_h, '__len__') and len(y_pred_h) > 0:
                    # Convert to Series for consistent handling, take last value
                    y_pred_h = pd.Series([y_pred_h[-1]])
                else:
                    logger.warning(f"Horizon {h}: y_pred_h is empty or unsupported type: {type(y_pred_h)}")
                    y_pred_h = pd.Series()
            
            # Extract corresponding test data (test_pos already validated above)
            if isinstance(y_test, pd.DataFrame):
                y_true_h = y_test.iloc[test_pos:test_pos+1].copy()
                # Extract target series if specified
                if target_series is not None and target_series in y_true_h.columns:
                    y_true_h = y_true_h[[target_series]]
                elif len(y_test.columns) == 1:
                    # Single column, use it
                    y_true_h = y_true_h.iloc[:, [0]]
            elif isinstance(y_test, pd.Series):
                y_true_h = y_test.iloc[test_pos:test_pos+1]
            else:
                y_true_h = y_test[test_pos:test_pos+1]
            
            # Check if we have valid data
            has_pred = len(y_pred_h) > 0 if hasattr(y_pred_h, '__len__') else (y_pred_h.size > 0 if hasattr(y_pred_h, 'size') else False)
            has_true = len(y_true_h) > 0 if hasattr(y_true_h, '__len__') else (y_true_h.size > 0 if hasattr(y_true_h, 'size') else False)
            
            # Enhanced logging for debugging n_valid=0 issue
            logger.info(f"Horizon {h}: After extraction - has_pred={has_pred}, has_true={has_true}")
            if not has_pred:
                logger.warning(f"Horizon {h}: y_pred_h is empty or invalid. Type={type(y_pred_h)}, length={len(y_pred_h) if hasattr(y_pred_h, '__len__') else 'N/A'}, size={getattr(y_pred_h, 'size', 'N/A')}")
            if not has_true:
                logger.warning(f"Horizon {h}: y_true_h is empty or invalid. Type={type(y_true_h)}, length={len(y_true_h) if hasattr(y_true_h, '__len__') else 'N/A'}, size={getattr(y_true_h, 'size', 'N/A')}, test_pos={test_pos}, y_test length={len(y_test)}")
            
            # Additional check: ensure shapes are compatible
            if has_pred and has_true:
                # Align shapes - both should be 1D or 2D with compatible dimensions
                if isinstance(y_pred_h, pd.DataFrame) and isinstance(y_true_h, pd.DataFrame):
                    # Ensure same number of columns
                    if y_pred_h.shape[1] != y_true_h.shape[1]:
                        logger.warning(f"Horizon {h}: Shape mismatch - y_pred_h has {y_pred_h.shape[1]} columns, y_true_h has {y_true_h.shape[1]} columns")
                        has_pred = False
                elif isinstance(y_pred_h, pd.Series) and isinstance(y_true_h, pd.Series):
                    # Both Series - should be compatible
                    pass
                elif isinstance(y_pred_h, pd.DataFrame) and isinstance(y_true_h, pd.Series):
                    # Convert Series to DataFrame for compatibility
                    y_true_h = y_true_h.to_frame()
                elif isinstance(y_pred_h, pd.Series) and isinstance(y_true_h, pd.DataFrame):
                    # Convert Series to DataFrame
                    y_pred_h = y_pred_h.to_frame()
            
            logger.debug(f"Horizon {h}: has_pred={has_pred}, has_true={has_true}, y_pred_h shape={y_pred_h.shape if hasattr(y_pred_h, 'shape') else 'N/A'}, y_true_h shape={y_true_h.shape if hasattr(y_true_h, 'shape') else 'N/A'}")
            
            if has_pred and has_true:
                try:
                    # Fix: If y_true_h is Series (not DataFrame), target_series must be None or int, not string
                    # When y_test is Series, we don't need to specify target_series since there's only one column
                    target_series_for_metrics = target_series
                    if isinstance(y_true_h, pd.Series) and isinstance(target_series, str):
                        # y_true_h is Series but target_series is string - set to None
                        target_series_for_metrics = None
                        logger.debug(f"Horizon {h}: y_true_h is Series, setting target_series=None (was string: {target_series})")
                    elif isinstance(y_true_h, pd.DataFrame) and len(y_true_h.columns) == 1:
                        # Single column DataFrame - can use None or column name
                        if isinstance(target_series, str) and target_series not in y_true_h.columns:
                            # target_series string doesn't match column name - set to None
                            target_series_for_metrics = None
                            logger.debug(f"Horizon {h}: y_true_h has 1 column but target_series '{target_series}' not in columns, setting target_series=None")
                    
                    metrics = calculate_standardized_metrics(
                        y_true_h, y_pred_h, y_train=y_train, target_series=target_series_for_metrics
                    )
                    results[h] = metrics
                except Exception as e:
                    logger.warning(f"Horizon {h}: Error calculating metrics: {e}")
                    results[h] = {
                        'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                        'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                        'sigma': np.nan, 'n_valid': 0
                    }
            else:
                logger.warning(f"Horizon {h}: Missing data - has_pred={has_pred}, has_true={has_true}")
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
        except Exception as e:
            # If prediction fails for this horizon, return NaN metrics
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Horizon {h}: Prediction failed with error: {e}")
            results[h] = {
                'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'sigma': np.nan, 'n_valid': 0
            }
    
    return results


# ========================================================================
# Model Comparison Functions
# ========================================================================

def compare_multiple_models(
    model_results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: Optional[str] = None
) -> Dict[str, Any]:
    """Compare results from multiple forecasting models.
    
    This function takes results from multiple model training runs and
    generates a comparison table with standardized metrics for each horizon.
    
    Parameters
    ----------
    model_results : dict
        Dictionary mapping model name to experiment results.
        Each result should have:
        - 'status': 'completed' or 'failed'
        - 'metrics': Dictionary with training metrics (converged, num_iter, loglik, etc.)
        - 'result': Model result object (optional, for extracting forecasts)
        - 'metadata': Model metadata (optional)
    horizons : List[int]
        List of forecast horizons to compare (e.g., [1, 7, 28])
    target_series : str, optional
        Target series name (for context and filtering)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'metrics_table': pd.DataFrame with metrics per model and horizon
        - 'summary': Summary statistics dictionary
        - 'best_model_per_horizon': Dictionary mapping horizon to best model name
        - 'best_model_overall': Best model across all horizons
        
    Notes
    -----
    - Only models with status='completed' and valid metrics are included
    - Metrics are extracted from training results (converged, num_iter, loglik)
    - For forecast metrics, models need to have prediction results available
    - Standardized metrics (sMSE, sMAE, sRMSE) are preferred for comparison
    """
    # Filter successful models
    successful_models = {
        name: result for name, result in model_results.items()
        if result.get('status') == 'completed' and result.get('metrics') is not None
    }
    
    if len(successful_models) == 0:
        return {
            'metrics_table': None,
            'summary': 'No successful models to compare',
            'best_model_per_horizon': {},
            'best_model_overall': None
        }
    
    # Extract metrics for each model
    comparison_data = []
    
    for model_name, result in successful_models.items():
        metrics = result.get('metrics', {})
        forecast_metrics = metrics.get('forecast_metrics', {})
        
        # Extract training metrics
        row = {
            'model': model_name,
            'converged': metrics.get('converged', False),
            'num_iter': metrics.get('num_iter', 0),
            'loglik': metrics.get('loglik', np.nan),
            'model_type': metrics.get('model_type', 'unknown')
        }
        
        # Extract forecast metrics for each horizon
        for horizon in horizons:
            horizon_metrics = forecast_metrics.get(horizon, {})
            if horizon_metrics:
                row[f'sMSE_h{horizon}'] = horizon_metrics.get('sMSE', np.nan)
                row[f'sMAE_h{horizon}'] = horizon_metrics.get('sMAE', np.nan)
                row[f'sRMSE_h{horizon}'] = horizon_metrics.get('sRMSE', np.nan)
            else:
                row[f'sMSE_h{horizon}'] = np.nan
                row[f'sMAE_h{horizon}'] = np.nan
                row[f'sRMSE_h{horizon}'] = np.nan
        
        comparison_data.append(row)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(comparison_data)
    
    # Generate summary statistics
    summary = {
        'total_models': len(successful_models),
        'converged_models': int(metrics_df['converged'].sum()),
        'avg_loglik': float(metrics_df['loglik'].mean()) if not metrics_df['loglik'].isna().all() else np.nan,
        'best_loglik': float(metrics_df['loglik'].max()) if not metrics_df['loglik'].isna().all() else np.nan,
        'best_model_by_loglik': metrics_df.loc[metrics_df['loglik'].idxmax(), 'model'] if not metrics_df['loglik'].isna().all() else None
    }
    
    # Determine best model per horizon based on sMSE (lower is better)
    best_model_per_horizon = {}
    for horizon in horizons:
        sMSE_col = f'sMSE_h{horizon}'
        if sMSE_col in metrics_df.columns:
            valid_models = metrics_df[metrics_df[sMSE_col].notna()]
            if len(valid_models) > 0:
                best_idx = valid_models[sMSE_col].idxmin()
                best_model_per_horizon[horizon] = metrics_df.loc[best_idx, 'model']
    
    # Best model overall (lowest average sMSE across all horizons)
    sMSE_cols = [f'sMSE_h{h}' for h in horizons if f'sMSE_h{h}' in metrics_df.columns]
    if sMSE_cols:
        metrics_df['avg_sMSE'] = metrics_df[sMSE_cols].mean(axis=1)
        valid_models = metrics_df[metrics_df['avg_sMSE'].notna()]
        if len(valid_models) > 0:
            best_model_overall = valid_models.loc[valid_models['avg_sMSE'].idxmin(), 'model']
        else:
            best_model_overall = summary.get('best_model_by_loglik')
    else:
        best_model_overall = summary.get('best_model_by_loglik')
    
    return {
        'metrics_table': metrics_df,
        'summary': summary,
        'best_model_per_horizon': best_model_per_horizon,
        'best_model_overall': best_model_overall,
        'target_series': target_series,
        'horizons': horizons
    }


def generate_comparison_table(
    comparison_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate a formatted comparison table from comparison results.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_multiple_models() function
    output_path : str, optional
        Path to save the comparison table (CSV format)
        
    Returns
    -------
    pd.DataFrame
        Formatted comparison table
    """
    metrics_table = comparison_results.get('metrics_table')
    
    if metrics_table is None:
        return pd.DataFrame()
    
    # Format table for better readability
    formatted_table = metrics_table.copy()
    
    # Round numeric columns
    numeric_cols = formatted_table.select_dtypes(include=[np.number]).columns
    formatted_table[numeric_cols] = formatted_table[numeric_cols].round(4)
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        formatted_table.to_csv(output_path_obj, index=False, encoding='utf-8')
    
    return formatted_table


def save_comparison_plots(
    comparison_results: Dict[str, Any],
    output_dir: str,
    target_series: str
) -> List[Path]:
    """Save comparison plots for model results.
    
    This function generates and saves visualization plots comparing
    multiple models' performance across different horizons.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_multiple_models() function
    output_dir : str
        Output directory for saving plots
    target_series : str
        Target series name (for plot titles)
        
    Returns
    -------
    List[Path]
        List of paths to saved plot files
        
    Note
    ----
    This function requires matplotlib. Plots are saved to nowcasting-report/images/
    if output_dir is set appropriately, otherwise to the specified output_dir.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return []
    
    metrics_table = comparison_results.get('metrics_table')
    if metrics_table is None or len(metrics_table) == 0:
        print("Warning: No metrics table available for plotting")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    # Plot 1: Log-likelihood comparison
    if 'loglik' in metrics_table.columns and not metrics_table['loglik'].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        models = metrics_table['model']
        logliks = metrics_table['loglik']
        
        ax.bar(models, logliks)
        ax.set_xlabel('Model')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title(f'Model Comparison: Log-Likelihood ({target_series})')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plot_path = output_path / f"comparison_loglik_{target_series}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
    
    # Plot 2: Convergence status
    if 'converged' in metrics_table.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        models = metrics_table['model']
        converged = metrics_table['converged'].astype(int)
        
        ax.bar(models, converged, color=['red' if not c else 'green' for c in metrics_table['converged']])
        ax.set_xlabel('Model')
        ax.set_ylabel('Converged (1=Yes, 0=No)')
        ax.set_title(f'Model Comparison: Convergence Status ({target_series})')
        ax.set_ylim([-0.1, 1.1])
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plot_path = output_path / f"comparison_convergence_{target_series}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
    
    # Also save to nowcasting-report/images/ if possible
    try:
        report_images_dir = Path(__file__).parent.parent / "nowcasting-report" / "images"
        if report_images_dir.exists():
            for plot_path in saved_plots:
                import shutil
                shutil.copy(plot_path, report_images_dir / plot_path.name)
    except Exception as e:
        print(f"Warning: Could not copy plots to nowcasting-report/images/: {e}")
    
    return saved_plots


# ========================================================================
# Result Aggregation Functions
# ========================================================================

def collect_all_comparison_results(outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Collect all comparison results from outputs/comparisons/."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    
    comparisons_dir = outputs_dir / "comparisons"
    if not comparisons_dir.exists():
        return {}
    
    all_results = {}
    
    for comparison_dir in comparisons_dir.iterdir():
        if not comparison_dir.is_dir():
            continue
        
        results_file = comparison_dir / "comparison_results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target_series = data.get('target_series')
            if target_series:
                if target_series not in all_results:
                    all_results[target_series] = []
                all_results[target_series].append(data)
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")
            continue
    
    return all_results


def aggregate_overall_performance(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate overall performance metrics across all models, targets, and horizons."""
    rows = []
    
    for target_series, results_list in all_results.items():
        for result_data in results_list:
            comparison = result_data.get('comparison', {})
            metrics_table = comparison.get('metrics_table')
            
            if metrics_table is None:
                continue
            
            # Convert to DataFrame if it's a dict
            if isinstance(metrics_table, dict):
                df = pd.DataFrame(metrics_table)
            elif isinstance(metrics_table, pd.DataFrame):
                df = metrics_table
            else:
                continue
            
            # Add target_series column
            df['target_series'] = target_series
            
            rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    aggregated = pd.concat(rows, ignore_index=True)
    return aggregated


def main_aggregator():
    """Main entry point for aggregator module."""
    print("=" * 70)
    print("Aggregating Experiment Results")
    print("=" * 70)
    
    # Collect all results
    all_results = collect_all_comparison_results()
    
    if not all_results:
        print("No comparison results found in outputs/comparisons/")
        return
    
    print(f"\nFound results for {len(all_results)} target series:")
    for target, results in all_results.items():
        print(f"  - {target}: {len(results)} comparison(s)")
    
    # Aggregate performance
    aggregated = aggregate_overall_performance(all_results)
    
    if aggregated.empty:
        print("\nNo metrics to aggregate.")
        return
    
    # Save aggregated results
    outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    experiments_dir = outputs_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = experiments_dir / "aggregated_results.csv"
    aggregated.to_csv(output_file, index=False)
    
    print(f"\n✓ Aggregated results saved to: {output_file}")
    print(f"  Total rows: {len(aggregated)}")
    print(f"  Columns: {', '.join(aggregated.columns)}")
