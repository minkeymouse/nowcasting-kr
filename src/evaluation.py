"""Evaluation helper functions for forecasting experiments.

This module provides standardized metric calculation functions for evaluating
forecasting model performance, including standardized MSE, MAE, and RMSE.
"""

from typing import Union, Dict, Optional, Tuple, Any, List
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

# Configure logging if not already configured (only once at module level)
if not hasattr(logging, '_src_evaluation_configured'):
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    logging._src_evaluation_configured = True

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

# Module-level logger (use this consistently across all functions)
_module_logger = logging.getLogger(__name__)


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
    logger = _module_logger
    
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
                try:
                    col_idx = y_true.columns.get_loc(target_series)
                except KeyError:
                    # target_series not in columns - try fallback strategies
                    if len(y_true.columns) == 1:
                        col_idx = 0
                        # Update columns list to match
                        columns = [y_true.columns[0]]
                    else:
                        # Multiple columns but target_series not found
                        # Fallback: try to find in y_pred if it exists
                        if isinstance(y_pred, pd.DataFrame) and target_series in y_pred.columns:
                            # Use y_pred column index, but extract from y_true using same position
                            pred_col_idx = y_pred.columns.get_loc(target_series)
                            # Use same column index in y_true (assuming same order)
                            if pred_col_idx < len(y_true.columns):
                                col_idx = pred_col_idx
                                logger.warning(f"target_series '{target_series}' not found in y_true.columns, using column index {col_idx} from y_pred")
                            else:
                                # Can't match, use first column as last resort
                                col_idx = 0
                                logger.warning(f"target_series '{target_series}' not found in y_true.columns={list(y_true.columns)}, using first column as fallback")
                        else:
                            # Last resort: use first column
                            col_idx = 0
                            logger.warning(f"target_series '{target_series}' not found in y_true.columns={list(y_true.columns)}, using first column as fallback")
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
                    try:
                        col_idx = y_train.columns.get_loc(target_series)
                    except KeyError:
                        # target_series not in y_train.columns - use same col_idx from y_true extraction
                        # This handles case where DFM/DDFM returns columns but target_series not in training data
                        if is_y_true_dataframe:
                            try:
                                col_idx = y_true.columns.get_loc(target_series)
                            except KeyError:
                                # Fallback: use first column if target_series not found in either
                                col_idx = 0
                                logger.warning(f"target_series '{target_series}' not found in y_train.columns={list(y_train.columns)} or y_true.columns={list(y_true.columns)}, using column 0")
                        else:
                            col_idx = 0
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
    
    # Debug logging for mask calculation (logger already defined at function start)
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
    
    # Validate for extreme values (numerical instability)
    # Threshold: values > 1e10 are considered unstable
    EXTREME_THRESHOLD = 1e10
    if isinstance(sMSE, (int, float)) and (abs(sMSE) > EXTREME_THRESHOLD or np.isinf(sMSE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sMSE detected: {sMSE}. This indicates numerical instability.")
        sMSE = np.nan
    if isinstance(sMAE, (int, float)) and (abs(sMAE) > EXTREME_THRESHOLD or np.isinf(sMAE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sMAE detected: {sMAE}. This indicates numerical instability.")
        sMAE = np.nan
    if isinstance(sRMSE, (int, float)) and (abs(sRMSE) > EXTREME_THRESHOLD or np.isinf(sRMSE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sRMSE detected: {sRMSE}. This indicates numerical instability.")
        sRMSE = np.nan
    
    # Validate for suspiciously good results (potential data leakage or numerical precision issues)
    # Threshold: sRMSE < 1e-4 is suspiciously good and may indicate issues
    # For VAR models, near-zero values at horizon 1 often indicate persistence predictions or numerical underflow
    SUSPICIOUSLY_GOOD_THRESHOLD = 1e-4
    if isinstance(sRMSE, (int, float)) and (0 < sRMSE < SUSPICIOUSLY_GOOD_THRESHOLD):
        logger.warning(f"calculate_standardized_metrics: Suspiciously good sRMSE detected: {sRMSE}. This may indicate data leakage, numerical precision issues, persistence predictions (VAR), or overfitting. Verify train-test split and model implementation.")
        # Mark as potentially invalid for VAR models at horizon 1 (likely persistence or numerical underflow)
        # Note: We don't automatically set to NaN here, but the warning helps identify the issue
    
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
        List of forecast horizons (e.g., [1, 11, 22] for 1, 11, 22 months ahead)
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
                
                # Additional validation: Check if prediction is suspiciously close to last training value
                # This can help detect if VAR is just predicting the last value (persistence)
                if h == 1 and y_train is not None:
                    try:
                        if isinstance(y_train, (pd.DataFrame, pd.Series)):
                            last_train_val = y_train.iloc[-1]
                            if isinstance(y_pred_h, (pd.DataFrame, pd.Series)):
                                pred_val = y_pred_h.iloc[0] if len(y_pred_h) > 0 else None
                                if pred_val is not None and isinstance(last_train_val, (pd.Series, pd.DataFrame)):
                                    # Compare values
                                    if isinstance(pred_val, pd.Series) and isinstance(last_train_val, pd.Series):
                                        diff = (pred_val - last_train_val).abs()
                                        if target_series_for_metrics is not None:
                                            if isinstance(target_series_for_metrics, str) and target_series_for_metrics in diff.index:
                                                diff_val = diff[target_series_for_metrics]
                                            elif isinstance(target_series_for_metrics, int) and target_series_for_metrics < len(diff):
                                                diff_val = diff.iloc[target_series_for_metrics]
                                            else:
                                                diff_val = diff.iloc[0] if len(diff) > 0 else None
                                        else:
                                            diff_val = diff.iloc[0] if len(diff) > 0 else None
                                        
                                        # If prediction is extremely close to last training value, log warning
                                        # For VAR models, this often indicates persistence predictions (predicting last value)
                                        # which can lead to suspiciously good results (near-zero sRMSE) that don't reflect true forecasting ability
                                        if diff_val is not None and diff_val < 1e-6:
                                            logger.warning(f"Horizon {h}: Prediction is extremely close to last training value (diff={diff_val}). This may indicate VAR is essentially predicting persistence (last value), which could explain suspiciously good results. For VAR models, this is often a sign of numerical underflow or model instability rather than true forecasting ability.")
                    except Exception as e:
                        logger.debug(f"Horizon {h}: Could not check prediction vs last training value: {e}")
                
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
    
    This function implements a single-step evaluation design where each forecast
    horizon is evaluated using exactly one test point. This is an intentional design
    choice rather than a limitation, as it provides focused assessment of model
    performance at each specific forecast horizon.
    
    Evaluation Design:
        - For each horizon h, the function extracts exactly one test point at position
          test_pos = h - 1 (since test data starts at train_end+1, horizon h corresponds
          to position h-1 in test data).
        - This results in n_valid=1 for all horizons, which is expected behavior.
        - The single-step design is appropriate for nowcasting applications where
          we want to assess model performance at specific forecast horizons rather
          than aggregating across multiple test points.
    
    Why Single-Step Evaluation?
        - Data limitation: After 80/20 train/test split, the test set may be too small
          for multi-step evaluation, especially for longer horizons (e.g., h=22).
        - Focused assessment: Single-step evaluation provides a clear, focused assessment
          of model performance at each specific forecast horizon without aggregation
          effects that could mask horizon-specific performance characteristics.
        - Consistency: All models are evaluated using the same single-point design,
          ensuring fair comparison across models.
    
    Note: This design limitation is documented in the report methodology section.
    Multi-step evaluation (evaluating multiple test points per horizon) is not used
    due to the data limitation mentioned above.
    
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
        Dictionary with horizon as key and metrics dict as value.
        Each metrics dict contains: sMSE, sMAE, sRMSE, MSE, MAE, RMSE, sigma, n_valid.
        Note: n_valid=1 for all valid horizons due to single-step evaluation design.
    """
    logger = _module_logger
    
    # Defense-in-depth: Validate train-test split to prevent data leakage
    # This provides additional validation beyond the checks in train.py
    if hasattr(y_train, 'index') and hasattr(y_test, 'index'):
        train_max = y_train.index.max() if len(y_train) > 0 else None
        test_min = y_test.index.min() if len(y_test) > 0 else None
        if train_max is not None and test_min is not None:
            if train_max >= test_min:
                raise ValueError(
                    f"Data leakage detected in evaluate_forecaster: "
                    f"Training period ends at {train_max} but test period starts at {test_min}. "
                    f"There must be a gap between training and test periods to prevent data leakage."
                )
            logger.debug(f"Train-test split validated: train ends at {train_max}, test starts at {test_min}")
    
    # Check if forecaster is already fitted to avoid re-training
    is_fitted = False
    if hasattr(forecaster, 'is_fitted'):
        # Check if it's a method (callable) or property
        if callable(forecaster.is_fitted):
            try:
                is_fitted = forecaster.is_fitted()
            except Exception:
                # If calling fails, try as property/attribute
                is_fitted = bool(forecaster.is_fitted) if not callable(forecaster.is_fitted) else False
        else:
            is_fitted = bool(forecaster.is_fitted)
    elif hasattr(forecaster, '_is_fitted'):
        is_fitted = bool(forecaster._is_fitted)
    elif hasattr(forecaster, '_fitted_forecaster'):
        is_fitted = forecaster._fitted_forecaster is not None
    elif hasattr(forecaster, '_y'):
        is_fitted = forecaster._y is not None
    
    # Only fit if not already fitted (to avoid re-training loaded models)
    if not is_fitted:
        logger.info("Forecaster not fitted, fitting on training data...")
        forecaster.fit(y_train)
    else:
        logger.debug("Forecaster already fitted, skipping fit() to avoid re-training")
    
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
            
            # VAR-specific stability check: detect unstable forecasts before calculating metrics
            # VAR models can become numerically unstable for long horizons, producing extreme values
            model_type = getattr(forecaster, '_fitted_forecaster', None)
            if model_type is not None:
                model_type_str = str(type(model_type)).lower()
                is_var = 'var' in model_type_str
            else:
                # Try to detect VAR from forecaster type
                forecaster_type_str = str(type(forecaster)).lower()
                is_var = 'var' in forecaster_type_str
            
            if is_var and h > 1:
                # Check if forecast values are extreme (indicating numerical instability)
                # Extract numeric values from y_pred_h
                pred_values = None
                if isinstance(y_pred_h, pd.DataFrame):
                    pred_values = y_pred_h.values.flatten()
                elif isinstance(y_pred_h, pd.Series):
                    pred_values = y_pred_h.values
                elif hasattr(y_pred_h, '__array__'):
                    pred_values = np.asarray(y_pred_h).flatten()
                
                if pred_values is not None and len(pred_values) > 0:
                    max_abs_pred = np.max(np.abs(pred_values))
                    # Use consistent threshold with aggregation (1e10) for consistency
                    # This ensures that prediction values that would lead to extreme metrics
                    # are caught early during evaluation, matching the validation in aggregation
                    VAR_PREDICTION_THRESHOLD = 1e10
                    if max_abs_pred > VAR_PREDICTION_THRESHOLD:
                        logger.warning(
                            f"Horizon {h}: VAR forecast contains extreme values (max_abs={max_abs_pred:.2e} > {VAR_PREDICTION_THRESHOLD:.0e}). "
                            f"This indicates numerical instability. Marking metrics as NaN."
                        )
                        results[h] = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': np.nan, 'n_valid': 0
                        }
                        continue
                    # Check for NaN or Inf in predictions
                    if np.any(~np.isfinite(pred_values)):
                        logger.warning(
                            f"Horizon {h}: VAR forecast contains NaN or Inf values. "
                            f"Marking metrics as NaN."
                        )
                        results[h] = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': np.nan, 'n_valid': 0
                        }
                        continue
            
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
                    elif target_series is not None and isinstance(y_test, pd.DataFrame) and target_series in y_test.columns:
                        # target_series not in y_pred_h.columns but in y_test.columns
                        # Use column index from y_test to extract from y_pred_h
                        col_idx = y_test.columns.get_loc(target_series)
                        if col_idx < y_pred_h.shape[1]:
                            y_pred_h = y_pred_h.iloc[:, [col_idx]]
                            logger.debug(f"Horizon {h}: Extracted target_series '{target_series}' from y_pred_h using column index {col_idx} (from y_test.columns)")
                        else:
                            logger.warning(f"Horizon {h}: Column index {col_idx} for target_series '{target_series}' out of bounds for y_pred_h (shape: {y_pred_h.shape})")
                            y_pred_h = pd.DataFrame()
                    elif isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
                        # Single column, use it
                        y_pred_h = y_pred_h.iloc[:, [0]]
                    elif target_series is not None:
                        # target_series specified but not found in either y_pred_h or y_test
                        logger.warning(f"Horizon {h}: target_series '{target_series}' not found in y_pred_h.columns={list(y_pred_h.columns)} or y_test.columns={list(y_test.columns) if isinstance(y_test, pd.DataFrame) else 'N/A'}")
                        # Try to keep all columns and let calculate_standardized_metrics handle it
                        pass
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
                    
                    # VAR-specific check for horizon 1: detect if VAR is just predicting persistence
                    # (last training value), which would result in suspiciously good results
                    var_persistence_detected = False
                    if is_var and h == 1:
                        # Extract last training value for comparison
                        if isinstance(y_train, pd.DataFrame):
                            if target_series and target_series in y_train.columns:
                                last_train_val = y_train[target_series].iloc[-1]
                                train_std = y_train[target_series].std()
                            elif len(y_train.columns) == 1:
                                last_train_val = y_train.iloc[-1, 0]
                                train_std = y_train.iloc[:, 0].std()
                            else:
                                last_train_val = None
                                train_std = None
                        elif isinstance(y_train, pd.Series):
                            last_train_val = y_train.iloc[-1]
                            train_std = y_train.std()
                        else:
                            last_train_val = None
                            train_std = None
                        
                        # Extract prediction value
                        if isinstance(y_pred_h, pd.DataFrame):
                            if target_series and target_series in y_pred_h.columns:
                                pred_val = y_pred_h[target_series].iloc[0]
                            elif len(y_pred_h.columns) == 1:
                                pred_val = y_pred_h.iloc[0, 0]
                            else:
                                pred_val = None
                        elif isinstance(y_pred_h, pd.Series):
                            pred_val = y_pred_h.iloc[0]
                        else:
                            pred_val = None
                        
                        # Check if prediction is very close to last training value (persistence)
                        # Use both relative difference and absolute difference normalized by std
                        if last_train_val is not None and pred_val is not None:
                            if np.isfinite(last_train_val) and np.isfinite(pred_val):
                                abs_diff = abs(pred_val - last_train_val)
                                rel_diff = abs_diff / (abs(last_train_val) + 1e-10)
                                
                                # Check relative difference (original check)
                                if rel_diff < 1e-6:
                                    var_persistence_detected = True
                                    logger.warning(
                                        f"Horizon {h}: VAR prediction ({pred_val:.6f}) is essentially identical to last training value "
                                        f"({last_train_val:.6f}), suggesting VAR is predicting persistence. "
                                        f"Marking metrics as NaN due to model limitation."
                                    )
                                # Also check if absolute difference is very small compared to training std
                                # This catches cases where VAR predicts something very close to last value
                                # even if relative difference is larger (e.g., for small values)
                                elif train_std is not None and train_std > 0:
                                    std_normalized_diff = abs_diff / (train_std + 1e-10)
                                    if std_normalized_diff < 1e-4:  # Very small compared to std
                                        var_persistence_detected = True
                                        logger.warning(
                                            f"Horizon {h}: VAR prediction ({pred_val:.6f}) is very close to last training value "
                                            f"({last_train_val:.6f}), with std-normalized difference {std_normalized_diff:.2e}. "
                                            f"Marking metrics as NaN due to persistence prediction."
                                        )
                    
                    metrics = calculate_standardized_metrics(
                        y_true_h, y_pred_h, y_train=y_train, target_series=target_series_for_metrics
                    )
                    
                    # If VAR persistence detected, mark all metrics as NaN
                    if var_persistence_detected:
                        metrics = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': metrics.get('sigma', np.nan),
                            'n_valid': 0
                        }
                    
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
        List of forecast horizons to compare (e.g., [1, 11, 22])
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


# ========================================================================
# Result Aggregation Functions
# ========================================================================

def collect_all_comparison_results(outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Collect all comparison results from outputs/comparisons/."""
    logger = _module_logger
    
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
    
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
            logger.warning(f"Could not load {results_file}: {e}")
            continue
    
    return all_results


def aggregate_overall_performance(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate overall performance metrics across all models, targets, and horizons."""
    rows = []
    
    for target_series, results_list in all_results.items():
        # Sort by timestamp and use the latest result for each target
        if results_list:
            # Sort by timestamp (newest first)
            results_list_sorted = sorted(
                results_list,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            result_data = results_list_sorted[0] if results_list_sorted else None
        else:
            result_data = None
        
        if not result_data:
            continue
        
        results = result_data.get('results', {})
        horizons = result_data.get('horizons', list(range(1, 23)))  # Default: horizons 1-22 (monthly: 2024-01 to 2025-10)
        
        # Extract metrics from each model
        for model_name, model_data in results.items():
            if not isinstance(model_data, dict):
                continue
            
            metrics = model_data.get('metrics', {})
            if not isinstance(metrics, dict):
                continue
            
            forecast_metrics = metrics.get('forecast_metrics', {})
            if not isinstance(forecast_metrics, dict):
                continue
            
            # Extract metrics for each horizon
            for horizon in horizons:
                horizon_str = str(horizon)
                if horizon_str not in forecast_metrics:
                    continue
                
                horizon_metrics = forecast_metrics[horizon_str]
                if not isinstance(horizon_metrics, dict):
                    continue
                
                n_valid = horizon_metrics.get('n_valid', 0)
                
                # Apply validation to filter extreme values (numerical instability)
                # This ensures extreme VAR values (e.g., > 1e10) are marked as NaN
                EXTREME_THRESHOLD = 1e10
                
                def validate_metric(val):
                    """Validate metric value and return NaN if extreme.
                    
                    This function filters out extreme values that indicate numerical instability,
                    such as VAR forecasts that become unstable for long horizons.
                    """
                    if val is None:
                        return np.nan
                    if isinstance(val, (int, float)):
                        if np.isnan(val) or np.isinf(val):
                            return np.nan
                        if abs(val) > EXTREME_THRESHOLD:
                            # Log warning for extreme values
                            _module_logger.warning(
                                f"aggregate_overall_performance: Extreme value detected for "
                                f"{model_name.upper()} {target_series} horizon {horizon}: {val:.2e}. "
                                f"Marking as NaN due to numerical instability."
                            )
                            return np.nan
                    # Handle string representations of numbers
                    if isinstance(val, str):
                        try:
                            val_float = float(val)
                            return validate_metric(val_float)
                        except (ValueError, TypeError):
                            return np.nan
                    return val
                
                # Validate all metrics (standardized and raw) to filter extreme values
                smse = validate_metric(horizon_metrics.get('sMSE'))
                smae = validate_metric(horizon_metrics.get('sMAE'))
                srmse = validate_metric(horizon_metrics.get('sRMSE'))
                mse = validate_metric(horizon_metrics.get('MSE'))
                mae = validate_metric(horizon_metrics.get('MAE'))
                rmse = validate_metric(horizon_metrics.get('RMSE'))
                
                # Check for suspiciously good results (potential data leakage, numerical issues, or single-point luck)
                # Threshold: sMSE < 1e-4 or sMAE < 1e-3 is suspiciously good for any model/horizon
                # This is especially important when n_valid=1 (single test point)
                # Note: Zero values (perfect predictions) are also considered suspicious
                SUSPICIOUSLY_GOOD_SMSE_THRESHOLD = 1e-4
                SUSPICIOUSLY_GOOD_SMAE_THRESHOLD = 1e-3
                if isinstance(smse, (int, float)) and not np.isnan(smse) and (0 <= abs(smse) < SUSPICIOUSLY_GOOD_SMSE_THRESHOLD):
                    _module_logger.warning(
                        f"aggregate_overall_performance: Suspiciously good sMSE detected for "
                        f"{model_name.upper()} {target_series} horizon {horizon}: {smse:.2e}. "
                        f"This may indicate data leakage, numerical precision issues, or single-point luck (n_valid={n_valid}). "
                        f"Marking as NaN for reliability."
                    )
                    smse = np.nan
                    smae = np.nan
                    srmse = np.nan
                elif isinstance(smae, (int, float)) and not np.isnan(smae) and (0 <= abs(smae) < SUSPICIOUSLY_GOOD_SMAE_THRESHOLD):
                    _module_logger.warning(
                        f"aggregate_overall_performance: Suspiciously good sMAE detected for "
                        f"{model_name.upper()} {target_series} horizon {horizon}: {smae:.2e}. "
                        f"This may indicate data leakage, numerical precision issues, or single-point luck (n_valid={n_valid}). "
                        f"Marking as NaN for reliability."
                    )
                    smse = np.nan
                    smae = np.nan
                    srmse = np.nan
                
                # Include all horizons, even if n_valid=0 (for complete 36-row dataset)
                # This allows the report to show NaN/N/A for unavailable combinations
                row = {
                    'target': target_series,
                    'model': model_name.upper(),
                    'horizon': horizon,
                    'sMSE': smse,
                    'sMAE': smae,
                    'sRMSE': srmse,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'sigma': horizon_metrics.get('sigma'),
                    'n_valid': n_valid
                }
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    # Create DataFrame from rows
    aggregated = pd.DataFrame(rows)
    return aggregated


def main_aggregator():
    """Main entry point for aggregator module."""
    logger = _module_logger
    logger.info("=" * 70)
    logger.info("Aggregating Experiment Results")
    logger.info("=" * 70)
    
    # Collect all results
    all_results = collect_all_comparison_results()
    
    if not all_results:
        logger.warning("No comparison results found in outputs/comparisons/")
        return
    
    logger.info(f"Found results for {len(all_results)} target series:")
    for target, results in all_results.items():
        logger.info(f"  - {target}: {len(results)} comparison(s)")
    
    # Aggregate performance
    aggregated = aggregate_overall_performance(all_results)
    
    if aggregated.empty:
        logger.warning("No metrics to aggregate.")
        return
    
    # Save aggregated results
    outputs_dir = Path(__file__).parent.parent / "outputs"
    experiments_dir = outputs_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = experiments_dir / "aggregated_results.csv"
    aggregated.to_csv(output_file, index=False)
    
    logger.info(f"Aggregated results saved to: {output_file}")
    logger.info(f"  Total rows: {len(aggregated)}")
    logger.info(f"  Columns: {', '.join(aggregated.columns)}")


# ========================================================================
# LaTeX Table Generation Functions
# ========================================================================

def generate_latex_table_forecasting_results(
    aggregated_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> str:
    """Generate LaTeX table for forecasting results (Table 2).
    
    Table structure: Rows = model-target combinations (12 rows: ARIMA-KOIPALL.G, ARIMA-KOEQUIPTE, 
    ARIMA-KOWRCCNSE, VAR-KOIPALL.G, ..., DDFM-KOWRCCNSE).
    Columns = metrics (2 columns: sMAE, sMSE).
    Each cell contains the average metric value across all horizons (1-22 months).
    Total: 12 rows × 3 columns (including model-target column).
    
    Parameters
    ----------
    aggregated_df : pd.DataFrame
        DataFrame from aggregate_overall_performance()
    output_path : Path, optional
        Path to save the LaTeX table file
        
    Returns
    -------
    str
        LaTeX table code
    """
    models = ['ARIMA', 'VAR', 'DFM', 'DDFM']
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
    metrics = ['sMAE', 'sMSE']
    
    # Calculate averages across all horizons for each model-target combination
    avg_data = {}  # (model, target) -> {metric: average_value}
    EXTREME_THRESHOLD = 1e10
    
    if not aggregated_df.empty:
        for model in models:
            for target in targets:
                # Get all horizons for this model-target combination
                model_target_data = aggregated_df[
                    (aggregated_df['model'] == model) & 
                    (aggregated_df['target'] == target)
                ]
                
                if len(model_target_data) == 0:
                    continue
                
                avg_values = {}
                for metric in metrics:
                    values = []
                    for _, row in model_target_data.iterrows():
                        val = row[metric]
                        if not pd.isna(val) and isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                            if abs(val) < EXTREME_THRESHOLD:
                                values.append(val)
                    
                    if values:
                        avg_values[metric] = np.mean(values)
                    else:
                        avg_values[metric] = np.nan
                
                if avg_values:
                    avg_data[(model, target)] = avg_values
    
    # Find minimum values for each metric (for bold formatting)
    min_values = {}  # metric -> min_value
    for metric in metrics:
        values = []
        for (model, target), metrics_dict in avg_data.items():
            if metric in metrics_dict:
                val = metrics_dict[metric]
                if not pd.isna(val):
                    values.append(val)
        if values:
            min_values[metric] = min(values)
    
    # Generate LaTeX with merged headers
    # Structure: Model (merged) | Target | sMAE | sMSE
    num_targets = len(targets)
    
    latex = """\\begin{table}[h]
\\centering
\\caption[Forecasting Results by Model-Target (Average across Horizons)]{Forecasting Results by Model-Target (Average across Horizons)\\footnote{Experiments evaluate all horizons from 1 to 22 months (2024--01 to 2025--10). This table shows the average metric values across all horizons for each model-target combination. Bold values indicate the best (lowest) metric for each metric type. Full results for all horizons are available in the appendix.}}
\\label{tab:forecasting_results}
\\begin{tabular}{l|l|cc}
\\toprule
\\multirow{2}{*}{Model} & Target & \\multicolumn{2}{c}{Metrics} \\\\
\\cmidrule(lr){3-4}
 & & sMAE & sMSE \\\\
\\midrule
"""
    
    EXTREME_THRESHOLD = 1e10
    
    def format_value(val):
        """Format value for LaTeX table."""
        if pd.isna(val) or (isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val))):
            return 'N/A'
        if isinstance(val, (int, float)):
            if abs(val) > EXTREME_THRESHOLD:
                return 'Unstable'
            return f"{val:.4f}"
        return str(val)
    
    # Generate rows for model-target combinations
    if len(avg_data) == 0:
        # Generate placeholder table
        for model in models:
            for target_idx, target in enumerate(targets):
                values = []
                for metric in metrics:
                    values.append('N/A')
                
                # First row of each model: show model name with multirow, subsequent rows: empty
                if target_idx == 0:
                    latex += f"\\multirow{{{num_targets}}}{{*}}{{{model}}} & {target} & {' & '.join(values)} \\\\\n"
                else:
                    latex += f" & {target} & {' & '.join(values)} \\\\\n"
                # Add horizontal line after each model group
                if target_idx == len(targets) - 1 and model != models[-1]:
                    latex += "\\midrule\n"
    else:
        # Generate rows for model-target combinations
        for model_idx, model in enumerate(models):
            for target_idx, target in enumerate(targets):
                values = []
                for metric in metrics:
                    key = (model, target)
                    if key in avg_data and metric in avg_data[key]:
                        val = avg_data[key][metric]
                        formatted_val = format_value(val)
                        
                        # Check if this is the minimum value for this metric
                        if formatted_val != 'N/A' and formatted_val != 'Unstable':
                            try:
                                val_float = float(val)
                                if metric in min_values:
                                    if abs(val_float - min_values[metric]) < 1e-6:  # Allow small floating point differences
                                        formatted_val = f"\\textbf{{{formatted_val}}}"
                            except (ValueError, TypeError):
                                pass
                        
                        values.append(formatted_val)
                    else:
                        values.append('N/A')
                
                # First row of each model: show model name with multirow, subsequent rows: empty
                if target_idx == 0:
                    latex += f"\\multirow{{{num_targets}}}{{*}}{{{model}}} & {target} & {' & '.join(values)} \\\\\n"
                else:
                    latex += f" & {target} & {' & '.join(values)} \\\\\n"
                # Add horizontal line after each model group (except last)
                if target_idx == len(targets) - 1 and model_idx < len(models) - 1:
                    latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
    
    return latex


def generate_latex_table_appendix_forecasting(
    aggregated_df: pd.DataFrame,
    target: Optional[str] = None,
    output_path: Optional[Path] = None
) -> str:
    """Generate LaTeX table for appendix forecasting results (all horizons).
    
    Table structure: Rows = horizons (22 rows: 1, 2, ..., 22 months).
    Columns = model-metric combinations (8 columns: ARIMA_sMAE, ARIMA_sMSE, VAR_sMAE, VAR_sMSE, 
    DFM_sMAE, DFM_sMSE, DDFM_sMAE, DDFM_sMSE).
    Total: 22 rows × 9 columns (including horizon column).
    
    If target is None, generates a table with all targets combined (averaged across targets).
    
    Parameters
    ----------
    aggregated_df : pd.DataFrame
        DataFrame from aggregate_overall_performance()
    target : str, optional
        Target series name (KOIPALL.G, KOEQUIPTE, KOWRCCNSE). If None, averages across all targets.
    output_path : Path, optional
        Path to save the LaTeX table file
        
    Returns
    -------
    str
        LaTeX table code
    """
    models = ['ARIMA', 'VAR', 'DFM', 'DDFM']
    metrics = ['sMAE', 'sMSE']
    horizons = list(range(1, 23))  # 1 to 22 months
    
    EXTREME_THRESHOLD = 1e10
    
    def format_value(val):
        """Format value for LaTeX table."""
        if pd.isna(val) or (isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val))):
            return 'N/A'
        if isinstance(val, (int, float)):
            if abs(val) > EXTREME_THRESHOLD:
                return 'Unstable'
            return f"{val:.4f}"
        return str(val)
    
    # Filter data by target if specified
    if target is not None:
        filtered_df = aggregated_df[aggregated_df['target'] == target].copy()
        caption_target = target
    else:
        # Average across all targets
        filtered_df = aggregated_df.copy()
        caption_target = "All Targets (Averaged)"
    
    # Create lookup dictionary: (horizon, model) -> {metric: value}
    data_lookup = {}
    if not filtered_df.empty:
        if target is None:
            # Average across targets for each horizon-model combination
            for horizon in horizons:
                for model in models:
                    horizon_model_data = filtered_df[
                        (filtered_df['horizon'] == horizon) & 
                        (filtered_df['model'] == model)
                    ]
                    if len(horizon_model_data) > 0:
                        avg_metrics = {}
                        for metric in metrics:
                            values = []
                            for _, row in horizon_model_data.iterrows():
                                val = row[metric]
                                if not pd.isna(val) and isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                                    if abs(val) < EXTREME_THRESHOLD:
                                        values.append(val)
                            if values:
                                avg_metrics[metric] = np.mean(values)
                            else:
                                avg_metrics[metric] = np.nan
                        if avg_metrics:
                            data_lookup[(horizon, model)] = avg_metrics
        else:
            # Single target
            for _, row in filtered_df.iterrows():
                key = (row['horizon'], row['model'])
                if key not in data_lookup:
                    data_lookup[key] = {}
                for metric in metrics:
                    val = row[metric]
                    if not pd.isna(val) and isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                        if abs(val) < EXTREME_THRESHOLD:
                            data_lookup[key][metric] = val
                        else:
                            data_lookup[key][metric] = np.nan
    
    # Generate LaTeX
    table_label = f"tab:appendix_forecasting_{target.lower().replace('.', '_')}" if target else "tab:appendix_forecasting_all"
    latex = f"""\\begin{{longtable}}{{l|cc|cc|cc|cc}}
\\caption{{Forecasting Results by Horizon for {caption_target} (All Horizons)}}\\label{{{table_label}}} \\\\
\\toprule
Horizon & \\multicolumn{{2}}{{c}}{{ARIMA}} & \\multicolumn{{2}}{{c}}{{VAR}} & \\multicolumn{{2}}{{c}}{{DFM}} & \\multicolumn{{2}}{{c}}{{DDFM}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}
 & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\\\
\\midrule
\\endfirsthead
\\multicolumn{{9}}{{c}}{{{{Continued from previous page}}}} \\\\
\\toprule
Horizon & \\multicolumn{{2}}{{c}}{{ARIMA}} & \\multicolumn{{2}}{{c}}{{VAR}} & \\multicolumn{{2}}{{c}}{{DFM}} & \\multicolumn{{2}}{{c}}{{DDFM}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}
 & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{9}}{{r}}{{{{Continued on next page}}}} \\\\
\\endfoot
\\bottomrule
\\endlastfoot
"""
    
    # Generate rows for each horizon
    for horizon in horizons:
        values = []
        for model in models:
            for metric in metrics:
                key = (horizon, model)
                if key in data_lookup and metric in data_lookup[key]:
                    val = data_lookup[key][metric]
                    formatted_val = format_value(val)
                    values.append(formatted_val)
                else:
                    values.append('N/A')
        
        latex += f"{horizon} & {' & '.join(values)} \\\\\n"
    
    latex += """\\end{longtable}"""
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
    
    return latex


def generate_latex_table_nowcasting_backtest(
    outputs_dir: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> str:
    """Generate LaTeX table for nowcasting backtest results (Table 3).
    
    Table structure: Rows = model-timepoint combinations (4 rows: DFM-4weeks, DFM-1week, 
    DDFM-4weeks, DDFM-1week). Note: ARIMA and VAR are not included as they do not support nowcasting.
    Columns = target-metric combinations (6 columns: KOIPALL.G_sMAE, KOIPALL.G_sMSE, 
    KOEQUIPTE_sMAE, KOEQUIPTE_sMSE, KOWRCCNSE_sMAE, KOWRCCNSE_sMSE).
    Total: 4 rows × 7 columns (including model-timepoint column).
    
    Parameters
    ----------
    outputs_dir : Path, optional
        Directory containing backtest results (default: project root / outputs)
    output_path : Path, optional
        Path to save the LaTeX table file
        
    Returns
    -------
    str
        LaTeX table code
    """
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
    
    backtest_dir = outputs_dir / "backtest"
    
    # Generate LaTeX
    latex = """\\begin{table}[h]
\\centering
\\caption[Nowcasting Backtest Results by Model-Timepoint and Target-Metric]{Nowcasting Backtest Results by Model-Timepoint and Target-Metric\\footnote{Train with data from 1985 to 2019, nowcast from Jan 2024 to Oct 2025 (22 months). For each target month, perform nowcasting at multiple time points (4 weeks before, 1 week before month end). By masking unavailable data based on release dates, generate 1 horizon forecast at each time point. Calculate sMSE, sMAE for each month and time point, then average across 22 months.}}
\\label{tab:nowcasting_backtest}
\\begin{tabular}{lcccccc}
\\toprule
Model-Timepoint & KOIPALL.G & KOIPALL.G & KOEQUIPTE & KOEQUIPTE & KOWRCCNSE & KOWRCCNSE \\\\
 & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\\\
\\midrule
"""
    
    # Load backtest results
    # Note: Only DFM and DDFM support nowcasting (ARIMA/VAR cannot handle missing data from release date masking)
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
    models = ['DFM', 'DDFM']
    timepoints = ['4weeks', '1weeks']  # Note: JSON uses "4weeks" and "1weeks" (not "1week")
    
    # Collect results: (model, timepoint) -> {target: {sMAE, sMSE}}
    results_data = {}
    
    for target in targets:
        for model in models:
            model_lower = model.lower()
            result_file = backtest_dir / f"{target}_{model_lower}_backtest.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check for "no_results" status
                    status = data.get('status', '')
                    if status == 'no_results':
                        # Skip this file - no valid results available
                        _module_logger.debug(f"Skipping {result_file.name}: status='no_results'")
                        continue
                    
                    results_by_timepoint = data.get('results_by_timepoint', {})
                    
                    # Only process if results_by_timepoint is not empty
                    if not results_by_timepoint:
                        _module_logger.debug(f"No results_by_timepoint in {result_file.name}")
                        continue
                    
                    for timepoint in timepoints:
                        if timepoint in results_by_timepoint:
                            tp_data = results_by_timepoint[timepoint]
                            key = (model, timepoint)
                            if key not in results_data:
                                results_data[key] = {}
                            
                            # Use overall metrics (averaged across months)
                            results_data[key][target] = {
                                'sMAE': tp_data.get('overall_sMAE'),
                                'sMSE': tp_data.get('overall_sMSE')
                            }
                except Exception as e:
                    _module_logger.warning(f"Error loading {result_file}: {e}")
                    continue
    
    # Generate table rows
    def format_value(val):
        """Format value for LaTeX table.
        
        Handles None, NaN, Inf, extreme values, and string representations of numbers.
        """
        if val is None:
            return 'N/A'
        # Handle string representations of numbers
        if isinstance(val, str):
            try:
                val = float(val)
            except (ValueError, TypeError):
                return 'N/A'
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return 'N/A'
            # Check for extreme values (numerical instability)
            if abs(val) > 1e10:
                return 'Unstable'
            return f"{val:.4f}"
        return str(val)
    
    # Map timepoint keys to display labels (JSON uses "1weeks" but table should show "1week")
    timepoint_labels = {'4weeks': '4weeks', '1weeks': '1week'}
    
    for model in models:
        for timepoint in timepoints:
            # Use display label for table (singular "1week" instead of "1weeks")
            row_label = f"{model}-{timepoint_labels.get(timepoint, timepoint)}"
            values = []
            for target in targets:
                key = (model, timepoint)
                if key in results_data and target in results_data[key]:
                    target_data = results_data[key][target]
                    values.append(format_value(target_data.get('sMAE')))
                    values.append(format_value(target_data.get('sMSE')))
                else:
                    values.extend(['N/A', 'N/A'])
            latex += f"{row_label} & {' & '.join(values)} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
    
    return latex


def generate_latex_table_dataset_params(
    config_dir: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> str:
    """Generate LaTeX table for dataset details and model parameters.
    
    Parameters
    ----------
    config_dir : Path, optional
        Directory containing config files (default: project root / config)
    output_path : Path, optional
        Path to save the LaTeX table file
        
    Returns
    -------
    str
        LaTeX table code
    """
    import yaml
    
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config"
    
    # Load model configs
    model_configs = {}
    for model_name in ['arima', 'var', 'dfm', 'ddfm']:
        config_file = config_dir / "model" / f"{model_name}.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    model_configs[model_name] = yaml.safe_load(f)
            except Exception:
                model_configs[model_name] = {}
    
    # Load experiment configs to get series counts
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    series_counts = {}
    for target in targets:
        config_name_map = {
            'KOEQUIPTE': 'investment_koequipte_report',
            'KOWRCCNSE': 'consumption_kowrccnse_report',
            'KOIPALL.G': 'production_koipallg_report'
        }
        config_file = config_dir / "experiment" / f"{config_name_map.get(target, '')}.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    exp_config = yaml.safe_load(f)
                    series_list = exp_config.get('series', [])
                    series_counts[target] = len(series_list) if isinstance(series_list, list) else 0
            except Exception:
                series_counts[target] = 0
        else:
            series_counts[target] = 0
    
    # Extract parameters
    arima_order = model_configs.get('arima', {}).get('order', [1, 1, 1])
    if isinstance(arima_order, list) and len(arima_order) >= 3:
        arima_params = f"({arima_order[0]}, {arima_order[1]}, {arima_order[2]})"
    else:
        arima_params = "(1, 1, 1)"
    
    var_lags = model_configs.get('var', {}).get('lag_order', 1)
    var_params = f"Lag: {var_lags}"
    
    dfm_factors = model_configs.get('dfm', {}).get('blocks', {}).get('Block_Global', {}).get('factors', 3)
    dfm_max_iter = model_configs.get('dfm', {}).get('max_iter', 5000)
    dfm_params = f"Factors: {dfm_factors}, Max Iter: {dfm_max_iter}"
    
    ddfm_layers = model_configs.get('ddfm', {}).get('encoder_layers', [64, 32])
    ddfm_factors = model_configs.get('ddfm', {}).get('num_factors', 2)
    ddfm_epochs = model_configs.get('ddfm', {}).get('epochs', 100)
    if isinstance(ddfm_layers, list):
        layers_str = '-'.join(map(str, ddfm_layers))
    else:
        layers_str = "64-32"
    ddfm_params = f"Layers: {layers_str}, Factors: {ddfm_factors}, Epochs: {ddfm_epochs}"
    
    # Calculate total series count (average across targets)
    avg_series_count = int(sum(series_counts.values()) / len(series_counts)) if series_counts else 0
    
    # Generate LaTeX
    latex = """\\begin{table}[h]
\\centering
\\caption[Dataset Details and Model Parameters]{Dataset Details and Model Parameters\\footnote{Dataset: 3 target variables (KOEQUIPTE, KOWRCCNSE, KOIPALL.G) with average of """ + str(avg_series_count) + """ series per target. Training period: 1985-2019, Nowcasting period: 2024-2025.}}
\\label{tab:dataset_params}
\\begin{tabular}{ll}
\\toprule
\\textbf{Item} & \\textbf{Value} \\\\
\\midrule
\\multicolumn{2}{l}{\\textbf{Dataset}} \\\\
\\quad Targets & 3 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G) \\\\
\\quad Series per Target & """ + str(avg_series_count) + """ (average) \\\\
\\quad Training Period & 1985-2019 \\\\
\\quad Nowcasting Period & 2024-2025 \\\\
\\midrule
\\multicolumn{2}{l}{\\textbf{Model Parameters}} \\\\
\\quad ARIMA & Order: """ + arima_params + """ \\\\
\\quad VAR & """ + var_params + """ \\\\
\\quad DFM & """ + dfm_params + """ \\\\
\\quad DDFM & """ + ddfm_params + """ \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
    
    return latex


def generate_all_latex_tables(
    outputs_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Generate all LaTeX tables from aggregated results.
    
    Parameters
    ----------
    outputs_dir : Path, optional
        Directory containing outputs/ (default: project root / outputs)
    tables_dir : Path, optional
        Directory to save LaTeX tables (default: outputs/experiments/tables/)
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping table names to LaTeX code
    """
    logger = _module_logger
    
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
    
    if tables_dir is None:
        # Save LaTeX tables to outputs/experiments/tables/ for consistency
        tables_dir = outputs_dir / "experiments" / "tables"
    
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config"
    
    # Ensure tables directory exists
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated results
    aggregated_file = outputs_dir / "experiments" / "aggregated_results.csv"
    
    if not aggregated_file.exists():
        logger.warning(f"{aggregated_file} not found. Generating placeholder tables.")
        aggregated_df = pd.DataFrame()
    else:
        aggregated_df = pd.read_csv(aggregated_file)
        # Filter extreme values (numerical instability) - in case CSV was generated before validation
        EXTREME_THRESHOLD = 1e10
        
        # Filter extreme values in all metrics (standardized and raw) for consistency
        all_metrics = ['sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
        for metric in all_metrics:
            if metric in aggregated_df.columns:
                # Convert to numeric, coercing errors to NaN
                aggregated_df[metric] = pd.to_numeric(aggregated_df[metric], errors='coerce')
                mask_extreme = aggregated_df[metric].abs() > EXTREME_THRESHOLD
                if mask_extreme.any():
                    n_extreme = mask_extreme.sum()
                    logger.warning(
                        f"generate_all_latex_tables: Found {n_extreme} extreme values in {metric} "
                        f"(>{EXTREME_THRESHOLD}). Marking as NaN."
                    )
                    aggregated_df.loc[mask_extreme, metric] = np.nan
        
        # Mark suspiciously good values as NaN (potential data leakage, numerical issues, or single-point luck)
        # This applies to all models and horizons, not just VAR-1
        SUSPICIOUSLY_GOOD_SMSE_THRESHOLD = 1e-4
        SUSPICIOUSLY_GOOD_SMAE_THRESHOLD = 1e-3
        
        # Check for suspiciously good sMSE values
        if 'sMSE' in aggregated_df.columns:
            smse_numeric = pd.to_numeric(aggregated_df['sMSE'], errors='coerce')
            suspicious_smse_mask = (
                smse_numeric.abs() < SUSPICIOUSLY_GOOD_SMSE_THRESHOLD
            ) & (smse_numeric.abs() >= 0) & (~smse_numeric.isna())
            if suspicious_smse_mask.any():
                n_suspicious = suspicious_smse_mask.sum()
                logger.warning(
                    f"generate_all_latex_tables: Found {n_suspicious} suspiciously good sMSE values "
                    f"(<{SUSPICIOUSLY_GOOD_SMSE_THRESHOLD}). These may indicate data leakage, numerical issues, "
                    f"or single-point luck. Marking as NaN."
                )
                all_metrics = ['sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
                for metric in all_metrics:
                    if metric in aggregated_df.columns:
                        aggregated_df.loc[suspicious_smse_mask, metric] = np.nan
        
        # Check for suspiciously good sMAE values
        if 'sMAE' in aggregated_df.columns:
            smae_numeric = pd.to_numeric(aggregated_df['sMAE'], errors='coerce')
            suspicious_smae_mask = (
                smae_numeric.abs() < SUSPICIOUSLY_GOOD_SMAE_THRESHOLD
            ) & (smae_numeric.abs() >= 0) & (~smae_numeric.isna())
            if suspicious_smae_mask.any():
                n_suspicious = suspicious_smae_mask.sum()
                logger.warning(
                    f"generate_all_latex_tables: Found {n_suspicious} suspiciously good sMAE values "
                    f"(<{SUSPICIOUSLY_GOOD_SMAE_THRESHOLD}). These may indicate data leakage, numerical issues, "
                    f"or single-point luck. Marking as NaN."
                )
                all_metrics = ['sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
                for metric in all_metrics:
                    if metric in aggregated_df.columns:
                        aggregated_df.loc[suspicious_smae_mask, metric] = np.nan
        
        # Mark VAR-1 persistence values as NaN (extremely small values indicate persistence prediction)
        PERSISTENCE_THRESHOLD_SMSE = 1e-6
        PERSISTENCE_THRESHOLD_SMAE = 1e-4
        var1_mask = (aggregated_df['model'] == 'VAR') & (aggregated_df['horizon'] == 1)
        if var1_mask.any():
            # Detect persistence: if either sMSE or sMAE indicates persistence, mark ALL metrics as NaN
            # This is consistent with evaluate_forecaster() which marks all metrics when persistence is detected
            persistence_detected = False
            if 'sMSE' in aggregated_df.columns:
                var1_smse_mask = var1_mask & (aggregated_df['sMSE'].abs() < PERSISTENCE_THRESHOLD_SMSE)
                if var1_smse_mask.any():
                    persistence_detected = True
                    n_persistence = var1_smse_mask.sum()
                    logger.warning(
                        f"generate_all_latex_tables: Found {n_persistence} VAR-1 sMSE persistence values "
                        f"(<{PERSISTENCE_THRESHOLD_SMSE}). Marking all metrics as NaN."
                    )
            if 'sMAE' in aggregated_df.columns:
                var1_smae_mask = var1_mask & (aggregated_df['sMAE'].abs() < PERSISTENCE_THRESHOLD_SMAE)
                if var1_smae_mask.any():
                    persistence_detected = True
                    n_persistence = var1_smae_mask.sum()
                    logger.warning(
                        f"generate_all_latex_tables: Found {n_persistence} VAR-1 sMAE persistence values "
                        f"(<{PERSISTENCE_THRESHOLD_SMAE}). Marking all metrics as NaN."
                    )
            
            # CRITICAL FIX: Build persistence_rows mask directly (don't rely on persistence_detected flag)
            # This ensures all VAR-1 rows with suspiciously small metrics are marked as NaN
            persistence_rows = pd.Series(False, index=aggregated_df.index)
            
            # Check sMSE persistence
            if 'sMSE' in aggregated_df.columns:
                smse_persistence = var1_mask & (aggregated_df['sMSE'].abs() < PERSISTENCE_THRESHOLD_SMSE)
                persistence_rows = persistence_rows | smse_persistence
            
            # Check sMAE persistence
            if 'sMAE' in aggregated_df.columns:
                smae_persistence = var1_mask & (aggregated_df['sMAE'].abs() < PERSISTENCE_THRESHOLD_SMAE)
                persistence_rows = persistence_rows | smae_persistence
            
            # Mark all metrics as NaN for rows with persistence
            # This ensures VAR-1 persistence predictions are not reported as valid results
            if persistence_rows.any():
                all_metrics = ['sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
                for metric in all_metrics:
                    if metric in aggregated_df.columns:
                        aggregated_df.loc[persistence_rows, metric] = np.nan
                logger.info(
                    f"generate_all_latex_tables: Marked {persistence_rows.sum()} VAR-1 rows as NaN due to persistence detection. "
                    f"This is expected behavior - VAR-1 often predicts the last training value, which is not a valid forecast."
                )
    
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
    
    tables = {}
    
    # Generate each table
    logger.info("Generating LaTeX tables...")
    
    # Table 1: Dataset and parameters
    logger.info("  - tab_dataset_params.tex")
    tables['dataset_params'] = generate_latex_table_dataset_params(
        config_dir,
        tables_dir / "tab_dataset_params.tex"
    )
    
    # Table 2: Forecasting results (model-horizon rows, target-metric columns)
    logger.info("  - tab_forecasting_results.tex")
    tables['forecasting_results'] = generate_latex_table_forecasting_results(
        aggregated_df,
        tables_dir / "tab_forecasting_results.tex"
    )
    
    # Table 3: Nowcasting backtest results
    logger.info("  - tab_nowcasting_backtest.tex")
    tables['nowcasting_backtest'] = generate_latex_table_nowcasting_backtest(
        outputs_dir,
        tables_dir / "tab_nowcasting_backtest.tex"
    )
    
    logger.info(f"All LaTeX tables generated and saved to: {tables_dir}")
    
    return tables
