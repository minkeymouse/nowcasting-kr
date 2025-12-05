"""Evaluation helper functions for forecasting experiments.

This module provides standardized metric calculation functions for evaluating
forecasting model performance, including standardized MSE, MAE, and RMSE.
"""

from typing import Union, Dict, Optional, Tuple
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
    # Convert to numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
        columns = y_true.columns
    elif isinstance(y_true, pd.Series):
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
            if isinstance(y_true, pd.DataFrame):
                col_idx = y_true.columns.get_loc(target_series)
            else:
                raise ValueError("target_series must be int if y_true is not DataFrame")
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
        
        # Extract predictions for horizon h (h-1 index since 0-indexed)
        if isinstance(y_pred, pd.DataFrame):
            y_pred_h = y_pred.iloc[h-1:h] if h <= len(y_pred) else pd.DataFrame()
        elif isinstance(y_pred, pd.Series):
            y_pred_h = y_pred.iloc[h-1:h] if h <= len(y_pred) else pd.Series()
        else:
            y_pred_h = y_pred[h-1:h] if h <= len(y_pred) else np.array([])
        
        if isinstance(y_true, pd.DataFrame):
            y_true_h = y_true.iloc[h-1:h] if h <= len(y_true) else pd.DataFrame()
        elif isinstance(y_true, pd.Series):
            y_true_h = y_true.iloc[h-1:h] if h <= len(y_true) else pd.Series()
        else:
            y_true_h = y_true[h-1:h] if h <= len(y_true) else np.array([])
        
        # Calculate metrics for this horizon
        if len(y_pred_h) > 0 and len(y_true_h) > 0:
            metrics = calculate_standardized_metrics(
                y_true_h, y_pred_h, y_train=y_train, target_series=target_series
            )
            results[h] = metrics
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
    
    # Generate predictions for all horizons
    horizons_arr = np.asarray(horizons)
    max_horizon = int(np.max(horizons_arr))
    
    # Predict up to max horizon
    fh = np.arange(1, max_horizon + 1)
    y_pred = forecaster.predict(fh=fh)
    
    # Calculate metrics per horizon
    results = calculate_metrics_per_horizon(
        y_test, y_pred, horizons, y_train=y_train, target_series=target_series
    )
    
    return results
