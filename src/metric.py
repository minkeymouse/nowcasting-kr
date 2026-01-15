"""Metric computation functions for evaluating forecast performance.

Includes standardized MSE/MAE computation, test data standard deviation calculation,
and result saving utilities.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


def _get_tent_weights(n_weeks: int, base_weights: np.ndarray) -> np.ndarray:
    """Get tent kernel weights for given number of weeks.
    
    Parameters
    ----------
    n_weeks : int
        Number of weeks in month
    base_weights : np.ndarray
        Base tent weights (typically [0.1, 0.2, 0.3, 0.4])
    
    Returns
    -------
    np.ndarray
        Weights for n_weeks, normalized to sum to 1
    """
    if n_weeks == 4:
        return base_weights
    elif n_weeks == 5:
        # 5-week month: first 4 weeks with weights, last week gets 0
        weights = np.concatenate([base_weights, [0.0]])
        return weights / weights.sum()
    elif n_weeks == 3:
        # 3-week month: use first 3 weights
        weights = base_weights[:3]
        return weights / weights.sum()
    else:
        # Linear interpolation for other cases
        weights = np.linspace(0.1, 0.4, n_weeks)
        return weights / weights.sum()


def compute_test_data_std(
    test_data: pd.DataFrame,
    target_series: List[str],
    monthly_series: Optional[set] = None,
    tent_weights: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Compute standard deviation of test data for metric normalization.
    
    For monthly series, test data is aggregated to monthly first using tent kernel weights
    before computing standard deviation. This ensures consistent normalization with
    monthly-evaluated metrics (sMSE, sMAE).
    
    Parameters
    ----------
    test_data : pd.DataFrame
        Full test dataset (weekly-aligned for model input)
    target_series : list
        Target series names
    monthly_series : set, optional
        Set of monthly series IDs. If None, all series are treated as weekly.
    tent_weights : np.ndarray, optional
        Tent kernel weights for monthly aggregation. Default: [0.1, 0.2, 0.3, 0.4]
    
    Returns
    -------
    np.ndarray or None
        Standard deviation per series, shape (len(target_series),) or None
    """
    if test_data is None or not target_series:
        return None
    
    available_targets = [t for t in target_series if t in test_data.columns]
    if not available_targets:
        return None
    
    # If no monthly series specified, compute std directly from weekly data
    if monthly_series is None or len(monthly_series) == 0:
        return test_data[available_targets].std(ddof=0).values
    
    # Aggregate monthly series to monthly before computing std
    if tent_weights is None:
        tent_weights = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Create DataFrame for aggregation
    test_data_with_month = test_data.copy()
    test_data_with_month['year_month'] = test_data_with_month.index.to_period('M')
    
    monthly_values_dict = {series: [] for series in available_targets}
    
    for year_month, month_group in test_data_with_month.groupby('year_month'):
        for series_name in available_targets:
            is_monthly = series_name in monthly_series
            
            if is_monthly:
                # Aggregate monthly series using tent kernel weights
                month_values = month_group[series_name].dropna().values
                if len(month_values) > 0:
                    weights = _get_tent_weights(len(month_values), tent_weights)
                    # Only use weights for available values (handle missing values in month)
                    if len(weights) == len(month_values):
                        aggregated_value = np.dot(month_values, weights)
                    else:
                        # If fewer values than weights, normalize available weights
                        available_weights = weights[:len(month_values)]
                        available_weights = available_weights / available_weights.sum()
                        aggregated_value = np.dot(month_values, available_weights)
                    monthly_values_dict[series_name].append(aggregated_value)
            else:
                # Weekly series: use all weekly values in the month
                month_values = month_group[series_name].dropna().values
                monthly_values_dict[series_name].extend(month_values.tolist())
    
    # Compute std from aggregated data
    std_values = []
    for series_name in available_targets:
        if len(monthly_values_dict[series_name]) > 0:
            std_val = np.std(monthly_values_dict[series_name], ddof=0)
            std_values.append(std_val)
        else:
            std_values.append(0.0)  # No data available
    
    return np.array(std_values)


def validate_std_for_metrics(
    test_data_std: Optional[np.ndarray],
    n_series: int
) -> Optional[np.ndarray]:
    """Validate test_data_std for metric computation.
    
    Returns None if std doesn't match series count, otherwise returns std.
    
    Parameters
    ----------
    test_data_std : np.ndarray or None
        Standard deviation array
    n_series : int
        Expected number of series
    
    Returns
    -------
    np.ndarray or None
        Validated std array, or None if invalid
    """
    if test_data_std is None or len(test_data_std) != n_series:
        return None
    return test_data_std


def compute_smse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: Optional[Any] = None,
    per_series: bool = False,
    test_data_std: Optional[np.ndarray] = None
) -> float:
    """Compute Standardized Mean Squared Error (sMSE).
    
    Formula: MSE normalized by test data standard deviation (per series, then average).
    sMSE = mean(MSE_i / std_i^2) where i indexes series
    
    If test_data_std is provided, uses that for normalization (ensures consistency across models).
    Otherwise, uses standard deviation computed from y_true.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_series) or (n_samples,)
    y_pred : np.ndarray
        Predicted values, same shape as y_true
    scaler : optional
        Scaler object (for consistency, not used here)
    per_series : bool, default False
        If True, return per-series sMSE. If False, return average.
    test_data_std : np.ndarray, optional
        Standard deviation of test data (per series), shape (n_series,).
        If provided, uses this for normalization instead of computing from y_true.
    
    Returns
    -------
    float or np.ndarray
        Standardized MSE (average if per_series=False, per-series if True)
    
    Examples
    --------
    >>> smse = compute_smse(y_true, y_pred, test_data_std=test_std)
    >>> smse  # Average standardized MSE across all series
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Ensure same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} and {y_pred.shape}")
    
    n_series = y_true.shape[1]
    
    # Mask out NaN values (compute metrics only on valid pairs)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Compute MSE per series (only on valid values)
    mse_per_series = []
    for i in range(n_series):
        valid = valid_mask[:, i]
        if np.sum(valid) == 0:
            # All NaN for this series - skip or set to 0
            mse_per_series.append(0.0)
        else:
            mse_per_series.append(np.mean((y_true[valid, i] - y_pred[valid, i]) ** 2))
    
    mse_per_series = np.array(mse_per_series)
    mse_per_series = np.atleast_1d(mse_per_series)
    
    # Get normalization factor: use test_data_std if provided, otherwise compute from y_true
    if test_data_std is not None:
        std_per_series = np.asarray(test_data_std)
        std_per_series = np.atleast_1d(std_per_series)
        if len(std_per_series) != n_series:
            raise ValueError(f"test_data_std must have length {n_series}, got {len(std_per_series)}")
        # Convert to variance for normalization
        var_per_series = std_per_series ** 2
    else:
        # Compute variance from actuals (fallback)
        var_per_series = []
        for i in range(n_series):
            valid = valid_mask[:, i]
            if np.sum(valid) == 0:
                var_per_series.append(1.0)
            else:
                var_per_series.append(np.var(y_true[valid, i], ddof=0))
        var_per_series = np.array(var_per_series)
        var_per_series = np.atleast_1d(var_per_series)
    
    # Handle zero variance (constant series)
    zero_var_mask = var_per_series < 1e-10
    if np.any(zero_var_mask):
        logger.warning(f"{np.sum(zero_var_mask)} series have zero variance. Setting variance to 1.0 for these series.")
        var_per_series[zero_var_mask] = 1.0
    
    # Standardized MSE per series: MSE / variance
    smse_per_series = mse_per_series / var_per_series
    
    # Handle inf (may occur if variance is zero), but keep NaN for missing data
    smse_per_series = np.where(np.isinf(smse_per_series), 0.0, smse_per_series)
    
    if per_series:
        return smse_per_series
    else:
        # Return NaN if all series have missing data, otherwise mean of valid values
        if np.all(np.isnan(smse_per_series)):
            return float(np.nan)
        else:
            return float(np.nanmean(smse_per_series))


def compute_smae(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    scaler: Optional[Any] = None,
    per_series: bool = False,
    test_data_std: Optional[np.ndarray] = None
) -> float:
    """Compute Standardized Mean Absolute Error (sMAE).
    
    Formula: MAE normalized by test data standard deviation (per series, then average).
    sMAE = mean(MAE_i / std_i) where i indexes series
    
    If test_data_std is provided, uses that for normalization (ensures consistency across models).
    Otherwise, uses standard deviation computed from y_true.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_series) or (n_samples,)
    y_pred : np.ndarray
        Predicted values, same shape as y_true
    scaler : optional
        Scaler object (for consistency with compute_smse, not used here)
    per_series : bool, default False
        If True, return per-series sMAE. If False, return average.
    test_data_std : np.ndarray, optional
        Standard deviation of test data (per series), shape (n_series,).
        If provided, uses this for normalization instead of computing from y_true.
    
    Returns
    -------
    float or np.ndarray
        Standardized MAE (average if per_series=False, per-series if True)
    
    Examples
    --------
    >>> smae = compute_smae(y_true, y_pred, test_data_std=test_std)
    >>> smae  # Average standardized MAE across all series
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle 1D case
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Ensure same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} and {y_pred.shape}")
    
    n_series = y_true.shape[1]
    
    # Mask out NaN values (compute metrics only on valid pairs)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Compute MAE per series (only on valid values)
    mae_per_series = []
    for i in range(n_series):
        valid = valid_mask[:, i]
        if np.sum(valid) == 0:
            # All NaN for this series - return NaN to indicate missing data
            mae_per_series.append(np.nan)
        else:
            mae_per_series.append(np.mean(np.abs(y_true[valid, i] - y_pred[valid, i])))
    
    mae_per_series = np.array(mae_per_series)
    mae_per_series = np.atleast_1d(mae_per_series)
    
    # Get normalization factor: use test_data_std if provided, otherwise compute from y_true
    if test_data_std is not None:
        std_per_series = np.asarray(test_data_std)
        std_per_series = np.atleast_1d(std_per_series)
        if len(std_per_series) != n_series:
            raise ValueError(f"test_data_std must have length {n_series}, got {len(std_per_series)}")
    else:
        # Compute standard deviation from actuals (fallback)
        std_per_series = []
        for i in range(n_series):
            valid = valid_mask[:, i]
            if np.sum(valid) == 0:
                std_per_series.append(1.0)
            else:
                std_per_series.append(np.std(y_true[valid, i], ddof=0))
        std_per_series = np.array(std_per_series)
        std_per_series = np.atleast_1d(std_per_series)
    
    # Handle zero standard deviation (constant series)
    zero_std_mask = std_per_series < 1e-10
    if np.any(zero_std_mask):
        logger.warning(f"{np.sum(zero_std_mask)} series have zero standard deviation. Setting std to 1.0 for these series.")
        std_per_series[zero_std_mask] = 1.0
    
    # Standardized MAE per series: MAE / std
    smae_per_series = mae_per_series / std_per_series
    
    # Handle inf (may occur if std is zero), but keep NaN for missing data
    smae_per_series = np.where(np.isinf(smae_per_series), 0.0, smae_per_series)
    
    if per_series:
        return smae_per_series
    else:
        # Return NaN if all series have missing data, otherwise mean of valid values
        if np.all(np.isnan(smae_per_series)):
            return float(np.nan)
        else:
            return float(np.nanmean(smae_per_series))


def save_experiment_results(
    output_dir: Path,
    predictions: np.ndarray,
    actuals: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    target_series: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Save experiment results to files.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save results
    predictions : np.ndarray
        Predictions, shape (n_samples, n_series) or (horizon, n_series)
    actuals : np.ndarray, optional
        Actual values, same shape as predictions
    dates : pd.DatetimeIndex, optional
        Dates for predictions
    target_series : list, optional
        Names of target series
    metrics : dict, optional
        Dictionary of metrics (e.g., {"smse": 0.5, "smae": 0.3})
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = np.asarray(predictions)
    
    # Create predictions DataFrame
    if target_series is None:
        target_series = [f"series_{i}" for i in range(predictions.shape[1] if predictions.ndim > 1 else 1)]
    
    pred_df = pd.DataFrame(predictions, index=dates, columns=target_series) if dates is not None else pd.DataFrame(predictions, columns=target_series)
    
    pred_df.to_csv(output_dir / "predictions.csv")
    
    # Save actuals if provided
    if actuals is not None:
        actuals = np.asarray(actuals)
        actual_df = pd.DataFrame(actuals, index=dates, columns=target_series) if dates is not None else pd.DataFrame(actuals, columns=target_series)
        actual_df.to_csv(output_dir / "actuals.csv")
    
    # Save metrics if provided
    if metrics:
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
