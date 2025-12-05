"""Nowcasting simulation helper functions for backtesting.

This module provides functions to simulate nowcasting scenarios by masking
recent observations to simulate publication lags and real-time data availability.
"""

from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def mask_recent_observations(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    mask_days: int,
    date_index: Optional[pd.DatetimeIndex] = None,
    target_date: Optional[Union[datetime, str]] = None,
    mask_columns: Optional[Union[list, str]] = None
) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], np.ndarray]:
    """Mask recent observations to simulate publication lag.
    
    This function masks the most recent `mask_days` days of data to simulate
    a nowcasting scenario where recent data is not yet available due to
    publication lags.
    
    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        Time series data (T × N) or (T,)
    mask_days : int
        Number of days to mask from the end (simulates publication lag)
    date_index : pd.DatetimeIndex, optional
        Date index for the data. If None and data is DataFrame/Series with
        DatetimeIndex, uses data.index
    target_date : datetime or str, optional
        Target date to mask from. If None, uses the last date in data.
        Data after (target_date - mask_days) will be masked.
    mask_columns : list or str, optional
        Columns to mask (for DataFrame). If None, masks all columns.
        If string, masks only that column.
        
    Returns
    -------
    masked_data : pd.DataFrame, pd.Series, or np.ndarray
        Data with recent observations masked (set to NaN)
    mask : np.ndarray
        Boolean mask indicating which observations were masked (True = masked)
        
    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> 
    >>> # Create sample data
    >>> dates = pd.date_range('2024-01-01', periods=100, freq='D')
    >>> data = pd.DataFrame({'series1': np.random.randn(100), 'series2': np.random.randn(100)}, index=dates)
    >>> 
    >>> # Mask last 25 days
    >>> masked_data, mask = mask_recent_observations(data, mask_days=25)
    >>> 
    >>> # Check: last 25 rows should be NaN
    >>> assert masked_data.iloc[-25:].isna().all().all()
    """
    # Get date index
    if date_index is None:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data.index, pd.DatetimeIndex):
                date_index = data.index
            else:
                # Integer index - create dummy dates
                date_index = pd.date_range('2000-01-01', periods=len(data), freq='D')
        else:
            # Numpy array - create dummy dates
            date_index = pd.date_range('2000-01-01', periods=len(data), freq='D')
    
    # Determine target date
    if target_date is None:
        target_date = date_index[-1]
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Calculate cutoff date
    cutoff_date = target_date - timedelta(days=mask_days)
    
    # Create mask: True for dates after cutoff (to be masked)
    mask = date_index > cutoff_date
    
    # Apply mask to data
    if isinstance(data, pd.DataFrame):
        masked_data = data.copy()
        if mask_columns is None:
            # Mask all columns
            masked_data.loc[mask, :] = np.nan
        elif isinstance(mask_columns, str):
            # Mask single column
            masked_data.loc[mask, mask_columns] = np.nan
        else:
            # Mask specified columns
            masked_data.loc[mask, mask_columns] = np.nan
    elif isinstance(data, pd.Series):
        masked_data = data.copy()
        masked_data.loc[mask] = np.nan
    else:
        # Numpy array
        masked_data = data.copy()
        if data.ndim == 1:
            masked_data[mask] = np.nan
        else:
            masked_data[mask, :] = np.nan
    
    return masked_data, mask


def create_nowcasting_splits(
    data: Union[pd.DataFrame, pd.Series],
    target_dates: list,
    mask_days: int,
    train_size: Optional[int] = None,
    test_size: int = 1
) -> list:
    """Create train/test splits for nowcasting backtesting.
    
    For each target date, creates a split where:
    - Training data: All data available up to (target_date - mask_days)
    - Test data: Target date (to be nowcasted)
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time series data with DatetimeIndex
    target_dates : list
        List of target dates (datetime or str) to nowcast
    mask_days : int
        Publication lag in days (data after target_date - mask_days is masked)
    train_size : int, optional
        Maximum number of training samples. If None, uses all available data.
    test_size : int, default=1
        Number of test samples (typically 1 for nowcasting)
        
    Returns
    -------
    list
        List of tuples (train_idx, test_idx) for each target date
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    splits = []
    
    for target_date in target_dates:
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Calculate cutoff date
        cutoff_date = target_date - timedelta(days=mask_days)
        
        # Find indices
        train_mask = data.index <= cutoff_date
        test_mask = data.index == target_date
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        # Limit training size if specified
        if train_size is not None and len(train_idx) > train_size:
            train_idx = train_idx[-train_size:]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def simulate_nowcasting_evaluation(
    forecaster,
    data: Union[pd.DataFrame, pd.Series],
    target_dates: list,
    mask_days: int,
    horizons: Union[list, np.ndarray] = [1],
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, Any]:
    """Simulate nowcasting evaluation with masked data.
    
    This function simulates a nowcasting scenario by:
    1. Masking recent observations (publication lag)
    2. Training on available data
    3. Evaluating on target dates
    4. Comparing with full-data forecasts
    
    Parameters
    ----------
    forecaster : sktime BaseForecaster or callable
        Forecaster to evaluate. If callable, should accept (y_train, fh) and return predictions.
    data : pd.DataFrame or pd.Series
        Full time series data with DatetimeIndex
    target_dates : list
        List of target dates to nowcast
    mask_days : int
        Publication lag in days
    horizons : list or np.ndarray, default=[1]
        Forecast horizons to evaluate
    target_series : str or int, optional
        Target series to evaluate (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary with evaluation results:
        - 'nowcast_metrics': Metrics for nowcasting (with masked data)
        - 'full_metrics': Metrics for full-data forecasting (baseline)
        - 'improvement': Improvement of full-data over nowcasting
        - 'target_dates': List of evaluated target dates
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    # Create splits
    splits = create_nowcasting_splits(data, target_dates, mask_days)
    
    nowcast_results = []
    full_results = []
    
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        
        # Get training and test data
        y_train = data.iloc[train_idx]
        y_test = data.iloc[test_idx]
        target_date = data.index[test_idx[0]]
        
        # Nowcasting: Use masked training data
        y_train_masked, _ = mask_recent_observations(
            y_train, mask_days=mask_days, target_date=target_date
        )
        
        # Remove rows that are all NaN
        y_train_masked = y_train_masked.dropna(how='all')
        
        if len(y_train_masked) == 0:
            continue
        
        # Fit and predict with masked data (nowcasting)
        try:
            if hasattr(forecaster, 'fit') and hasattr(forecaster, 'predict'):
                # sktime forecaster
                forecaster.fit(y_train_masked)
                fh = np.asarray(horizons)
                y_pred_nowcast = forecaster.predict(fh=fh)
            else:
                # Callable
                y_pred_nowcast = forecaster(y_train_masked, fh=horizons)
            
            # Calculate nowcasting metrics
            from .evaluation import calculate_metrics_per_horizon
            nowcast_metrics = calculate_metrics_per_horizon(
                y_test, y_pred_nowcast, horizons, y_train=y_train_masked,
                target_series=target_series
            )
            nowcast_results.append({
                'target_date': target_date,
                'metrics': nowcast_metrics
            })
        except Exception as e:
            print(f"Error in nowcasting evaluation for {target_date}: {e}")
            continue
        
        # Full data: Use all training data (baseline)
        try:
            if hasattr(forecaster, 'fit') and hasattr(forecaster, 'predict'):
                forecaster.fit(y_train)
                y_pred_full = forecaster.predict(fh=fh)
            else:
                y_pred_full = forecaster(y_train, fh=horizons)
            
            # Calculate full-data metrics
            full_metrics = calculate_metrics_per_horizon(
                y_test, y_pred_full, horizons, y_train=y_train,
                target_series=target_series
            )
            full_results.append({
                'target_date': target_date,
                'metrics': full_metrics
            })
        except Exception as e:
            print(f"Error in full-data evaluation for {target_date}: {e}")
            continue
    
    # Aggregate results across target dates
    # Average metrics across all target dates
    if len(nowcast_results) > 0 and len(full_results) > 0:
        # Get all horizons
        all_horizons = set()
        for result in nowcast_results + full_results:
            all_horizons.update(result['metrics'].keys())
        
        nowcast_avg = {}
        full_avg = {}
        
        for h in all_horizons:
            nowcast_vals = [r['metrics'].get(h, {}) for r in nowcast_results]
            full_vals = [r['metrics'].get(h, {}) for r in full_results]
            
            # Average each metric
            nowcast_avg[h] = {
                'sMSE': np.nanmean([v.get('sMSE', np.nan) for v in nowcast_vals]),
                'sMAE': np.nanmean([v.get('sMAE', np.nan) for v in nowcast_vals]),
                'sRMSE': np.nanmean([v.get('sRMSE', np.nan) for v in nowcast_vals]),
            }
            full_avg[h] = {
                'sMSE': np.nanmean([v.get('sMSE', np.nan) for v in full_vals]),
                'sMAE': np.nanmean([v.get('sMAE', np.nan) for v in full_vals]),
                'sRMSE': np.nanmean([v.get('sRMSE', np.nan) for v in full_vals]),
            }
        
        # Calculate improvement (negative = nowcasting worse, positive = better)
        improvement = {}
        for h in all_horizons:
            improvement[h] = {
                'sMSE': full_avg[h]['sMSE'] - nowcast_avg[h]['sMSE'],
                'sMAE': full_avg[h]['sMAE'] - nowcast_avg[h]['sMAE'],
                'sRMSE': full_avg[h]['sRMSE'] - nowcast_avg[h]['sRMSE'],
            }
    else:
        nowcast_avg = {}
        full_avg = {}
        improvement = {}
    
    return {
        'nowcast_metrics': nowcast_avg,
        'full_metrics': full_avg,
        'improvement': improvement,
        'target_dates': [r['target_date'] for r in nowcast_results],
        'n_evaluations': len(nowcast_results)
    }
