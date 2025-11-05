"""Forecasting functions for DFM models.

This module implements forward prediction using AR dynamics, reverse transformations,
and confidence interval calculation for multi-horizon forecasts.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from datetime import date, timedelta
import pandas as pd

from .config import ModelConfig


def forecast_factors(
    Z_last: np.ndarray,
    A: np.ndarray,
    V_last: np.ndarray,
    Q: np.ndarray,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast factors forward using AR dynamics.
    
    For h-step ahead forecast:
        Z_{t+h} = A^h @ Z_t
        Var[Z_{t+h}] = A^h @ V_t @ (A^h)^T + sum_{j=0}^{h-1} A^j @ Q @ (A^j)^T
    
    Parameters
    ----------
    Z_last : np.ndarray
        Last smoothed factor estimate (m,)
    A : np.ndarray
        Transition matrix (m x m)
    V_last : np.ndarray
        Last smoothed factor covariance (m x m)
    Q : np.ndarray
        Factor innovation covariance (m x m)
    horizon : int
        Forecast horizon (number of steps ahead)
        
    Returns
    -------
    Z_forecast : np.ndarray
        Forecasted factors (horizon x m)
    V_forecast : np.ndarray
        Forecasted factor covariances (horizon x m x m)
    """
    m = Z_last.shape[0]
    Z_forecast = np.zeros((horizon, m))
    V_forecast = np.zeros((horizon, m, m))
    
    Z_current = Z_last.copy()
    V_current = V_last.copy()
    
    for h in range(horizon):
        # Forecast: Z_{t+h} = A @ Z_{t+h-1}
        Z_forecast[h] = A @ Z_current
        
        # Variance: Var[Z_{t+h}] = A @ Var[Z_{t+h-1}] @ A^T + Q
        V_forecast[h] = A @ V_current @ A.T + Q
        
        # Update for next iteration
        Z_current = Z_forecast[h]
        V_current = V_forecast[h]
    
    return Z_forecast, V_forecast


def forecast_data(
    Z_forecast: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    V_forecast: np.ndarray,
    Mx: np.ndarray,
    Wx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project forecasted factors to data space.
    
    Parameters
    ----------
    Z_forecast : np.ndarray
        Forecasted factors (horizon x m)
    C : np.ndarray
        Loading matrix (N x m)
    R : np.ndarray
        Observation noise covariance (N x N)
    V_forecast : np.ndarray
        Forecasted factor covariances (horizon x m x m)
    Mx : np.ndarray
        Data means (N,) for reverse standardization
    Wx : np.ndarray
        Data standard deviations (N,) for reverse standardization
        
    Returns
    -------
    y_forecast : np.ndarray
        Forecasted data (horizon x N) in standardized space
    y_forecast_var : np.ndarray
        Forecasted data variances (horizon x N)
    """
    horizon, m = Z_forecast.shape
    N = C.shape[0]
    
    # Project factors to data space: y = C @ Z
    y_forecast = Z_forecast @ C.T  # (horizon x m) @ (m x N) = (horizon x N)
    
    # Calculate forecast variance: Var[y] = C @ Var[Z] @ C^T + R
    y_forecast_var = np.zeros((horizon, N))
    for h in range(horizon):
        # Var[y_h] = C @ V_h @ C^T + R
        y_var_h = C @ V_forecast[h] @ C.T + R
        y_forecast_var[h] = np.diag(y_var_h)
    
    return y_forecast, y_forecast_var


def inverse_transform_series(
    y_transformed: np.ndarray,
    y_original: np.ndarray,
    transformation: str,
    frequency: str,
    step: int = 1
) -> np.ndarray:
    """Reverse transformation to get levels from transformed values.
    
    Parameters
    ----------
    y_transformed : np.ndarray
        Transformed values (horizon,)
    y_original : np.ndarray
        Original level values (T,) - used for cumulative transformations
    transformation : str
        Transformation code: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'log'
    frequency : str
        Frequency: 'm' (monthly) or 'q' (quarterly)
    step : int
        Step size (3 for quarterly, 1 for monthly)
        
    Returns
    -------
    y_levels : np.ndarray
        Values in original level (horizon,)
    """
    horizon = len(y_transformed)
    y_levels = np.zeros(horizon)
    
    # Get last original value for cumulative transformations
    last_original = y_original[-1] if len(y_original) > 0 else 0
    
    if transformation == 'lin':
        # No transformation: already in levels
        y_levels = y_transformed.copy()
    
    elif transformation == 'chg':
        # First difference: y_t = y_{t-1} + Δy_t
        y_levels[0] = last_original + y_transformed[0]
        for h in range(1, horizon):
            y_levels[h] = y_levels[h-1] + y_transformed[h]
    
    elif transformation == 'ch1':
        # Year-over-year difference: y_t = y_{t-12} + Δy_t
        # Need historical data - simplified: use last value
        if len(y_original) >= 12:
            y_levels = y_original[-12] + y_transformed
        else:
            y_levels = last_original + y_transformed
    
    elif transformation == 'pch':
        # Percent change: y_t = y_{t-1} * (1 + Δy_t/100)
        y_levels[0] = last_original * (1 + y_transformed[0] / 100)
        for h in range(1, horizon):
            y_levels[h] = y_levels[h-1] * (1 + y_transformed[h] / 100)
    
    elif transformation == 'pc1':
        # Year-over-year percent change: y_t = y_{t-12} * (1 + Δy_t/100)
        if len(y_original) >= 12:
            y_levels = y_original[-12] * (1 + y_transformed / 100)
        else:
            y_levels = last_original * (1 + y_transformed / 100)
    
    elif transformation == 'pca':
        # Percent change annualized: y_t = y_{t-1} * (1 + Δy_t/100)^(1/n)
        n = step / 12  # Annualization factor
        y_levels[0] = last_original * ((1 + y_transformed[0] / 100) ** (1/n))
        for h in range(1, horizon):
            y_levels[h] = y_levels[h-1] * ((1 + y_transformed[h] / 100) ** (1/n))
    
    elif transformation == 'log':
        # Natural log: y_t = exp(log_y_t)
        y_levels = np.exp(y_transformed)
    
    else:
        # Unknown transformation: return as-is
        y_levels = y_transformed.copy()
    
    return y_levels


def forecast_with_intervals(
    Res,
    config: ModelConfig,
    horizon: int,
    series_indices: Optional[List[int]] = None,
    confidence_level: float = 0.95
) -> Dict[str, np.ndarray]:
    """Generate forecasts with confidence intervals for specified series.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results
    config : ModelConfig
        Model configuration
    horizon : int
        Forecast horizon (number of steps ahead)
    series_indices : List[int], optional
        Indices of series to forecast (if None, forecasts all series)
    confidence_level : float, default 0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    Dict with keys:
        'forecast': (horizon x N) forecasted values (standardized)
        'forecast_levels': (horizon x N) forecasted values (original levels)
        'forecast_var': (horizon x N) forecast variances
        'lower_bound': (horizon x N) lower confidence bounds (levels)
        'upper_bound': (horizon x N) upper confidence bounds (levels)
        'forecast_dates': (horizon,) forecast dates
    """
    # Get last smoothed factors and covariance
    if hasattr(Res, 'Z') and Res.Z is not None:
        Z_last = Res.Z[-1, :]  # Last smoothed factor (m,)
    else:
        raise ValueError("DFMResult must have smoothed factors (Z)")
    
    # Get last smoothed factor covariance
    # Note: DFMResult doesn't store V_t|T for each time step
    # Use steady-state approximation: V_ss = V_0 (stationary) or compute from A, Q
    # For proper implementation, we'd need to store V_t|T from smoother
    # For now, use V_0 as approximation (reasonable for stationary processes)
    # Alternative: solve Lyapunov equation for steady-state covariance
    try:
        from scipy.linalg import solve_discrete_lyapunov
        # Steady-state covariance: V_ss = A @ V_ss @ A^T + Q
        V_steady = solve_discrete_lyapunov(Res.A, Res.Q)
        V_last = V_steady
    except (ImportError, Exception):
        # Fallback: use V_0 if Lyapunov solver not available
        V_last = Res.V_0.copy()
    
    # Forecast factors forward
    Z_forecast, V_forecast = forecast_factors(
        Z_last, Res.A, V_last, Res.Q, horizon
    )
    
    # Project to data space
    y_forecast, y_forecast_var = forecast_data(
        Z_forecast, Res.C, Res.R, V_forecast, Res.Mx, Res.Wx
    )
    
    # Select series if specified
    if series_indices is not None:
        y_forecast = y_forecast[:, series_indices]
        y_forecast_var = y_forecast_var[:, series_indices]
        Mx = Res.Mx[series_indices]
        Wx = Res.Wx[series_indices]
        transformations = [config.Transformation[i] for i in series_indices]
        frequencies = [config.Frequency[i] for i in series_indices]
    else:
        Mx = Res.Mx
        Wx = Res.Wx
        transformations = config.Transformation
        frequencies = config.Frequency
    
    # Reverse standardization: y_levels = y_std * Wx + Mx
    y_forecast_levels = y_forecast * Wx + Mx
    
    # Reverse transformations to original scale
    # Note: Need original data for cumulative transformations
    # For now, use last smoothed values as approximation
    if hasattr(Res, 'X_sm'):
        X_sm = Res.X_sm
    else:
        X_sm = Res.x_sm * Res.Wx + Res.Mx
    
    y_forecast_final = np.zeros_like(y_forecast_levels)
    for i in range(len(transformations)):
        step = 3 if frequencies[i] == 'q' else 1
        y_forecast_final[:, i] = inverse_transform_series(
            y_forecast_levels[:, i],
            X_sm[:, series_indices[i] if series_indices else i],
            transformations[i],
            frequencies[i],
            step
        )
    
    # Calculate confidence intervals
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    forecast_std = np.sqrt(y_forecast_var) * Wx  # Scale variance by Wx
    
    lower_bound = y_forecast_final - z_score * forecast_std
    upper_bound = y_forecast_final + z_score * forecast_std
    
    return {
        'forecast': y_forecast,
        'forecast_levels': y_forecast_final,
        'forecast_var': y_forecast_var,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'forecast_dates': None  # Will be set by caller with actual dates
    }


def generate_forecast_dates(
    last_date: pd.Timestamp,
    frequency: str,
    horizon: int
) -> pd.DatetimeIndex:
    """Generate forecast dates based on frequency.
    
    Parameters
    ----------
    last_date : pd.Timestamp
        Last observation date
    frequency : str
        Frequency: 'm' (monthly) or 'q' (quarterly) or 'd' (daily)
    horizon : int
        Forecast horizon (number of steps)
        
    Returns
    -------
    pd.DatetimeIndex
        Forecast dates
    """
    if frequency == 'q':
        # Quarterly: add 3 months per step
        dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=horizon,
            freq='Q'
        )
    elif frequency == 'm':
        # Monthly: add 1 month per step
        dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='M'
        )
    elif frequency == 'd':
        # Daily: add 1 day per step
        dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=horizon,
            freq='D'
        )
    else:
        # Default: monthly
        dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='M'
        )
    
    return dates

