"""Nowcasting utility functions.

This module contains helper functions for nowcasting operations:
- Frequency and date calculations
- Forecast horizon configuration
- Data transformations
- Configuration validation
- News decomposition utilities
- Data view creation (nowcasting-specific)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING, Union, Iterator
from datetime import datetime, timedelta
import logging
from calendar import monthrange
import pandas as pd
from dataclasses import dataclass

from dfm_python.config import DFMConfig
from dfm_python.utils.time import TimeIndex, parse_timestamp, to_python_datetime, get_latest_time, clock_to_datetime_freq, parse_period_string
from dfm_python.utils.helpers import get_series_ids, get_clock_frequency
from dfm_python.config.utils import FREQUENCY_HIERARCHY, get_periods_per_year
from dfm_python.logger import get_logger

# Optional sktime imports
try:
    from sktime.transformations.base import BaseTransformer
    from sktime.split.base import BaseSplitter
    from sktime.forecasting.base import BaseForecaster
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    BaseTransformer = None
    BaseSplitter = None
    BaseForecaster = None

_logger = get_logger(__name__)


def sort_data(Z: np.ndarray, Mnem: List[str], config: DFMConfig) -> Tuple[np.ndarray, List[str]]:
    """Sort data columns to match configuration order.
    
    Parameters
    ----------
    Z : np.ndarray
        Data matrix (T x N)
    Mnem : List[str]
        Series identifiers (mnemonics) from data file
    config : DFMConfig
        Model configuration with series order
        
    Returns
    -------
    Z_sorted : np.ndarray
        Sorted data matrix (T x N)
    Mnem_sorted : List[str]
        Sorted series identifiers
    """
    series_ids = get_series_ids(config)
    
    # Create mapping from series_id to index in data
    mnem_to_idx = {m: i for i, m in enumerate(Mnem)}
    
    # Find permutation
    perm = []
    Mnem_filt = []
    for sid in series_ids:
        if sid in mnem_to_idx:
            perm.append(mnem_to_idx[sid])
            Mnem_filt.append(sid)
        else:
            _logger.warning(f"Series '{sid}' from config not found in data")
    
    if len(perm) == 0:
        raise ValueError("No matching series found between config and data")
    
    # Apply permutation
    Z_filt = Z[:, perm]
    
    return Z_filt, Mnem_filt


def rem_nans_spline(X: np.ndarray, method: int = 2, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Treat NaNs in dataset for DFM estimation using standard interpolation methods.
    
    This function implements standard econometric practice for handling missing data
    in time series, following the approach used in FRBNY Nowcasting Model and similar
    DFM implementations.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (T x N)
    method : int
        Missing data handling method:
        - 1: Replace all missing values using spline interpolation
        - 2: Remove >80% NaN rows, then fill (default, recommended)
        - 3: Only remove all-NaN rows
        - 4: Remove all-NaN rows, then fill
        - 5: Fill missing values
    k : int
        Spline interpolation order (default: 3 for cubic spline)
        
    Returns
    -------
    X : np.ndarray
        Data with NaNs treated
    indNaN : np.ndarray
        Boolean mask indicating original NaN positions
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import lfilter
    
    # Ensure X is a numeric numpy array
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        try:
            X = X.astype(np.float64)
        except (ValueError, TypeError):
            try:
                X_df = pd.DataFrame(X)
                X = X_df.select_dtypes(include=[np.number]).to_numpy()
                if X.size == 0:
                    raise ValueError("Input data contains no numeric columns")
                if X.shape != X_df.shape:
                    _logger.warning(f"Non-numeric columns removed. Shape changed from {X_df.shape} to {X.shape}")
            except ImportError:
                raise TypeError(f"Cannot convert input data to numeric. dtype: {X.dtype}")
    
    T, N = X.shape
    indNaN = np.isnan(X)
    
    def _remove_leading_trailing(threshold: float):
        """Remove rows with NaN count above threshold."""
        rem = np.sum(indNaN, axis=1) > (N * threshold if threshold < 1 else threshold)
        nan_lead = np.cumsum(rem) == np.arange(1, T + 1)
        nan_end = np.cumsum(rem[::-1]) == np.arange(1, T + 1)[::-1]
        return ~(nan_lead | nan_end)
    
    def _fill_missing(x: np.ndarray, mask: np.ndarray):
        """Fill missing values using spline interpolation and moving average."""
        if len(mask) != len(x):
            mask = mask[:len(x)]
        
        non_nan = np.where(~mask)[0]
        if len(non_nan) < 2:
            return x
        
        x_filled = x.copy()
        if non_nan[-1] >= len(x):
            non_nan = non_nan[non_nan < len(x)]
        if len(non_nan) < 2:
            return x
        
        x_filled[non_nan[0]:non_nan[-1]+1] = CubicSpline(non_nan, x[non_nan])(np.arange(non_nan[0], min(non_nan[-1]+1, len(x))))
        x_filled[mask[:len(x_filled)]] = np.nanmedian(x_filled)
        
        # Moving average filter
        pad = np.concatenate([np.full(k, x_filled[0]), x_filled, np.full(k, x_filled[-1])])
        ma = lfilter(np.ones(2*k+1)/(2*k+1), 1, pad)[2*k+1:]
        if len(ma) == len(x_filled):
            x_filled[mask[:len(x_filled)]] = ma[mask[:len(x_filled)]]
        return x_filled
    
    if method == 1:
        # Replace all missing values
        for i in range(N):
            mask = indNaN[:, i]
            x = X[:, i].copy()
            x[mask] = np.nanmedian(x)
            pad = np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])])
            ma = lfilter(np.ones(2*k+1)/(2*k+1), 1, pad)[2*k+1:]
            x[mask] = ma[mask]
            X[:, i] = x
    
    elif method == 2:
        # Remove >80% NaN rows, then fill
        mask = _remove_leading_trailing(0.8)
        X = X[mask]
        indNaN = np.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 3:
        # Only remove all-NaN rows
        mask = _remove_leading_trailing(N)
        X = X[mask]
        indNaN = np.isnan(X)
    
    elif method == 4:
        # Remove all-NaN rows, then fill
        mask = _remove_leading_trailing(N)
        X = X[mask]
        indNaN = np.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 5:
        # Fill missing values
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    return X, indNaN


def calculate_release_date(release_date: int, period: datetime) -> datetime:
    """Calculate release date relative to the period.
    
    Parameters
    ----------
    release_date : int
        Release date offset (day of month, or negative days before end of previous month)
    period : datetime
        Period for which to calculate release date
        
    Returns
    -------
    datetime
        Calculated release date
    """
    if release_date is None:
        return period
    
    if release_date >= 1:
        # Day of current month
        last_day = monthrange(period.year, period.month)[1]
        day = min(release_date, last_day)
        return datetime(period.year, period.month, day)
    
    # release_date < 0 => days before end of previous month
    if period.month == 1:
        prev_year = period.year - 1
        prev_month = 12
    else:
        prev_year = period.year
        prev_month = period.month - 1
    last_day_prev_month = monthrange(prev_year, prev_month)[1]
    day = last_day_prev_month + release_date + 1
    day = max(1, day)
    return datetime(prev_year, prev_month, day)


def create_data_view(
    X: np.ndarray,
    Time: Union[TimeIndex, Any],
    Z: Optional[np.ndarray] = None,
    config: Optional[DFMConfig] = None,
    view_date: Union[datetime, str, None] = None,
    *,
    X_frame: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
    """Create data view at a specific view date.
    
    This function masks data that is not yet available at the view_date based on
    release date information in the config.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    Time : TimeIndex or array-like
        Time index for the data
    Z : np.ndarray, optional
        Original untransformed data (T x N)
    config : DFMConfig, optional
        Model configuration with series release date information
    view_date : datetime or str, optional
        Date for which to create the view. If None, uses latest available date.
    X_frame : pd.DataFrame, optional
        Pandas DataFrame representation of X (for efficient masking)
        
    Returns
    -------
    X_view : np.ndarray
        Masked data matrix (T x N) with NaN for unavailable observations
    Time : TimeIndex or array-like
        Time index (unchanged)
    Z_view : np.ndarray or None
        Masked original data (T x N) or None if Z was not provided
    """
    if isinstance(view_date, str):
        view_date = parse_timestamp(view_date)
    elif view_date is None:
        view_date = get_latest_time(Time)
    
    if not isinstance(view_date, datetime):
        view_date = parse_timestamp(view_date)
    
    if config is None or not hasattr(config, 'series') or not config.series:
        return X.copy(), Time, Z.copy() if Z is not None else None
    
    # Prepare time list
    if isinstance(Time, TimeIndex):
        time_list = [to_python_datetime(t) for t in Time]
    else:
        time_list = []
        for t in Time:
            if isinstance(t, datetime):
                time_list.append(t)
            elif isinstance(t, pd.Timestamp):
                time_list.append(t.to_pydatetime())
            elif hasattr(t, 'to_pydatetime'):
                time_list.append(t.to_pydatetime())
            elif hasattr(t, 'to_python'):
                time_list.append(t.to_python())
            else:
                time_list.append(parse_timestamp(t))
    
    # Build pandas DataFrame reference
    try:
        series_ids = get_series_ids(config)
    except ValueError:
        series_ids = [f'series_{i}' for i in range(X.shape[1])]
    
    if X_frame is not None:
        df = X_frame.copy()
    else:
        df = pd.DataFrame(X, columns=series_ids[:X.shape[1]])
    df['_view_time'] = time_list
    
    # Track masks for applying to numpy fallbacks
    series_masks: Dict[int, np.ndarray] = {}
    
    for i, series_cfg in enumerate(config.series):
        if i >= len(df.columns) - 1:  # exclude time column
            continue
        release_offset = getattr(series_cfg, 'release_date', None)
        if release_offset is None:
            continue
        
        release_dates = [calculate_release_date(release_offset, t) for t in time_list]
        mask = np.array([view_date >= rd for rd in release_dates], dtype=bool)
        series_masks[i] = mask
        
        # Apply mask using pandas where
        df[series_ids[i]] = df[series_ids[i]].where(mask, None)
    
    df_view = df.drop(columns=['_view_time'])
    X_view = df_view.to_numpy()
    
    if Z is not None:
        Z_view = Z.copy()
        for i, mask in series_masks.items():
            Z_view[~mask, i] = np.nan
    else:
        Z_view = None
    
    return X_view, Time, Z_view


def get_higher_frequency(clock: str) -> Optional[str]:
    """Get frequency one step faster than clock.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    str or None
        Frequency one step faster than clock, or None if no higher frequency available
        
    Examples
    --------
    >>> get_higher_frequency('m')
    'w'
    >>> get_higher_frequency('q')
    'm'
    >>> get_higher_frequency('d')
    None
    """
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    target_h = clock_h - 1
    
    if target_h < 1:
        return None  # No higher frequency available (clock is already fastest)
    
    # Find frequency with target hierarchy
    for freq, h in FREQUENCY_HIERARCHY.items():
        if h == target_h:
            return freq
    
    return None  # No higher frequency found


def calc_backward_date(
    target_date: datetime,
    step: int,
    freq: str
) -> datetime:
    """Calculate backward date with accurate calendar handling.
    
    Parameters
    ----------
    target_date : datetime
        Target date to go backward from
    step : int
        Number of steps to go backward
    freq : str
        Frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    datetime
        Calculated backward date
        
    Examples
    --------
    >>> from datetime import datetime
    >>> calc_backward_date(datetime(2024, 3, 15), 1, 'm')
    datetime(2024, 2, 15)
    >>> calc_backward_date(datetime(2024, 3, 15), 2, 'q')
    datetime(2023, 9, 15)
    """
    try:
        from dateutil.relativedelta import relativedelta
        use_relativedelta = True
    except ImportError:
        use_relativedelta = False
        relativedelta = None  # type: ignore
        _logger.debug("dateutil.relativedelta not available, using timedelta approximation")
    
    if freq == 'd':
        return target_date - timedelta(days=step)
    elif freq == 'w':
        return target_date - timedelta(weeks=step)
    elif freq == 'm':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step)
        else:
            # Approximate: 30 days per month
            return target_date - timedelta(days=step * 30)
    elif freq == 'q':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 3)
        else:
            # Approximate: 90 days per quarter
            return target_date - timedelta(days=step * 90)
    elif freq == 'sa':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 6)
        else:
            # Approximate: 180 days per semi-annual
            return target_date - timedelta(days=step * 180)
    elif freq == 'a':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(years=step)
        else:
            # Approximate: 365 days per year
            return target_date - timedelta(days=step * 365)
    else:
        # Fallback for unknown frequencies
        _logger.warning(f"Unknown frequency '{freq}', using 30-day approximation")
        return target_date - timedelta(days=step * 30)


def get_forecast_horizon(clock: str, horizon: Optional[int] = None) -> Tuple[int, str]:
    """Get forecast horizon configuration based on clock frequency.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
    horizon : int, optional
        Number of periods for forecast horizon. If None, defaults to 1 timestep.
        
    Returns
    -------
    Tuple[int, str]
        (horizon_periods, datetime_freq) where:
        - horizon_periods: Number of periods to forecast
        - datetime_freq: Frequency string for datetime_range() ('D', 'W', 'ME', 'QE', 'YE')
        
    Examples
    --------
    >>> get_forecast_horizon('m', 3)
    (3, 'ME')
    >>> get_forecast_horizon('q')
    (1, 'QE')
    """
    if horizon is None:
        horizon = 1  # Default: 1 timestep based on clock frequency
    
    # Map clock frequency to datetime frequency string
    datetime_freq = clock_to_datetime_freq(clock)
    
    # For semi-annual, we need 6 months per period
    if clock == 'sa' and horizon > 0:
        horizon = horizon * 6  # Convert to months
    
    return horizon, datetime_freq


def check_config(saved_config: Any, current_config: DFMConfig) -> None:
    """Check if saved config is consistent with current config.
    
    Parameters
    ----------
    saved_config : Any
        Saved configuration object (may be DFMConfig or dict-like)
    current_config : DFMConfig
        Current configuration object
        
    Notes
    -----
    - Issues a warning if configs differ significantly
    - Does not raise exceptions (allows computation to continue)
    
    Examples
    --------
    >>> check_config(saved_config, current_config)
    # May issue warnings if configs differ
    """
    try:
        # Basic checks
        if hasattr(saved_config, 'series') and hasattr(current_config, 'series'):
            if len(saved_config.series) != len(current_config.series):
                _logger.warning(
                    f"Config mismatch: saved config has {len(saved_config.series)} series, "
                    f"current config has {len(current_config.series)} series"
                )
        
        if hasattr(saved_config, 'block_names') and hasattr(current_config, 'block_names'):
            if saved_config.block_names != current_config.block_names:
                _logger.warning(
                    f"Config mismatch: block names differ. "
                    f"Saved: {saved_config.block_names}, Current: {current_config.block_names}"
                )
    except Exception as e:
        _logger.debug(f"Config consistency check failed (non-critical): {str(e)}")
        # If comparison fails, continue anyway


def extract_news(
    singlenews: np.ndarray,
    weight: np.ndarray,
    series_ids: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """Extract summary statistics from news decomposition.
    
    Parameters
    ----------
    singlenews : np.ndarray
        News contributions (N,) or (N, n_targets)
    weight : np.ndarray
        Weights (N,) or (N, n_targets)
    series_ids : List[str]
        Series IDs
    top_n : int, default 5
        Number of top contributors to include
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'total_impact', 'top_contributors', etc.
        
    Examples
    --------
    >>> summary = extract_news(singlenews, weight, series_ids, top_n=10)
    >>> print(summary['top_contributors'])
    """
    # Handle both 1D and 2D arrays
    if singlenews.ndim == 1:
        news_contributions = singlenews
        weights = weight
    else:
        # If 2D, use first target (column 0)
        news_contributions = singlenews[:, 0]
        weights = weight[:, 0] if weight.ndim > 1 else weight
    
    # Calculate total impact
    total_impact = np.nansum(news_contributions)
    
    # Get top contributors
    abs_contributions = np.abs(news_contributions)
    top_indices = np.argsort(abs_contributions)[::-1][:top_n]
    
    # Build list of top contributors
    top_contributors = []
    for idx in top_indices:
        if idx < len(series_ids):
            top_contributors.append({
                'series_id': series_ids[idx],
                'contribution': float(news_contributions[idx]),
                'weight': float(weights[idx]) if idx < len(weights) else 0.0
            })
    
    return {
        'total_impact': float(total_impact),
        'top_contributors': top_contributors,
        # Note: revision_impact and release_impact are placeholders for future implementation.
        # revision_impact: Measures impact of data revisions (currently returns total_impact).
        # release_impact: Measures impact of new data releases (currently returns 0.0).
        # These features are not critical for core nowcasting functionality and can be
        # implemented incrementally when needed. See STATUS.md for tracking.
        'revision_impact': float(total_impact),  # Placeholder: returns total_impact
        'release_impact': 0.0  # Placeholder: returns 0.0
    }


# ============================================================================
# News decomposition and backtest classes (merged from helpers.py)
# ============================================================================

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nowcast import NowcastResult

from dfm_python.config.results import DFMResult
from dfm_python.config import DFMConfig
from dfm_python.utils.time import TimeIndex, parse_timestamp, clock_to_datetime_freq
from dfm_python.utils.helpers import (
    get_series_id,
    get_frequencies,
    get_series_ids,
    safe_get_attr,
)
from dfm_python.config.utils import FREQUENCY_HIERARCHY, get_periods_per_year
from dfm_python.logger import get_logger

_logger_helpers = get_logger(__name__)


@dataclass
class NewsDecompResult:
    """Result from news decomposition calculation.
    
    This dataclass contains all information about how new data releases
    affect the nowcast, including the forecast update and contributions
    from each data series.
    
    Attributes
    ----------
    y_old : float
        Nowcast value using old data view
    y_new : float
        Nowcast value using new data view
    change : float
        Forecast update (y_new - y_old)
    singlenews : np.ndarray
        News contributions per series (N,) or (N, n_targets)
    top_contributors : List[Tuple[str, float]]
        Top contributors to the forecast update, sorted by absolute impact
    actual : np.ndarray
        Actual values of newly released data
    forecast : np.ndarray
        Forecasted values for new data (from old view)
    weight : np.ndarray
        Weights for news contributions (N,) or (N, n_targets)
    t_miss : np.ndarray
        Time indices of new data releases
    v_miss : np.ndarray
        Variable indices of new data releases
    innov : np.ndarray
        Innovation terms (standardized differences between actual and forecast)
    """
    y_old: float
    y_new: float
    change: float
    singlenews: np.ndarray
    top_contributors: List[Tuple[str, float]]
    actual: np.ndarray
    forecast: np.ndarray
    weight: np.ndarray
    t_miss: np.ndarray
    v_miss: np.ndarray
    innov: np.ndarray


def para_const(X: np.ndarray, result: DFMResult, lag: int = 0) -> Dict[str, Any]:
    """Implement Kalman filter for news calculation with fixed parameters.
    
    This function applies the Kalman filter and smoother to a data matrix X
    using pre-estimated model parameters from a DFMResult. It is used in
    news decomposition when model parameters are already known.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N) with potentially missing values (NaN)
    result : DFMResult
        DFM result containing estimated parameters (A, C, Q, R, Mx, Wx, Z_0, V_0)
    lag : int, default 0
        Maximum lag for calculating Plag (smoothed factor covariances)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'Plag': List of smoothed factor covariances for different lags
        - 'P': Smoothed factor covariance matrix
        - 'X_sm': Smoothed data matrix (T x N)
        - 'F': Smoothed factors (T x r)
        - 'Z': Smoothed factors (T+1 x r, includes initial state)
        - 'V': Smoothed factor covariances (T+1 x r x r)
    
    Notes
    -----
    This function is based on the MATLAB para_const() function from the
    FRBNY Nowcasting Model. It implements Kalman filtering with fixed
    parameters for use in news decomposition calculations.
    
    The function standardizes the input data using Mx and Wx from the
    result, applies the Kalman filter and smoother, then transforms
    the smoothed factors back to observation space.
    """
    # Extract parameters from result
    Z_0 = result.Z_0
    V_0 = result.V_0
    A = result.A
    C = result.C
    Q = result.Q
    R = result.R
    Mx = result.Mx
    Wx = result.Wx
    
    T, N = X.shape
    r = A.shape[0]  # Number of factors
    
    # Standardize data: Y = (X - Mx) / Wx
    # Handle division by zero
    Wx_safe = np.where(Wx == 0, 1.0, Wx)
    Y = ((X - Mx) / Wx_safe).T  # Transpose to (N x T) for Kalman filter
    
    # Use PyTorch Lightning KalmanFilter
    try:
        import torch
        from dfm_python.ssm.kalman import KalmanFilter
        
        # Convert to torch tensors
        device = torch.device('cpu')  # Use CPU for nowcasting
        Y_torch = torch.tensor(Y, dtype=torch.float64, device=device)
        A_torch = torch.tensor(A, dtype=torch.float64, device=device)
        C_torch = torch.tensor(C, dtype=torch.float64, device=device)
        Q_torch = torch.tensor(Q, dtype=torch.float64, device=device)
        R_torch = torch.tensor(R, dtype=torch.float64, device=device)
        Z_0_torch = torch.tensor(Z_0, dtype=torch.float64, device=device)
        V_0_torch = torch.tensor(V_0, dtype=torch.float64, device=device)
        
        # Apply Kalman filter and smoother
        kalman = KalmanFilter()
        zsmooth, Vsmooth, VVsmooth, _ = kalman(
            Y_torch, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch
        )
        
        # Convert back to numpy
        Zsmooth = zsmooth.T.cpu().numpy()  # (T+1) x r
        Vsmooth = Vsmooth.cpu().numpy()  # r x r x (T+1)
        VVsmooth = VVsmooth.cpu().numpy()  # r x r x T
        
        # Get filtered state for Vf calculation
        Sf = kalman.filter_forward(Y_torch, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch)
        Vf = Sf.VmU.cpu().numpy()  # r x r x (T+1)
        
        # Smoothed factor covariances for transition matrix
        # Vs is V_{t|T} for t = 1, ..., T (skip initial state)
        Vs = Vsmooth[:, :, 1:].transpose(2, 0, 1)  # T x r x r
        Vf = Vf[:, :, 1:]  # r x r x T (filtered posterior covariance, skip initial)
        
    except ImportError:
        raise ImportError(
            "PyTorch is required for para_const. Install with: pip install torch"
        )
    
    # Calculate Plag (smoothed factor covariances for different lags)
    # MATLAB: Plag{1} = Vs; then for jk = 1:lag, jt = size(Plag{1},3):-1:lag+1
    # Note: MATLAB uses 1-based indexing, we use 0-based
    # size(Plag{1},3) in MATLAB = T, so jt goes from T down to lag+1
    # In 0-based: jt goes from T-1 down to lag (equivalent to lag+1 to T in 1-based)
    Plag = [Vs]  # Plag[0] = Vs (lag 0)
    
    if lag > 0:
        for jk in range(1, lag + 1):
            Plag_jk = np.zeros_like(Vs)
            # MATLAB: for jt = size(Plag{1},3):-1:lag+1
            # In 0-based: jt from T-1 down to lag (inclusive)
            for jt in range(T - 1, lag - 1, -1):  # Backwards from T-1 to lag
                # Calculate smoothed covariance for lag jk at time jt
                # MATLAB: As = Vf(:,:,jt-jk)*A'*pinv(A*Vf(:,:,jt-jk)*A'+Q)
                # Note: Vf is r x r x T, we need to index it correctly
                # jt-jk in MATLAB (1-based) = jt-jk in Python (0-based) if jt-jk >= 0
                if jt - jk >= 0:
                    # Vf is r x r x T, index as Vf[:, :, jt - jk]
                    V_t_jk = Vf[:, :, jt - jk]
                else:
                    # Use first smoothed covariance as fallback
                    V_t_jk = Vs[0]
                
                try:
                    # MATLAB: As = Vf(:,:,jt-jk)*A'*pinv(A*Vf(:,:,jt-jk)*A'+Q)
                    As = V_t_jk @ A.T @ np.linalg.pinv(A @ V_t_jk @ A.T + Q)
                    # MATLAB: Plag{jk+1}(:,:,jt) = As*Plag{jk}(:,:,jt)
                    # Plag[jk-1] is previous lag (jk-1 in 0-based = jk in 1-based)
                    Plag_jk[jt] = As @ Plag[jk - 1][jt]
                except (np.linalg.LinAlgError, ValueError):
                    # Fallback if inversion fails
                    Plag_jk[jt] = Plag[jk - 1][jt]
            Plag.append(Plag_jk)
    
    # Transform factors to observation space
    # x_sm = Z * C' (standardized)
    x_sm = Zsmooth[1:, :] @ C.T  # T x N (skip initial state)
    
    # Unstandardize: X_sm = x_sm * Wx + Mx
    X_sm = x_sm * Wx + Mx  # T x N
    
    return {
        'Plag': Plag,
        'P': Vsmooth[1:, :, :],  # T x r x r (skip initial state)
        'X_sm': X_sm,  # T x N
        'F': Zsmooth[1:, :],  # T x r (smoothed factors, skip initial state)
        'Z': Zsmooth,  # (T+1) x r (includes initial state)
        'V': Vsmooth,  # (T+1) x r x r (includes initial state)
    }


@dataclass
class BacktestResult:
    """Result from backtest evaluation of nowcasting model.
    
    This dataclass contains all information from a pseudo real-time backtest,
    including nowcasts at different view dates, news decomposition between steps,
    and evaluation metrics.
    """
    target_series: str
    target_date: datetime
    backward_steps: int
    higher_freq: bool
    backward_freq: str
    view_list: List  # List[DataView] - avoiding circular import
    nowcast_results: List["NowcastResult"]
    news_results: List[Optional["NewsDecompResult"]]
    actual_values: np.ndarray
    errors: np.ndarray
    mae_per_step: np.ndarray
    mse_per_step: np.ndarray
    rmse_per_step: np.ndarray
    overall_mae: Optional[float]
    overall_rmse: Optional[float]
    overall_mse: Optional[float]
    failed_steps: List[int]
    
    def plot(self, save_path: Optional[str] = None, show: bool = True):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Nowcast values vs actual
            ax1 = axes[0]
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax1.plot(view_dates, nowcast_values, 'o-', label='Nowcast', color='blue')
            if not np.all(np.isnan(self.actual_values)):
                ax1.axhline(y=self.actual_values[0], color='red', linestyle='--', label='Actual')
            ax1.set_xlabel('View Date')
            ax1.set_ylabel('Value')
            ax1.set_title(f'Backtest Results: {self.target_series} at {self.target_date}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Error metrics
            ax2 = axes[1]
            steps = range(self.backward_steps)
            ax2.plot(steps, self.rmse_per_step, 'o-', label='RMSE', color='green')
            ax2.set_xlabel('Backward Step')
            ax2.set_ylabel('Error')
            ax2.set_title('Error Metrics per Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    def plot_trajectory(self, save_path: Optional[str] = None, show: bool = True):
        """Plot nowcast trajectory over backward steps."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax.plot(view_dates, nowcast_values, 'o-', label='Nowcast Trajectory', color='blue', linewidth=2, markersize=8)
            
            if not np.all(np.isnan(self.actual_values)):
                ax.axhline(y=self.actual_values[0], color='red', linestyle='--', linewidth=2, label='Actual')
            
            ax.set_xlabel('View Date', fontsize=12)
            ax.set_ylabel('Nowcast Value', fontsize=12)
            ax.set_title(f'Nowcast Trajectory: {self.target_series} at {self.target_date}', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


# ============================================================================
# DataView class for pseudo real-time data slices
# ============================================================================

@dataclass
class DataView:
    """Lightweight descriptor for pseudo real-time data slices.
    
    This class provides a convenient way to create and manage time-specific
    views of data for nowcasting and forecasting. It encapsulates the data
    and configuration needed to create a view at a specific point in time.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    Time : TimeIndex or array-like
        Time index for the data
    Z : np.ndarray, optional
        Original untransformed data (T x N)
    config : DFMConfig, optional
        Model configuration with series release date information
    view_date : datetime or str, optional
        Date for which to create the view. If None, uses latest available date.
    description : str, optional
        Optional description of this data view
    X_frame : pd.DataFrame, optional
        Pandas DataFrame representation of X (for efficient masking)
    """
    X: np.ndarray
    Time: Union[TimeIndex, Any]
    Z: Optional[np.ndarray]
    config: Optional[DFMConfig]
    view_date: Optional[Union[datetime, str]] = None
    description: Optional[str] = None
    X_frame: Optional[pd.DataFrame] = None
    
    def materialize(self) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
        """Return the masked arrays for this data view.
        
        This method creates a time-specific view of the data by masking
        observations that are not yet available at the view_date based on
        release date information in the config.
        
        Returns
        -------
        X_view : np.ndarray
            Masked data matrix (T x N) with NaN for unavailable observations
        Time : TimeIndex or array-like
            Time index (unchanged)
        Z_view : np.ndarray or None
            Masked original data (T x N) or None if Z was not provided
        """
        return create_data_view(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=self.view_date,
            X_frame=self.X_frame
        )
    
    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        Time: Union[TimeIndex, Any],
        Z: Optional[np.ndarray],
        config: Optional[DFMConfig],
        view_date: Optional[Union[datetime, str]] = None,
        description: Optional[str] = None,
        X_frame: Optional[pd.DataFrame] = None
    ) -> 'DataView':
        """Create DataView from arrays.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N)
        Time : TimeIndex or array-like
            Time index for the data
        Z : np.ndarray, optional
            Original untransformed data
        config : DFMConfig, optional
            Model configuration
        view_date : datetime or str, optional
            View date
        description : str, optional
            Description of this view
        X_frame : pd.DataFrame, optional
            Pandas DataFrame representation
        
        Returns
        -------
        DataView
            New DataView instance
        """
        return cls(
            X=X,
            Time=Time,
            Z=Z,
            config=config,
            view_date=view_date,
            description=description,
            X_frame=X_frame
        )
    
    def with_view_date(self, view_date: Union[datetime, str]) -> 'DataView':
        """Return a shallow copy with a different view date.
        
        Parameters
        ----------
        view_date : datetime or str
            New view date
        
        Returns
        -------
        DataView
            New DataView instance with updated view_date
        """
        return DataView(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=view_date,
            description=self.description,
            X_frame=self.X_frame
        )


# ============================================================================
# Sktime transformers and splitters for nowcasting operations
# ============================================================================

if HAS_SKTIME:
    class PublicationLagMasker(BaseTransformer):
        """Transformer that masks data based on publication lags.
        
        This transformer applies publication lag logic to create data views at
        specific dates, using SeriesConfig.release_date to determine availability.
        It leverages the existing create_data_view() function internally.
        
        Parameters
        ----------
        config : DFMConfig
            Model configuration containing series release date information
        view_date : datetime or str
            Date when data snapshot is taken (data available at this date)
        time_index : TimeIndex or array-like, optional
            Time index for the data. If None, inferred from input data.
        
        Examples
        --------
        >>> from src.nowcast.utils import PublicationLagMasker
        >>> from datetime import datetime
        >>> 
        >>> masker = PublicationLagMasker(
        ...     config=model.config,
        ...     view_date=datetime(2024, 1, 15)
        ... )
        >>> 
        >>> # Fit and transform
        >>> X_masked = masker.fit_transform(X)
        """
        
        _tags = {
            "scitype:transform-input": "Series",
            "scitype:transform-output": "Series",
            "scitype:instancewise": False,
            "X_inner_mtype": "pd.DataFrame",
            "y_inner_mtype": "pd.DataFrame",
            "univariate-only": False,
            "requires-y": False,
            "enforce_index_type": None,
            "fit_is_empty": True,
            "transform-returns-same-time-index": True,
            "skip-inverse-transform": True,  # Masking is not invertible
        }
        
        def __init__(
            self,
            config: DFMConfig,
            view_date: Union[datetime, str],
            time_index: Optional[Union[TimeIndex, list, np.ndarray]] = None
        ):
            super().__init__()
            
            self.config = config
            self.view_date = view_date
            self.time_index = time_index
            self._fitted_time_index = None
        
        def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
            """Fit the transformer (no-op, but stores time index)."""
            # Store time index for transform
            if self.time_index is None:
                # Infer from X index
                if isinstance(X.index, pd.DatetimeIndex):
                    self._fitted_time_index = [t.to_pydatetime() for t in X.index]
                else:
                    # Try to parse index
                    self._fitted_time_index = [parse_timestamp(str(t)) for t in X.index]
            else:
                # Use provided time index
                if isinstance(self.time_index, TimeIndex):
                    self._fitted_time_index = [to_python_datetime(t) for t in self.time_index]
                else:
                    self._fitted_time_index = list(self.time_index)
            
            return self
        
        def _transform(
            self,
            X: pd.DataFrame,
            y: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
            """Transform data by applying publication lag masking."""
            # Parse view_date if string
            if isinstance(self.view_date, str):
                view_date = parse_timestamp(self.view_date)
            else:
                view_date = self.view_date
            
            # Convert X to numpy
            X_array = X.values
            
            # Get time index
            if self._fitted_time_index is not None:
                time_index = self._fitted_time_index
            elif isinstance(X.index, pd.DatetimeIndex):
                time_index = [t.to_pydatetime() for t in X.index]
            else:
                time_index = [parse_timestamp(str(t)) for t in X.index]
            
            # Convert to TimeIndex if needed
            time_index_obj = TimeIndex(time_index)
            
            # Use create_data_view to get masked data
            X_view, Time_view, _ = create_data_view(
                X=X_array,
                Time=time_index_obj,
                config=self.config,
                view_date=view_date,
                X_frame=X
            )
            
            # Convert back to DataFrame with same index and columns
            X_masked = pd.DataFrame(
                X_view,
                index=X.index,
                columns=X.columns
            )
            
            return X_masked
        
        def _inverse_transform(
            self,
            X: pd.DataFrame,
            y: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
            """Inverse transform (returns original data)."""
            return X


    class NewsDecompositionTransformer(BaseTransformer):
        """Transformer that computes news decomposition between two data views."""
        
        _tags = {
            "scitype:transform-input": "Series",
            "scitype:transform-output": "Series",
            "scitype:instancewise": False,
            "X_inner_mtype": "pd.DataFrame",
            "y_inner_mtype": "pd.DataFrame",
            "univariate-only": False,
            "requires-y": True,  # Needs both old and new views
            "enforce_index_type": None,
            "fit_is_empty": True,
            "transform-returns-same-time-index": False,
            "skip-inverse-transform": True,
        }
        
        def __init__(
            self,
            nowcast_manager: Any,
            target_series: str,
            target_period: Union[datetime, str],
            view_date_old: Union[datetime, str],
            view_date_new: Union[datetime, str]
        ):
            super().__init__()
            
            self.nowcast_manager = nowcast_manager
            self.target_series = target_series
            self.target_period = target_period
            self.view_date_old = view_date_old
            self.view_date_new = view_date_new
        
        def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
            """Fit the transformer (no-op)."""
            return self
        
        def _transform(
            self,
            X: pd.DataFrame,
            y: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
            """Transform by computing news decomposition."""
            # Get news decomposition
            news_result = self.nowcast_manager.decompose(
                self.target_series,
                self.target_period,
                self.view_date_old,
                self.view_date_new,
                return_dict=False
            )
            
            # Convert to DataFrame for sktime compatibility
            if isinstance(news_result, NewsDecompResult):
                # Create DataFrame with news decomposition summary
                result_df = pd.DataFrame({
                    'y_old': [news_result.y_old],
                    'y_new': [news_result.y_new],
                    'change': [news_result.change],
                    'total_news': [news_result.change],  # Alias
                })
                
                # Add top contributors as additional columns
                if news_result.top_contributors:
                    for i, (series_id, impact) in enumerate(news_result.top_contributors[:5]):
                        result_df[f'top_contributor_{i+1}'] = [series_id]
                        result_df[f'top_impact_{i+1}'] = [impact]
                
                return result_df
            else:
                # Fallback if dict format
                result_df = pd.DataFrame([news_result])
                return result_df


    class NowcastingSplitter(BaseSplitter):
        """Custom splitter for nowcasting backtesting with publication lags."""
        
        def __init__(
            self,
            target_periods: List[datetime],
            backward_steps: int,
            config: DFMConfig,
            time_index: Union[TimeIndex, List, np.ndarray],
            higher_freq: bool = False,
            clock: Optional[str] = None
        ):
            super().__init__()
            
            self.target_periods = target_periods
            self.backward_steps = backward_steps
            self.config = config
            self.higher_freq = higher_freq
            
            # Convert time_index to list of datetimes
            if isinstance(time_index, TimeIndex):
                self.time_list = [to_python_datetime(t) for t in time_index]
            else:
                self.time_list = []
                for t in time_index:
                    if isinstance(t, datetime):
                        self.time_list.append(t)
                    elif isinstance(t, pd.Timestamp):
                        self.time_list.append(t.to_pydatetime())
                    else:
                        self.time_list.append(parse_timestamp(t))
            
            # Get clock frequency
            if clock is None:
                self.clock = get_clock_frequency(config, 'm')
            else:
                self.clock = clock
            
            # Get backward frequency
            if higher_freq:
                self.backward_freq = get_higher_frequency(self.clock)
                if self.backward_freq is None:
                    # No higher frequency available, use clock
                    self.backward_freq = self.clock
            else:
                self.backward_freq = self.clock
            
            # Pre-compute all splits
            self._splits: List[Tuple[np.ndarray, np.ndarray]] = []
            self._view_dates: List[datetime] = []
            self._target_dates: List[datetime] = []
            self._compute_splits()
        
        def _compute_splits(self):
            """Pre-compute all train/test splits."""
            self._splits = []
            self._view_dates = []
            self._target_dates = []
            
            for target_period in self.target_periods:
                # Iterate through backward steps
                for step in range(self.backward_steps, -1, -1):
                    # Calculate view date (when data snapshot is taken)
                    view_date = calc_backward_date(
                        target_period,
                        step,
                        self.backward_freq
                    )
                    
                    # Get available data mask at view_date
                    available_mask = self._get_available_mask(view_date)
                    
                    # Train indices: all available data up to view_date
                    train_idx = np.where(available_mask)[0]
                    
                    # Test index: target period (backward-looking nowcast)
                    test_idx = self._find_time_index(target_period)
                    
                    if test_idx is not None and len(train_idx) > 0:
                        self._splits.append((train_idx, np.array([test_idx])))
                        self._view_dates.append(view_date)
                        self._target_dates.append(target_period)
        
        def _get_available_mask(self, view_date: datetime) -> np.ndarray:
            """Get mask for data available at view_date based on publication lags."""
            if self.config is None or not hasattr(self.config, 'series') or not self.config.series:
                # No config, assume all data available
                return np.ones(len(self.time_list), dtype=bool)
            
            # For each time period, check if data would be available at view_date
            available_mask = np.zeros(len(self.time_list), dtype=bool)
            
            for t_idx, period in enumerate(self.time_list):
                # Check each series to see if it would be available
                period_available = False
                
                for series_cfg in self.config.series:
                    release_offset = getattr(series_cfg, 'release_date', None)
                    if release_offset is None:
                        # No release date specified, assume available
                        period_available = True
                        break
                    
                    # Calculate when this series would be released for this period
                    release_date = calculate_release_date(release_offset, period)
                    
                    # Data is available if view_date >= release_date
                    if view_date >= release_date:
                        period_available = True
                        break
                
                available_mask[t_idx] = period_available
            
            return available_mask
        
        def _find_time_index(self, target_date: datetime) -> Optional[int]:
            """Find index of target_date in time_list."""
            # Try exact match first
            if target_date in self.time_list:
                return self.time_list.index(target_date)
            
            # Try to find closest match (within same period)
            for i, t in enumerate(self.time_list):
                if (t.year == target_date.year and 
                    t.month == target_date.month):
                    return i
            
            return None
        
        def split(self, y: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            """Generate train/test splits for nowcasting backtest."""
            # Return pre-computed splits
            for train_idx, test_idx in self._splits:
                yield train_idx, test_idx
        
        def get_n_splits(self, y: Optional[pd.DataFrame] = None) -> int:
            """Return the number of splits."""
            return len(self._splits)
        
        def get_split_params(self, split_idx: int) -> dict:
            """Get parameters for a specific split."""
            if split_idx >= len(self._splits):
                raise IndexError(f"Split index {split_idx} out of range [0, {len(self._splits)})")
            
            return {
                'view_date': self._view_dates[split_idx],
                'target_date': self._target_dates[split_idx]
            }


    class NowcastForecaster(BaseForecaster):
        """sktime-compatible forecaster wrapper for nowcasting."""
        
        _tags = {
            "requires-fh-in-fit": False,
            "handles-missing-data": True,
            "y_inner_mtype": "pd.DataFrame",
            "X_inner_mtype": "pd.DataFrame",
            "scitype:y": "both",
        }
        
        def __init__(
            self,
            nowcast_manager: Any,
            target_series: str,
            target_period: Union[datetime, str]
        ):
            super().__init__()
            
            self.nowcast_manager = nowcast_manager
            self.target_series = target_series
            self.target_period = target_period
            self._view_date = None
            self._is_fitted = False
        
        def _fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None, fh=None):
            """Fit the forecaster (stores view_date from metadata)."""
            # Extract view_date from y metadata or index
            if hasattr(y, 'attrs') and 'view_date' in y.attrs:
                self._view_date = y.attrs['view_date']
            elif hasattr(y, 'view_date'):
                self._view_date = getattr(y, 'view_date')
            else:
                # Use latest date in index as view_date
                if isinstance(y.index, pd.DatetimeIndex):
                    self._view_date = y.index[-1].to_pydatetime()
                else:
                    # Try to parse
                    self._view_date = parse_timestamp(str(y.index[-1]))
            
            self._is_fitted = True
            return self
        
        def _predict(self, fh, X: Optional[pd.DataFrame] = None) -> pd.Series:
            """Generate nowcast prediction."""
            if not self._is_fitted or self._view_date is None:
                raise ValueError(
                    "Forecaster must be fitted before prediction. Call fit() first."
                )
            
            # Get target_period from X metadata if provided, otherwise use stored
            target_period = self.target_period
            if X is not None:
                if hasattr(X, 'attrs') and 'target_period' in X.attrs:
                    target_period = X.attrs['target_period']
                elif hasattr(X, 'target_period'):
                    target_period = getattr(X, 'target_period')
            
            # Use nowcast manager to compute nowcast
            nowcast_value = self.nowcast_manager(
                self.target_series,
                view_date=self._view_date,
                target_period=target_period
            )
            
            # Convert target_period to datetime if string
            if isinstance(target_period, str):
                clock = get_clock_frequency(self.nowcast_manager.model.config, 'm')
                target_date = parse_period_string(target_period, clock)
            else:
                target_date = target_period
            
            # Return as Series with target_period as index
            if isinstance(target_date, str):
                target_date = parse_timestamp(target_date)
            # Ensure target_date is datetime
            if not isinstance(target_date, datetime):
                target_date = to_python_datetime(target_date)
            # Create DatetimeIndex and return Series
            index = pd.DatetimeIndex([target_date])
            return pd.Series([nowcast_value], index=index, name=self.target_series)
        
        def set_view_date(self, view_date: Union[datetime, str]):
            """Set view date for prediction."""
            if isinstance(view_date, str):
                self._view_date = parse_timestamp(view_date)
            else:
                self._view_date = view_date
else:
    # Placeholder classes when sktime is not available
    PublicationLagMasker = None
    NewsDecompositionTransformer = None
    NowcastingSplitter = None
    NowcastForecaster = None

