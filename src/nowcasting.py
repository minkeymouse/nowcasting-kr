"""Nowcasting module - current period estimation with incomplete data.

This module provides NOWCASTING functionality only (not forecasting).
Forecasting is handled separately via train.py compare (see run_forecast.sh).

NOWCAST vs FORECAST:
- NOWCAST: Estimate current period using incomplete data (publication lag considered)
- FORECAST: Predict future periods using complete historical data

This module combines:
- Nowcasting functionality (current period estimation)
- News decomposition (forecast update attribution)
- Backtesting for nowcasting (pseudo real-time evaluation)
- sktime integration for nowcasting splits and evaluation

Helper functions for nowcasting operations:
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
        'revision_impact': float(total_impact),
        'release_impact': 0.0
    }


# ============================================================================
# News decomposition and backtest classes (merged from helpers.py)
# ============================================================================

from dataclasses import dataclass
from typing import TYPE_CHECKING

# NowcastResult is defined in this file below

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





# ============================================================================
# Nowcast Class
# ============================================================================

"""Nowcasting and news decomposition for DFM models.

This module implements nowcasting functionality and news decomposition framework for 
understanding how new data releases affect nowcasts. The "news" is defined as the 
difference between the new data release and the model's previous forecast, decomposed
into contributions from each data series.

The module provides:
- Nowcasting: Generate current-period estimates before official data release
- News decomposition: Forecast updates when new data arrives
- Attribution: Forecast changes attributed to specific data series
- Understanding: Which data releases are most informative

This is essential for nowcasting applications where policymakers need to
understand the drivers of forecast revisions.
"""

import numpy as np
from scipy.linalg import pinv, inv
from typing import Tuple, Optional, Dict, Union, List, Any, Callable
from datetime import datetime, timedelta
import warnings
import logging
from dfm_python.logger import get_logger
import pandas as pd
import time

from dfm_python.config import DFMConfig
from dfm_python.config.results import DFMResult
from dfm_python.utils.time import (
    TimeIndex,
    parse_timestamp,
    datetime_range,
    days_in_month,
    clock_to_datetime_freq,
    get_next_period_end,
    find_time_index,
    parse_period_string,
    get_latest_time,
    convert_to_timestamp,
    to_python_datetime,
    calculate_rmse,
    calculate_mae,
)
from dfm_python.utils.helpers import (
    safe_get_attr,
    get_series_ids,
    get_series_names,
    find_series_index,
    get_series_id,
    get_frequencies,
    get_clock_frequency,
)
from dataclasses import dataclass
from dfm_python.config.results import para_const

# Set up logger
_logger = get_logger(__name__)

DEFAULT_FALLBACK_DATE: str = '2017-01-01'


# ============================================================================
# Data Classes for Backtest Results
# ============================================================================

@dataclass
class NowcastResult:
    """Result from a single nowcast calculation."""
    target_series: str
    target_period: datetime
    view_date: datetime
    nowcast_value: float
    confidence_interval: Optional[Tuple[float, float]] = None  # (lower, upper)
    factors_at_view: Optional[np.ndarray] = None  # Factor values at view_date
    dfm_result: Optional[DFMResult] = None  # Full DFM result for this view
    data_availability: Optional[Dict[str, int]] = None  # n_available, n_missing




# ============================================================================
# Helper Functions
# ============================================================================
from dfm_python.utils.helpers import (
    calc_backward_date,
    get_forecast_horizon,
    check_config,
    extract_news,
)


class Nowcast:
    """Nowcasting and news decomposition manager.
    
    This class provides a unified interface for nowcasting operations,
    news decomposition, and forecast updates. It takes a DFM model instance
    and provides methods for calculating nowcasts and decomposing forecast

# ============================================================================
# Nowcast Class
# ============================================================================

    ... )
    >>> print(f"Change: {news.change:.2f}")
    >>> print(f"Top contributors: {news.top_contributors}")
    """
    
    def __init__(self, model: Any, data_module: Optional[Any] = None):  # type: ignore
        """Initialize Nowcast manager.
        
        Parameters
        ----------
        model : DFM
            Trained DFM model instance (validation is done in DFM.nowcast property)
        data_module : DFMDataModule, optional
            DataModule containing data. If None, will try to get from model._data_module.
        
        Note
        ----
        Validation is performed in DFM.nowcast property before creating instance.
        This class uses DataModule to access data instead of model.data.
        """
        self.model = model
        # Get DataModule from model if not provided
        if data_module is None:
            data_module = getattr(model, '_data_module', None)
        if data_module is None:
            raise ValueError("DataModule must be provided either directly or via model._data_module")
        self.data_module = data_module
        # Caching for performance
        self._para_const_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._data_view_cache: Dict[str, Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]] = {}
    
    def get_data_view(self, view_date: Union[datetime, str]) -> Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]:
        """Get data view at specific date (with caching).
        
        Parameters
        ----------
        view_date : datetime or str
            View date for data availability
            
        Returns
        -------
        Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]
            (X_view, Time_view, Z_view) - data available at view_date
        """
        view_date_str = str(view_date)
        if view_date_str in self._data_view_cache:
            return self._data_view_cache[view_date_str]
        
        start = time.perf_counter()
        # Get raw data from DataModule
        raw_data = self.data_module.data
        time_index = self.data_module.time_index
        
        # Convert to numpy if needed
        if hasattr(raw_data, 'to_numpy'):
            X = raw_data.to_numpy()
        else:
            X = np.asarray(raw_data)
        
        X_view, Time_view, Z_view = create_data_view(
            X=X,
            Time=time_index,
            Z=X,  # Use same data for Z (original data)
            config=self.model.config,
            view_date=view_date,
            X_frame=raw_data if isinstance(raw_data, pd.DataFrame) else None
        )
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "create_data_view[%s] completed in %.3fs",
                view_date_str,
                time.perf_counter() - start
        )
        self._data_view_cache[view_date_str] = (X_view, Time_view, Z_view)
        return X_view, Time_view, Z_view
    
    def _get_kalman_result(self, cache_key: str, X_view: np.ndarray) -> Dict[str, Any]:
        """Cache-aware wrapper around para_const for profiling."""
        key = (cache_key, X_view.shape[0])
        if key in self._para_const_cache:
            return self._para_const_cache[key]
        start = time.perf_counter()
        self._para_const_cache[key] = para_const(X_view, self.model.result, 0)
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "para_const[%s] completed in %.3fs",
                cache_key,
                time.perf_counter() - start
            )
        return self._para_const_cache[key]
    
    def _parse_target_date(
        self,
        target_date: Union[datetime, str],
        target_series: Optional[str] = None
    ) -> datetime:
        """Parse target date from string or datetime.
        
        Parameters
        ----------
        target_date : datetime or str
            Target date to parse
        target_series : str, optional
            Target series ID (used to determine frequency for string parsing)
            
        Returns
        -------
        datetime
            Parsed target date
        """
        if isinstance(target_date, datetime):
            return target_date
        elif isinstance(target_date, str):
            clock = get_clock_frequency(self.model.config, 'm')
            if target_series is not None:
                frequencies = get_frequencies(self.model.config)
                i_series = find_series_index(self.model.config, target_series)
                freq = frequencies[i_series] if i_series < len(frequencies) else clock
            else:
                freq = clock
            return parse_period_string(target_date, freq)
        else:
            return parse_timestamp(target_date)
    
    def _extract_float(self, value: Union[float, np.ndarray]) -> float:
        """Extract float value from forecast core output (handles scalar and array returns).
        
        Parameters
        ----------
        value : float or np.ndarray
            Value from _forecast_core (can be scalar or array)
            
        Returns
        -------
        float
            Extracted float value
        """
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(value[0])
        else:
            return float(value)
    
    def _prepare_target(
        self,
        target_series: str,
        target_period: Optional[Union[datetime, str]],
        view_date: datetime,
        Time_view: TimeIndex
    ) -> Tuple[datetime, int]:
        """Prepare target period and time index for forecast operations.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        target_period : datetime or str, optional
            Target period. If None, uses latest available.
        view_date : datetime
            View date for data availability
        Time_view : TimeIndex
            Time index for the view
            
        Returns
        -------
        Tuple[datetime, int]
            (target_period, t_fcst) where t_fcst is the time index
        """
        # Determine target period
        if target_period is None:
            # Handle empty Time_view gracefully
            if len(Time_view) == 0:
                # If Time_view is empty, we can't determine target_period
                # This should be handled by __call__ method, but for now use view_date as fallback
                _logger.warning(f"Time_view is empty, using view_date {view_date} as target_period fallback")
                target_period = view_date if isinstance(view_date, datetime) else parse_timestamp(str(view_date))
                t_fcst = 0  # Placeholder - __call__ will handle extension
                return target_period, t_fcst
            try:
                target_period = get_latest_time(Time_view)
            except (ValueError, IndexError, AttributeError) as e:
                # If get_latest_time fails, use view_date as fallback
                _logger.warning(f"Could not get latest time from Time_view: {e}, using view_date {view_date} as fallback")
                target_period = view_date if isinstance(view_date, datetime) else parse_timestamp(str(view_date))
                t_fcst = len(Time_view) - 1 if len(Time_view) > 0 else 0
                return target_period, t_fcst
        else:
            target_period = self._parse_target_date(target_period, target_series)
        
        # Handle empty Time_view before find_time_index
        if len(Time_view) == 0:
            _logger.warning(f"Time_view is empty, using target_period {target_period} directly (will be extended in __call__)")
            t_fcst = 0  # Placeholder - __call__ will handle extension
            return target_period, t_fcst
        
        # Find time index for target period
        t_fcst = find_time_index(Time_view, target_period)
        if t_fcst is None:
            # If exact match not found, try to find closest available date at or before target_period
            # This handles cases where data is weekly/monthly but target_period is month-end
            target_period_dt = target_period if isinstance(target_period, datetime) else parse_timestamp(str(target_period))
            closest_idx = None
            closest_date = None
            min_diff = None
            
            for i, t in enumerate(Time_view):
                if not isinstance(t, datetime):
                    try:
                        if isinstance(t, pd.Timestamp):
                            t = t.to_pydatetime()
                        elif hasattr(t, 'to_pydatetime'):
                            t = t.to_pydatetime()
                        elif hasattr(t, 'to_python'):
                            t = t.to_python()
                        else:
                            t = parse_timestamp(t)
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                if isinstance(t, datetime):
                    # Find closest date at or before target_period
                    if t <= target_period_dt:
                        diff = (target_period_dt - t).total_seconds()
                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                            closest_idx = i
                            closest_date = t
            
            if closest_idx is not None:
                # Use closest available date instead of exact target_period
                target_period = closest_date
                t_fcst = closest_idx
                _logger.warning(
                    f"Target period {target_period_dt} not found in Time index. "
                    f"Using closest available date: {closest_date}"
                )
            else:
                # If no date at or before target_period found, target_period is in the future
                # For nowcasting, we need to extend the time index to include the target_period
                # Calculate t_fcst as an offset from the latest available date
                if len(Time_view) > 0:
                    # Use the last index in Time_view as the base (most recent available date)
                    # This handles cases where get_latest_time might not match exactly
                    latest_idx = len(Time_view) - 1
                    latest_time = Time_view[latest_idx]
                    
                    # Convert latest_time to datetime if needed
                    if not isinstance(latest_time, datetime):
                        try:
                            if isinstance(latest_time, pd.Timestamp):
                                latest_time = latest_time.to_pydatetime()
                            elif hasattr(latest_time, 'to_pydatetime'):
                                latest_time = latest_time.to_pydatetime()
                            elif hasattr(latest_time, 'to_python'):
                                latest_time = latest_time.to_python()
                            else:
                                latest_time = parse_timestamp(str(latest_time))
                        except (ValueError, TypeError, AttributeError):
                            # Fallback: use last element directly
                            pass
                    
                    # Calculate number of periods between latest_time and target_period
                    # This will be used to extend the data in __call__
                    # For now, use the latest index and let __call__ extend the data
                    # Keep original target_period for the nowcast calculation
                    t_fcst = latest_idx
                    # Calculate how many periods ahead we need to forecast
                    # This is a rough estimate - the actual extension happens in __call__
                    _logger.warning(
                        f"Target period {target_period_dt} is in the future relative to Time index. "
                        f"Using latest available date {latest_time} (index {latest_idx}) as base. "
                        f"Time_view range: {Time_view[0] if len(Time_view) > 0 else 'empty'} to "
                        f"{Time_view[-1] if len(Time_view) > 0 else 'empty'}. "
                        f"Data will be extended in __call__ to accommodate future target_period."
                    )
                else:
                    # CRITICAL FIX: Instead of raising an error, use a fallback approach
                    # This handles cases where Time_view is empty or has no valid dates
                    # For nowcasting, we can still proceed by using the target_period directly
                    # and letting __call__ handle the extension
                    _logger.warning(
                        f"Target period {target_period_dt} not found in Time index and Time_view is empty. "
                        f"Using target_period directly and will extend data in __call__."
                    )
                    # Use 0 as a placeholder index - __call__ will handle the extension
                    t_fcst = 0
        
        return target_period, t_fcst
    
    def _create_nowcast_result(
        self,
        target_series: str,
        target_period: datetime,
        view_date: datetime,
        nowcast_value: float,
        X_view: np.ndarray,
        Time_view: TimeIndex
    ) -> NowcastResult:
        """Create NowcastResult with metadata (data availability, factors, etc.).
        
        This is a consolidated helper method to create NowcastResult objects
        with consistent metadata extraction.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        target_period : datetime
            Target period for nowcast
        view_date : datetime
            View date for data availability
        nowcast_value : float
            Calculated nowcast value
        X_view : np.ndarray
            Data view matrix (T x N)
        Time_view : TimeIndex
            Time index for the view
            
        Returns
        -------
        NowcastResult
            NowcastResult with all metadata populated
        """
        # Calculate data availability
        n_total = X_view.size
        n_missing = int(np.sum(np.isnan(X_view)))
        n_available = n_total - n_missing
        data_availability = {
            'n_available': n_available,
            'n_missing': n_missing
        }
        
        # Extract factors and DFM result
        factors_at_view = None
        dfm_result = None
        if self.model.result is not None and hasattr(self.model.result, 'Z'):
            t_view = find_time_index(Time_view, view_date)
            if t_view is not None and t_view < self.model.result.Z.shape[0]:
                factors_at_view = self.model.result.Z[t_view, :].copy()
            dfm_result = self.model.result
        
        return NowcastResult(
            target_series=target_series,
            target_period=target_period,
            view_date=view_date,
            nowcast_value=nowcast_value,
            confidence_interval=None,  # Could be calculated from Kalman filter covariance (C @ Var(Z) @ C.T + R)
            factors_at_view=factors_at_view,
            dfm_result=dfm_result,
            data_availability=data_availability
        )
    
    
    def _forecast_core(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        t_fcst: int,
        v_news: Union[int, np.ndarray, List[int]],
        *,
        Res_old_cache: Optional[Dict[str, Any]] = None,
        Res_new_cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray], np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Core nowcast engine for nowcasting, news decomposition (internal method).
        
        This is the shared computational core used by nowcast() and decompose() methods.
        It performs Kalman filtering/smoothing on data views and calculates nowcast
        values and news contributions.
        
        NOTE: This is for NOWCASTING (current period estimation), not forecasting (future prediction).
        
        - For nowcast: X_old = X_new (same data, computes y_new only)
        - For decompose: X_old != X_new (different views, computes y_old, y_new, and news)
        
        This method is the foundation for all nowcasting operations, ensuring consistent
        calculations across single-view nowcasts and multi-view news decomposition.
        
        Parameters
        ----------
        X_old : np.ndarray
            Old data matrix (T x N) - data available at earlier view_date
        X_new : np.ndarray
            New data matrix (T x N) - data available at later view_date
        t_fcst : int
            Target time index for forecast/nowcast
        v_news : int, np.ndarray, or List[int]
            Target variable index(s) to forecast
            
        Returns
        -------
        Tuple containing 9 elements:
        y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov
            - y_old: Nowcast value using X_old (old data view)
            - y_new: Nowcast value using X_new (new data view, this is the nowcast when X_old = X_new)
            - singlenews: News contributions per series
            - actual, forecast, weight, t_miss, v_miss, innov: Additional decomposition details
            Note: 'forecast' here refers to the model's prediction of new data (for news decomposition),
                  not a future forecast. This is still nowcasting (current period estimation).
        """
        # Input validation
        if not isinstance(X_old, np.ndarray) or X_old.ndim != 2:
            raise ValueError(f"X_old must be a 2D numpy array, got {type(X_old)}")
        if not isinstance(X_new, np.ndarray) or X_new.ndim != 2:
            raise ValueError(f"X_new must be a 2D numpy array, got {type(X_new)}")
        if X_old.shape[1] != X_new.shape[1]:
            raise ValueError(f"X_old and X_new must have same number of series. Got {X_old.shape[1]} and {X_new.shape[1]}")
        if not isinstance(t_fcst, (int, np.integer)) or t_fcst < 0:
            raise ValueError(f"t_fcst must be a non-negative integer, got {t_fcst}")
        if t_fcst >= X_new.shape[0]:
            raise ValueError(f"t_fcst ({t_fcst}) must be less than number of time periods ({X_new.shape[0]})")
        
        # Normalize v_news to array
        v_news_arr = np.atleast_1d(v_news)
        is_scalar = isinstance(v_news, (int, np.integer)) or (isinstance(v_news, np.ndarray) and v_news.ndim == 0)
        n_targets = len(v_news_arr)
        
        # Validate v_news indices
        if np.any(v_news_arr < 0) or np.any(v_news_arr >= X_new.shape[1]):
            raise ValueError(f"v_news indices must be in range [0, {X_new.shape[1]}), got {v_news_arr}")
        
        r = self.model.result.C.shape[1]
        _, N = X_new.shape
        
        def _resolve_res(cache: Optional[Dict[str, Any]], X_mat: np.ndarray, lag: int = 0) -> Dict[str, Any]:
            if cache is not None and lag == 0:
                return cache
            return para_const(X_mat, self.model.result, lag)
        
        # Check if targets are already observed
        targets_observed = np.array([not np.isnan(X_new[t_fcst, v]) for v in v_news_arr])
        
        if np.all(targets_observed):
            # NO FORECAST CASE: Already values for all target variables at time t_fcst
            Res_old = _resolve_res(Res_old_cache, X_old)
            
            # Initialize output arrays
            if is_scalar:
                singlenews = np.zeros(N)
                singlenews[v_news_arr[0]] = X_new[t_fcst, v_news_arr[0]] - Res_old['X_sm'][t_fcst, v_news_arr[0]]
                y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                y_new = X_new[t_fcst, v_news_arr[0]]
            else:
                singlenews = np.zeros((N, n_targets))
                for i, v in enumerate(v_news_arr):
                    singlenews[v, i] = X_new[t_fcst, v] - Res_old['X_sm'][t_fcst, v]
                y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                y_new = np.array([X_new[t_fcst, v] for v in v_news_arr])
            
            actual = np.array([])
            forecast = np.array([])
            weight = np.array([])
            t_miss = np.array([])
            v_miss = np.array([])
            innov = np.array([])
        else:
            # FORECAST CASE
            Mx = self.model.result.Mx
            Wx = self.model.result.Wx
            
            # Calculate indicators for missing values
            miss_old = np.isnan(X_old)
            miss_new = np.isnan(X_new)
            
            # Indicator for missing: 1 = new data available, -1 = old data available but not new
            i_miss = miss_old.astype(int) - miss_new.astype(int)
            
            # Time/variable indices where new data is available
            t_miss, v_miss = np.where(i_miss == 1)
            t_miss = t_miss.flatten()
            v_miss = v_miss.flatten()
            
            if len(v_miss) == 0:
                # NO NEW INFORMATION
                Res_old = _resolve_res(Res_old_cache, X_old)
                Res_new = _resolve_res(Res_new_cache, X_new)
                
                if is_scalar:
                    y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                    y_new = y_old
                    singlenews = np.array([])
                else:
                    y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                    y_new = y_old
                    singlenews = np.array([]).reshape(0, n_targets)
                
                actual = np.array([])
                forecast = np.array([])
                weight = np.array([])
                t_miss = np.array([])
                v_miss = np.array([])
                innov = np.array([])
            else:
                # NEW INFORMATION
                # Difference between forecast time and new data time
                lag = t_fcst - t_miss
                
                # Biggest time interval
                lag_abs_max = float(np.max(np.abs(lag)))
                lag_range = float(np.max(lag) - np.min(lag))
                k = int(max(lag_abs_max, lag_range))
                
                C = self.model.result.C
                R_cov = self.model.result.R.T
                
                n_news = len(lag)
                
                # Smooth old dataset
                Res_old = _resolve_res(Res_old_cache if k == 0 else None, X_old, k)
                Plag = Res_old['Plag']
                
                # Smooth new dataset
                Res_new = _resolve_res(Res_new_cache, X_new, 0)
                
                # Get nowcasts for all target variables
                if is_scalar:
                    y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                    y_new = Res_new['X_sm'][t_fcst, v_news_arr[0]]
                else:
                    y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                    y_new = np.array([Res_new['X_sm'][t_fcst, v] for v in v_news_arr])
                
                # Calculate projection matrices
                P1 = []
                for i in range(n_news):
                    h = abs(t_fcst - t_miss[i])
                    m = max(t_miss[i], t_fcst)
                    
                    if t_miss[i] > t_fcst:
                        Pp = Plag[h + 1][:, :, m] if h + 1 < len(Plag) else Plag[-1][:, :, m]
                    else:
                        Pp = Plag[h + 1][:, :, m].T if h + 1 < len(Plag) else Plag[-1][:, :, m].T
                    
                    P1.append(Pp @ C[v_miss[i], :r].T)
                
                P1 = np.hstack(P1) if len(P1) > 0 else np.zeros((r, 1))
                
                # Calculate innovations
                innov = np.zeros(n_news)
                for i in range(n_news):
                    X_new_norm = (X_new[t_miss[i], v_miss[i]] - Mx[v_miss[i]]) / Wx[v_miss[i]]
                    X_sm_norm = (Res_old['X_sm'][t_miss[i], v_miss[i]] - Mx[v_miss[i]]) / Wx[v_miss[i]]
                    innov[i] = X_new_norm - X_sm_norm
                
                # Calculate P2 (covariance of innovations)
                P2 = np.zeros((n_news, n_news))
                for i in range(n_news):
                    for j in range(n_news):
                        h = abs(lag[i] - lag[j])
                        m = max(t_miss[i], t_miss[j])
                        
                        if t_miss[j] > t_miss[i]:
                            Pp = Plag[h + 1][:, :, m] if h + 1 < len(Plag) else Plag[-1][:, :, m]
                        else:
                            Pp = Plag[h + 1][:, :, m].T if h + 1 < len(Plag) else Plag[-1][:, :, m].T
                        
                        if v_miss[i] == v_miss[j] and t_miss[i] != t_miss[j]:
                            WW = 0
                        else:
                            WW = R_cov[v_miss[i], v_miss[j]]
                        
                        P2[i, j] = C[v_miss[i], :r] @ Pp @ C[v_miss[j], :r].T + WW
                
                # Calculate weights and news for each target variable
                if n_news > 0 and P2.size > 0:
                    try:
                        P2_inv = inv(P2)
                        # Calculate gain for each target variable
                        if is_scalar:
                            v_idx = v_news_arr[0]
                            gain = (Wx[v_idx] * C[v_idx, :r] @ P1 @ P2_inv).reshape(-1)
                            totnews = Wx[v_idx] * C[v_idx, :r] @ P1 @ P2_inv @ innov
                        else:
                            gain = np.zeros((n_targets, n_news))
                            totnews = np.zeros(n_targets)
                            for idx, v in enumerate(v_news_arr):
                                gain[idx, :] = Wx[v] * C[v, :r] @ P1 @ P2_inv
                                totnews[idx] = Wx[v] * C[v, :r] @ P1 @ P2_inv @ innov
                    except (np.linalg.LinAlgError, ValueError) as e:
                        # If inversion fails, use simpler approach
                        _logger.warning(
                            f"Matrix inversion failed for P2, using fallback values. "
                            f"Error: {type(e).__name__}: {str(e)}"
                        )
                        if is_scalar:
                            gain = np.ones(n_news) * 0.1
                            totnews = np.sum(innov) * 0.1
                        else:
                            gain = np.ones((n_targets, n_news)) * 0.1
                            totnews = np.ones(n_targets) * np.sum(innov) * 0.1
                else:
                    if is_scalar:
                        gain = np.zeros(n_news)
                        totnews = 0
                    else:
                        gain = np.zeros((n_targets, n_news))
                        totnews = np.zeros(n_targets)
                
                # Organize output
                if is_scalar:
                    singlenews = np.full(N, np.nan)
                    actual = np.full(N, np.nan)
                    forecast = np.full(N, np.nan)
                    weight = np.full(N, np.nan)
                    
                    for i in range(n_news):
                        actual[v_miss[i]] = X_new[t_miss[i], v_miss[i]]
                        forecast[v_miss[i]] = Res_old['X_sm'][t_miss[i], v_miss[i]]
                        if i < len(gain):
                            singlenews[v_miss[i]] = gain[i] * innov[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                            weight[v_miss[i]] = gain[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                else:
                    # Multiple targets: singlenews is (N, n_targets)
                    singlenews = np.full((N, n_targets), np.nan)
                    actual = np.full(N, np.nan)
                    forecast = np.full(N, np.nan)
                    weight = np.full((N, n_targets), np.nan)
                    
                    for i in range(n_news):
                        actual[v_miss[i]] = X_new[t_miss[i], v_miss[i]]
                        forecast[v_miss[i]] = Res_old['X_sm'][t_miss[i], v_miss[i]]
                        for idx in range(n_targets):
                            if i < len(gain[idx]):
                                singlenews[v_miss[i], idx] = gain[idx, i] * innov[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                                weight[v_miss[i], idx] = gain[idx, i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                
                # Remove duplicates from v_miss
                v_miss = np.unique(v_miss)
        
        return y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov
    
    def __call__(
        self,
        target_series: str,
        view_date: Optional[Union[datetime, str]] = None,
        target_period: Optional[Union[datetime, str]] = None,
        return_result: bool = False
    ) -> Union[float, NowcastResult]:
        """Calculate nowcast for target series (callable interface).
        
        Parameters
        ----------
        target_series : str
            Target series ID
        view_date : datetime or str, optional
            Data view date. If None, uses latest available.
        target_period : datetime or str, optional
            Target period for nowcast. If None, uses latest.
        return_result : bool, default False
            If True, returns NowcastResult with additional information.
            If False, returns only the nowcast value (float).
            
        Returns
        -------
        float or NowcastResult
            Nowcast value if return_result=False, or NowcastResult if return_result=True
            
        Examples
        --------
        >>> nowcast = Nowcast(model)
        >>> value = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
        >>> # Or get full result
        >>> result = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1', return_result=True)
        >>> print(f"Nowcast: {result.nowcast_value}, Factors: {result.factors_at_view}")
        """
        if view_date is None:
            view_date = get_latest_time(self.model.time)
        elif isinstance(view_date, str):
            view_date = parse_timestamp(view_date)
        
        X_view, Time_view, _ = self.get_data_view(view_date)
        
        # CRITICAL FIX: Detect and handle Inf values in data view before processing
        # Inf values can cause numerical instability in Kalman filter
        if X_view.size > 0 and np.any(np.isinf(X_view)):
            inf_count = np.sum(np.isinf(X_view))
            _logger.warning(f"Data view contains {inf_count} Inf values for target {target_series} at view_date {view_date}, replacing with NaN")
            # Replace Inf with NaN - NaN will be handled by missing data logic
            X_view = np.where(np.isinf(X_view), np.nan, X_view)
        
        # Get series index
        i_series = find_series_index(self.model.config, target_series)
        
        # Get frequency
        frequencies = get_frequencies(self.model.config)
        freq = frequencies[i_series] if i_series < len(frequencies) else 'm'
        
        # Prepare target period and time index
        target_period, t_nowcast = self._prepare_target(
            target_series, target_period, view_date, Time_view
        )
        
        # Get forecast horizon based on clock frequency
        clock = get_clock_frequency(self.model.config, 'm')
        forecast_horizon, _ = get_forecast_horizon(clock, horizon=None)
        
        # Check if target_period is in the future relative to Time_view
        # If so, we need to extend the data and calculate the correct t_nowcast
        T, N = X_view.shape
        if T == 0:
            # CRITICAL FIX: Instead of raising ValueError, return NaN or handle gracefully
            # This allows the calling code to skip this month instead of crashing
            _logger.warning(f"Data view is empty (no data available at view_date {view_date})")
            # Return NaN to signal that nowcast cannot be calculated
            if return_result:
                return NowcastResult(
                    target_series=target_series,
                    target_period=target_period if isinstance(target_period, datetime) else parse_timestamp(str(target_period)),
                    view_date=view_date,
                    nowcast_value=np.nan
                )
            else:
                return np.nan
        
        target_period_dt = target_period if isinstance(target_period, datetime) else parse_timestamp(str(target_period))
        if len(Time_view) == 0:
            # CRITICAL FIX: Instead of raising ValueError, handle gracefully
            # For nowcasting, we can extend the data to accommodate future target_period
            _logger.warning(f"Time index is empty (no time points available at view_date {view_date}), will extend data")
            # Use a default extension - __call__ will handle this
            latest_time = view_date
        else:
            try:
                latest_time = get_latest_time(Time_view)
            except (ValueError, IndexError, AttributeError) as e:
                # CRITICAL FIX: Instead of raising ValueError, use view_date as fallback
                _logger.warning(f"Could not get latest time from Time index: {e}, using view_date {view_date} as fallback")
                latest_time = view_date
        
        if target_period_dt > latest_time:
            # Target period is in the future - need to extend data and calculate correct index
            # Calculate number of periods between latest_time and target_period
            from dateutil.relativedelta import relativedelta
            if clock == 'm':
                # Monthly: calculate months difference
                months_diff = (target_period_dt.year - latest_time.year) * 12 + (target_period_dt.month - latest_time.month)
                periods_ahead = max(months_diff, forecast_horizon)
            elif clock == 'q':
                # Quarterly: calculate quarters difference
                quarters_diff = (target_period_dt.year - latest_time.year) * 4 + ((target_period_dt.month - 1) // 3 - (latest_time.month - 1) // 3)
                periods_ahead = max(quarters_diff, forecast_horizon)
            else:
                # Default: use forecast_horizon
                periods_ahead = forecast_horizon
            
            # Extend data to accommodate future target_period
            X_extended = np.vstack([X_view, np.full((periods_ahead, N), np.nan)])
            # Calculate correct index for target_period in extended data
            t_nowcast = T - 1 + periods_ahead
        elif t_nowcast >= T - forecast_horizon:
            # Need to extend data for forecast horizon
            X_extended = np.vstack([X_view, np.full((forecast_horizon, N), np.nan)])
        else:
            X_extended = X_view
        
        # Use _forecast_core to calculate nowcast
        # For nowcast calculation, we use the same data for old and new (no update)
        try:
            nowcast_cache_key = f"nowcast:{view_date}"
            res_cache = self._get_kalman_result(nowcast_cache_key, X_extended)
            y_new = self._forecast_core(
                X_extended,
                X_extended,
                t_nowcast,
                i_series,
                Res_old_cache=res_cache,
                Res_new_cache=res_cache
            )[1]
            
            # Extract float value from forecast core output
            nowcast_value = self._extract_float(y_new)
            
            # Return simple float if not requested
            if not return_result:
                return nowcast_value
            
            # Return full NowcastResult
            # Use helper method pattern for consistency
            return self._create_nowcast_result(
                target_series=target_series,
                target_period=target_period,
                view_date=view_date,
                nowcast_value=nowcast_value,
                X_view=X_view,
                Time_view=Time_view
            )
            
        except Exception as e:
            _logger.error(f"Nowcast calculation failed: {e}")
            raise
    
    def decompose(
        self,
        target_series: str,
        target_period: Union[datetime, str],
        view_date_old: Union[datetime, str],
        view_date_new: Union[datetime, str],
        return_dict: bool = False
    ) -> Union[NewsDecompResult, Dict[str, Any]]:
        """Decompose nowcast update into news contributions (nowcast update decomposition).
        
        This method analyzes how new data releases affect the nowcast by comparing
        forecasts from two different view dates. It calculates both the old and new
        nowcast values (y_old and y_new) and decomposes the change into contributions
        
        Note: The result includes y_new, which is the nowcast value at view_date_new.
        
        Parameters
        ----------
        target_series : str
            Target series ID to nowcast
        target_period : datetime or str
            Target period for nowcast (e.g., '2024Q1')
        view_date_old : datetime or str
            Older data view date (baseline for comparison)
        view_date_new : datetime or str
            Newer data view date (contains additional data releases)
        return_dict : bool, default False
            If True, returns dictionary format.
            If False, returns NewsDecompResult dataclass.
            
        Returns
        -------
        NewsDecompResult or Dict[str, Any]
            News decomposition result containing:
            - 'y_old': float - nowcast using view_date_old data
            - 'y_new': float - nowcast using view_date_new data (reusable in backtest)
            - 'change': float - forecast update (y_new - y_old)
            - 'singlenews': np.ndarray - news contributions per series
            - 'top_contributors': List[Tuple[str, float]] - top contributors to the update
            - 'actual': np.ndarray - actual values of newly released data
            - 'forecast': np.ndarray - forecasted values for new data
            - 'weight': np.ndarray - weights for news contributions
            - 't_miss': np.ndarray - time indices of new data
            - 'v_miss': np.ndarray - variable indices of new data
            - 'innov': np.ndarray - innovation terms
        """
        # Get data views
        X_old, Time_old, _ = self.get_data_view(view_date_old)
        X_new, Time_new, _ = self.get_data_view(view_date_new)
        
        # Prepare target period and time index
        i_series = find_series_index(self.model.config, target_series)
        target_date, t_fcst = self._prepare_target(
            target_series, target_period, view_date_new, Time_new
        )
        
        # Ensure same time dimension
        T_old = X_old.shape[0]
        T_new = X_new.shape[0]
        if T_new > T_old:
            X_old = np.vstack([X_old, np.full((T_new - T_old, X_old.shape[1]), np.nan)])
        
        # Use consistent cache key format
        cache_key_old = f"decompose:{view_date_old}"
        cache_key_new = f"decompose:{view_date_new}"
        Res_old_cache = self._get_kalman_result(cache_key_old, X_old)
        Res_new_cache = self._get_kalman_result(cache_key_new, X_new)
        
        # Call _forecast_core
        try:
            y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = \
                self._forecast_core(
                    X_old,
                    X_new,
                    t_fcst,
                    i_series,
                    Res_old_cache=Res_old_cache,
                    Res_new_cache=Res_new_cache
                )
            
            # Extract float values consistently
            y_old_float = self._extract_float(y_old)
            y_new_float = self._extract_float(y_new)
            
            # Extract summary
            series_ids = get_series_ids(self.model.config)
            summary = self._extract_news(singlenews, weight, series_ids, top_n=5)
            
            # Create NewsDecompResult
            news_result = NewsDecompResult(
                y_old=y_old_float,
                y_new=y_new_float,
                change=y_new_float - y_old_float,
                singlenews=singlenews,
                top_contributors=summary['top_contributors'],
                actual=actual,
                forecast=forecast,
                weight=weight,
                t_miss=t_miss,
                v_miss=v_miss,
                innov=innov
            )
            
            if return_dict:
                # Return dictionary format
                return {
                    'y_old': news_result.y_old,
                    'y_new': news_result.y_new,
                    'change': news_result.change,
                    'singlenews': news_result.singlenews,
                    'top_contributors': news_result.top_contributors,
                    'actual': news_result.actual,
                    'forecast': news_result.forecast,
                    'weight': news_result.weight,
                    't_miss': news_result.t_miss,




                    'v_miss': news_result.v_miss,
                    'innov': news_result.innov
                }
            
            return news_result
            
        except Exception as e:
            _logger.error(f"News decomposition calculation failed: {e}")
            raise
    
    def _extract_news(
        self,
        singlenews: np.ndarray,
        weight: np.ndarray,
        series_ids: List[str],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Extract news summary from news decomposition results.
        
        Parameters
        ----------
        singlenews : np.ndarray
            Individual news contributions (N,) or (N, n_targets)
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
        """
        # Handle both 1D and 2D arrays
        if singlenews.ndim == 1:
            news_contributions = singlenews
            weights = weight
        else:
            # If 2D, use first target (column 0)
            news_contributions = singlenews[:, 0]
            weights = weight[:, 0] if weight.ndim > 1 else weight
        
        return extract_news(singlenews, weight, series_ids, top_n)
    