"""Data reading utilities and transformation functions for DFM preprocessing.

This module provides functions for reading time series data from files,
transformation functions for time series data, and utilities for creating
transformers from DFM configuration.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

# Set up paths using centralized utility (relative import since we're in src/)
from ..utils.config_parser import setup_paths
setup_paths(include_app=True)

# Import custom exceptions for error handling
from ..utils.config_parser import ValidationError, ConfigError

# ============================================================================
# Sktime Optional Dependency Handling
# ============================================================================

try:
    # sktime 0.40+ uses ColumnEnsembleTransformer instead of ColumnTransformer
    from sktime.transformations.compose import (
        TransformerPipeline,
        ColumnwiseTransformer,
        ColumnEnsembleTransformer
    )
    from sktime.transformations.base import BaseTransformer
    
    from sktime.transformations.series.log import LogTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import StandardScaler
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    ColumnEnsembleTransformer = None
    BaseTransformer = None
    TransformerPipeline = None
    ColumnwiseTransformer = None
    LogTransformer = None
    Differencer = None
    FunctionTransformer = None
    StandardScaler = None


def check_sktime_available():
    """Check if sktime is available and raise ImportError if not.
    
    Raises
    ------
    ImportError
        If sktime is not installed, with helpful installation message.
    """
    # Re-check at runtime in case import failed due to missing components
    try:
        import sktime
        # Check if essential components are available
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.func_transform import FunctionTransformer
        from sktime.transformations.series.difference import Differencer
        # If we get here, sktime is available
        return
    except ImportError as e:
        raise ImportError(
            f"sktime is required for sktime transformers. "
            f"Install it with: pip install sktime. "
            f"Original error: {e}"
        )


# ============================================================================
# Validation Constants
# ============================================================================

VALID_TRANSFORMATION_TYPES = ['lin', 'log', 'chg', 'ch1', 'cha', 'pch', 'pc1', 'pca']
"""Valid transformation types for time series preprocessing."""

VALID_FREQUENCY_CODES = ['m', 'q', 'sa', 'a', 'd', 'w']
"""Valid frequency codes for time series data.
- 'm': monthly
- 'q': quarterly
- 'sa': semi-annual
- 'a': annual
- 'd': daily
- 'w': weekly
"""


# ============================================================================
# Validation Helper Functions
# ============================================================================

def validate_transformation_type(trans: str, series_id: str) -> None:
    """Validate transformation type.
    
    Parameters
    ----------
    trans : str
        Transformation type to validate
    series_id : str
        Series ID (for error messages)
        
    Raises
    ------
    ValidationError
        If transformation type is invalid
    """
    if trans not in VALID_TRANSFORMATION_TYPES:
        valid_str = ', '.join(VALID_TRANSFORMATION_TYPES)
        raise ValidationError(
            f"Invalid transformation type '{trans}' for series '{series_id}'. "
            f"Valid values: {valid_str}"
        )


def validate_frequency_code(freq: str, series_id: str) -> None:
    """Validate frequency code.
    
    Parameters
    ----------
    freq : str
        Frequency code to validate
    series_id : str
        Series ID (for error messages)
        
    Raises
    ------
    ValidationError
        If frequency code is invalid
    """
    if freq not in VALID_FREQUENCY_CODES:
        valid_str = ', '.join(VALID_FREQUENCY_CODES)
        raise ValidationError(
            f"Invalid frequency code '{freq}' for series '{series_id}'. "
            f"Valid values: {valid_str}"
        )


def validate_series_ids_unique(series_ids: List[str]) -> None:
    """Validate that series IDs are unique.
    
    Parameters
    ----------
    series_ids : List[str]
        List of series IDs to validate
        
    Raises
    ------
    ValidationError
        If duplicate series IDs are found
    """
    seen = set()
    duplicates = []
    for idx, series_id in enumerate(series_ids):
        if series_id in seen:
            duplicates.append((idx, series_id))
        seen.add(series_id)
    
    if duplicates:
        dup_str = ', '.join(f"index {idx} ('{sid}')" for idx, sid in duplicates)
        raise ValidationError(
            f"Duplicate series IDs found: {dup_str}. "
            f"Each series must have a unique identifier."
        )


# ============================================================================
# Data Reading Functions
# ============================================================================

def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp string to datetime.
    
    Parameters
    ----------
    date_str : str
        Date string in various formats
        
    Returns
    -------
    datetime
        Parsed datetime object
    """
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, try pandas parsing
    try:
        import pandas as pd
        return pd.to_datetime(date_str).to_pydatetime()
    except (ValueError, TypeError) as e:
        # Pandas parsing failed with expected exceptions
        raise ValidationError(f"Could not parse date string: {date_str}: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions from pandas
        raise ValidationError(f"Unexpected error parsing date string: {date_str}: {e}")


class TimeIndex:
    """Time index for time series data."""
    
    def __init__(self, dates: Union[List[datetime], pl.Series]):
        """Initialize TimeIndex.
        
        Parameters
        ----------
        dates : list of datetime or polars Series
            List of datetime objects
        """
        if isinstance(dates, pl.Series):
            self.dates = dates.to_list()
        else:
            self.dates = list(dates)
    
    def filter(self, mask: np.ndarray) -> 'TimeIndex':
        """Filter time index by boolean mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean mask
            
        Returns
        -------
        TimeIndex
            Filtered time index
        """
        if isinstance(mask, pl.Series):
            mask = mask.to_numpy()
        filtered_dates = [d for i, d in enumerate(self.dates) if mask[i]]
        return TimeIndex(filtered_dates)
    
    def __len__(self) -> int:
        return len(self.dates)
    
    def __getitem__(self, idx):
        return self.dates[idx]
    
    def __iter__(self):
        return iter(self.dates)


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, TimeIndex, List[str]]:
    """Read time series data from file.
    
    Supports tabular data formats with dates and series values.
    Automatically detects date column and handles various data layouts.
    
    Expected format:
    - First column: Date (YYYY-MM-DD format or datetime-parseable)
    - Subsequent columns: Series data (one column per series)
    - Header row: Series IDs
    
    Alternative format (long format):
    - Metadata columns: series_id, series_name, etc.
    - Date columns: Starting from first date column
    - One row per series, dates as columns
    
    Parameters
    ----------
    datafile : str or Path
        Path to data file
        
    Returns
    -------
    Z : np.ndarray
        Data matrix (T x N) with T time periods and N series
    Time : TimeIndex
        Time index for the data
    mnemonics : List[str]
        Series identifiers (column names)
    """
    from ..utils.config_parser import validate_data_file
    datafile = Path(datafile)
    validate_data_file(datafile)
    
    # Read data file
    try:
        # Use infer_schema_length=None to infer all rows, and try_parse_dates=False
        # to avoid parsing issues with mixed numeric/string columns
        df = pl.read_csv(datafile, infer_schema_length=None, try_parse_dates=False)
    except Exception as e:
        raise ValidationError(f"Failed to read data file {datafile}: {e}")
    
    # Check if first column is a date column or metadata
    first_col = df.columns[0]
    
    # Try to parse first column as date
    try:
        first_val = df[first_col][0]
        if first_val is None:
            is_date_first = False
        else:
            parse_timestamp(str(first_val))
            is_date_first = True
    except (ValueError, TypeError, IndexError):
        is_date_first = False
    
    # If first column is not a date, check if data is in "long" format (one row per series)
    # Skip this check if first column is integer (likely date_id) - treat as standard format
    if not is_date_first:
        first_col_type = df[first_col].dtype
        is_integer_id = first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
        
        # Only check for long format if first column is not an integer ID
        if not is_integer_id:
            # Look for date columns (starting from a certain column)
            date_cols = []
            for col in df.columns:
                try:
                    parse_timestamp(str(df[col][0]))
                    date_cols.append(col)
                except (ValueError, TypeError):
                    pass
            
            if len(date_cols) > 0:
                # Long format: transpose and use first date column as index
                first_date_col = date_cols[0]
                date_col_idx = df.columns.index(first_date_col)
                date_cols_all = df.columns[date_col_idx:]
                
                # Extract dates from column names (they are dates in long format)
                dates = []
                for col in date_cols_all:
                    try:
                        dates.append(parse_timestamp(col))
                    except (ValueError, TypeError):
                        # Skip invalid date columns
                        pass
                
                # Transpose: rows become series, columns become time
                # Select date columns and transpose
                date_data = df.select(date_cols_all)
                Z = date_data.to_numpy().T.astype(float)
                Time = TimeIndex(dates)
                mnemonics = df[first_col].to_list() if first_col in df.columns else [f"series_{i}" for i in range(len(df))]
                
                return Z, Time, mnemonics
    
    # Standard format: first column is date, rest are series
    # Handle integer date_id columns (treat as sequential time index)
    try:
        # Check if first column is integer (date_id format)
        first_col_type = df[first_col].dtype
        if first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
            # Integer date_id: use as sequential index, generate synthetic dates
            n_periods = len(df)
            from datetime import timedelta
            # Start from a default date and increment by day
            start_date = datetime(2000, 1, 1)
            dates = [start_date + timedelta(days=int(df[first_col][i])) for i in range(n_periods)]
            Time = TimeIndex(dates)
        else:
            # Try to parse as date
            time_series = df[first_col].cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
            # If that fails, try other formats
            if time_series.null_count() > 0:
                # Try parsing as string first
                time_series = df[first_col].str.strptime(pl.Datetime, strict=False)
            dates = [parse_timestamp(str(d)) for d in time_series]
            Time = TimeIndex(dates)
    except (ValueError, TypeError) as e:
        # If date parsing fails, treat first column as integer date_id
        try:
            first_col_type = df[first_col].dtype
            if first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                n_periods = len(df)
                from datetime import timedelta
                start_date = datetime(2000, 1, 1)
                dates = [start_date + timedelta(days=int(df[first_col][i])) for i in range(n_periods)]
                Time = TimeIndex(dates)
            else:
                raise ValidationError(f"Failed to parse date column '{first_col}': {e}")
        except (ValueError, TypeError) as e2:
            # Integer date_id parsing failed with expected exceptions
            raise ValidationError(f"Failed to parse date column '{first_col}' as integer date_id: {e2}")
        except Exception as e2:
            # Catch any other unexpected exceptions
            raise ValidationError(f"Unexpected error parsing date column '{first_col}': {e2}")
    
    # Extract series data (all columns except first)
    series_cols = [col for col in df.columns if col != first_col]
    series_data = df.select(series_cols)
    Z = series_data.to_numpy().astype(float)
    mnemonics = series_cols
    
    return Z, Time, mnemonics


# ============================================================================
# Transformation Functions
# ============================================================================
# These functions provide custom transformations for time series data.
# They are implemented as pickleable functions for use with sktime's FunctionTransformer.

# Frequency → lag mappings (for transformation functions only)
# Note: Extended to include 'd' and 'w' for daily/weekly frequencies
FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12, 'd': 1, 'w': 1}


def get_periods_per_year(frequency: str) -> int:
    """Get number of periods per year for a given frequency.
    
    Parameters
    ----------
    frequency : str
        Frequency code ('d', 'w', 'm', 'q', 'sa', 'a')
        
    Returns
    -------
    int
        Number of periods per year
    """
    freq_map = {
        'd': 365,
        'w': 52,
        'm': 12,
        'q': 4,
        'sa': 2,
        'a': 1
    }
    return freq_map.get(frequency.lower(), 12)


def get_annual_factor(frequency: str, step: int = 1) -> float:
    """Get annualization factor for a given frequency and step.
    
    Parameters
    ----------
    frequency : str
        Frequency code ('d', 'w', 'm', 'q', 'sa', 'a')
    step : int, default 1
        Number of periods to difference
        
    Returns
    -------
    float
        Annualization factor (periods_per_year / step)
    """
    periods_per_year = get_periods_per_year(frequency)
    return periods_per_year / step


# ============================================================================
# Transformation Functions (consolidated from transformations.py)
# ============================================================================

def pch_transform(X, step: int = 1) -> np.ndarray:
    """Percent change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
        
    Returns
    -------
    np.ndarray
        Transformed series (percent change)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = 100.0 * (X[step:] - X[:-step]) / np.abs(X[:-step] + 1e-10)
    return result


def pc1_transform(X, year_step: int = 12) -> np.ndarray:
    """Year-over-year percent change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    year_step : int, default 12
        Number of periods for year-over-year (12 for monthly, 4 for quarterly)
        
    Returns
    -------
    np.ndarray
        Transformed series (year-over-year percent change)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > year_step:
        result[year_step:] = 100.0 * (X[year_step:] - X[:-year_step]) / np.abs(X[:-year_step] + 1e-10)
    return result


def pca_transform(X, step: int = 1, annual_factor: float = 12.0) -> np.ndarray:
    """Percent change annualized transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    np.ndarray
        Transformed series (percent change annualized)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = annual_factor * 100.0 * (X[step:] - X[:-step]) / np.abs(X[:-step] + 1e-10)
    return result


def cch_transform(X, step: int = 1) -> np.ndarray:
    """Continuously compounded rate of change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
        
    Returns
    -------
    np.ndarray
        Transformed series (continuously compounded rate)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = 100.0 * (
            np.log(np.abs(X[step:]) + 1e-10) - 
            np.log(np.abs(X[:-step]) + 1e-10)
        )
    return result


def cca_transform(X, step: int = 1, annual_factor: float = 12.0) -> np.ndarray:
    """Continuously compounded annual rate of change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    np.ndarray
        Transformed series (continuously compounded annual rate)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = annual_factor * 100.0 * (
            np.log(np.abs(X[step:]) + 1e-10) - 
            np.log(np.abs(X[:-step]) + 1e-10)
        )
    return result


def log_transform(X) -> np.ndarray:
    """Log transformation (with absolute value + epsilon).
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
        
    Returns
    -------
    np.ndarray
        Transformed series (log of absolute value)
    """
    X = np.asarray(X).flatten()
    return np.log(np.abs(X) + 1e-10)


def identity_transform(X) -> np.ndarray:
    """Identity transformation (no-op).
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
        
    Returns
    -------
    np.ndarray
        Unchanged input
    """
    return np.asarray(X).flatten()


def identity_with_index(X):
    """Identity transformation preserving pandas index (for FunctionTransformer).
    
    This is a module-level function to avoid pickle errors when saving transformers.
    
    Parameters
    ----------
    X : pd.Series or array-like
        Input data
        
    Returns
    -------
    pd.Series or np.ndarray
        Same as input, preserving index if Series
    """
    import pandas as pd
    if isinstance(X, pd.Series):
        result_values = identity_transform(X.values)
        return pd.Series(result_values, index=X.index, name=X.name)
    else:
        return identity_transform(X)


def log_with_index(X):
    """Log transformation preserving pandas index (for FunctionTransformer).
    
    This is a module-level function to avoid pickle errors when saving transformers.
    
    Parameters
    ----------
    X : pd.Series or array-like
        Input data
        
    Returns
    -------
    pd.Series or np.ndarray
        Log-transformed data, preserving index if Series
    """
    import pandas as pd
    if isinstance(X, pd.Series):
        result_values = log_transform(X.values)
        return pd.Series(result_values, index=X.index, name=X.name)
    else:
        return log_transform(X)


def cha_with_index(X, step: int = 1, annual_factor: float = 12.0):
    """Change annual rate transformation preserving pandas index (for FunctionTransformer).
    
    This is a module-level function to avoid pickle errors when saving transformers.
    
    Parameters
    ----------
    X : pd.Series or array-like
        Input data
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    pd.Series or np.ndarray
        Change annual rate transformed data, preserving index if Series
    """
    import pandas as pd
    import numpy as np
    if isinstance(X, pd.Series):
        # cha_transform_func returns 2D array, flatten to 1D
        result_2d = cha_transform_func(X.values, step=step, annual_factor=annual_factor)
        result_values = result_2d.flatten()
        return pd.Series(result_values, index=X.index, name=X.name)
    else:
        return cha_transform_func(X, step=step, annual_factor=annual_factor)


def cha_transform_func(X, step: int = 1, annual_factor: float = 12.0) -> np.ndarray:
    """Change annual rate transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    np.ndarray
        Transformed series (change annual rate) as 2D array (T, 1)
    """
    X = np.asarray(X)
    # Ensure 2D input for transformer compatibility
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_flat = X.flatten()
    T = len(X_flat)
    result = np.full((T, 1), np.nan)
    if T > step:
        # Annualized change: (X[t] / X[t-step])^(1/n) - 1
        ratio = X_flat[step:] / (np.abs(X_flat[:-step]) + 1e-10)
        result[step:, 0] = 100.0 * (np.power(ratio, 1.0 / annual_factor) - 1.0)
    return result


# Factory functions (pickleable closures)

def _get_function_transformer():
    """Get FunctionTransformer from sktime, raising ImportError if unavailable."""
    try:
        from sktime.transformations.series.func_transform import FunctionTransformer
        return FunctionTransformer
    except ImportError:
        raise ImportError(
            "sktime is required for FunctionTransformer. "
            "Install it with: pip install sktime"
        )


def make_pch_transformer(step: int):
    """Create pch transformer with step parameter.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for percent change that preserves index
    """
    FunctionTransformer = _get_function_transformer()
    def pch_with_index(X, step=step):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        if isinstance(X, pd.Series):
            result_values = pch_transform(X.values, step=step)
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return pch_transform(X, step=step)
    return FunctionTransformer(func=lambda X: pch_with_index(X, step=step))


def make_pc1_transformer(year_step: int):
    """Create pc1 transformer with year_step parameter.
    
    Parameters
    ----------
    year_step : int
        Number of periods for year-over-year
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for year-over-year percent change that preserves index
    """
    FunctionTransformer = _get_function_transformer()
    def pc1_with_index(X, year_step=year_step):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        if isinstance(X, pd.Series):
            result_values = pc1_transform(X.values, year_step=year_step)
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return pc1_transform(X, year_step=year_step)
    return FunctionTransformer(func=lambda X: pc1_with_index(X, year_step=year_step))


def make_pca_transformer(step: int, annual_factor: float):
    """Create pca transformer with step and annual_factor parameters.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
    annual_factor : float
        Annualization factor
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for percent change annualized
    """
    FunctionTransformer = _get_function_transformer()
    def pca_with_index(X, step=step, annual_factor=annual_factor):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        if isinstance(X, pd.Series):
            result_values = pca_transform(X.values, step=step, annual_factor=annual_factor)
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return pca_transform(X, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=lambda X: pca_with_index(X, step=step, annual_factor=annual_factor))


def make_cch_transformer(step: int):
    """Create cch transformer with step parameter.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for continuously compounded rate
    """
    FunctionTransformer = _get_function_transformer()
    def cch_with_index(X, step=step):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        if isinstance(X, pd.Series):
            result_values = cch_transform(X.values, step=step)
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return cch_transform(X, step=step)
    return FunctionTransformer(func=lambda X: cch_with_index(X, step=step))


def make_cca_transformer(step: int, annual_factor: float):
    """Create cca transformer with step and annual_factor parameters.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
    annual_factor : float
        Annualization factor
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for continuously compounded annual rate
    """
    FunctionTransformer = _get_function_transformer()
    def cca_with_index(X, step=step, annual_factor=annual_factor):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        if isinstance(X, pd.Series):
            result_values = cca_transform(X.values, step=step, annual_factor=annual_factor)
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return cca_transform(X, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=lambda X: cca_with_index(X, step=step, annual_factor=annual_factor))


def make_cha_transformer(step: int, annual_factor: float):
    """Create cha transformer with step and annual_factor parameters.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
    annual_factor : float
        Annualization factor
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for change annual rate
    """
    import functools
    FunctionTransformer = _get_function_transformer()
    # Use functools.partial with module-level function for proper pickle serialization
    # This avoids lambda closure issues that cause pickle errors
    cha_func = functools.partial(cha_with_index, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=cha_func)


# Index-Preserving Column Ensemble Transformer

if HAS_SKTIME:
    class IndexPreservingColumnEnsembleTransformer(BaseTransformer):
        """Wrapper for ColumnEnsembleTransformer that preserves index type.
        
        ColumnEnsembleTransformer uses pd.concat internally, which can convert
        DatetimeIndex to base Index when concatenating Series with different index types.
        This wrapper ensures the output always has a compatible index type
        (DatetimeIndex, PeriodIndex, or RangeIndex).
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
            "fit_is_empty": False,  # Must be False to call _fit
            "transform-returns-same-time-index": True,
        }
        
        def __init__(self, transformers, remainder="drop", feature_names_out="auto"):
            """Initialize IndexPreservingColumnEnsembleTransformer.
            
            Parameters
            ----------
            transformers : list of tuples
                Same as ColumnEnsembleTransformer.transformers
            remainder : str or estimator, default "drop"
                Same as ColumnEnsembleTransformer.remainder
            feature_names_out : str, default "auto"
                Same as ColumnEnsembleTransformer.feature_names_out
            """
            super().__init__()
            self.transformers = transformers
            self.remainder = remainder
            self.feature_names_out = feature_names_out
            self._column_transformer = ColumnEnsembleTransformer(
                transformers=transformers,
                remainder=remainder,
                feature_names_out=feature_names_out
            )
        
        def _fit(self, X: "pd.DataFrame", y: Optional["pd.DataFrame"] = None):
            """Fit the transformer."""
            # Store original index type for transform (before fitting)
            if isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
                self._original_index_type = type(X.index)
                self._original_index = X.index.copy()
            else:
                # Try to convert to DatetimeIndex
                try:
                    self._original_index = pd.to_datetime(X.index)
                    self._original_index_type = pd.DatetimeIndex
                except (ValueError, TypeError):
                    # Fallback to RangeIndex
                    self._original_index = pd.RangeIndex(start=0, stop=len(X))
                    self._original_index_type = pd.RangeIndex
            
            # Fit the underlying transformer
            self._column_transformer.fit(X, y)
            return self
        
        def _transform(self, X: "pd.DataFrame", y: Optional["pd.DataFrame"] = None) -> "pd.DataFrame":
            """Transform X and ensure output has compatible index.
            
            This method intercepts the transform call and ensures each transformer's
            output has the same index type before concatenation, preventing pd.concat
            from creating a base Index.
            """
            # Get transformers and transform each column separately
            # This allows us to ensure each output has compatible index before concat
            Xts = []
            keys = []
            
            # Get fitted transformers
            # ColumnEnsembleTransformer stores fitted transformers in transformers_ attribute
            fitted_transformers = getattr(self._column_transformer, 'transformers_', [])
            
            for name, est, index in fitted_transformers:
                # Transform this column
                Xt_col = est.transform(X.loc[:, index], y)
                
                # Ensure output has compatible index (same as input X.index)
                # Always use input X.index to ensure all outputs have same index
                if isinstance(Xt_col, pd.Series):
                    # Use input index if length matches, otherwise align
                    if len(Xt_col) == len(X.index):
                        Xt_col.index = X.index
                    else:
                        # Length mismatch - try to align or use RangeIndex
                        if len(Xt_col) <= len(X.index):
                            Xt_col.index = X.index[:len(Xt_col)]
                        else:
                            # Output longer than input - use RangeIndex
                            Xt_col.index = pd.RangeIndex(start=0, stop=len(Xt_col))
                    Xts.append(Xt_col)
                elif isinstance(Xt_col, pd.DataFrame):
                    # Use input index if length matches
                    if len(Xt_col) == len(X.index):
                        Xt_col.index = X.index
                    else:
                        if len(Xt_col) <= len(X.index):
                            Xt_col.index = X.index[:len(Xt_col)]
                        else:
                            Xt_col.index = pd.RangeIndex(start=0, stop=len(Xt_col))
                    Xts.append(Xt_col)
                else:
                    # Convert numpy array to Series/DataFrame with input index
                    Xt_col = np.asarray(Xt_col)
                    if len(Xt_col) == len(X.index):
                        if Xt_col.ndim == 1:
                            Xt_col = pd.Series(Xt_col, index=X.index)
                        else:
                            col_names = [f'{name}_{i}' for i in range(Xt_col.shape[1])]
                            Xt_col = pd.DataFrame(Xt_col, index=X.index, columns=col_names)
                    else:
                        # Length mismatch - use RangeIndex
                        if Xt_col.ndim == 1:
                            Xt_col = pd.Series(Xt_col, index=pd.RangeIndex(start=0, stop=len(Xt_col)))
                        else:
                            col_names = [f'{name}_{i}' for i in range(Xt_col.shape[1])]
                            Xt_col = pd.DataFrame(Xt_col, index=pd.RangeIndex(start=0, stop=len(Xt_col)), columns=col_names)
                    Xts.append(Xt_col)
                
                keys.append(name)
            
            # Concatenate with compatible indices (all should have same index now)
            if Xts:
                # Ensure all have same index before concat
                # Use the first transformer's index (or input index) as reference
                reference_index = X.index if isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)) else pd.RangeIndex(start=0, stop=len(X))
                
                # Align all outputs to reference index
                aligned_Xts = []
                for i, Xt_col in enumerate(Xts):
                    if len(Xt_col) == len(reference_index):
                        if isinstance(Xt_col, pd.Series):
                            Xt_col.index = reference_index
                        elif isinstance(Xt_col, pd.DataFrame):
                            Xt_col.index = reference_index
                    aligned_Xts.append(Xt_col)
                
                Xt = pd.concat(aligned_Xts, axis=1, keys=keys)
                
                # Final check: ensure output index is compatible and sorted
                if not isinstance(Xt.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
                    Xt.index = reference_index
                elif not Xt.index.is_monotonic_increasing:
                    # Index is not sorted - reindex to sorted version
                    if isinstance(Xt.index, pd.DatetimeIndex):
                        Xt = Xt.sort_index()
                    else:
                        # For non-DatetimeIndex, use reference index
                        Xt.index = reference_index
            else:
                # No transformers - return empty DataFrame with compatible index
                Xt = pd.DataFrame(index=X.index if isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)) else pd.RangeIndex(start=0, stop=len(X)))
            
            return Xt
else:
    # Fallback when sktime is not available
    IndexPreservingColumnEnsembleTransformer = None


# ============================================================================
# Transformer Factory from Config
# ============================================================================

def create_transformer_from_config(config: Any) -> Any:
    """Create sktime ColumnEnsembleTransformer from DFMConfig.
    
    This function creates a ColumnEnsembleTransformer that applies per-series
    transformations based on the DFMConfig, followed by standardization.
    The transformer is wrapped in a TransformerPipeline with StandardScaler.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object containing series configurations
        
    Returns
    -------
    TransformerPipeline
        Transformer pipeline with ColumnEnsembleTransformer and StandardScaler.
        Supports Polars output via set_output(transform="polars").
        
    Raises
    ------
    ImportError
        If sktime is not installed
    ValidationError
        If config is invalid, missing required fields, or contains invalid
        transformation types, frequency codes, or duplicate series IDs
    """
    # Check sktime availability
    check_sktime_available()  # Keep public API for preprocess module
    
    # Import components, handling version differences
    try:
        from sktime.transformations.compose import (
            TransformerPipeline,
            ColumnEnsembleTransformer
        )
    except ImportError:
        raise ImportError("TransformerPipeline or ColumnEnsembleTransformer not available in sktime")
    
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.impute import Imputer
    from sktime.forecasting.naive import NaiveForecaster
    from sklearn.preprocessing import StandardScaler
    
    # Validate config
    if config is None:
        raise ValidationError("config cannot be None")
    if not hasattr(config, 'series') or not config.series:
        raise ValidationError("config must have a non-empty 'series' attribute")
    
    # Frequency → lag mappings
    FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
    FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12, 'd': 1, 'w': 1}
    
    # Get series IDs from config
    try:
        series_ids = config.get_series_ids()
    except AttributeError:
        # Fallback: generate series IDs from series list
        series_ids = [s.series_id if hasattr(s, 'series_id') and s.series_id else f"series_{i}" 
                     for i, s in enumerate(config.series)]
    
    # Validate series IDs are unique
    validate_series_ids_unique(series_ids)
    
    # Get global imputation method (fallback if series-specific is null)
    global_impute = None
    if hasattr(config, 'preprocess') and hasattr(config.preprocess, 'global_preprocessing'):
        global_preproc = config.preprocess.global_preprocessing
        if hasattr(global_preproc, 'imputation') and hasattr(global_preproc.imputation, 'method'):
            global_impute = global_preproc.imputation.method
    
    # Create per-series transformers
    transformers = []
    for i, series_config in enumerate(config.series):
        trans = series_config.transformation.lower() if hasattr(series_config, 'transformation') else 'lin'
        freq = series_config.frequency.lower() if hasattr(series_config, 'frequency') else 'm'
        series_id = series_ids[i] if i < len(series_ids) else f"series_{i}"
        
        # Get series-specific imputation method
        series_impute = None
        try:
            # Check if series_config has impute attribute
            if hasattr(series_config, 'impute'):
                series_impute = series_config.impute
            # Also check if it's a dict-like object
            elif isinstance(series_config, dict) and 'impute' in series_config:
                series_impute = series_config['impute']
        except (AttributeError, TypeError):
            pass
        
        # Use global or default if series-specific is null/None
        if series_impute is None or series_impute == 'null' or series_impute == '':
            if global_impute:
                series_impute = global_impute
            else:
                # Default: ffill_bfill (use ffill then bfill)
                series_impute = 'ffill_bfill'
        
        # Create imputation transformer(s) for this series
        imputation_steps = []
        if series_impute:
            if series_impute == 'ffill_bfill':
                # Use both ffill and bfill
                imputation_steps.append(Imputer(method="ffill"))
                imputation_steps.append(Imputer(method="bfill"))
            elif series_impute == 'ffill':
                imputation_steps.append(Imputer(method="ffill"))
            elif series_impute == 'bfill':
                imputation_steps.append(Imputer(method="bfill"))
            elif series_impute == 'naive' or series_impute == 'forecaster':
                # Use naive forecaster for imputation
                imputation_steps.append(Imputer(
                    method="forecaster",
                    forecaster=NaiveForecaster(strategy="last")
                ))
                # Add bfill as safety net
                imputation_steps.append(Imputer(method="bfill"))
            else:
                # Unknown method, use default ffill_bfill
                imputation_steps.append(Imputer(method="ffill"))
                imputation_steps.append(Imputer(method="bfill"))
        
        # Validate transformation type and frequency code
        validate_transformation_type(trans, series_id)
        validate_frequency_code(freq, series_id)
        
        # Create transformer based on transformation type
        if trans == 'lin':
            # Identity transformation (no-op) - preserve index
            # Direct reference to module-level function for proper pickle serialization
            # Use globals() to get the function from current module namespace
            transformer = FunctionTransformer(func=globals()['identity_with_index'])
        elif trans == 'log':
            # Log transformation - preserve index
            # Direct reference to module-level function for proper pickle serialization
            # Use globals() to get the function from current module namespace
            transformer = FunctionTransformer(func=globals()['log_with_index'])
        elif trans == 'chg':
            # Change (difference)
            lag = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = Differencer(lags=lag)  # Differencer accepts int, not list
        elif trans == 'ch1':
            # Year over year change
            lag = FREQ_TO_LAG_YOY.get(freq, 12)
            transformer = Differencer(lags=lag)  # Differencer accepts int, not list
        elif trans == 'cha':
            # Change (annual rate) - custom function transformer
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            transformer = make_cha_transformer(step, annual_factor)
        elif trans == 'pch':
            # Percent change
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = make_pch_transformer(step)
        elif trans == 'pc1':
            # Year over year percent change
            year_step = FREQ_TO_LAG_YOY.get(freq, 12)
            transformer = make_pc1_transformer(year_step)
        elif trans == 'pca':
            # Percent change annualized
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            transformer = make_pca_transformer(step, annual_factor)
        else:
            # This should never happen due to validation above, but keep as safety check
            # If we reach here, it means validation was bypassed somehow
            raise ValidationError(
                f"Unknown transformation '{trans}' for series '{series_id}'. "
                f"This should have been caught by validation. "
                f"Valid values: {', '.join(VALID_TRANSFORMATION_TYPES)}"
            )
        
        # Combine imputation and transformation into a pipeline for this series
        if imputation_steps:
            # Create a pipeline: imputation → transformation
            series_pipeline_steps = []
            for j, impute_step in enumerate(imputation_steps):
                series_pipeline_steps.append((f"impute_{j}", impute_step))
            series_pipeline_steps.append(("transform", transformer))
            series_transformer = TransformerPipeline(series_pipeline_steps)
        else:
            # No imputation, just transformation
            series_transformer = transformer
        
        # Add to transformers list: (name, transformer, column_index)
        # ColumnEnsembleTransformer accepts column index (int) or column name (str/list)
        transformers.append((series_id, series_transformer, i))
    
    # Create ColumnEnsembleTransformer with index preservation wrapper
    # ColumnEnsembleTransformer uses pd.concat internally, which can convert
    # DatetimeIndex to base Index when concatenating Series with different index types.
    # We wrap it to ensure output always has compatible index (DatetimeIndex, PeriodIndex, or RangeIndex).
    try:
        column_transformer = IndexPreservingColumnEnsembleTransformer(transformers=transformers)
    except (NameError, TypeError):
        # Fallback to standard ColumnEnsembleTransformer if wrapper not available
        column_transformer = ColumnEnsembleTransformer(transformers=transformers)
    
    # Create full pipeline: ColumnEnsembleTransformer → StandardScaler
    pipeline = TransformerPipeline([
        ("transform", column_transformer),
        ("scaler", StandardScaler())
    ])
    
    # Set pandas output (required for sktime mtype compatibility)
    # pandas output ensures index is preserved (DatetimeIndex, PeriodIndex, or RangeIndex)
    try:
        pipeline.set_output(transform="pandas")
    except AttributeError:
        # Older sktime versions might not support set_output
        # This is okay - DFMDataModule will handle it
        pass
    
    return pipeline

