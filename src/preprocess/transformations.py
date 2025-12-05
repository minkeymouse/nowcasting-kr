"""Custom transformation functions for DFM.

This module provides transformation functions implemented as
pickleable functions for use with sktime's FunctionTransformer.
Also includes IndexPreservingColumnEnsembleTransformer wrapper.
"""

import numpy as np
from typing import Callable, Any, Optional
import pandas as pd

from sktime.transformations.compose import ColumnEnsembleTransformer
from sktime.transformations.base import BaseTransformer

# Frequency → lag mappings (for transformation functions only)
FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1}
FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12}

# Import get_periods_per_year and get_annual_factor from dfm-python config.utils
# These are the canonical implementations
try:
    from dfm_python.config.utils import get_periods_per_year, get_annual_factor
except ImportError:
    # Fallback: define locally if dfm-python not available
    def get_periods_per_year(frequency: str) -> int:
        """Get periods per year for a frequency code."""
        freq_map = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
        return freq_map.get(frequency.lower(), 12)
    
    def get_annual_factor(frequency: str, step: int) -> float:
        """Get annualization factor for a frequency and step."""
        periods_per_year = get_periods_per_year(frequency)
        return periods_per_year / step


# Transformation functions

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
    FunctionTransformer = _get_function_transformer()
    def cha_with_index(X, step=step, annual_factor=annual_factor):
        """Wrapper that preserves index when transforming."""
        import pandas as pd
        import numpy as np  # Ensure np is available in closure
        if isinstance(X, pd.Series):
            # cha_transform_func returns 2D array, flatten to 1D
            result_2d = cha_transform_func(X.values, step=step, annual_factor=annual_factor)
            result_values = result_2d.flatten()
            return pd.Series(result_values, index=X.index, name=X.name)
        else:
            return cha_transform_func(X, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=lambda X: cha_with_index(X, step=step, annual_factor=annual_factor))


# ========================================================================
# Index-Preserving Column Ensemble Transformer
# ========================================================================

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
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
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
    
    def _transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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

