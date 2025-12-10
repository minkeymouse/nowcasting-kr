"""Data preprocessing module - simplified and focused on core functionality.

This module provides essential preprocessing functions for time series models:
- Data preparation for different model types (ARIMA, VAR, DFM, DDFM)
- Resampling (weekly to monthly)
- Missing data imputation
- Transformer creation from config
"""

import logging
from typing import Optional, Tuple, Any, Dict
import numpy as np
import pandas as pd

from src.utils import ValidationError

_logger = logging.getLogger(__name__)

# ============================================================================
# Sktime Imports
# ============================================================================

try:
    from sktime.transformations.compose import TransformerPipeline, ColumnEnsembleTransformer
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.impute import Imputer
    from sktime.forecasting.naive import NaiveForecaster
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # Try to import LogTransformer (location may vary by sktime version)
    try:
        from sktime.transformations.series.log import LogTransformer
    except ImportError:
        try:
            from sktime.transformations.series import LogTransformer
        except ImportError:
            LogTransformer = None  # Will use FunctionTransformer as fallback
    
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    raise ImportError("sktime is required. Install with: pip install sktime[forecasting]")

# ============================================================================
# Core Preprocessing Functions
# ============================================================================

def resample_to_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """Resample weekly data to monthly using window averaging."""
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    # Select only numeric columns for resampling (exclude object/string columns)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        _logger.warning("No numeric columns found for resampling")
        return data
    
    # Resample only numeric columns
    try:
        resampled = data[numeric_cols].resample('ME').mean()
    except ValueError:
        resampled = data[numeric_cols].resample('M').mean()
    
    return resampled


def resample_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """Resample monthly data to weekly frequency for mixed-frequency DFM.
    
    When clock='w' and series are monthly (frequency='m'), we need to convert
    monthly data to weekly frequency. For mixed_freq=True with tent kernel,
    monthly data should be placed at weekly intervals with NaN values between.
    
    This function:
    1. Creates a weekly index covering the data period
    2. Places monthly values at appropriate weekly positions (typically end of month)
    3. Fills remaining weeks with NaN (for tent kernel processing)
    
    Parameters
    ----------
    data : pd.DataFrame
        Monthly data with DatetimeIndex
        
    Returns
    -------
    pd.DataFrame
        Weekly data with monthly values placed at appropriate positions
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        _logger.warning("No numeric columns found for resampling")
        return data
    
    # Create weekly index covering the data period
    # Start from first data point, end at last data point
    start_date = data.index.min()
    end_date = data.index.max()
    
    # Create weekly index (end of week, typically Sunday)
    try:
        weekly_index = pd.date_range(start=start_date, end=end_date, freq='W')
    except ValueError:
        # Fallback to 'W-SUN' if 'W' fails
        weekly_index = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    
    # Create weekly DataFrame with NaN
    weekly_data = pd.DataFrame(index=weekly_index, columns=numeric_cols)
    weekly_data[:] = np.nan
    
    # For each monthly observation, find the closest weekly date (typically end of month)
    # and place the value there
    for date in data.index:
        # Find the weekly date that is closest to this monthly date
        # Typically, monthly data is at end of month, so find the last week of that month
        month_end = date + pd.offsets.MonthEnd(0)
        
        # Find the closest weekly date (within the same month)
        closest_week_idx = weekly_index[weekly_index <= month_end]
        if len(closest_week_idx) > 0:
            # Use the last week of the month
            target_week = closest_week_idx[-1]
            if target_week in weekly_data.index:
                weekly_data.loc[target_week, numeric_cols] = data.loc[date, numeric_cols].values
    
    return weekly_data


def set_dataframe_frequency(y_train: pd.DataFrame) -> pd.DataFrame:
    """Set frequency on DataFrame index if DatetimeIndex."""
    if not isinstance(y_train.index, pd.DatetimeIndex):
        return y_train
    
    if y_train.index.freq is not None:
        return y_train
    
    inferred_freq = pd.infer_freq(y_train.index)
    if inferred_freq:
        try:
            y_train = y_train.asfreq(inferred_freq, method='ffill')
        except (TypeError, ValueError):
            y_train = y_train.asfreq(inferred_freq)
            y_train = y_train.ffill()
    
    return y_train


def apply_scaling(y_train: pd.DataFrame, scaler_type: str = 'robust') -> Tuple[pd.DataFrame, Any]:
    """Apply scaling to data.
    
    Parameters
    ----------
    y_train : pd.DataFrame
        Data to scale
    scaler_type : str
        Type of scaler ('robust', 'standard', or None for no scaling)
        
    Returns
    -------
    Tuple[pd.DataFrame, Any]
        (scaled_data, scaler) tuple
    """
    if scaler_type is None or scaler_type == 'null':
        return y_train, None
    
    scaler_type_lower = scaler_type.lower() if scaler_type else 'robust'
    
    if scaler_type_lower == 'robust':
        scaler = RobustScaler()
    elif scaler_type_lower == 'standard':
        scaler = StandardScaler()
    else:
        _logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler")
        scaler = RobustScaler()
    
    # Fit and transform
    scaled_values = scaler.fit_transform(y_train.values)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=y_train.index,
        columns=y_train.columns
    )
    
    return scaled_df, scaler


def impute_missing_values(y_train: pd.DataFrame, model_type: str = "var") -> pd.DataFrame:
    """Impute missing values using forward-fill, backward-fill, and forecaster."""
    if y_train.isnull().sum().sum() == 0:
        return y_train
    
    _logger.warning(f"{model_type.upper()} data contains NaN values. Applying imputation...")
    
    imputer_ffill = Imputer(method="ffill")
    imputer_bfill = Imputer(method="bfill")
    imputer_forecaster = Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))
    
    for col in y_train.columns:
        col_series = y_train[[col]]
        col_imputed = imputer_ffill.fit_transform(col_series)
        
        if isinstance(col_imputed, pd.DataFrame) and col_imputed.isnull().sum().sum() > 0:
            col_imputed = imputer_bfill.fit_transform(col_imputed)
        
        if isinstance(col_imputed, pd.DataFrame) and col_imputed.isnull().sum().sum() > 0:
            try:
                col_imputed = imputer_forecaster.fit_transform(col_imputed)
            except Exception as e:
                _logger.warning(f"Forecaster imputation failed for {col}: {e}")
        
        if isinstance(col_imputed, pd.DataFrame):
            y_train[col] = col_imputed[col]
    
    # Drop remaining NaN rows
    nan_count = int(y_train.isnull().sum().sum())
    if nan_count > 0:
        nan_rows = int(y_train.isnull().any(axis=1).sum())
        _logger.warning(f"Dropping {nan_rows} rows with remaining NaN values...")
        y_train = y_train.dropna()
        if len(y_train) == 0:
            raise ValidationError(f"{model_type.upper()}: All data was dropped after imputation.")
    
    return y_train


def apply_transformations(data: pd.DataFrame, config_path: Optional[str] = None, series_ids: Optional[list] = None) -> pd.DataFrame:
    """Apply transformations based on series config files (config/series/{series_id}.yaml).
    
    This function reads transformation information from Hydra series config files,
    maintaining consistency with the Hydra config system.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to transform
    config_path : Optional[str]
        Path to config directory (default: config/ relative to project root)
    series_ids : Optional[list]
        List of series IDs to process. If None, uses all columns in data.
        
    Returns
    -------
    pd.DataFrame
        Transformed data
    """
    import yaml
    from pathlib import Path
    
    # Get config path
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / "config")
    
    config_dir = Path(config_path)
    series_dir = config_dir / "series"
    
    # Get series IDs to process
    if series_ids is None:
        series_ids = list(data.columns)
    
    # Create transformation mapping from series config files
    trans_map = {}
    freq_map = {}
    
    for series_id in series_ids:
        if series_id not in data.columns:
            continue
        
        # Try to load from series config file
        series_config_file = series_dir / f"{series_id}.yaml"
        if series_config_file.exists():
            try:
                with open(series_config_file, 'r') as f:
                    series_config = yaml.safe_load(f) or {}
                
                trans = series_config.get('transformation', 'lin')
                freq = series_config.get('frequency', 'm')
                trans_map[series_id] = str(trans).lower() if trans else 'lin'
                freq_map[series_id] = str(freq).lower() if freq else 'm'
                _logger.debug(f"Loaded transformation for {series_id} from {series_config_file}: {trans_map[series_id]}")
            except Exception as e:
                _logger.warning(f"Failed to load series config for {series_id} from {series_config_file}: {e}. Using defaults.")
                trans_map[series_id] = 'lin'
                freq_map[series_id] = 'm'
        else:
            # Fallback: try metadata.csv if series config not found
            _logger.debug(f"Series config not found for {series_id} at {series_config_file}. Trying metadata.csv fallback...")
            metadata_path = Path(__file__).parent.parent / "data" / "metadata.csv"
            if metadata_path.exists():
                try:
                    metadata = pd.read_csv(metadata_path)
                    meta_row = metadata[metadata['SeriesID'] == series_id]
                    if len(meta_row) > 0:
                        trans = meta_row.iloc[0].get('Transformation', 'lin')
                        freq = meta_row.iloc[0].get('Frequency', 'm')
                        trans_map[series_id] = str(trans).lower() if pd.notna(trans) else 'lin'
                        freq_map[series_id] = str(freq).lower() if pd.notna(freq) else 'm'
                        _logger.debug(f"Loaded transformation for {series_id} from metadata.csv: {trans_map[series_id]}")
                    else:
                        trans_map[series_id] = 'lin'
                        freq_map[series_id] = 'm'
                except Exception as e:
                    _logger.warning(f"Failed to load metadata for {series_id}: {e}. Using default (lin).")
                    trans_map[series_id] = 'lin'
                    freq_map[series_id] = 'm'
            else:
                # No config found, use default
                trans_map[series_id] = 'lin'
                freq_map[series_id] = 'm'
    
    # Apply transformations
    result_data = data.copy()
    
    for col in data.columns:
        if col not in trans_map:
            continue
        
        trans = trans_map[col]
        freq = freq_map.get(col, 'm')
        
        if trans == 'chg':
            # Difference transformation
            # Applied at original series frequency, not clock frequency
            # For weekly: lag=1, for monthly: lag=1, for quarterly: lag=1
            lag = 1  # Period-over-period difference at series frequency
            
            series = result_data[col].dropna()
            if len(series) > lag:
                # Calculate difference at series frequency
                diff_values = series.diff(periods=lag)
                result_data[col] = diff_values
                _logger.debug(f"Applied chg transformation to {col} (series freq={freq}, lag={lag})")
        
        elif trans == 'ch1':
            # Year-over-year change
            # Applied at original series frequency
            # For weekly: lag=52 (52 weeks = 1 year)
            # For monthly: lag=12 (12 months = 1 year)
            # For quarterly: lag=4 (4 quarters = 1 year)
            lag = 52 if freq == 'w' else (12 if freq == 'm' else (4 if freq == 'q' else 1))
            series = result_data[col].dropna()
            if len(series) > lag:
                diff_values = series.diff(periods=lag)
                result_data[col] = diff_values
                _logger.debug(f"Applied ch1 transformation to {col} (series freq={freq}, lag={lag})")
        
        elif trans == 'log':
            # Log transformation
            series = result_data[col].dropna()
            if len(series) > 0 and (series > 0).all():
                result_data[col] = np.log(series)
                _logger.debug(f"Applied log transformation to {col}")
            else:
                _logger.warning(f"Cannot apply log transformation to {col}: non-positive values")
        
        elif trans == 'pch':
            # Percent change
            # Applied at original series frequency
            lag = 1  # Period-over-period change at series frequency
            series = result_data[col].dropna()
            if len(series) > lag:
                pct_change = series.pct_change(periods=lag) * 100
                result_data[col] = pct_change
                _logger.debug(f"Applied pch transformation to {col} (series freq={freq}, lag={lag})")
        
        # 'lin' (linear/no transformation) - no change needed
    
    return result_data


def prepare_multivariate_data(
    data: pd.DataFrame,
    config_dict: Optional[dict],
    cfg: Any,
    target_series: Optional[str],
    model_type: str
) -> Tuple[pd.DataFrame, list[str]]:
    """Prepare multivariate training data for DFM/DDFM/VAR models."""
    from omegaconf import OmegaConf
    from pathlib import Path
    
    # Get config path for series config files
    project_root = Path(__file__).parent.parent
    config_path = str(project_root / "config")
    
    # Check if model uses weekly frequency (clock='w')
    # For DFM: weekly clock means no resampling (keep weekly data)
    # For DDFM: weekly clock means no resampling (keep weekly data), monthly clock means resample to monthly
    should_resample = True
    if config_dict and config_dict.get('clock') == 'w':
        should_resample = False

    # ------------------------------------------------------------------
    # Normalize index to avoid duplicate labels:
    # - Weekly clock ('w'): use date_w if available (unique weekly dates)
    # - Monthly clock: use first column (date); drop duplicate dates
    # ------------------------------------------------------------------
    data = data.copy()
    if not should_resample and 'date_w' in data.columns:
        idx = pd.to_datetime(data['date_w'], errors='coerce')
    else:
        idx = pd.to_datetime(data.iloc[:, 0], errors='coerce')
    data.index = idx
    data = data[~data.index.isna()]
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep='last')]
    # Remove date columns from features
    for col in ['date', 'date_w']:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    # Get series list from config
    series_ids = []
    if config_dict and 'series' in config_dict:
        series_ids = [
            s.get('series_id', s) if isinstance(s, dict) else s 
            for s in config_dict['series']
        ]
    elif hasattr(cfg, 'experiment') and 'series' in cfg.experiment:
        series_ids = OmegaConf.to_container(cfg.experiment.series, resolve=True)
        if not isinstance(series_ids, list):
            series_ids = []
    
    # Filter to available series
    if series_ids:
        available_series = [s for s in series_ids if s in data.columns]
        if target_series and target_series in data.columns and target_series not in available_series:
            available_series.append(target_series)
        
        if not available_series:
            raise ValidationError(
                f"{model_type.upper()}: No series found in data. "
                f"Config series: {series_ids[:5]}..., Data columns: {list(data.columns)[:5]}..."
            )
        
        selected_data = data[available_series].dropna(how='all')
        
        # Apply transformations from series config files (at original series frequency)
        selected_data = apply_transformations(selected_data, config_path=config_path, series_ids=available_series)
        
        if should_resample:
            processed_data = resample_to_monthly(selected_data)
        else:
            processed_data = selected_data
        return processed_data, available_series
    else:
        # Use all numeric columns
        y_train = data.select_dtypes(include=[np.number])
        if target_series and target_series in data.columns and target_series not in y_train.columns:
            y_train[target_series] = data[target_series]
        
        y_train = y_train.dropna(how='all')
        
        # Apply transformations from series config files (at original series frequency)
        y_train = apply_transformations(y_train, config_path=config_path, series_ids=list(y_train.columns))
        
        if should_resample:
            processed_data = resample_to_monthly(y_train)
        else:
            processed_data = y_train
        available_series = list(processed_data.columns)
        return processed_data, available_series


def prepare_univariate_data(data: pd.DataFrame, target_series: str, config_path: Optional[str] = None) -> pd.Series:
    """Prepare univariate training data for ARIMA models."""
    from pathlib import Path
    
    if target_series not in data.columns:
        raise ValidationError(f"Target series '{target_series}' not found in data columns")
    
    # Get config path for series config files
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / "config")
    
    y_train = data[[target_series]]
    
    # Apply transformations from series config files (at original series frequency)
    y_train = apply_transformations(y_train, config_path=config_path, series_ids=[target_series])
    
    y_train_monthly = resample_to_monthly(y_train)
    y_train_monthly = set_dataframe_frequency(y_train_monthly)
    
    return y_train_monthly.iloc[:, 0]


# ============================================================================
# Transformation Functions (Index-preserving for sktime)
# ============================================================================

def _identity_transform(X):
    """Identity transformation preserving pandas index."""
    if isinstance(X, pd.Series):
        return X
    return np.asarray(X)


def _pch_transform(X, step: int = 1):
    """Percent change transformation preserving index."""
    if isinstance(X, pd.Series):
        X_vals = X.values
        index = X.index
        name = X.name
    else:
        X_vals = np.asarray(X).flatten()
        index = None
        name = None
    
    T = len(X_vals)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = 100.0 * (X_vals[step:] - X_vals[:-step]) / (np.abs(X_vals[:-step]) + 1e-10)
    
    if index is not None:
        return pd.Series(result, index=index, name=name)
    return result


def _cha_transform(X, step: int = 1, annual_factor: float = 12.0):
    """Change annualized transformation preserving index.
    
    Annualized change: (X[t] / X[t-step])^(1/annual_factor) - 1
    """
    if isinstance(X, pd.Series):
        X_vals = X.values
        index = X.index
        name = X.name
    else:
        X_vals = np.asarray(X).flatten()
        index = None
        name = None
    
    T = len(X_vals)
    result = np.full(T, np.nan)
    if T > step:
        ratio = X_vals[step:] / (np.abs(X_vals[:-step]) + 1e-10)
        result[step:] = 100.0 * (np.power(ratio, 1.0 / annual_factor) - 1.0)
    
    if index is not None:
        return pd.Series(result, index=index, name=name)
    return result


def _get_periods_per_year(frequency: str) -> int:
    """Get number of periods per year for a given frequency."""
    freq_map = {'d': 365, 'w': 52, 'm': 12, 'q': 4, 'sa': 2, 'a': 1}
    return freq_map.get(frequency.lower(), 12)


def _get_annual_factor(frequency: str, step: int = 1) -> float:
    """Get annualization factor for a given frequency and step."""
    periods_per_year = _get_periods_per_year(frequency)
    return periods_per_year / step


def create_transformer_from_config(config: Any) -> Any:
    """Create sktime transformer pipeline from DFMConfig."""
    if config is None or not hasattr(config, 'series') or not config.series:
        raise ValidationError("config must have a non-empty 'series' attribute")
    
    # Frequency to lag mappings
    FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
    FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12, 'd': 1, 'w': 1}
    
    # Get series IDs
    try:
        series_ids = config.get_series_ids()
    except AttributeError:
        series_ids = [s.series_id if hasattr(s, 'series_id') and s.series_id else f"series_{i}" 
                     for i, s in enumerate(config.series)]
    
    # Get global imputation method
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
        
        # Get imputation method
        series_impute = None
        if hasattr(series_config, 'impute'):
            series_impute = series_config.impute
        elif isinstance(series_config, dict) and 'impute' in series_config:
            series_impute = series_config['impute']
        
        if not series_impute or series_impute in ('null', ''):
            series_impute = global_impute or 'ffill_bfill'
        
        # Create imputation steps
        imputation_steps = []
        if series_impute == 'ffill_bfill':
            imputation_steps = [Imputer(method="ffill"), Imputer(method="bfill")]
        elif series_impute == 'ffill':
            imputation_steps = [Imputer(method="ffill")]
        elif series_impute == 'bfill':
            imputation_steps = [Imputer(method="bfill")]
        elif series_impute in ('naive', 'forecaster'):
            imputation_steps = [
                Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last")),
                Imputer(method="bfill")
            ]
        else:
            imputation_steps = [Imputer(method="ffill"), Imputer(method="bfill")]
        
        # Create transformation using sktime transformers
        if trans == 'lin':
            # Identity transformation - no change
            transformer = FunctionTransformer(func=_identity_transform)
        
        elif trans == 'log':
            # Log transformation - use LogTransformer if available, otherwise FunctionTransformer
            if LogTransformer is not None:
                transformer = LogTransformer()
            else:
                # Fallback: use FunctionTransformer with log function
                def _log_func(X):
                    if isinstance(X, pd.Series):
                        return pd.Series(
                            np.log(np.abs(X.values) + 1e-10), 
                            index=X.index, 
                            name=X.name
                        )
                    return np.log(np.abs(X) + 1e-10)
                transformer = FunctionTransformer(func=_log_func)
        
        elif trans == 'chg':
            # Change (difference) - use Differencer
            lag = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = Differencer(lags=lag)
        
        elif trans == 'ch1':
            # Year-over-year change - use Differencer
            lag = FREQ_TO_LAG_YOY.get(freq, 12)
            transformer = Differencer(lags=lag)
        
        elif trans == 'pch':
            # Percent change - custom function transformer
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = FunctionTransformer(
                func=lambda X: _pch_transform(X, step=step),
                inverse_func=None
            )
        
        elif trans == 'pc1':
            # Year-over-year percent change
            year_step = FREQ_TO_LAG_YOY.get(freq, 12)
            transformer = FunctionTransformer(
                func=lambda X: _pch_transform(X, step=year_step),
                inverse_func=None
            )
        
        elif trans == 'cha':
            # Change annualized - custom function transformer
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = _get_annual_factor(freq, step)
            transformer = FunctionTransformer(
                func=lambda X: _cha_transform(X, step=step, annual_factor=annual_factor),
                inverse_func=None
            )
        
        elif trans == 'pca':
            # Percent change annualized
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = _get_annual_factor(freq, step)
            # pca = pch * annual_factor
            transformer = FunctionTransformer(
                func=lambda X: _pch_transform(X, step=step) * annual_factor,
                inverse_func=None
            )
        
        else:
            raise ValidationError(
                f"Unknown transformation '{trans}' for series '{series_id}'. "
                f"Valid transformations: lin, log, chg, ch1, cha, pch, pc1, pca"
            )
        
        # Combine imputation and transformation
        if imputation_steps:
            steps: list = [(f"impute_{j}", step) for j, step in enumerate(imputation_steps)]
            steps.append(("transform", transformer))
            series_transformer = TransformerPipeline(steps)
        else:
            series_transformer = transformer
        
        transformers.append((series_id, series_transformer, i))
    
    # Create ColumnEnsembleTransformer
    column_transformer = ColumnEnsembleTransformer(transformers=transformers)
    
    # Create full pipeline: ColumnEnsembleTransformer → StandardScaler
    pipeline = TransformerPipeline([
        ("transform", column_transformer),
        ("scaler", StandardScaler())
    ])
    
    return pipeline
