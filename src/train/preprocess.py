"""Data preprocessing module for time series models."""

import logging
from typing import Optional, Tuple, Any, Dict
import numpy as np
import pandas as pd

from src.utils import ValidationError, SCALER_ROBUST, SCALER_STANDARD

_logger = logging.getLogger(__name__)

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


def impute_weekly_with_monthly_value(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing weekly values within each month using that month's value.
    
    For mixed frequency data: if there's at least 1 value in a month (e.g., JAN),
    that value (or mean of values) is imputed to all other missing weekly values
    in that month.
    
    Parameters
    ----------
    data : pd.DataFrame
        Weekly data with DatetimeIndex
        
    Returns
    -------
    pd.DataFrame
        Data with weekly values imputed within each month
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        _logger.warning("No numeric columns found for monthly imputation")
        return data
    
    result = data.copy()
    
    # Group by year-month and impute within each month
    for col in numeric_cols:
        # Create year-month groups
        year_month = result.index.to_period('M')
        
        for period in year_month.unique():
            # Get all indices for this month
            month_mask = year_month == period
            month_indices = result.index[month_mask]
            month_values = result.loc[month_indices, col]
            
            # Check if there's at least one non-null value in this month
            non_null_values = month_values.dropna()
            
            if len(non_null_values) > 0:
                # Use the first non-null value (or mean if multiple values exist)
                # User requested: "if there is 1 value in JAN, that value is imputed"
                # We'll use the first value, or mean if multiple exist
                if len(non_null_values) == 1:
                    fill_value = non_null_values.iloc[0]
                else:
                    # If multiple values exist, use mean (more robust)
                    fill_value = non_null_values.mean()
                
                # Fill missing values in this month with the fill_value
                null_mask = month_values.isna()
                if null_mask.any():
                    result.loc[month_indices[null_mask], col] = fill_value
                    _logger.debug(f"Imputed {null_mask.sum()} missing weekly values in {period} for {col} with value {fill_value:.4f}")
    
    return result


def resample_to_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """Resample weekly data to monthly using window averaging.
    
    Note: This function expects weekly data that has been preprocessed with
    impute_weekly_with_monthly_value() to fill missing weekly values within each month.
    """
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
    """Resample monthly data to weekly frequency for mixed-frequency DFM."""
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


def _set_frequency_from_config(data: pd.DataFrame, config_path: Optional[str] = None, series_ids: Optional[list] = None) -> pd.DataFrame:
    """Set frequency on DataFrame index from series config files."""
    import yaml
    from pathlib import Path
    
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    if data.index.freq is not None:
        return data
    
    # Get config path
    if config_path is None:
        from src.utils import get_config_path
        config_path = str(get_config_path())
    
    config_dir = Path(config_path)
    series_dir = config_dir / "series"
    
    # Get series IDs to process
    if series_ids is None:
        series_ids = list(data.columns)
    
    # Get frequency from series config (use most common frequency or 'w' for weekly)
    freq_from_config = None
    freq_counts = {}
    
    for series_id in series_ids:
        if series_id not in data.columns:
            continue
        
        # Try to load from series config file
        series_config_file = series_dir / f"{series_id}.yaml"
        if series_config_file.exists():
            try:
                with open(series_config_file, 'r') as f:
                    series_config = yaml.safe_load(f) or {}
                freq = str(series_config.get('frequency', 'm')).lower()
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
            except Exception:
                freq_counts['m'] = freq_counts.get('m', 0) + 1
        else:
            # Try metadata.csv
            from src.utils import get_project_root
            metadata_path = get_project_root() / "data" / "metadata.csv"
            if metadata_path.exists():
                try:
                    metadata = pd.read_csv(metadata_path)
                    meta_row = metadata[metadata['SeriesID'] == series_id]
                    if len(meta_row) > 0:
                        freq = str(meta_row.iloc[0].get('Frequency', 'm')).lower()
                        freq_counts[freq] = freq_counts.get(freq, 0) + 1
                    else:
                        freq_counts['m'] = freq_counts.get('m', 0) + 1
                except Exception:
                    freq_counts['m'] = freq_counts.get('m', 0) + 1
            else:
                freq_counts['m'] = freq_counts.get('m', 0) + 1
    
    # Use most common frequency, default to 'w' (weekly) for DFM
    if freq_counts:
        freq_from_config = max(freq_counts, key=freq_counts.get)
    else:
        freq_from_config = 'w'  # Default to weekly for DFM
    
    # Map frequency string to pandas frequency
    freq_map = {
        'w': 'W',
        'm': 'M',
        'q': 'Q',
        'y': 'Y',
        'd': 'D'
    }
    pandas_freq = freq_map.get(freq_from_config, 'W')
    
    # Set frequency on index
    try:
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            # Use inferred frequency if it matches config, otherwise use config
            if pandas_freq in inferred_freq or inferred_freq.startswith(pandas_freq):
                data = data.asfreq(inferred_freq, method='ffill')
            else:
                data = data.asfreq(pandas_freq, method='ffill')
        else:
            # Create new index with frequency
            data = data.asfreq(pandas_freq, method='ffill')
    except (TypeError, ValueError):
        # Fallback: try to set frequency directly
        try:
            data = data.asfreq(pandas_freq)
            data = data.ffill()
        except Exception:
            _logger.warning(f"Could not set frequency {pandas_freq} from config, using inferred frequency")
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                data = data.asfreq(inferred_freq, method='ffill')
    
    return data


def set_dataframe_frequency(y_train: pd.DataFrame) -> pd.DataFrame:
    """Set frequency on DataFrame index if DatetimeIndex (fallback: infer from index).
    
    For weekly data (detected by date_w index name or ~weekly spacing), sets W-SUN frequency.
    """
    if not isinstance(y_train.index, pd.DatetimeIndex):
        return y_train
    
    if y_train.index.freq is not None:
        return y_train
    
    # Check if this is weekly data (by index name or spacing)
    is_weekly = False
    if hasattr(y_train.index, 'name') and y_train.index.name == 'date_w':
        is_weekly = True
        _logger.debug("Detected weekly data from date_w index name")
    else:
        # Check spacing: if average spacing is ~7 days, it's weekly
        if len(y_train.index) > 1:
            avg_spacing = (y_train.index[-1] - y_train.index[0]).days / (len(y_train.index) - 1)
            if 6.5 <= avg_spacing <= 7.5:  # Approximately weekly
                is_weekly = True
                _logger.debug(f"Detected weekly data from spacing: {avg_spacing:.2f} days")
    
    inferred_freq = pd.infer_freq(y_train.index)
    if inferred_freq:
        try:
            y_train = y_train.asfreq(inferred_freq, method='ffill')
        except (TypeError, ValueError):
            # If inferred freq doesn't work, try W-SUN for weekly data
            if is_weekly:
                try:
                    y_train = y_train.asfreq('W-SUN', method='ffill')
                except (TypeError, ValueError):
                    # If that also fails, create new index with freq
                    try:
                        y_train.index = pd.DatetimeIndex(y_train.index, freq='W-SUN')
                    except:
                        # Last resort: just set freq attribute (may not work for all operations)
                        pass
            else:
                y_train = y_train.asfreq(inferred_freq)
                y_train = y_train.ffill()
    elif is_weekly:
        # No inferred freq but we know it's weekly, try W-SUN
        try:
            y_train = y_train.asfreq('W-SUN', method='ffill')
        except (TypeError, ValueError):
            # If asfreq fails, create new index with freq
            try:
                y_train.index = pd.DatetimeIndex(y_train.index, freq='W-SUN')
            except:
                # Last resort: try to set freq directly (may not work)
                pass
    
    return y_train


def apply_scaling(y_train: pd.DataFrame, scaler_type: str = SCALER_ROBUST) -> Tuple[pd.DataFrame, Any]:
    """Apply scaling to data."""
    if scaler_type is None or scaler_type == 'null':
        return y_train, None
    
    scaler_type_lower = scaler_type.lower() if scaler_type else SCALER_ROBUST
    
    if scaler_type_lower == SCALER_ROBUST:
        scaler = RobustScaler()
    elif scaler_type_lower == SCALER_STANDARD:
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
    """Impute missing values using forward-fill, backward-fill, and forecaster.
    
    Note: DataFrame index should have frequency set (via set_dataframe_frequency) before calling this.
    """
    if y_train.isnull().sum().sum() == 0:
        return y_train
    
    _logger.warning(f"{model_type.upper()} data contains NaN values. Applying imputation...")
    
    # Check if frequency is set (required by sktime)
    if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is None:
        _logger.warning("Index frequency not set. Attempting to infer...")
        inferred_freq = pd.infer_freq(y_train.index)
        if inferred_freq:
            # Create new index with frequency
            y_train.index = pd.DatetimeIndex(y_train.index, freq=inferred_freq)
        else:
            _logger.warning("Could not infer frequency. Using pandas ffill/bfill instead of sktime Imputer.")
            # Fallback to pandas methods if frequency can't be set
            y_train = y_train.fillna(method='ffill').fillna(method='bfill')
            return y_train
    
    imputer_ffill = Imputer(method="ffill")
    imputer_bfill = Imputer(method="bfill")
    imputer_forecaster = Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))
    
    for col in y_train.columns:
        # Convert to Series (sktime Imputer works better with Series)
        col_series = y_train[col]
        
        col_imputed = imputer_ffill.fit_transform(col_series)
        
        # Convert back to Series if DataFrame
        if isinstance(col_imputed, pd.DataFrame):
            if col in col_imputed.columns:
                col_imputed = col_imputed[col]
            else:
                col_imputed = col_imputed.iloc[:, 0]
        
        if col_imputed.isnull().sum() > 0:
            col_imputed = imputer_bfill.fit_transform(col_imputed)
            if isinstance(col_imputed, pd.DataFrame):
                if col in col_imputed.columns:
                    col_imputed = col_imputed[col]
                else:
                    col_imputed = col_imputed.iloc[:, 0]
        
        if col_imputed.isnull().sum() > 0:
            try:
                col_imputed = imputer_forecaster.fit_transform(col_imputed)
                if isinstance(col_imputed, pd.DataFrame):
                    if col in col_imputed.columns:
                        col_imputed = col_imputed[col]
                    else:
                        col_imputed = col_imputed.iloc[:, 0]
            except Exception as e:
                _logger.warning(f"Forecaster imputation failed for {col}: {e}")
        
        y_train[col] = col_imputed
    
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
    """Apply transformations based on series config files."""
    import yaml
    from pathlib import Path
    
    # Get config path
    if config_path is None:
        from src.utils import get_config_path
        config_path = str(get_config_path())
    
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
                trans_map[series_id] = str(series_config.get('transformation', 'lin')).lower()
                freq_map[series_id] = str(series_config.get('frequency', 'm')).lower()
            except Exception:
                trans_map[series_id] = 'lin'
                freq_map[series_id] = 'm'
        else:
            from src.utils import get_project_root
            metadata_path = get_project_root() / "data" / "metadata.csv"
            if metadata_path.exists():
                try:
                    metadata = pd.read_csv(metadata_path)
                    meta_row = metadata[metadata['SeriesID'] == series_id]
                    if len(meta_row) > 0:
                        trans_map[series_id] = str(meta_row.iloc[0].get('Transformation', 'lin')).lower()
                        freq_map[series_id] = str(meta_row.iloc[0].get('Frequency', 'm')).lower()
                    else:
                        trans_map[series_id] = 'lin'
                        freq_map[series_id] = 'm'
                except Exception:
                    trans_map[series_id] = 'lin'
                    freq_map[series_id] = 'm'
            else:
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
            lag = 1
            series = result_data[col].dropna()
            if len(series) > lag:
                result_data[col] = series.diff(periods=lag)
        elif trans == 'ch1':
            lag = 52 if freq == 'w' else (12 if freq == 'm' else (4 if freq == 'q' else 1))
            series = result_data[col].dropna()
            if len(series) > lag:
                result_data[col] = series.diff(periods=lag)
        elif trans == 'log':
            series = result_data[col].dropna()
            if len(series) > 0 and (series > 0).all():
                result_data[col] = np.log(series)
        elif trans == 'pch':
            series = result_data[col].dropna()
            if len(series) > 1:
                result_data[col] = series.pct_change(periods=1) * 100
    
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
    from src.utils import get_config_path
    config_path = str(get_config_path())
    
    # Unified preprocessing: all models use weekly data (no resampling)
    
    # Normalize index to avoid duplicate labels
    data = data.copy()
    if 'date_w' in data.columns:
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
        
        # Check if original data is weekly BEFORE transformations
        is_weekly_data = False
        if isinstance(selected_data.index, pd.DatetimeIndex):
            original_freq = pd.infer_freq(selected_data.index)
            if original_freq and ('W' in original_freq or original_freq.startswith('W')):
                is_weekly_data = True
        
        # For mixed frequency data: impute missing weekly values within each month
        # BEFORE transformations (so transformations have complete data to work with)
        if is_weekly_data and isinstance(selected_data.index, pd.DatetimeIndex):
            selected_data = impute_weekly_with_monthly_value(selected_data)
        
        # NOTE: For DDFM, apply transformations here (matching previous implementation)
        # Previous implementation: transformations were applied at src/ level, then manually inverse transformed after prediction
        # Current: transformations are applied here, preprocessing pipeline only handles standardization
        # This prevents double differencing while maintaining the previous working approach
        # For DFM, transformations can be handled by pipeline or here (configurable)
        # For other models (VAR, etc.), apply transformations here as before
        if model_type.lower() == 'ddfm':
            # For DDFM: Apply transformations here (matching previous implementation)
            # Preprocessing pipeline will only handle standardization (no transformations)
            selected_data = apply_transformations(selected_data, config_path=config_path, series_ids=available_series)
        elif model_type.lower() != 'dfm':
            # For non-DFM/DDFM models, apply transformations here as before
            selected_data = apply_transformations(selected_data, config_path=config_path, series_ids=available_series)
        # For DFM: transformations can be handled by preprocessing pipeline (optional)
        
        # Unified preprocessing: all models use weekly data (no resampling)
        processed_data = selected_data
        # Set frequency from series config BEFORE imputation (required by sktime)
        processed_data = _set_frequency_from_config(processed_data, config_path=config_path, series_ids=available_series)
        # Final imputation for any remaining NaNs (unified preprocessing step)
        if processed_data.isnull().sum().sum() > 0:
            processed_data = impute_missing_values(processed_data, model_type=model_type)
        return processed_data, available_series
    else:
        # Use all numeric columns
        y_train = data.select_dtypes(include=[np.number])
        if target_series and target_series in data.columns and target_series not in y_train.columns:
            y_train[target_series] = data[target_series]
        
        y_train = y_train.dropna(how='all')
        
        # Check if original data is weekly BEFORE transformations
        is_weekly_data = False
        if isinstance(y_train.index, pd.DatetimeIndex):
            original_freq = pd.infer_freq(y_train.index)
            if original_freq and ('W' in original_freq or original_freq.startswith('W')):
                is_weekly_data = True
        
        # For mixed frequency data: impute missing weekly values within each month
        # BEFORE transformations (so transformations have complete data to work with)
        if is_weekly_data and isinstance(y_train.index, pd.DatetimeIndex):
            y_train = impute_weekly_with_monthly_value(y_train)
        
        # NOTE: For DDFM, apply transformations here (matching previous implementation)
        # Previous implementation: transformations were applied at src/ level, then manually inverse transformed after prediction
        # Current: transformations are applied here, preprocessing pipeline only handles standardization
        # This prevents double differencing while maintaining the previous working approach
        # For DFM, transformations can be handled by pipeline or here (configurable)
        # For other models (VAR, etc.), apply transformations here as before
        if model_type.lower() == 'ddfm':
            # For DDFM: Apply transformations here (matching previous implementation)
            # Preprocessing pipeline will only handle standardization (no transformations)
            y_train = apply_transformations(y_train, config_path=config_path, series_ids=list(y_train.columns))
        elif model_type.lower() != 'dfm':
            # For non-DFM/DDFM models, apply transformations here as before
            y_train = apply_transformations(y_train, config_path=config_path, series_ids=list(y_train.columns))
        # For DFM: transformations can be handled by preprocessing pipeline (optional)
        
        # Unified preprocessing: all models use weekly data (no resampling)
        processed_data = y_train
        # Set frequency from series config BEFORE imputation (required by sktime)
        processed_data = _set_frequency_from_config(processed_data, config_path=config_path, series_ids=list(y_train.columns))
        # Final imputation for any remaining NaNs (unified preprocessing step)
        if processed_data.isnull().sum().sum() > 0:
            processed_data = impute_missing_values(processed_data, model_type=model_type)
        available_series = list(processed_data.columns)
        return processed_data, available_series


def prepare_univariate_data(data: pd.DataFrame, target_series: str, config_path: Optional[str] = None) -> pd.Series:
    """Prepare univariate training data for models.
    
    All models now train on weekly data. Forecasts are made weekly and then
    aggregated to monthly for evaluation against monthly targets.
    
    For mixed-frequency data (weekly data with monthly index), uses date_w column
    as index to preserve weekly frequency, similar to prepare_multivariate_data.
    """
    from pathlib import Path
    
    if target_series not in data.columns:
        raise ValidationError(f"Target series '{target_series}' not found in data columns")
    
    # Get config path for series config files
    if config_path is None:
        from src.utils import get_config_path
        config_path = str(get_config_path())
    
    # Normalize index to avoid duplicate labels (similar to prepare_multivariate_data)
    # Use date_w column if available to preserve weekly frequency
    y_train = data[[target_series]].copy()
    if 'date_w' in data.columns:
        # Use date_w as index to preserve weekly frequency
        idx = pd.to_datetime(data['date_w'], errors='coerce')
        y_train.index = idx
        y_train = y_train[~y_train.index.isna()]
        if y_train.index.has_duplicates:
            y_train = y_train[~y_train.index.duplicated(keep='last')]
        # Set weekly frequency explicitly using asfreq
        if isinstance(y_train.index, pd.DatetimeIndex):
            # Try to infer freq, but default to W-SUN for weekly data
            inferred_freq = pd.infer_freq(y_train.index)
            if inferred_freq and ('W' in inferred_freq or inferred_freq.startswith('W')):
                try:
                    y_train = y_train.asfreq(inferred_freq, method='ffill')
                except (TypeError, ValueError):
                    # If asfreq fails, try W-SUN
                    try:
                        y_train = y_train.asfreq('W-SUN', method='ffill')
                    except (TypeError, ValueError):
                        _logger.warning("Could not set weekly frequency, keeping as-is")
            else:
                # Default to W-SUN for weekly data
                try:
                    y_train = y_train.asfreq('W-SUN', method='ffill')
                except (TypeError, ValueError):
                    _logger.warning("Could not set W-SUN frequency, keeping as-is")
        _logger.info(f"Using date_w column as index to preserve weekly frequency (length: {len(y_train)}, freq: {y_train.index.freq if hasattr(y_train.index, 'freq') else 'None'})")
    elif isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.has_duplicates:
        # If index has duplicates but no date_w, try to infer weekly dates
        # This handles cases where data is weekly but indexed by month-end dates
        _logger.warning(f"Index has duplicates but no date_w column. Attempting to preserve weekly structure...")
        # Keep all rows (don't drop duplicates) to preserve weekly data
        # The duplicates indicate multiple weekly observations per month
    
    # Check if original data is weekly BEFORE transformations
    # (transformations might introduce NaNs that break frequency inference)
    is_weekly_data = False
    if isinstance(y_train.index, pd.DatetimeIndex):
        original_freq = pd.infer_freq(y_train.index)
        if original_freq and ('W' in original_freq or original_freq.startswith('W')):
            is_weekly_data = True
        elif y_train.index.has_duplicates:
            # If index has duplicates, it's likely weekly data indexed by month-end
            # Check if we have multiple observations per unique date
            unique_dates = y_train.index.nunique()
            if len(y_train) > unique_dates * 3:  # Rough heuristic: more than 3x unique dates suggests weekly
                is_weekly_data = True
                _logger.info(f"Detected weekly data structure: {len(y_train)} observations, {unique_dates} unique dates")
    
    # For mixed frequency data: impute missing weekly values within each month
    # BEFORE transformations (so transformations have complete data to work with)
    if is_weekly_data and isinstance(y_train.index, pd.DatetimeIndex):
        y_train = impute_weekly_with_monthly_value(y_train)
    
    # Apply transformations from series config files (at original series frequency)
    y_train = apply_transformations(y_train, config_path=config_path, series_ids=[target_series])
    
    # Keep weekly data (no resampling to monthly)
    # IMPORTANT: Don't call _set_frequency_from_config here as it may resample to monthly
    # Instead, use set_dataframe_frequency which preserves the existing structure
    # Final imputation for any remaining NaNs (unified preprocessing step)
    if y_train.isnull().sum().sum() > 0:
        y_train = impute_missing_values(y_train, model_type='univariate')
    
    # Ensure index is sorted (required by sktime)
    y_train = y_train.sort_index()
    
    # Set frequency
    y_train = set_dataframe_frequency(y_train)
    
    # If freq is still None but we have date_w index, try to set it explicitly
    if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is None:
        if hasattr(y_train.index, 'name') and y_train.index.name == 'date_w':
            # Try to create a new index with W-SUN freq
            try:
                # Check if dates are approximately weekly spaced
                if len(y_train.index) > 1:
                    avg_spacing = (y_train.index[-1] - y_train.index[0]).days / (len(y_train.index) - 1)
                    if 6.5 <= avg_spacing <= 7.5:
                        # Create new DatetimeIndex with freq (this may fail if dates don't align perfectly)
                        # Instead, just ensure it's sorted and let sktime handle it
                        pass
            except:
                pass
    
    return y_train.iloc[:, 0]



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
        
        # Create transformation
        if trans == 'lin':
            transformer = FunctionTransformer(func=_identity_transform)
        elif trans == 'log':
            if LogTransformer is not None:
                transformer = LogTransformer()
            else:
                def _log_func(X):
                    if isinstance(X, pd.Series):
                        return pd.Series(np.log(np.abs(X.values) + 1e-10), index=X.index, name=X.name)
                    return np.log(np.abs(X) + 1e-10)
                transformer = FunctionTransformer(func=_log_func)
        elif trans == 'chg':
            transformer = Differencer(lags=FREQ_TO_LAG_STEP.get(freq, 1))
        elif trans == 'ch1':
            transformer = Differencer(lags=FREQ_TO_LAG_YOY.get(freq, 12))
        elif trans == 'pch':
            transformer = FunctionTransformer(func=lambda X: _pch_transform(X, step=FREQ_TO_LAG_STEP.get(freq, 1)), inverse_func=None)
        elif trans == 'pc1':
            transformer = FunctionTransformer(func=lambda X: _pch_transform(X, step=FREQ_TO_LAG_YOY.get(freq, 12)), inverse_func=None)
        elif trans == 'cha':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = FunctionTransformer(func=lambda X: _cha_transform(X, step=step, annual_factor=_get_annual_factor(freq, step)), inverse_func=None)
        elif trans == 'pca':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            transformer = FunctionTransformer(func=lambda X: _pch_transform(X, step=step) * _get_annual_factor(freq, step), inverse_func=None)
        else:
            raise ValidationError(f"Unknown transformation '{trans}' for series '{series_id}'")
        
        if imputation_steps:
            steps = [(f"impute_{j}", step) for j, step in enumerate(imputation_steps)]
            steps.append(("transform", transformer))
            series_transformer = TransformerPipeline(steps)
        else:
            series_transformer = transformer
        
        transformers.append((series_id, series_transformer, i))
    
    column_transformer = ColumnEnsembleTransformer(transformers=transformers)
    return TransformerPipeline([("transform", column_transformer), ("scaler", StandardScaler())])
