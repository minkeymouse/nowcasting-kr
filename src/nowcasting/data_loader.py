"""Data loading, specification parsing, and transformation functions."""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import date
import warnings
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)

try:
    from scipy.io import loadmat, savemat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

# MATLAB datenum offset: datenum('1970-01-01') = 719529
_MATLAB_DATENUM_OFFSET = 719529


def load_config_from_yaml(configfile: Union[str, Path]) -> ModelConfig:
    """Load model configuration from YAML file.
    
    Parameters
    ----------
    configfile : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    ModelConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist
    ImportError
        If omegaconf is not available
    ValueError
        If configuration is invalid
    """
    if not OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    cfg = OmegaConf.load(configfile)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Handle nested structure (if config has @package model: directive)
    if 'series' in cfg_dict and 'block_names' in cfg_dict:
        # Direct model config structure
        return ModelConfig.from_dict(cfg_dict)
    elif 'model' in cfg_dict:
        # Nested under 'model' key (from @package model:)
        return ModelConfig.from_dict(cfg_dict['model'])
    else:
        # Try to construct from top-level keys
        return ModelConfig.from_dict(cfg_dict)


def load_config_from_csv(configfile: Union[str, Path]) -> ModelConfig:
    """Load model configuration from CSV file.
    
    CSV format should have columns:
    - series_id, series_name, frequency, transformation, category, units
    - Block columns (named after block names, e.g., Global, Consumption, Investment, External)
    - Block columns contain 0 or 1 (1 = series loads on that block)
    
    Example:
        series_id,series_name,frequency,transformation,category,units,Global,Consumption,Investment,External
        BOK_200Y001,GDP Growth Rate,q,pch,GDP,PCT,1,0,0,0
        BOK_200Y002,Consumption,q,pch,Consumption,PCT,1,1,0,0
    
    Parameters
    ----------
    configfile : str or Path
        Path to CSV specification file
        
    Returns
    -------
    ModelConfig
        Model configuration object
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist
    ValueError
        If required columns are missing or data is invalid
    """
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    try:
        df = pd.read_csv(configfile)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {configfile}: {e}")
    
    # Required fields
    required_fields = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
    optional_fields = ['api_code', 'api_source']  # Optional but useful for database agent
    
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Detect block columns (all columns that are not in required_fields or optional_fields)
    all_columns = set(df.columns)
    excluded_fields = set(required_fields) | set(optional_fields)
    block_columns = sorted([col for col in all_columns if col not in excluded_fields])
    
    if not block_columns:
        raise ValueError("No block columns found. Expected columns like 'Global', 'Consumption', etc.")
    
    # Validate block columns contain only 0 or 1
    for block_col in block_columns:
        if not df[block_col].isin([0, 1]).all():
            raise ValueError(f"Block column '{block_col}' must contain only 0 or 1")
    
    # Ensure all series load on at least one block (first block should always be 1)
    if block_columns[0] not in df.columns:
        raise ValueError(f"First block column '{block_columns[0]}' is required")
    
    if not (df[block_columns[0]] == 1).all():
        raise ValueError(f"All series must load on the first block '{block_columns[0]}' (Global)")
    
    # Build blocks array (N x n_blocks)
    blocks_data = df[block_columns].values.astype(int)
    
    # Convert to ModelConfig format
    from .config import SeriesConfig
    
    series_list = []
    for idx, row in df.iterrows():
        # Build block array from block columns
        blocks = [int(row[col]) for col in block_columns]
        
        series_list.append(SeriesConfig(
            series_id=row['series_id'],
            series_name=row['series_name'],
            frequency=row['frequency'],
            transformation=row['transformation'],
            category=row['category'],
            units=row['units'],
            blocks=blocks,
            api_code=row.get('api_code'),  # Optional field
            api_source=row.get('api_source')  # Optional field
        ))
    
    return ModelConfig(series=series_list, block_names=block_columns)


# Note: Excel support removed - use CSV or YAML configs instead


def load_config(configfile: Union[str, Path]) -> ModelConfig:
    """Load model configuration from file (auto-detects YAML or CSV).
    
    Parameters
    ----------
    configfile : str or Path
        Path to configuration file (.yaml, .yml, or .csv)
        
    Returns
    -------
    ModelConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist
    ValueError
        If file format is not supported or configuration is invalid
    """
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    suffix = configfile.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile)
    elif suffix == '.csv':
        return load_config_from_csv(configfile)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .csv")


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Apply transformation to a single series."""
    T = Z.shape[0]
    X = np.full(T, np.nan)
    t1 = step
    n = step / 12
    
    if formula == 'lin':
        X[:] = Z
    elif formula == 'chg':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = Z[idx[1:]] - Z[idx[:-1]]
    elif formula == 'ch1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            X[idx] = Z[idx] - Z[idx - 12]
    elif formula == 'pch':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = 100 * (Z[idx[1:]] / Z[idx[:-1]] - 1)
    elif formula == 'pc1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx] = 100 * (Z[idx] / Z[idx - 12] - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'pca':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx[1:]] = 100 * ((Z[idx[1:]] / Z[idx[:-1]]) ** (1/n) - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'log':
        with np.errstate(invalid='ignore'):
            X[:] = np.log(Z)
    else:
        X[:] = Z
    
    return X


def transform_data(Z: np.ndarray, Time: pd.DatetimeIndex, config: ModelConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Transform each data series based on specification."""
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    for i in range(N):
        step = 3 if config.Frequency[i] == 'q' else 1
        X[:, i] = _transform_series(Z[:, i], config.Transformation[i], config.Frequency[i], step)
    
    # Drop first quarter of observations (4 months) since transformations cause missing values
    # Note: MATLAB drops 4 rows (Time(4:end)), not 12
    drop = 4
    if T > drop:
        return X[drop:], Time[drop:], Z[drop:]
    return X, Time, Z


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Read data from CSV file.
    
    CSV file should have:
    - First column: Date (YYYY-MM-DD format or pandas parseable)
    - Subsequent columns: Series data (one column per series)
    - Header row: Series IDs (matching config.SeriesID)
    """
    datafile = Path(datafile)
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read CSV file
    try:
        df = pd.read_csv(datafile, index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {datafile}: {e}")
    
    mnemonics = df.columns.tolist()
    Time = df.index
    Z = df.values.astype(float)
    
    return Z, Time, mnemonics


def sort_data(Z: np.ndarray, Mnem: List[str], config: ModelConfig) -> Tuple[np.ndarray, List[str]]:
    """Sort series to match configuration order."""
    in_config = [m in config.SeriesID for m in Mnem]
    Mnem_filt = [m for m, in_c in zip(Mnem, in_config) if in_c]
    Z_filt = Z[:, in_config]
    
    perm = [Mnem_filt.index(sid) for sid in config.SeriesID]
    return Z_filt[:, perm], [Mnem_filt[i] for i in perm]


def _datenum_to_pandas(Time_mat: np.ndarray) -> pd.DatetimeIndex:
    """Convert MATLAB datenum to pandas DatetimeIndex.
    
    MATLAB datenum: days since 0000-01-01
    Conversion: Time_pandas = 1970-01-01 + (Time_mat - 719529) days
    """
    if Time_mat.ndim > 1:
        Time_mat = Time_mat.flatten()
    return pd.Timestamp('1970-01-01') + pd.to_timedelta(Time_mat - _MATLAB_DATENUM_OFFSET, unit='D')


def _pandas_to_datenum(Time: pd.DatetimeIndex) -> np.ndarray:
    """Convert pandas DatetimeIndex to MATLAB datenum format.
    
    Conversion: Time_mat = (Time_pandas - 1970-01-01).days + 719529
    """
    if not isinstance(Time, pd.DatetimeIndex):
        Time = pd.to_datetime(Time)
    return (Time - pd.Timestamp('1970-01-01')).days.values.astype(float) + _MATLAB_DATENUM_OFFSET


def _extract_matlab_variables(mat_data: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Extract Z, Time, Mnem from MATLAB .mat file data.
    
    Returns (Z, Time_mat, Mnem) or (None, None, None) if not found.
    """
    Z = mat_data.get('Z')
    Time_mat = mat_data.get('Time')
    Mnem = mat_data.get('Mnem')
    
    # Try alternative keys if not found (scipy sometimes adds prefixes)
    if Z is None or Time_mat is None or Mnem is None:
        for key in mat_data.keys():
            if key.startswith('__'):
                continue  # Skip metadata
            key_upper = key.upper()
            if Z is None and 'Z' in key_upper and 'TIME' not in key_upper and 'MNEM' not in key_upper:
                Z = mat_data[key]
            elif Time_mat is None and 'TIME' in key_upper:
                Time_mat = mat_data[key]
            elif Mnem is None and 'MNEM' in key_upper:
                Mnem = mat_data[key]
    
    # Convert Mnem from MATLAB format to Python list
    if Mnem is not None:
        if isinstance(Mnem, np.ndarray):
            Mnem = [str(item[0]) if isinstance(item, np.ndarray) and item.size > 0 
                   else str(item) for item in Mnem.flatten()]
        elif not isinstance(Mnem, list):
            Mnem = [str(Mnem)]
    
    return Z, Time_mat, Mnem


def _load_from_mat_cache(datafile_mat: Path) -> Optional[Tuple[np.ndarray, pd.DatetimeIndex, List[str]]]:
    """Load data from cached .mat file.
    
    Returns (Z, Time, Mnem) if successful, None otherwise.
    """
    if not SCIPY_AVAILABLE or not datafile_mat.exists():
        return None
    
    try:
        mat_data = loadmat(str(datafile_mat))
        Z, Time_mat, Mnem = _extract_matlab_variables(mat_data)
        
        if Z is None or Time_mat is None or Mnem is None:
            return None
        
        # Convert MATLAB datenum to pandas DatetimeIndex
        Time = _datenum_to_pandas(Time_mat)
        
        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        print(f'  Loaded from cached .mat file: {datafile_mat}')
        return Z, Time, Mnem
    except Exception as e:
        warnings.warn(f"Failed to load from .mat cache ({e}). Reading from CSV instead.")
        return None


def _save_to_mat_cache(Z: np.ndarray, Time: pd.DatetimeIndex, Mnem: List[str], 
                       mat_dir: Path, datafile_mat: Path) -> None:
    """Save data to .mat cache file."""
    if not SCIPY_AVAILABLE:
        return
    
    try:
        mat_dir.mkdir(exist_ok=True)
        Time_mat = _pandas_to_datenum(Time)
        Mnem_array = np.array([m.encode('utf-8') if isinstance(m, str) else m 
                             for m in Mnem], dtype=object)
        savemat(str(datafile_mat), {'Z': Z, 'Time': Time_mat, 'Mnem': Mnem_array})
        print(f'  Cached to .mat file: {datafile_mat}')
    except Exception as e:
        warnings.warn(f"Failed to save .mat cache ({e}). Continuing without cache.")


def load_data(datafile: Union[str, Path], config: ModelConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load and transform data from CSV file.
    
    This function reads data from a CSV file, sorts it to match the model
    configuration (from CSV or YAML), and applies the specified transformations.
    
    Note: For production use, prefer `load_data_from_db()` which loads data
    directly from the database.
    
    Parameters
    ----------
    datafile : str or Path
        Path to CSV data file (.csv)
    config : ModelConfig
        Model configuration object (from CSV or YAML)
    sample_start : pd.Timestamp or str, optional
        Start date for sample (YYYY-MM-DD). If None, uses all available data.
        Data before this date will be dropped.
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N), ready for DFM estimation
    Time : pd.DatetimeIndex
        Time index for the data
    Z : np.ndarray
        Original untransformed data (T x N), for reference
    """
    print('Loading data...')
    
    datafile = Path(datafile)
    if datafile.suffix.lower() != '.csv':
        raise ValueError('Only CSV files supported. Use database for production data loading.')
    
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Create path for cached .mat file (matching MATLAB behavior)
    mat_dir = datafile.parent / 'mat'
    datafile_mat = mat_dir / f'{datafile.stem}.mat'
    
    # Try to load from cache first
    cache_data = _load_from_mat_cache(datafile_mat)
    
    # Read from CSV if cache load failed
    if cache_data is None:
        Z, Time, Mnem = read_data(datafile)
        _save_to_mat_cache(Z, Time, Mnem, mat_dir, datafile_mat)
    else:
        Z, Time, Mnem = cache_data
    
    # Process data: sort to match config order, then transform
    Z, _ = sort_data(Z, Mnem, config)
    X, Time, Z = transform_data(Z, Time, config)
    
    # Apply sample_start filtering
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.to_datetime(sample_start)
        mask = Time >= sample_start
        Time, X, Z = Time[mask], X[mask], Z[mask]
    
    return X, Time, Z


def _get_db_client(client: Optional[object] = None):
    """Get database client, raising ImportError if database module unavailable."""
    if client is not None:
        return client
    try:
        from database import get_client
        return get_client()
    except ImportError as e:
        raise ImportError(
            "Database module not available. Install dependencies or use file-based loading."
        ) from e


def _normalize_date(date_input: Optional[Union[date, str, pd.Timestamp]]) -> Optional[date]:
    """Normalize date input to date object."""
    if date_input is None:
        return None
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        return pd.to_datetime(date_input).date()
    if isinstance(date_input, pd.Timestamp):
        return date_input.date()
    return None


def _resolve_vintage_id(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    client: Optional[object] = None
) -> Optional[int]:
    """Resolve vintage_id from vintage_date if needed."""
    if vintage_id is not None:
        return vintage_id
    
    if vintage_date is not None:
        try:
            from database import get_latest_vintage_id
            normalized_date = _normalize_date(vintage_date)
            resolved_id = get_latest_vintage_id(vintage_date=normalized_date, client=client)
            if resolved_id is None:
                logger.warning(f"No vintage found for date {normalized_date}")
            return resolved_id
        except ImportError:
            logger.error("Database module not available. Cannot resolve vintage_id from date.")
            return None
        except Exception as e:
            logger.error(f"Error resolving vintage_id from date {vintage_date}: {e}")
            return None
    
    return None


def _apply_transformations_from_metadata(
    Z: np.ndarray,
    Time: pd.DatetimeIndex,
    data_df: pd.DataFrame,
    series_metadata_df: pd.DataFrame,
    config: Optional[ModelConfig] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Apply transformations using series metadata, falling back to config if needed.
    
    This function consolidates transformation logic to work with both database
    metadata and ModelConfig objects (from CSV or YAML).
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Build metadata lookup from database
    metadata_dict = {}
    if not series_metadata_df.empty and 'series_id' in series_metadata_df.columns:
        for _, row in series_metadata_df.iterrows():
            series_id = row['series_id']
            metadata_dict[series_id] = {
                'transformation': row.get('transformation', 'lin'),
                'frequency': row.get('frequency', 'm')
            }
    
    # Apply transformations using metadata or config
    if config is not None:
        # Use config (works with CSV or YAML configs)
        for i in range(N):
            if i < len(config.SeriesID):
                series_id = config.SeriesID[i]
                # Prefer metadata if available, otherwise use config
                if series_id in metadata_dict:
                    transform = metadata_dict[series_id]['transformation']
                    freq = metadata_dict[series_id]['frequency']
                else:
                    transform = config.Transformation[i]
                    freq = config.Frequency[i]
            else:
                # Fallback for series not in config
                transform = 'lin'
                freq = 'm'
            
            step = 3 if freq == 'q' else 1
            X[:, i] = _transform_series(Z[:, i], transform, freq, step)
    else:
        # Use metadata only (no config available)
        for i, series_id in enumerate(data_df.columns):
            if series_id in metadata_dict:
                transform = metadata_dict[series_id]['transformation']
                freq = metadata_dict[series_id]['frequency']
            else:
                transform = 'lin'
                freq = 'm'
            
            step = 3 if freq == 'q' else 1
            X[:, i] = _transform_series(Z[:, i], transform, freq, step)
    
    # Drop first 4 observations (transformations cause missing values)
    drop = 4
    if T > drop:
        return X[drop:], Time[drop:]
    return X, Time


def _handle_missing_series(
    data_df: pd.DataFrame,
    expected_series: List[str],
    vintage_id: int,
    strict_mode: bool
) -> pd.DataFrame:
    """Handle missing series by filling with NaN or raising error."""
    actual_series = set(data_df.columns)
    missing_series = set(expected_series) - actual_series
    
    if missing_series:
        if strict_mode:
            raise ValueError(
                f"Missing series in vintage {vintage_id}: {sorted(missing_series)}"
            )
        logger.warning(
            f"Missing series in vintage {vintage_id}, filling with NaN: "
            f"{sorted(missing_series)}"
        )
        # Add NaN columns for missing series
        for series_id in missing_series:
            data_df[series_id] = np.nan
    
    # Reorder columns to match expected order
    ordered_series = [s for s in expected_series if s in data_df.columns]
    if len(ordered_series) != len(expected_series):
        logger.warning(
            f"Series order mismatch: expected {len(expected_series)}, got {len(ordered_series)}"
        )
    
    return data_df[ordered_series] if ordered_series else data_df


def _fetch_vintage_data(
    vintage_id: int,
    config_id: Optional[int],
    config_series_ids: Optional[List[str]],
    start_date: Optional[date],
    end_date: Optional[date],
    strict_mode: bool,
    client: object
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """Fetch vintage data and metadata from database.
    
    Uses config-specific function if available, otherwise falls back to general function.
    """
    try:
        from database import (
            get_vintage_data,
            get_vintage_data_for_config,
            get_series_metadata_bulk
        )
    except ImportError:
        raise
    
    # Prefer config-specific function (uses model_block_assignments for ordering)
    if config_id is not None:
        try:
            return get_vintage_data_for_config(
                vintage_id=vintage_id,
                config_id=config_id,
                start_date=start_date,
                end_date=end_date,
                strict_mode=strict_mode,
                client=client
            )
        except (TypeError, AttributeError) as e:
            # Fallback if function signature differs or not available
            logger.warning(f"get_vintage_data_for_config failed ({e}), using general function")
    
    # Use general function with series IDs
    result = get_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        start_date=start_date,
        end_date=end_date,
        client=client
    )
    
    # Handle both return signatures (2-tuple or 3-tuple)
    if len(result) == 3:
        return result  # (data_df, time_index, series_metadata_df)
    else:
        data_df, time_index = result
        # Fetch metadata separately if needed
        series_metadata_df = (
            get_series_metadata_bulk(config_series_ids, client=client)
            if config_series_ids else pd.DataFrame()
        )
        return data_df, time_index, series_metadata_df


def load_data_from_db(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    config: Optional[ModelConfig] = None,
    config_id: Optional[int] = None,
    config_series_ids: Optional[List[str]] = None,
    start_date: Optional[Union[date, str, pd.Timestamp]] = None,
    end_date: Optional[Union[date, str, pd.Timestamp]] = None,
    sample_start: Optional[Union[pd.Timestamp, str]] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load vintage data from database and format for DFM.
    
    This function loads data from Supabase database and converts it to the format
    expected by DFM functions (numpy arrays). It applies transformations based on
    series metadata from the database.
    
    Parameters
    ----------
    vintage_id : int, optional
        Vintage ID (if None, uses vintage_date or latest vintage)
    vintage_date : date, str, or Timestamp, optional
        Vintage date (alternative to vintage_id)
    config : ModelConfig, optional
        Model configuration object (required for transformations)
    config_id : int, optional
        Model configuration ID in database (uses model_block_assignments for ordering)
    config_series_ids : List[str], optional
        List of series IDs in exact order (if None and config provided, uses config.SeriesID)
    start_date : date, str, or Timestamp, optional
        Start date for data filtering
    end_date : date, str, or Timestamp, optional
        End date for data filtering
    sample_start : Timestamp or str, optional
        Start date for estimation sample (data before this will be dropped)
    strict_mode : bool, default False
        If True, raise error for missing series. If False, fill with NaN and log warnings.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : pd.DatetimeIndex
        Time index for observations
    Z : np.ndarray
        Raw (untransformed) data matrix (T x N)
        
    Raises
    ------
    ValueError
        If config is required but not provided, or if strict_mode=True and series are missing
    ImportError
        If database module is not available
    """
    # Get database client
    client = _get_db_client(client)
    
    # Resolve vintage_id
    resolved_vintage_id = _resolve_vintage_id(vintage_id, vintage_date, client)
    if resolved_vintage_id is None:
        raise ValueError("Must provide either vintage_id or vintage_date")
    
    # Normalize date inputs
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    
    # Determine series IDs
    if config is not None and config_series_ids is None:
        config_series_ids = config.SeriesID
    
    # Fetch data from database
    try:
        data_df, time_index, series_metadata_df = _fetch_vintage_data(
            vintage_id=resolved_vintage_id,
            config_id=config_id,
            config_series_ids=config_series_ids,
            start_date=normalized_start,
            end_date=normalized_end,
            strict_mode=strict_mode,
            client=client
        )
    except ImportError as e:
        raise ImportError(
            "Database module not available. Install dependencies or use file-based loading."
        ) from e
    except Exception as e:
        logger.error(f"Error loading vintage data: {e}")
        raise
    
    if data_df.empty:
        logger.warning(f"Empty vintage data for vintage_id {resolved_vintage_id}")
        empty_shape = (0, 0)
        return (np.array([]).reshape(empty_shape), pd.DatetimeIndex([]),
                np.array([]).reshape(empty_shape))
    
    # Handle missing series and reorder if config provided
    if config is not None:
        data_df = _handle_missing_series(
            data_df, config.SeriesID, resolved_vintage_id, strict_mode
        )
        # Reorder metadata to match data_df columns
        if not series_metadata_df.empty and 'series_id' in series_metadata_df.columns:
            series_metadata_df = series_metadata_df.set_index('series_id')
            series_metadata_df = series_metadata_df.reindex(data_df.columns)
            series_metadata_df = series_metadata_df.reset_index()
    
    # Convert to numpy array
    Z = data_df.values.astype(float)
    
    # Apply transformations using config (CSV or YAML) with metadata fallback
    if config is not None:
        X, Time = _apply_transformations_from_metadata(
            Z, time_index, data_df, series_metadata_df, config
        )
        # Adjust Z to match X length (drop first 4 observations)
        drop = 4
        if len(Z) > drop:
            Z = Z[drop:]
    else:
        # No config available - use raw data without transformations
        logger.warning("No config provided - skipping transformations. Results may be incorrect.")
        X = Z.copy()
        Time = time_index
    
    # Apply sample_start filtering
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.to_datetime(sample_start)
        mask = Time >= sample_start
        Time = Time[mask]
        X = X[mask]
        Z = Z[mask]
    
    logger.info(
        f"Loaded data from database: vintage_id={resolved_vintage_id}, "
        f"shape={X.shape}, time_range=[{Time[0] if len(Time) > 0 else 'N/A'}, "
        f"{Time[-1] if len(Time) > 0 else 'N/A'}]"
    )
    
    return X, Time, Z


# Removed: load_config_from_db() and related helpers
# Model configs are now loaded from Hydra YAML files, not database
