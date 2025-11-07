"""Data loading, specification parsing, and transformation functions."""

import numpy as np
import pandas as pd
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import date
import warnings
import logging

from .config import DFMConfig
# Backward compatibility
ModelConfig = DFMConfig

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


def load_config_from_yaml(configfile: Union[str, Path]) -> DFMConfig:
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
        return DFMConfig.from_dict(cfg_dict)
    elif 'model' in cfg_dict:
        # Nested under 'model' key (from @package model:)
        return DFMConfig.from_dict(cfg_dict['model'])
    else:
        # Try to construct from top-level keys
        return DFMConfig.from_dict(cfg_dict)


def load_config_from_csv(configfile: Union[str, Path]) -> DFMConfig:
    """Load model configuration from CSV file.
    
    CSV format should have columns:
    - series_id (or 'id' as alias), series_name, frequency, transformation, category, units
    - Block columns (named after block names, e.g., Global, Consumption, Investment, External, or Block_Global, Block_Consumption, etc.)
    - Block columns contain 0 or 1 (1 = series loads on that block)
    
    Note: API/database-specific columns (data_code, item_id, api_source, country, is_kpi, etc.)
    are ignored by the generic DFM module. They should be handled by application-specific adapters.
    
    Example:
        id,series_name,frequency,transformation,category,units,Block_Global,Block_Consumption,Block_Invest,Block_Extern
        0,Real GDP (Quarterly),q,pca,GDP,Billion Won,1,1,0,0
        1,Nominal GDP (Quarterly),q,pca,GDP,Billion Won,1,1,0,0
    
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
    
    return _load_config_from_dataframe(df)


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load DFMConfig from a pandas DataFrame (shared by CSV and BytesIO loading).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CSV columns
        
    Returns
    -------
    DFMConfig
        Model configuration
    """
    # Handle 'id' as alias for 'series_id' (for backward compatibility)
    # If CSV has data_code, item_id, api_source, generate series_id from them
    # Otherwise, use 'id' as fallback
    if 'id' in df.columns and 'series_id' not in df.columns:
        # Try to generate series_id from data_code, item_id, api_source if available
        if all(col in df.columns for col in ['data_code', 'item_id', 'api_source']):
            # Generate series_id: {api_source}_{data_code}_{item_id}
            df['series_id'] = df.apply(
                lambda row: f"{row.get('api_source', '')}_{row.get('data_code', '')}_{row.get('item_id', '')}",
                axis=1
            )
        else:
            # Fallback: use 'id' as series_id if series_id column doesn't exist
            df['series_id'] = df['id'].astype(str)
    elif 'id' in df.columns and 'series_id' in df.columns:
        # Both exist - prefer series_id, but if it's empty, use id
        df['series_id'] = df['series_id'].fillna(df['id'].astype(str))
    
    # Required fields (DFM-relevant only - no API/database fields)
    required_fields = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
    optional_fields = []  # No optional fields in generic DFM module
    
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Detect block columns (all columns that are not in required_fields or optional_fields)
    # Also exclude 'id' (if it was used as alias, we already created series_id from it)
    # and other non-DFM metadata columns (API/database fields are ignored by generic DFM module)
    # Preserve original column order from DataFrame - first block should be Global
    excluded_fields = set(required_fields) | set(optional_fields) | {
        'id', 'country', 'data_code', 'item_id', 'api_source', 'api_code', 
        'api_group_id', 'is_kpi', 'description', 'priority', 'is_active', 'metadata'
    }
    # Use DataFrame column order (not sorted) to preserve CSV structure
    block_columns = [col for col in df.columns if col not in excluded_fields]
    
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
        
        # Create SeriesConfig with only DFM-relevant fields
        # API/database fields (data_code, item_id, api_source, etc.) are ignored by generic DFM module
        series_list.append(SeriesConfig(
            series_id=str(row['series_id']),  # Ensure string type
            series_name=row['series_name'],
            frequency=row['frequency'],
            transformation=row['transformation'],
            category=row['category'],
            units=row['units'],
            blocks=blocks
        ))
    
    return DFMConfig(series=series_list, block_names=block_columns)


# Note: Excel support removed - use CSV or YAML configs instead


def load_config(configfile: Union[str, Path, io.BytesIO]) -> DFMConfig:
    """Load model configuration from file (auto-detects YAML or CSV).
    
    Parameters
    ----------
    configfile : str, Path, or BytesIO
        Path to configuration file (.yaml, .yml, or .csv), or BytesIO object with CSV content
        
    Returns
    -------
    ModelConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist (for file paths)
    ValueError
        If file format is not supported or configuration is invalid
    """
    # Handle BytesIO (for database storage downloads)
    if isinstance(configfile, io.BytesIO):
        # Read CSV from BytesIO
        configfile.seek(0)  # Reset to beginning
        df = pd.read_csv(configfile)
        # Use the same logic as load_config_from_csv but with DataFrame
        return _load_config_from_dataframe(df)
    
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


def transform_data(Z: np.ndarray, Time: pd.DatetimeIndex, config: DFMConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
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


def sort_data(Z: np.ndarray, Mnem: List[str], config: DFMConfig) -> Tuple[np.ndarray, List[str]]:
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


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load and transform data from CSV file.
    
    This function reads data from a CSV file, sorts it to match the model
    configuration (from CSV or YAML), and applies the specified transformations.
    
    Note: For database-backed applications, create adapters that implement
    the same interface (return X, Time, Z arrays).
    
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


# Database adapters are application-specific and should be implemented separately.
# This keeps the DFM module generic and database-agnostic.
