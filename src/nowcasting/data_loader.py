"""Data loading, specification parsing, and transformation functions."""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

from .config import ModelConfig

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
        Model configuration (Pydantic model with validation)
        
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


def load_config_from_excel(configfile: Union[str, Path]) -> ModelConfig:
    """Load model configuration from Excel file (backward compatibility).
    
    Parameters
    ----------
    configfile : str or Path
        Path to Excel specification file
        
    Returns
    -------
    ModelConfig
        Model configuration (Pydantic model with validation)
        
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
    
    # Try openpyxl first (for .xlsx), then xlrd (for .xls)
    try:
        df = pd.read_excel(configfile, sheet_name=0, engine='openpyxl')
    except Exception as e:
        try:
            df = pd.read_excel(configfile, sheet_name=0, engine='xlrd')
        except Exception as e2:
            raise ValueError(f"Failed to read Excel file {configfile}: {e}. Also tried xlrd: {e2}")
    
    if 'Model' not in df.columns:
        raise ValueError("'Model' column missing from model specification.")
    df = df[df['Model'] == 1].copy()
    
    required_fields = ['SeriesID', 'SeriesName', 'Frequency', 'Units', 'Transformation', 'Category']
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    
    block_cols = [col for col in df.columns if col.startswith('Block')]
    if not block_cols:
        raise ValueError("No Block columns found.")
    
    blocks_data = np.nan_to_num(df[block_cols].values, nan=0.0).astype(int)
    if not np.all(blocks_data[:, 0] == 1):
        raise ValueError("All variables must load on global block.")
    
    # Sort by frequency order
    freq_order = ['d', 'w', 'm', 'q', 'sa', 'a']
    perm = [i for freq in freq_order for i, f in enumerate(df['Frequency']) if f == freq]
    
    # Convert to legacy format dict for ModelConfig.from_dict()
    spec_dict = {
        'SeriesID': [df['SeriesID'].iloc[i] for i in perm],
        'SeriesName': [df['SeriesName'].iloc[i] for i in perm],
        'Frequency': [df['Frequency'].iloc[i] for i in perm],
        'Units': [df['Units'].iloc[i] for i in perm],
        'Transformation': [df['Transformation'].iloc[i] for i in perm],
        'Category': [df['Category'].iloc[i] for i in perm],
        'Blocks': blocks_data[perm, :],
        'BlockNames': [re.sub(r'Block\d+-', '', col) for col in block_cols]
    }
    
    # Map transformations to readable units
    transform_map = {
        'lin': 'Levels (No Transformation)', 'chg': 'Change (Difference)',
        'ch1': 'Year over Year Change (Difference)', 'pch': 'Percent Change',
        'pc1': 'Year over Year Percent Change', 'pca': 'Percent Change (Annual Rate)',
        'cch': 'Continuously Compounded Rate of Change',
        'cca': 'Continuously Compounded Annual Rate of Change', 'log': 'Natural Log'
    }
    spec_dict['UnitsTransformed'] = [transform_map.get(t, t) for t in spec_dict['Transformation']]
    
    print('\n\n\nTable 1: Model specification')
    try:
        print(pd.DataFrame({
            'SeriesID': spec_dict['SeriesID'],
            'SeriesName': spec_dict['SeriesName'],
            'Units': spec_dict['Units'],
            'UnitsTransformed': spec_dict['UnitsTransformed']
        }).to_string(index=False))
    except:
        pass
    print('\n\n\n')
    
    return ModelConfig.from_dict(spec_dict)


def load_config(configfile: Union[str, Path]) -> ModelConfig:
    """Load model configuration from file (auto-detects YAML or Excel).
    
    Parameters
    ----------
    configfile : str or Path
        Path to configuration file (.yaml, .yml, .xlsx, or .xls)
        
    Returns
    -------
    ModelConfig
        Model configuration (Pydantic model with validation)
        
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
    elif suffix in ['.xlsx', '.xls']:
        return load_config_from_excel(configfile)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, .xlsx, or .xls")


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
    """Read data from Excel file."""
    datafile = Path(datafile)
    # Try openpyxl first (for .xlsx), then xlrd (for .xls)
    try:
        df = pd.read_excel(datafile, sheet_name='data', engine='openpyxl')
    except:
        df = pd.read_excel(datafile, sheet_name='data', engine='xlrd')
    mnemonics = df.columns[1:].tolist()
    
    # Parse dates
    date_col = df.iloc[:, 0]
    try:
        Time = pd.to_datetime(date_col, format='%m/%d/%Y')
    except:
        try:
            Time = pd.Timestamp('1899-12-30') + pd.to_timedelta(date_col - 1, unit='D')
        except:
            Time = pd.to_datetime(date_col, errors='coerce')
    
    return df.iloc[:, 1:].values.astype(float), Time, mnemonics


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
        warnings.warn(f"Failed to load from .mat cache ({e}). Reading from Excel instead.")
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
              sample_start: Optional[Union[pd.Timestamp, str]] = None,
              load_excel: bool = False) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load vintage of data from file and format.
    
    This function supports caching to .mat files for faster subsequent loads,
    matching MATLAB's behavior. If a cached .mat file exists and load_excel=False,
    data will be loaded from the cache instead of reading Excel.
    
    Parameters
    ----------
    datafile : str or Path
        Path to Excel data file (.xlsx or .xls)
    config : ModelConfig
        Model configuration object
    sample_start : Timestamp or str, optional
        Start date for estimation sample. Data before this date will be dropped.
    load_excel : bool, default False
        If True, forces reading from Excel even if cached .mat file exists.
        If False and .mat cache exists, loads from cache.
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : pd.DatetimeIndex
        Time index for observations
    Z : np.ndarray
        Raw (untransformed) data matrix (T x N)
    """
    print('Loading data...')
    
    datafile = Path(datafile)
    if datafile.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError('Only Microsoft Excel workbook files supported.')
    
    # Create path for cached .mat file (matching MATLAB behavior)
    mat_dir = datafile.parent / 'mat'
    datafile_mat = mat_dir / f'{datafile.stem}.mat'
    
    # Try to load from cache first
    cache_data = None if load_excel else _load_from_mat_cache(datafile_mat)
    
    # Read from Excel if cache load failed or was skipped
    if cache_data is None:
        Z, Time, Mnem = read_data(datafile)
        _save_to_mat_cache(Z, Time, Mnem, mat_dir, datafile_mat)
    else:
        Z, Time, Mnem = cache_data
    
    # Process data: sort, transform, filter
    Z, _ = sort_data(Z, Mnem, config)
    X, Time, Z = transform_data(Z, Time, config)
    
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.to_datetime(sample_start)
        mask = Time >= sample_start
        Time, X, Z = Time[mask], X[mask], Z[mask]
    
    return X, Time, Z
