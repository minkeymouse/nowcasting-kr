"""Common utilities for CLI setup, path management, and Hydra config parsing.

This module combines:
- CLI environment setup (path management)
- Path utilities (project root, dfm-python, src, app)
- Hydra configuration parsing
- Config validation and extraction
"""

import sys
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

# ============================================================================
# Exception Classes
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


# ============================================================================
# Centralized Logging Configuration
# ============================================================================

# Global flag to ensure logging is configured only once
_logging_configured = False


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    force: bool = False,
    log_dir: Optional[Path] = None,
    log_file: Optional[Path] = None
) -> None:
    """Centralized logging configuration for the entire project.
    
    This function ensures logging is configured exactly once, preventing
    duplicate log messages from multiple handlers.
    
    **Key Design Decisions:**
    1. **No root logger handlers**: We don't use `basicConfig()` which adds
       handlers to the root logger, as this causes duplicate messages when
       combined with package-specific loggers.
    2. **Package-specific loggers**: Each package (src, dfm_python) manages
       its own logger hierarchy without propagating to root.
    3. **Single configuration point**: All modules should call this function
       instead of `logging.basicConfig()`.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level
    format_string : str, optional
        Custom format string. If None, uses default format.
    force : bool, default False
        Force reconfiguration even if already configured. Use with caution.
    log_dir : Path, optional
        Directory to save log files. If None, only logs to stdout.
    log_file : Path, optional
        Specific log file path. If provided, uses this instead of creating train_{timestamp}.log.
        Pattern should match: {TARGET}_{MODEL}_{TIMESTAMP}.log for consistency with DFM/DDFM.
        
    Notes
    -----
    - This function is idempotent - safe to call multiple times
    - Prevents duplicate handlers by checking a global flag
    - Removes any existing root logger handlers to prevent conflicts
    """
    global _logging_configured
    
    if _logging_configured and not force:
        return
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # CRITICAL: Remove any existing root logger handlers to prevent duplicates
    # Root logger handlers are added by basicConfig() or other code
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Remove all root handlers to prevent duplicate messages
        # Package-specific loggers (src, dfm_python) will handle their own output
        root_logger.handlers.clear()
    
    # Configure root logger level (but no handlers - let packages handle their own)
    root_logger.setLevel(level)
    
    # Configure src package logger (separate from dfm_python)
    src_logger = logging.getLogger('src')
    src_logger.setLevel(level)
    src_logger.propagate = False  # Don't propagate to root
    
    # Add handler to src logger if not already present
    if not src_logger.handlers:
        # Console handler (always add)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        )
        src_logger.addHandler(console_handler)
        
        # File handler (if log_dir or log_file is specified)
        if log_dir is not None or log_file is not None:
            if log_file is not None:
                # Use provided log file path
                log_file_path = Path(log_file)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                # Extract model name from log file for cleanup
                import re
                match = re.match(r'^(.+?)_(.+?)_(\d{8}_\d{6})\.log$', log_file_path.name)
                if match:
                    model_name_for_cleanup = match.group(2)  # Extract model name
                    _cleanup_old_logs(log_file_path.parent, model_name=model_name_for_cleanup, keep_count=2)
            else:
                # Use log_dir with default train_{timestamp}.log pattern
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                # Clean up old logs before creating new one
                _cleanup_old_logs(log_dir, model_name='train', keep_count=2)
                # Create log file with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file_path = log_dir / f'train_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(
                logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
            )
            src_logger.addHandler(file_handler)
    
    _logging_configured = True


def _cleanup_old_logs(log_dir: Path, model_name: Optional[str] = None, keep_count: int = 2) -> None:
    """Clean up old log files, keeping only the latest N files per model.
    
    Parameters
    ----------
    log_dir : Path
        Directory containing log files
    model_name : str, optional
        If provided, only clean logs for this model. Otherwise, clean all models.
    keep_count : int, default 2
        Number of latest log files to keep per model
    """
    if not log_dir.exists():
        return
    
    import re
    from collections import defaultdict
    _logger = logging.getLogger(__name__)
    
    # Pattern: {target}_{model}_{timestamp}.log or {number}_{model}_{timestamp}.log or train_{timestamp}.log or {target}_compare_{timestamp}.log
    # Note: Numbered logs like 0_tft_20251211_115212.log should be grouped by model name (tft)
    log_pattern = re.compile(r'^(.+?)_(.+?)_(\d{8}_\d{6})\.log$|^train_(\d{8}_\d{6})\.log$')
    
    # Group logs by model
    logs_by_model = defaultdict(list)
    
    for log_file in log_dir.glob('*.log'):
        match = log_pattern.match(log_file.name)
        if match:
            if match.group(4):  # train_{timestamp}.log pattern
                model = 'train'
                timestamp = match.group(4)
            else:  # {target}_{model}_{timestamp}.log or {number}_{model}_{timestamp}.log
                # Extract model name (could be 'compare' or actual model name)
                # For numbered logs (0_tft, 1_tft), group(2) extracts the model name
                model = match.group(2)  # Extract model name
                timestamp = match.group(3)
            
            # Filter by model_name if provided
            if model_name is not None and model != model_name:
                continue
            
            logs_by_model[model].append((log_file, timestamp))
    
    # Keep only latest keep_count files per model
    for model, log_files in logs_by_model.items():
        if len(log_files) > keep_count:
            # Sort by timestamp (newest first)
            log_files.sort(key=lambda x: x[1], reverse=True)
            
            # Delete older files
            for log_file, _ in log_files[keep_count:]:
                try:
                    log_file.unlink()
                    _logger.debug(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    _logger.warning(f"Failed to delete log file {log_file.name}: {e}")


def cleanup_logs(log_dir: Path, keep_count: int = 2) -> None:
    """Public function to clean up old log files.
    
    Parameters
    ----------
    log_dir : Path
        Directory containing log files
    keep_count : int, default 2
        Number of latest log files to keep per model
    """
    _cleanup_old_logs(log_dir, model_name=None, keep_count=keep_count)


# ============================================================================
# DDFM Default Constants
# ============================================================================

# Default DDFM encoder layers
DEFAULT_DDFM_ENCODER_LAYERS = [64, 32]

# Default DDFM number of factors
DEFAULT_DDFM_NUM_FACTORS = 5

# Default DDFM training epochs
DEFAULT_DDFM_EPOCHS = 100


# ============================================================================
# Path Setup Utilities
# ============================================================================

# Cache for computed paths to avoid repeated calculations
_path_cache: dict = {}


def get_project_root() -> Path:
    """Get the project root directory.
    
    Assumes this file is in src/, so project root is 1 level up.
    
    Returns
    -------
    Path
        Project root directory
    """
    if 'project_root' not in _path_cache:
        # This file is in src/, so go up 1 level
        _path_cache['project_root'] = Path(__file__).parent.parent.resolve()
    return _path_cache['project_root']


def get_dfm_python_path() -> Path:
    """Get the dfm-python src directory path.
    
    Returns
    -------
    Path
        Path to dfm-python/src directory
    """
    if 'dfm_python_path' not in _path_cache:
        project_root = get_project_root()
        _path_cache['dfm_python_path'] = project_root / "dfm-python" / "src"
    return _path_cache['dfm_python_path']


def get_src_path() -> Path:
    """Get the src directory path.
    
    Returns
    -------
    Path
        Path to src directory
    """
    if 'src_path' not in _path_cache:
        _path_cache['src_path'] = Path(__file__).parent.resolve()
    return _path_cache['src_path']




def setup_paths(
    include_dfm_python: bool = True,
    include_src: bool = False,
    include_app: bool = False
) -> None:
    """Set up Python paths for the project.
    
    This function adds necessary paths to sys.path in a controlled manner.
    It should be called once at the start of a module or script.
    
    Parameters
    ----------
    include_dfm_python : bool, default True
        Whether to add dfm-python/src to sys.path
    include_src : bool, default False
        Whether to add src/ to sys.path
    include_app : bool, default False
        Whether to add app/ to sys.path
        
    Notes
    -----
    Paths are added to the beginning of sys.path to ensure they take precedence
    over installed packages with the same names.
    """
    paths_to_add = []
    
    if include_dfm_python:
        dfm_path = get_dfm_python_path()
        if dfm_path.exists():
            dfm_path_str = str(dfm_path)
            if dfm_path_str not in sys.path:
                paths_to_add.append(dfm_path_str)
        else:
            raise ImportError(
                f"dfm-python path not found: {dfm_path}. "
                "Ensure dfm-python submodule is initialized."
            )
    
    if include_src:
        src_path = get_src_path()
        if src_path.exists():
            src_path_str = str(src_path)
            if src_path_str not in sys.path:
                paths_to_add.append(src_path_str)
    
    if include_app:
        project_root = get_project_root()
        app_path = project_root / "app"
        if app_path.exists():
            app_path_str = str(app_path)
            if app_path_str not in sys.path:
                paths_to_add.append(app_path_str)
    
    # Insert at the beginning to ensure precedence
    for path in reversed(paths_to_add):
        sys.path.insert(0, path)


def setup_cli_paths() -> Path:
    """Set up paths for CLI scripts and return project root.
    
    This function sets up sys.path to allow scripts to be run directly
    (e.g., `python3 src/train.py`) while still supporting imports from
    the src package.
    
    Returns
    -------
    Path
        Project root directory path
    """
    # Get script directory (e.g., src/)
    script_dir = Path(__file__).parent.resolve()
    # Get project root (parent of src/)
    project_root = script_dir.parent.resolve()
    
    # Add to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    return project_root


def setup_cli_environment() -> None:
    """Set up complete CLI environment including paths and imports.
    
    This function performs all necessary setup for CLI scripts:
    1. Sets up sys.path
    2. Configures dfm-python, src, and app paths
    """
    # Set up paths
    setup_cli_paths()
    # Configure package paths
    setup_paths(include_dfm_python=True, include_src=True, include_app=True)


# ============================================================================
# General Helper Functions
# ============================================================================

def validate_data_file(data_path: Path) -> None:
    """Validate that data file exists, raising ValidationError if not.
    
    Parameters
    ----------
    data_path : Path
        Path to data file
        
    Raises
    ------
    ValidationError
        If file does not exist
    """
    if not data_path.exists():
        raise ValidationError(f"Data file not found: {data_path}")


# ============================================================================
# Hydra Config Parsing
# ============================================================================


def parse_experiment_config(
    config_name: str,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> DictConfig:
    """Parse experiment configuration from Hydra.
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/investment_koequipte_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    overrides : list of str, optional
        Hydra config overrides (e.g., ['model_overrides.dfm.max_iter=10'])
        
    Returns
    -------
    DictConfig
        Parsed Hydra configuration
        
    Raises
    ------
    ValueError
        If config cannot be loaded
    """
    if config_dir is None:
        # Default to config/ directory relative to project root
        project_root = get_project_root()
        config_dir = str(project_root / "config")
    
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
        # Convert to regular DictConfig (not struct) to allow overrides
        # This allows CLI overrides to work properly
        if OmegaConf.is_struct(cfg):
            OmegaConf.set_struct(cfg, False)
        # Also handle nested experiment config
        if 'experiment' in cfg and OmegaConf.is_struct(cfg.experiment):
            OmegaConf.set_struct(cfg.experiment, False)
        return cfg


def extract_experiment_params(cfg: DictConfig) -> Dict[str, Any]:
    """Extract common experiment parameters from config.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
        
    Returns
    -------
    dict
        Dictionary containing:
        - target_series: str
        - models: List[str]
        - horizons: List[int]
        - data_path: Optional[str]
        - exp_cfg: DictConfig (experiment section or full config)
    """
    # Get experiment section (Hydra may wrap in 'experiment' key)
    if 'experiment' in cfg:
        exp_cfg = cfg.experiment
    else:
        exp_cfg = cfg
    
    # Extract target_series
    target_series = exp_cfg.get('target_series')
    
    # Extract models list
    models = []
    models_raw = exp_cfg.get('models')
    if models_raw:
        models_container = OmegaConf.to_container(models_raw, resolve=True)
        if isinstance(models_container, list):
            models = [str(m) for m in models_container]
        elif isinstance(models_container, str):
            models = [models_container]
    
    # Extract horizons
    horizons = list(range(1, 25))  # Default: horizons 1-24 (monthly: 1 horizon = 1 month = 4 weeks)
    horizons_raw = exp_cfg.get('forecast_horizons')
    if horizons_raw:
        horizons_container = OmegaConf.to_container(horizons_raw, resolve=True)
        if isinstance(horizons_container, list):
            horizons = [int(str(h)) for h in horizons_container]
        elif isinstance(horizons_container, (int, str)):
            horizons = [int(str(horizons_container))]
    
    # Extract data_path
    data_path = exp_cfg.get('data_path')
    
    return {
        'target_series': target_series,
        'models': models,
        'horizons': horizons,
        'data_path': data_path,
        'exp_cfg': exp_cfg
    }


def validate_experiment_config(cfg: DictConfig, require_target: bool = True, require_models: bool = True) -> None:
    """Validate experiment configuration has required fields.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    require_target : bool, default True
        Whether target_series is required
    require_models : bool, default True
        Whether models list is required
        
    Raises
    ------
    ValueError
        If required fields are missing
    """
    params = extract_experiment_params(cfg)
    
    if require_target and not params['target_series']:
        config_name = cfg.get('_config_name_', 'unknown')
        raise ValueError(f"Config {config_name} must specify 'target_series'")
    
    if require_models:
        if not params['models']:
            config_name = cfg.get('_config_name_', 'unknown')
            raise ValueError(f"Config {config_name} must specify 'models' list")
        if len(params['models']) == 0:
            config_name = cfg.get('_config_name_', 'unknown')
            raise ValueError(f"Config {config_name} must specify at least one model in 'models' list")


# ============================================================================
# Config Helper Functions
# ============================================================================

def get_experiment_cfg(cfg: DictConfig) -> DictConfig:
    """Extract experiment config section from Hydra config."""
    return cfg.experiment if 'experiment' in cfg else cfg


def get_config_path() -> Path:
    """Get path to config directory.
    
    Returns
    -------
    Path
        Path to config directory
    """
    return get_project_root() / "config"


def detect_model_type(model_name: str) -> Optional[str]:
    """Detect model type from model name."""
    valid_types = ['arima', 'var', 'dfm', 'ddfm', 'tft', 'lstm', 'chronos']
    parts = model_name.lower().split('_')
    for part in parts:
        if part in valid_types:
            return part
    return None


def disable_omegaconf_struct(cfg: DictConfig) -> None:
    """Disable OmegaConf struct mode to allow modifications."""
    if OmegaConf.is_struct(cfg):
        OmegaConf.set_struct(cfg, False)
    if 'experiment' in cfg and OmegaConf.is_struct(cfg.experiment):
        OmegaConf.set_struct(cfg.experiment, False)


def resolve_data_path(data_path: Optional[str] = None) -> Path:
    """Resolve data file path with fallback to defaults.
    
    Parameters
    ----------
    data_path : Optional[str]
        Provided data path
        
    Returns
    -------
    Path
        Resolved data file path
        
    Raises
    ------
    ValidationError
        If no data file can be found
    """
    project_root = get_project_root()
    
    if data_path and Path(data_path).exists():
        return Path(data_path)
    
    # Try default locations
    for filename in ['data.csv', 'sample_data.csv']:
        default_path = project_root / "data" / filename
        if default_path.exists():
            return default_path
    
    raise ValidationError(f"Data file not found. Tried: {data_path}, data/data.csv, data/sample_data.csv")


def get_checkpoint_path(
    target_series: str,
    model_name: str,
    checkpoint_dir: Optional[str] = None,
    config_overrides: Optional[List[str]] = None
) -> Path:
    """Get checkpoint path for a model.
    
    Parameters
    ----------
    target_series : str
        Target series name
    model_name : str
        Model name
    checkpoint_dir : Optional[str]
        Checkpoint directory
    config_overrides : Optional[List[str]]
        Config overrides to extract checkpoint_dir from
        
    Returns
    -------
    Path
        Path to model checkpoint file
    """
    if checkpoint_dir is None and config_overrides:
        for override in config_overrides:
            if override.startswith('checkpoint_dir=') or override.startswith('+checkpoint_dir='):
                checkpoint_dir = override.split('=', 1)[1]
                break
    
    if checkpoint_dir is None:
        checkpoint_dir = 'checkpoints'
    
    return Path(checkpoint_dir) / f"{target_series}_{model_name}" / "model.pkl"


def load_and_filter_data(
    data_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    resample_freq: Optional[str] = None
) -> pd.DataFrame:
    """Load data file and filter by date range.
    
    Parameters
    ----------
    data_path : Path
        Path to data file
    start_date : pd.Timestamp
        Start date (inclusive)
    end_date : pd.Timestamp
        End date (inclusive)
    resample_freq : Optional[str]
        Resampling frequency (e.g., 'ME' for monthly end)
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if resample_freq:
        from src.train.preprocess import resample_to_monthly
        if resample_freq == 'ME':
            data = resample_to_monthly(data)
    
    return data


def set_forecaster_attributes(forecaster: Any, y_train: pd.DataFrame) -> None:
    """Set common forecaster attributes for compatibility."""
    if not hasattr(forecaster, 'is_fitted'):
        forecaster.is_fitted = True
    if not hasattr(forecaster, '_y'):
        forecaster._y = y_train


def extract_model_metrics(forecaster: Any) -> Dict[str, Any]:
    """Extract metrics from trained forecaster.
    
    Returns
    -------
    Dict[str, Any]
        Metrics dictionary with converged, num_iter, loglik
    """
    try:
        result = forecaster.get_result()
    except (AttributeError, RuntimeError):
        result = None
    
    return {
        'converged': result.get('converged', True) if isinstance(result, dict) else getattr(result, 'converged', True) if result else True,
        'num_iter': result.get('num_iter', 0) if isinstance(result, dict) else getattr(result, 'num_iter', 0) if result else 0,
        'loglik': result.get('loglik', np.nan) if isinstance(result, dict) else getattr(result, 'loglik', np.nan) if result else np.nan
    }


# ============================================================================
# Constants
# ============================================================================

# Date constants
TRAIN_START = pd.Timestamp('1985-01-01')
TRAIN_END = pd.Timestamp('2019-12-31')
RECENT_START = pd.Timestamp('2020-01-01')
RECENT_END = pd.Timestamp('2023-12-31')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2025-10-31')

# Frequency constants
FREQ_WEEKLY = 'w'
FREQ_MONTHLY = 'm'

# Scaler constants
SCALER_ROBUST = 'robust'
SCALER_STANDARD = 'standard'

# Model type constants
MODEL_ARIMA = 'arima'
MODEL_VAR = 'var'
MODEL_DFM = 'dfm'
MODEL_DDFM = 'ddfm'
MODEL_TFT = 'tft'
MODEL_LSTM = 'lstm'
MODEL_CHRONOS = 'chronos'


# ============================================================================
# Weekly to Monthly Aggregation Utilities
# ============================================================================

def aggregate_weekly_to_monthly(
    weekly_forecast: Union[pd.Series, pd.DataFrame],
    weeks_per_month: int = 4
) -> Union[pd.Series, pd.DataFrame]:
    """Aggregate weekly forecasts to monthly by averaging.
    
    Parameters
    ----------
    weekly_forecast : pd.Series or pd.DataFrame
        Weekly forecast data with DatetimeIndex (weekly frequency)
    weeks_per_month : int, default 4
        Number of weeks per month for aggregation
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Monthly aggregated forecast with DatetimeIndex (monthly frequency)
    """
    if not isinstance(weekly_forecast.index, pd.DatetimeIndex):
        raise ValueError("weekly_forecast must have DatetimeIndex")
    
    # Resample weekly to monthly using mean aggregation
    if isinstance(weekly_forecast, pd.Series):
        monthly_forecast = weekly_forecast.resample('ME').mean()
    else:
        monthly_forecast = weekly_forecast.resample('ME').mean()
    
    # Use module-level logger
    _logger = logging.getLogger(__name__)
    _logger.debug(f"Aggregated {len(weekly_forecast)} weekly forecasts to {len(monthly_forecast)} monthly forecasts")
    
    return monthly_forecast


def convert_horizon_months_to_weeks(horizon_months: int, weeks_per_month: int = 4) -> int:
    """Convert forecast horizon from months to weeks.
    
    Parameters
    ----------
    horizon_months : int
        Forecast horizon in months
    weeks_per_month : int, default 4
        Number of weeks per month
        
    Returns
    -------
    int
        Forecast horizon in weeks
    """
    return horizon_months * weeks_per_month

