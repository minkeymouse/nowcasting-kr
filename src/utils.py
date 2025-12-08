"""Common utilities for CLI setup, path management, and Hydra config parsing.

This module combines:
- CLI environment setup (path management)
- Path utilities (project root, dfm-python, src, app)
- Hydra configuration parsing
- Config validation and extraction
"""

import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

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
    force: bool = False
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
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        )
        src_logger.addHandler(handler)
    
    _logging_configured = True


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
    horizons = list(range(1, 23))  # Default: horizons 1-22 (monthly: 2024-01 to 2025-10)
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

