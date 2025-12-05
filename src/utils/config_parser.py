"""Common utilities for parsing Hydra experiment configurations and path setup."""

import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

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
    
    Assumes this file is in src/utils/, so project root is 2 levels up.
    
    Returns
    -------
    Path
        Project root directory
    """
    if 'project_root' not in _path_cache:
        # This file is in src/utils/, so go up 2 levels
        _path_cache['project_root'] = Path(__file__).parent.parent.parent.resolve()
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
        _path_cache['src_path'] = Path(__file__).parent.parent.resolve()
    return _path_cache['src_path']


def get_app_path() -> Path:
    """Get the app directory path.
    
    Returns
    -------
    Path
        Path to app directory
    """
    if 'app_path' not in _path_cache:
        project_root = get_project_root()
        _path_cache['app_path'] = project_root / "app"
    return _path_cache['app_path']


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
        app_path = get_app_path()
        if app_path.exists():
            app_path_str = str(app_path)
            if app_path_str not in sys.path:
                paths_to_add.append(app_path_str)
    
    # Insert at the beginning to ensure precedence
    for path in reversed(paths_to_add):
        sys.path.insert(0, path)


def ensure_paths_setup() -> None:
    """Ensure paths are set up, but only if dfm-python is not already importable.
    
    This is a safe function that can be called multiple times without side effects.
    It checks if dfm_python can be imported before adding paths.
    """
    try:
        import dfm_python
        # Already importable, no need to set up paths
        return
    except ImportError:
        # Not importable, set up paths
        setup_paths(include_dfm_python=True)


# ============================================================================
# General Helper Functions
# ============================================================================

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load YAML config file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file
        
    Returns
    -------
    dict
        Config dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_model_file(model_dir: str) -> Optional[Path]:
    """Find model.pkl file in directory.
    
    Parameters
    ----------
    model_dir : str
        Directory to search
        
    Returns
    -------
    Path or None
        Path to model.pkl if found
    """
    model_path = Path(model_dir) / "model.pkl"
    if model_path.exists():
        return model_path
    return None


def get_experiment_output_dir(
    base_dir: str,
    experiment_id: str,
    create: bool = True
) -> Path:
    """Get output directory for experiment.
    
    Parameters
    ----------
    base_dir : str
        Base output directory
    experiment_id : str
        Experiment ID
    create : bool, default True
        Whether to create directory if it doesn't exist
        
    Returns
    -------
    Path
        Output directory path
    """
    output_dir = Path(base_dir) / "experiments" / experiment_id
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_data_path(data_path: str) -> bool:
    """Validate that data file exists.
    
    Parameters
    ----------
    data_path : str
        Path to data file
        
    Returns
    -------
    bool
        True if file exists
    """
    return Path(data_path).exists()


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


def get_series_list_from_config(config: Dict[str, Any]) -> List[str]:
    """Extract series list from experiment config.
    
    Parameters
    ----------
    config : dict
        Experiment config dictionary
        
    Returns
    -------
    list
        List of series IDs
    """
    return config.get('series', [])


def get_target_series_from_config(config: Dict[str, Any]) -> Optional[str]:
    """Extract target series from experiment config.
    
    Parameters
    ----------
    config : dict
        Experiment config dictionary
        
    Returns
    -------
    str or None
        Target series ID
    """
    return config.get('target')


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
        Experiment config name (e.g., 'experiment/kogdp_report')
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
    horizons = [1, 7, 28]  # Default
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

