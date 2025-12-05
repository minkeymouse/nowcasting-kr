"""Common utilities for parsing Hydra experiment configurations."""

from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")


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
        project_root = Path(__file__).parent.parent.parent
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

