"""Helper functions for loading model configurations from CSV or YAML.

This module provides utilities for loading model configurations, with CSV as the
primary format for researchers to update. CSV configs are easier to edit in Excel
and maintain for large model specifications.
"""

from pathlib import Path
from typing import Union
from omegaconf import DictConfig

from .config import ModelConfig
from .data_loader import load_config


def load_model_config_from_hydra(cfg_model: DictConfig, use_db: bool = True) -> ModelConfig:
    """Load model configuration from database (latest) or CSV/YAML file.
    
    Priority order:
    1. Database (if use_db=True) - loads latest spec from DB
    2. CSV file (if config_path specified) - from migrations/001_initial_spec.csv
    3. YAML file (fallback) - from config/model/*.yaml
    
    Researchers update the spec file/database, and this function always uses the latest.
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
    use_db : bool, default True
        If True, try to load latest spec from database first
        
    Returns
    -------
    ModelConfig
        Model configuration object
        
    Raises
    ------
    FileNotFoundError
        If config_path is specified but file doesn't exist
    """
    # Try to load from database first (latest spec)
    if use_db:
        try:
            from database import get_client, load_model_config
            
            client = get_client()
            config_name = cfg_model.get('config_name', 'kr_dfm_v1')
            
            db_config = load_model_config(config_name, client=client)
            if db_config and 'config_json' in db_config:
                # Convert DB config to ModelConfig
                from .config import ModelConfig
                config_dict = db_config['config_json']
                # Handle block_names from DB
                if 'block_names' not in config_dict and 'block_names' in db_config:
                    config_dict['block_names'] = db_config['block_names']
                return ModelConfig.from_dict(config_dict)
        except (ImportError, Exception) as e:
            # Database not available or config not found - fall back to file
            pass
    
    # Load from CSV or YAML file
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        config_file = Path(model_config_path)
        if not config_file.is_absolute():
            # Relative path - resolve from project root
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / model_config_path
                if candidate.exists():
                    config_file = candidate
                    break
            else:
                # Fallback: assume relative to script location
                config_file = Path(__file__).parent.parent.parent / model_config_path
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}\n"
                f"Researchers should update: migrations/001_initial_spec.csv"
            )
        
        from .data_loader import load_config
        return load_config(config_file)
    else:
        # Fallback to YAML config (legacy support)
        from omegaconf import OmegaConf
        from .config import ModelConfig
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))


def get_default_config_path() -> Path:
    """Get default CSV config path for researchers to update.
    
    Returns
    -------
    Path
        Path to migrations/001_initial_spec.csv
    """
    # Try to find project root (where migrations/ exists)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        candidate = parent / 'migrations' / '001_initial_spec.csv'
        if candidate.exists():
            return candidate
    
    # Fallback: assume relative to this file
    return Path(__file__).parent.parent.parent / 'migrations' / '001_initial_spec.csv'
