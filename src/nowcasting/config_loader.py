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


def load_model_config_from_hydra(cfg_model: DictConfig) -> ModelConfig:
    """Load model configuration from Hydra config, supporting CSV or YAML.
    
    Researchers update migrations/001_initial_spec.csv for model specifications.
    This function handles both CSV and YAML configs transparently.
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
        
    Returns
    -------
    ModelConfig
        Model configuration object
        
    Raises
    ------
    FileNotFoundError
        If config_path is specified but file doesn't exist
    """
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        # Load from CSV or YAML file (researchers update CSV)
        config_file = Path(model_config_path)
        if not config_file.is_absolute():
            # Relative path - resolve from project root
            # Try to find project root (where migrations/ exists)
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
        
        return load_config(config_file)
    else:
        # Fallback to YAML config (legacy support)
        from omegaconf import OmegaConf
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
