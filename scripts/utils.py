"""Shared utilities for DFM training and forecasting scripts.

This module contains application-specific utilities that are shared between
training and forecasting scripts but kept separate from the generic DFM module.
"""

from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from src.nowcasting import load_config, ModelConfig


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> ModelConfig:
    """Load model configuration from database (latest) or CSV/YAML file.
    
    Application-specific config loader for Hydra workflows.
    Priority: Database → CSV → YAML
    
    This function is kept in scripts (not in DFM module) to keep DFM generic.
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
    use_db : bool, default=True
        Whether to try loading from database first
    script_path : Path, optional
        Path to the calling script (for relative path resolution)
        If None, uses current working directory
    
    Returns
    -------
    ModelConfig
        Loaded model configuration
    
    Raises
    ------
    FileNotFoundError
        If config file is specified but not found
    """
    # Try database first (if enabled)
    if use_db:
        try:
            from database import get_client, load_model_config
            
            client = get_client()
            config_name = cfg_model.get('config_name', '001-initial-spec')
            
            db_config = load_model_config(config_name, client=client)
            if db_config and 'config_json' in db_config:
                config_dict = db_config['config_json']
                if 'block_names' not in config_dict and 'block_names' in db_config:
                    config_dict['block_names'] = db_config['block_names']
                return ModelConfig.from_dict(config_dict)
        except (ImportError, Exception):
            pass  # Fall back to file
    
    # Load from CSV or YAML file
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        config_file = Path(model_config_path)
        
        # Resolve relative paths
        if not config_file.is_absolute():
            # Start from script location or current directory
            if script_path:
                base_dir = script_path.parent.parent
            else:
                base_dir = Path.cwd()
            
            # Try multiple possible locations
            search_paths = [
                base_dir / model_config_path,  # Relative to project root
                Path.cwd() / model_config_path,  # Relative to current dir
            ]
            
            # Also try parent directories
            for parent in [base_dir] + list(base_dir.parents):
                search_paths.append(parent / model_config_path)
            
            # Find first existing path
            for candidate in search_paths:
                if candidate.exists():
                    config_file = candidate
                    break
            else:
                # If not found, use default location relative to script
                if script_path:
                    config_file = script_path.parent.parent / model_config_path
                else:
                    config_file = Path(model_config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}\n"
                f"Researchers should update: src/spec/001_initial_spec.csv"
            )
        
        return load_config(config_file)
    else:
        # Fallback to YAML config (convert DictConfig to dict, then to ModelConfig)
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))


