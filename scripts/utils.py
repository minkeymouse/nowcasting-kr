"""Shared utilities for DFM scripts.

This module consolidates common functionality used across train_dfm.py,
nowcast_dfm.py, and other scripts to reduce code duplication.
"""

import logging
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from src.nowcasting import load_config, ModelConfig

logger = logging.getLogger(__name__)


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> ModelConfig:
    """Load model configuration from database (latest) or CSV/YAML file.
    
    Application-specific config loader for Hydra workflows.
    Priority: Database → CSV → YAML
    
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
        
        # Load config from file
        model_config = load_config(config_file)
        
        # If loading from CSV and use_db is enabled, save blocks to database
        if use_db and config_file.suffix.lower() == '.csv':
            try:
                from adapters.adapter_database import save_blocks_to_db
                
                # Derive config_name from CSV filename
                # Example: '001_initial_spec.csv' → '001-initial-spec'
                config_name = config_file.stem.replace('_', '-')
                
                # Save blocks to database
                save_blocks_to_db(model_config, config_name)
            except (ImportError, Exception) as e:
                # Log warning but don't fail - block saving is optional
                logger.warning(
                    f"Could not save blocks to database for {config_file.name}: {e}. "
                    f"Continuing without saving blocks."
                )
        
        return model_config
    else:
        # Fallback to YAML config (convert DictConfig to dict, then to ModelConfig)
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))


def get_db_client():
    """Get database client with consistent error handling.
    
    Returns
    -------
    Client
        Database client instance
    
    Raises
    ------
    ImportError
        If database module is not available
    Exception
        If client initialization fails
    """
    try:
        from adapters.adapter_database import _get_db_client
        return _get_db_client()
    except ImportError:
        # Fallback to database.get_client
        from database import get_client
        return get_client()



