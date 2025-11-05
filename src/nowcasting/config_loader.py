"""Configuration loading for DFM models.

This module provides functions to load model configurations from various sources:
- CSV files (local specification files)
- Database (latest specification from database)
- YAML files (Hydra configs)

The DFM module should use get_latest_spec_from_db() to always pull the latest
specification from the database for forecasting.
"""

import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
from omegaconf import DictConfig

from .config import ModelConfig
from .data_loader import load_config

logger = logging.getLogger(__name__)


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
            config_name = cfg_model.get('config_name', '001-initial-spec')
            
            db_config = load_model_config(config_name, client=client)
            if db_config and 'config_json' in db_config:
                # Convert DB config to ModelConfig
                config_dict = db_config['config_json']
                # Handle block_names from DB
                if 'block_names' not in config_dict and 'block_names' in db_config:
                    config_dict['block_names'] = db_config['block_names']
                return ModelConfig.from_dict(config_dict)
        except (ImportError, Exception) as e:
            # Database not available or config not found - fall back to file
            logger.debug(f"Could not load from database: {e}. Falling back to file.")
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


def load_config_from_db(
    config_name: str = '001-initial-spec',
    client=None
) -> ModelConfig:
    """
    Load model configuration directly from database by config name.
    
    This is a direct alternative to load_model_config_from_hydra() when you
    don't have a Hydra config dict. The DFM module can use this to always pull
    the latest spec from the database.
    
    Parameters
    ----------
    config_name : str
        Name of the model configuration in database
    client : Client, optional
        Supabase client instance (if None, will be created automatically)
        
    Returns
    -------
    ModelConfig
        Model configuration object
        
    Raises
    ------
    ValueError
        If config_name not found in database
    ImportError
        If database module not available
    """
    try:
        from database import get_latest_spec_from_db
    except ImportError:
        raise ImportError(
            "Database module not available. Install required dependencies or use CSV file."
        )
    
    # Get latest spec from database as DataFrame
    df = get_latest_spec_from_db(config_name=config_name, client=client)
    
    # Convert DataFrame to ModelConfig using existing CSV loader
    # (reuse the CSV parsing logic)
    from io import StringIO
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Use existing load_config_from_csv logic (import here to avoid Hydra issues)
    from .data_loader import load_config_from_csv
    return load_config_from_csv(csv_buffer)


def load_config_from_file_or_db(
    config_path: Optional[Union[str, Path]] = None,
    config_name: Optional[str] = '001-initial-spec',
    prefer_db: bool = True
) -> ModelConfig:
    """
    Load configuration from database or file, with preference for database.
    
    This is a convenience function that tries to load from database first,
    then falls back to file if database is not available.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to CSV file (fallback)
    config_name : str, optional
        Database config name (default: '001-initial-spec')
    prefer_db : bool
        If True, try database first, then file. If False, try file first.
        
    Returns
    -------
    ModelConfig
        Model configuration object
    """
    if prefer_db:
        # Try database first
        try:
            return load_config_from_db(config_name=config_name)
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not load from database: {e}. Falling back to file.")
            if config_path:
                return load_config(config_path)
            else:
                raise ValueError("No database config found and no file path provided")
    else:
        # Try file first
        if config_path:
            try:
                return load_config(config_path)
            except Exception as e:
                logger.warning(f"Could not load from file: {e}. Falling back to database.")
                return load_config_from_db(config_name=config_name)
        else:
            return load_config_from_db(config_name=config_name)
