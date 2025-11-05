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

logger = logging.getLogger(__name__)


def load_config_from_db(
    config_name: str = '001-initial-spec',
    client=None
):
    """
    Load model configuration from database.
    
    This is the recommended method for DFM module to get the latest spec,
    ensuring it always uses the most up-to-date configuration from the database.
    
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
):
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
                from .data_loader import load_config_from_csv
                return load_config_from_csv(config_path)
            else:
                raise ValueError("No database config found and no file path provided")
    else:
        # Try file first
        if config_path:
            try:
                from .data_loader import load_config_from_csv
                return load_config_from_csv(config_path)
            except Exception as e:
                logger.warning(f"Could not load from file: {e}. Falling back to database.")
                return load_config_from_db(config_name=config_name)
        else:
            return load_config_from_db(config_name=config_name)
