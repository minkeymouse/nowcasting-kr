"""Specification file management for DFM model configurations.

This module handles the synchronization between CSV specification files and
the database model configurations. It allows:
1. Exporting database configs to CSV format
2. Updating CSV files from database changes
3. Retrieving latest spec for DFM module
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .client import get_client
from .helpers import (
    ensure_client,
    get_block_assignments_for_config,
    get_block_names_for_config,
)
from .operations import (
    load_model_config,
    get_model_config_series_ids,
    get_series_metadata_bulk,
    TABLES,
)

logger = logging.getLogger(__name__)


def export_config_to_csv(
    config_name: str,
    output_path: Optional[Path] = None,
    client=None
) -> Path:
    """
    Export a model configuration from database to CSV format.
    
    This creates/updates the CSV spec file based on the database configuration,
    including all series metadata and block assignments.
    
    Parameters
    ----------
    config_name : str
        Name of the model configuration in database
    output_path : Path, optional
        Output CSV file path. If None, uses src/spec/001_initial_spec.csv
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Path
        Path to the exported CSV file
        
    Raises
    ------
    ValueError
        If config_name not found in database
    """
    client = ensure_client(client)
    
    # Load model config from database
    config = load_model_config(config_name, client=client)
    if not config:
        raise ValueError(f"Model configuration '{config_name}' not found in database")
    
    config_json = config.get('config_json', {})
    # Get block names from blocks table
    block_names = get_block_names_for_config(config_name, client=client)
    
    # Get series list from config
    series_list = config_json.get('series', [])
    if not series_list:
        raise ValueError(f"No series found in configuration '{config_name}'")
    
    # Get series IDs
    series_ids = [s['series_id'] for s in series_list]
    
    # Get detailed series metadata from database
    series_metadata_df = get_series_metadata_bulk(series_ids, client=client)
    
    # Create a mapping of series_id to metadata
    metadata_dict = series_metadata_df.set_index('series_id').to_dict('index')
    
    # Build CSV rows
    csv_rows = []
    for series_item in series_list:
        series_id = series_item['series_id']
        
        # Get metadata from database (preferred) or config (fallback)
        if series_id in metadata_dict:
            meta = metadata_dict[series_id]
            series_name = meta.get('series_name') or series_item.get('series_name', '')
            frequency = meta.get('frequency') or series_item.get('frequency', '')
            transformation = meta.get('transformation') or series_item.get('transformation', '')
            category = meta.get('category') or series_item.get('category', '')
            units = meta.get('units') or series_item.get('units', '')
            api_code = meta.get('api_code') or series_item.get('api_code', '')
            api_source = meta.get('api_source') or series_item.get('api_source', '')
        else:
            # Fallback to config values
            series_name = series_item.get('series_name', '')
            frequency = series_item.get('frequency', '')
            transformation = series_item.get('transformation', '')
            category = series_item.get('category', '')
            units = series_item.get('units', '')
            api_code = series_item.get('api_code', '')
            api_source = series_item.get('api_source', '')
        
        # Get block assignments
        blocks = series_item.get('blocks', [])
        if not blocks and len(block_names) > 0:
            # Get block assignments from blocks table using helper
            try:
                block_assignments = get_block_assignments_for_config(config_name, client=client)
                series_blocks = block_assignments.get(series_id, [])
                
                # Build blocks array (1 if series is in block, 0 otherwise)
                blocks = [0] * len(block_names)
                for block_name in series_blocks:
                    if block_name in block_names:
                        idx = block_names.index(block_name)
                        blocks[idx] = 1
            except Exception as e:
                logger.warning(f"Could not get block assignments for {series_id}: {e}")
                blocks = [0] * len(block_names)
        
        # Build row
        row = {
            'series_id': series_id,
            'series_name': series_name,
            'frequency': frequency,
            'transformation': transformation,
            'category': category,
            'units': units,
            'api_code': api_code,
            'api_source': api_source,
        }
        
        # Add block columns
        for block_name in block_names:
            if block_name in block_names:
                idx = block_names.index(block_name)
                row[block_name] = blocks[idx] if idx < len(blocks) else 0
        
        csv_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_rows)
    
    # Ensure block columns are in correct order
    column_order = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units', 'api_code', 'api_source'] + block_names
    df = df[[col for col in column_order if col in df.columns]]
    
    # Determine output path
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / 'src' / 'spec' / '001_initial_spec.csv'
    else:
        output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Exported configuration '{config_name}' to {output_path}")
    logger.info(f"  - Series count: {len(df)}")
    logger.info(f"  - Block names: {block_names}")
    
    return output_path


def get_latest_spec_from_db(
    config_name: str = '001-initial-spec',
    client=None
) -> pd.DataFrame:
    """
    Get the latest specification from database as DataFrame.
    
    This is the main function for the DFM module to pull the latest spec
    from the database instead of reading from CSV file.
    
    Parameters
    ----------
    config_name : str
        Name of the model configuration (default: '001-initial-spec')
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: series_id, series_name, frequency, transformation,
        category, units, api_code, api_source, and block columns
        
    Raises
    ------
    ValueError
        If config_name not found in database
    """
    client = ensure_client(client)
    
    # Load model config from database
    config = load_model_config(config_name, client=client)
    if not config:
        raise ValueError(f"Model configuration '{config_name}' not found in database")
    
    config_json = config.get('config_json', {})
    # Get block names from blocks table
    block_names = get_block_names_for_config(config_name, client=client)
    
    # Get series list from config
    series_list = config_json.get('series', [])
    if not series_list:
        raise ValueError(f"No series found in configuration '{config_name}'")
    
    # Get series IDs
    series_ids = [s['series_id'] for s in series_list]
    
    # Get detailed series metadata from database
    series_metadata_df = get_series_metadata_bulk(series_ids, client=client)
    metadata_dict = series_metadata_df.set_index('series_id').to_dict('index')
    
    # Get block assignments from blocks table using helper
    block_assignments_dict = get_block_assignments_for_config(config_name, client=client)
    
    # Convert to block index format for compatibility (series_id -> list of 0/1)
    block_assignments = {}
    for series_id, block_names_list in block_assignments_dict.items():
        blocks = [0] * len(block_names)
        for block_name in block_names_list:
            if block_name in block_names:
                idx = block_names.index(block_name)
                blocks[idx] = 1
        block_assignments[series_id] = blocks
    
    # Build DataFrame rows
    csv_rows = []
    for series_item in series_list:
        series_id = series_item['series_id']
        
        # Get metadata from database or config
        if series_id in metadata_dict:
            meta = metadata_dict[series_id]
            series_name = meta.get('series_name') or series_item.get('series_name', '')
            frequency = meta.get('frequency') or series_item.get('frequency', '')
            transformation = meta.get('transformation') or series_item.get('transformation', '')
            category = meta.get('category') or series_item.get('category', '')
            units = meta.get('units') or series_item.get('units', '')
            api_code = meta.get('api_code') or series_item.get('api_code', '')
            api_source = meta.get('api_source') or series_item.get('api_source', '')
        else:
            series_name = series_item.get('series_name', '')
            frequency = series_item.get('frequency', '')
            transformation = series_item.get('transformation', '')
            category = series_item.get('category', '')
            units = series_item.get('units', '')
            api_code = series_item.get('api_code', '')
            api_source = series_item.get('api_source', '')
        
        # Get blocks
        blocks = block_assignments.get(series_id, [0] * len(block_names))
        
        # Build row
        row = {
            'series_id': series_id,
            'series_name': series_name,
            'frequency': frequency,
            'transformation': transformation,
            'category': category,
            'units': units,
            'api_code': api_code,
            'api_source': api_source,
        }
        
        # Add block columns
        for block_name in block_names:
            idx = block_names.index(block_name) if block_name in block_names else -1
            row[block_name] = blocks[idx] if 0 <= idx < len(blocks) else 0
        
        csv_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_rows)
    
    # Ensure column order
    column_order = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units', 'api_code', 'api_source'] + block_names
    df = df[[col for col in column_order if col in df.columns]]
    
    return df


def sync_csv_from_db(
    config_name: str = '001-initial-spec',
    csv_path: Optional[Path] = None,
    client=None
) -> Path:
    """
    Sync CSV file from database configuration.
    
    This is a convenience function that exports the latest database config
    to the CSV file, keeping it in sync.
    
    Parameters
    ----------
    config_name : str
        Name of the model configuration in database
    csv_path : Path, optional
        Path to CSV file. If None, uses src/spec/001_initial_spec.csv
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Path
        Path to the updated CSV file
    """
    return export_config_to_csv(config_name, csv_path, client)

