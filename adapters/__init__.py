"""
Database adapter module for DFM.

This module provides adapters between the generic DFM module and the Supabase database.
All database-specific logic is isolated here, keeping the DFM module generic.
"""

from .adapter_database import (
    load_data_from_db,
    save_nowcast_to_db,
    save_blocks_to_db,
    save_factors_to_db,
    export_data_to_csv,
    upload_model_weights_to_storage,
    download_model_weights_from_storage,
    download_spec_csv_from_storage,
    csv_spec_to_hydra_config,
)

__all__ = [
    'load_data_from_db',
    'save_nowcast_to_db',
    'save_blocks_to_db',
    'save_factors_to_db',
    'export_data_to_csv',
    'upload_model_weights_to_storage',
    'download_model_weights_from_storage',
    'download_spec_csv_from_storage',
    'csv_spec_to_hydra_config',
]
