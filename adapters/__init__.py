"""
Database adapter module for DFM.

This module provides adapters between the generic DFM module and the Supabase database.
All database-specific logic is isolated here, keeping the DFM module generic.
"""

from .database import (
    load_data_from_db,
    save_nowcast_to_db,
    save_blocks_to_db,
    export_data_to_csv
)

__all__ = [
    'load_data_from_db',
    'save_nowcast_to_db',
    'save_blocks_to_db',
    'export_data_to_csv',
]
