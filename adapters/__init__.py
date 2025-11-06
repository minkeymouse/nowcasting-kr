"""Adapters for bridging generic DFM module with application-specific systems.

This package contains adapters that connect the generic DFM module (src/nowcasting)
with application-specific infrastructure like databases, APIs, etc.
"""

from .database import (
    load_data_from_db,
    save_nowcast_to_db,
    save_blocks_to_db,
)

__all__ = [
    'load_data_from_db',
    'save_nowcast_to_db',
    'save_blocks_to_db',
]

