"""Nowcasting simulation helper functions for backtesting.

DEPRECATED: This module has been consolidated into src.infer.
All functions are now available from src.infer:
- mask_recent_observations
- create_nowcasting_splits
- simulate_nowcasting_evaluation

This module is kept for backward compatibility and re-exports from src.infer.
"""

import warnings
warnings.warn(
    "src.nowcasting is deprecated. Import from src.infer instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from infer.py for backward compatibility
from .infer import (
    mask_recent_observations,
    create_nowcasting_splits,
    simulate_nowcasting_evaluation
)

__all__ = [
    'mask_recent_observations',
    'create_nowcasting_splits',
    'simulate_nowcasting_evaluation'
]
