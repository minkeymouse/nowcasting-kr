"""Custom transformation functions for DFM.

DEPRECATED: This module has been consolidated into src.preprocess.utils.
All functions are now available from src.preprocess.utils:
- pch_transform, pc1_transform, pca_transform, cch_transform, cca_transform
- log_transform, identity_transform, cha_transform_func
- make_pch_transformer, make_pc1_transformer, make_pca_transformer
- make_cch_transformer, make_cca_transformer, make_cha_transformer
- IndexPreservingColumnEnsembleTransformer
- FREQ_TO_LAG_YOY, FREQ_TO_LAG_STEP

This module is kept for backward compatibility and re-exports from src.preprocess.utils.
"""

import warnings
warnings.warn(
    "src.preprocess.transformations is deprecated. Import from src.preprocess.utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from utils.py for backward compatibility
from .utils import (
    pch_transform,
    pc1_transform,
    pca_transform,
    cch_transform,
    cca_transform,
    log_transform,
    identity_transform,
    cha_transform_func,
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
    make_cha_transformer,
    IndexPreservingColumnEnsembleTransformer,
    FREQ_TO_LAG_YOY,
    FREQ_TO_LAG_STEP,
)

__all__ = [
    'pch_transform',
    'pc1_transform',
    'pca_transform',
    'cch_transform',
    'cca_transform',
    'log_transform',
    'identity_transform',
    'cha_transform_func',
    'make_pch_transformer',
    'make_pc1_transformer',
    'make_pca_transformer',
    'make_cch_transformer',
    'make_cca_transformer',
    'make_cha_transformer',
    'IndexPreservingColumnEnsembleTransformer',
    'FREQ_TO_LAG_YOY',
    'FREQ_TO_LAG_STEP',
]
