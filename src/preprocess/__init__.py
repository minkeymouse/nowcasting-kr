"""Preprocessing module for DFM data transformations and scaling.

This module provides transformation utilities and data reading functions.
Users should provide their own sktime transformers to DFMDataModule.
"""

from .utils import (
    # Data reading utilities
    read_data,
    TimeIndex,
    parse_timestamp,
    # Transformer factory
    create_transformer_from_config,
    # Frequency utilities
    get_periods_per_year,
    get_annual_factor,
)
from .transformations import (
    # Transformation functions
    identity_transform,
    log_transform,
    pch_transform,
    pc1_transform,
    pca_transform,
    cch_transform,
    cca_transform,
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
)
from .sktime import check_sktime_available

# Optional sktime imports (may be None if sktime is not installed)
try:
    from .sktime import (
        ColumnTransformer,
        TransformerPipeline,
        FunctionTransformer,
        StandardScaler,
    )
    _has_sktime_imports = True
except ImportError:
    _has_sktime_imports = False
    ColumnTransformer = None
    TransformerPipeline = None
    FunctionTransformer = None
    StandardScaler = None

__all__ = [
    # Frequency utilities
    'get_periods_per_year',
    'get_annual_factor',
    # Transformation functions
    'identity_transform',
    'log_transform',
    'pch_transform',
    'pc1_transform',
    'pca_transform',
    'cch_transform',
    'cca_transform',
    'make_pch_transformer',
    'make_pc1_transformer',
    'make_pca_transformer',
    'make_cch_transformer',
    'make_cca_transformer',
    # Data reading
    'read_data',
    'TimeIndex',
    'parse_timestamp',
    # Transformer factory
    'create_transformer_from_config',
    # Sktime utilities
    'check_sktime_available',
    'ColumnTransformer',
    'TransformerPipeline',
    'FunctionTransformer',
    'StandardScaler',
]

