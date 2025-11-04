"""FRBNY Nowcasting Framework - Python Implementation

This package implements the Federal Reserve Bank of New York's nowcasting framework
for forecasting macroeconomic variables using dynamic factor models with mixed-frequency data.
"""

__version__ = "0.1.0"

# Import main modules for easy access
from .nowcasting import (
    ModelConfig,
    DataConfig,
    DFMConfig,
    AppConfig,
    load_config,
    load_config_from_yaml,
    load_config_from_excel,
    load_data,
    dfm,
    DFMResult,
    update_nowcast,
    # Deprecated aliases
    ModelSpec,
    load_spec,
)
from .utils import summarize

__all__ = [
    'ModelConfig',
    'DataConfig',
    'DFMConfig',
    'AppConfig',
    'load_config',
    'load_config_from_yaml',
    'load_config_from_excel',
    'load_data',
    'dfm',
    'DFMResult',
    'update_nowcast',
    'summarize',
    # Deprecated aliases
    'ModelSpec',
    'load_spec',
]

