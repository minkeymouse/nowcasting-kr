"""Training modules for different model types.

This package provides training functionality separated by model type:
- train_sktime: ARIMA, VAR, TFT, LSTM, Chronos
- train_dfm_python: DFM, DDFM
- train_common: Unified interface and common functions
"""

from .train_common import (
    train,
    train_model,
    compare_models,
    compare_models_by_config
)
from .train_sktime import train_sktime_model
from .train_dfm_python import train_dfm_python_model

__all__ = [
    'train',
    'train_model',
    'compare_models',
    'compare_models_by_config',
    'train_sktime_model',
    'train_dfm_python_model',
]
