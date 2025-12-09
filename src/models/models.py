"""DFM and DDFM model wrappers for sktime compatibility.

This module provides sktime-compatible forecasters for DFM and DDFM models.
All classes and functions are re-exported from sub-modules.
"""

from .models_forecasters import DFMForecaster, DDFMForecaster
from .models_utils import (
    load_config,
    create_data_module_from_dataframe,
    create_data_module,
    train_model,
    save_to_outputs
)

# For backward compatibility, provide DFM and DDFM as aliases to forecasters
DFM = DFMForecaster
DDFM = DDFMForecaster

__all__ = [
    'DFM',
    'DDFM',
    'DFMForecaster',
    'DDFMForecaster',
    'load_config',
    'create_data_module_from_dataframe',
    'create_data_module',
    'train_model',
    'save_to_outputs'
]
